#!/usr/bin/env python3
"""
CLARA training script integrated with real SPICE circuits.
Trains RL models on actual analog circuits from the programmable PLL subcircuits.
"""

import os
import numpy as np
import torch
from typing import Dict, Any, List
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
import wandb
from datetime import datetime
import random
from pathlib import Path

from analog_layout_env import AnalogLayoutEnv
from reward import AdaptiveRewardCalculator
from enhanced_spice_parser import EnhancedSpiceParser, parse_multiple_spice_files
import networkx as nx


class SpiceCircuitManager:
    """Manages SPICE circuits for training."""
    
    def __init__(self, spice_directory: str):
        self.spice_directory = spice_directory
        self.parser = EnhancedSpiceParser()
        self.circuits = {}
        self.suitable_circuits = {}
        self.load_circuits()

        
    
    def load_circuits(self):
        """Load and parse all SPICE circuits."""
        print(f"Loading SPICE circuits from {self.spice_directory}")
        
        # Parse all SPICE files
        results = parse_multiple_spice_files(self.spice_directory)
        
        # Filter circuits suitable for RL training
        for filename, data in results.items():
            if 'error' not in data:
                self.circuits[filename] = data
                
                # Only use circuits with reasonable size for RL training
                if 3 <= data['num_components'] <= 2000:
                    self.suitable_circuits[filename] = data
                    print(f"{filename}: {data['num_components']} components ({data['subcircuit_name']})")
                else:
                    print(f"{filename}: {data['num_components']} components (skipped - too large/small)")
            else:
                print(f"{filename}: Parse error")
        
        print(f"Loaded {len(self.circuits)} total circuits, {len(self.suitable_circuits)} suitable for training")
        
        if len(self.suitable_circuits) == 0:
            raise ValueError("No suitable circuits found for training!")
    
    def get_random_circuit(self) -> nx.Graph:
        """Get a random circuit for training."""
        circuit_name = random.choice(list(self.suitable_circuits.keys()))
        circuit_data = self.suitable_circuits[circuit_name]
        
        # Convert to NetworkX graph compatible with CLARA
        return self.convert_to_networkx_graph(circuit_data)
    
    def convert_to_networkx_graph(self, circuit_data: dict) -> nx.Graph:
        """Convert parsed SPICE data to NetworkX graph compatible with CLARA."""
        G = nx.Graph()
        
        components = circuit_data['components']
        
        # Add nodes with CLARA-compatible attributes
        for i, comp in enumerate(components):
            # Map SPICE component types to CLARA types
            clara_type = comp['type_value']  # Already mapped: NMOS=0, PMOS=1, etc.
            
            # Use normalized component dimensions to prevent grid overflow
            # Apply square root scaling to reduce extreme sizes
            raw_width = max(0.1, comp['width'])
            raw_length = max(0.1, comp['length'])
            
            # Square root scaling with reasonable bounds
            width = max(1, min(8, int(np.sqrt(raw_width) + 0.5)))
            height = max(1, min(8, int(np.sqrt(raw_length) + 0.5)))
            
            # Find matched components for symmetry
            matched_component = self._find_matched_component(comp, components, i)
            
            G.add_node(i,
                      component_type=clara_type,
                      width=width,
                      height=height,
                      matched_component=matched_component,
                      device_model=comp['device_model'],
                      spice_name=comp['name'])
        
        # Add edges from connectivity matrix
        adjacency_matrix = circuit_data['connectivity_matrix']
        for i in range(len(adjacency_matrix)):
            for j in range(i + 1, len(adjacency_matrix)):
                if adjacency_matrix[i][j] == 1:
                    G.add_edge(i, j)
        
        return G
    
    def _find_matched_component(self, comp, all_components, current_index):
        """Find matched component for differential pairs, current mirrors, etc."""
        # Look for components with same type and similar dimensions
        for j, other_comp in enumerate(all_components):
            if (j != current_index and 
                other_comp['type_value'] == comp['type_value'] and
                abs(other_comp['width'] - comp['width']) < 0.1 and
                other_comp['device_model'] == comp['device_model']):
                return j
        return -1
    
    def get_circuit_stats(self):
        """Get statistics about loaded circuits."""
        if not self.suitable_circuits:
            return {}
        
        sizes = [data['num_components'] for data in self.suitable_circuits.values()]
        
        # Component type distribution
        type_counts = {}
        for circuit_data in self.suitable_circuits.values():
            for comp in circuit_data['components']:
                comp_type = comp['type']
                type_counts[comp_type] = type_counts.get(comp_type, 0) + 1
        
        return {
            'total_circuits': len(self.suitable_circuits),
            'component_range': [min(sizes), max(sizes)],
            'avg_components': np.mean(sizes),
            'component_types': type_counts,
            'circuit_names': list(self.suitable_circuits.keys())
        }


class AnalogLayoutSpiceEnvWrapper(AnalogLayoutEnv):
    """Environment wrapper that uses real SPICE circuits for training."""
    
    def __init__(self, circuit_manager: SpiceCircuitManager, *args, **kwargs):
        self.circuit_manager = circuit_manager
        self.reward_calculator = AdaptiveRewardCalculator()
        self.episode_count = 0
        self.current_circuit_name = "unknown"
        # Enable action masking by default for SPICE circuits
        if 'enable_action_masking' not in kwargs:
            kwargs['enable_action_masking'] = True
        super().__init__(*args, **kwargs)
    
    def reset(self, **kwargs):
        """Reset environment with a random SPICE circuit."""
        self.episode_count += 1
        
        # Get a random real circuit instead of generating one
        if 'circuit_graph' not in kwargs:
            circuit_graph = self.circuit_manager.get_random_circuit()
            kwargs['circuit_graph'] = circuit_graph
            
            # Store circuit info for logging
            self.current_circuit_name = f"spice_circuit_{self.episode_count}"
        
        # Update reward calculator weights
        if hasattr(self, 'reward_calculator'):
            self.reward_calculator.update_weights_for_episode(self.episode_count)
        
        return super().reset(**kwargs)
    
    def step(self, action):
        """Step with adaptive reward calculation."""
        result = super().step(action)
        if len(result) == 5:  # Gymnasium format
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:  # Gym format
            obs, reward, done, info = result
        
        # Use adaptive reward calculator
        if hasattr(self, 'reward_calculator'):
            total_reward, reward_components = self.reward_calculator.calculate_total_reward(
                circuit=self.circuit,
                component_positions=self.component_positions,
                grid_size=self.grid_size,
                num_components=self.num_components,
                placed_count=np.sum(self.placed_mask[:self.num_components]),
                is_valid_action=info.get('valid_action', True),
                is_valid_placement=True
            )
            
            info['reward_components'] = {
                'symmetry': reward_components.symmetry,
                'compactness': reward_components.compactness,
                'connectivity': reward_components.connectivity,
                'completion': reward_components.completion,
                'placement_step': reward_components.placement_step
            }
            info['circuit_name'] = self.current_circuit_name
            
            # Return in the same format as received
            if len(result) == 5:  # Gymnasium format
                return obs, total_reward, done, False, info  # terminated, truncated
            else:  # Gym format
                return obs, total_reward, done, info
        
        # Return in the same format as received
        if len(result) == 5:  # Gymnasium format
            return obs, reward, done, False, info
        else:  # Gym format
            return obs, reward, done, info


class SpiceTensorBoardCallback(BaseCallback):
    """Custom callback for logging SPICE training metrics."""
    
    def __init__(self, circuit_manager: SpiceCircuitManager, verbose=0):
        super().__init__(verbose)
        self.circuit_manager = circuit_manager
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.circuit_success = {}
        
    def _on_step(self) -> bool:
        # Log episode metrics when episodes end
        for env_idx, done in enumerate(self.locals.get('dones', [])):
            if done:
                env = self.training_env.envs[env_idx]
                if hasattr(env, 'episode_count'):
                    info = self.locals.get('infos', [{}])[env_idx]
                    
                    # Log reward components
                    if 'reward_components' in info:
                        for component, value in info['reward_components'].items():
                            self.logger.record(f"reward/{component}", value)
                    
                    # Log environment metrics
                    self.logger.record("env/episode_count", env.episode_count)
                    self.logger.record("env/components_placed", 
                                     np.sum(env.placed_mask[:env.num_components]))
                    self.logger.record("env/total_components", env.num_components)
                    
                    # Log SPICE-specific metrics
                    if 'circuit_name' in info:
                        circuit_name = info['circuit_name']
                        success = np.sum(env.placed_mask[:env.num_components]) == env.num_components
                        
                        if circuit_name not in self.circuit_success:
                            self.circuit_success[circuit_name] = []
                        self.circuit_success[circuit_name].append(success)
                        
                        # Log success rate for each circuit type
                        if len(self.circuit_success[circuit_name]) >= 10:  # After 10 episodes
                            success_rate = np.mean(self.circuit_success[circuit_name][-10:])
                            self.logger.record(f"circuit_success/{circuit_name}", success_rate)
        
        return True


def setup_spice_training_environment(config: Dict[str, Any], circuit_manager: SpiceCircuitManager) -> VecEnv:
    """Setup the training environment with SPICE circuits."""
    
    def make_env():
        return AnalogLayoutSpiceEnvWrapper(
            circuit_manager=circuit_manager,
            grid_size=config['grid_size'],
            max_components=config['max_components']
        )
    
    # Create vectorized environment
    env = make_vec_env(make_env, n_envs=config['n_envs'])
    
    return env


def main():
    """Main training function with SPICE circuit integration."""
    
    # Configuration - optimized for real circuits
    config = {
        'grid_size': 20,      # Adequate size for real circuits
        'max_components': 100,  # Support larger real circuits
        'n_envs': 4,          # Parallel environments
        'total_timesteps': 200000,  # Longer training for real circuits
        'learning_rate': 5e-4,      # Moderate learning rate
        'batch_size': 128,          # Good batch size
        'n_steps': 512,            # Rollout buffer size
        'n_epochs': 4,             # Training epochs per update
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.1,
        'ent_coef': 0.01,
        'vf_coef': 0.25,
        'max_grad_norm': 0.5,
        'seed': 42
    }
    
    print("CLARA Training with Real SPICE Circuits")
    print("=" * 60)
    
    # Set random seeds
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    # Initialize SPICE circuit manager
    spice_directory = "/home/eli/Documents/Internship/CLARA/data/netlists/programmable_pll_subcircuits"
    circuit_manager = SpiceCircuitManager(spice_directory)
    
    # Print circuit statistics
    stats = circuit_manager.get_circuit_stats()
    print(f"\nSPICE Circuit Training Data:")
    print(f"   Total circuits: {stats['total_circuits']}")
    print(f"   Component range: {stats['component_range'][0]}-{stats['component_range'][1]}")
    print(f"   Average components: {stats['avg_components']:.1f}")
    print(f"   Component types: {stats['component_types']}")
    print(f"   Available circuits: {', '.join(stats['circuit_names'][:5])}...")
    
    # Initialize Weights & Biases (optional)
    use_wandb = os.getenv('USE_WANDB', 'false').lower() == 'true'
    if use_wandb:
        wandb.init(
            project="clara-spice-circuits",
            config=config,
            name=f"clara_spice_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Create training environment with SPICE circuits
    print(f"\nSetting up training environment...")
    env = setup_spice_training_environment(config, circuit_manager)
    
    # Create evaluation environment
    eval_env = setup_spice_training_environment({**config, 'n_envs': 1}, circuit_manager)
    
    # Initialize PPO
    print(f"Initializing PPO model...")
    model = PPO(
        "MultiInputPolicy",  # Use standard policy compatible with grid-agnostic format
        env,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_range=config['clip_range'],
        ent_coef=config['ent_coef'],
        vf_coef=config['vf_coef'],
        max_grad_norm=config['max_grad_norm'],
        verbose=1,
        device='auto'
    )
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/clara_spice_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logger
    model.set_logger(configure(log_dir, ["stdout", "tensorboard"]))
    
    # Setup callbacks
    callbacks = []
    
    # Custom callback for SPICE metrics
    tb_callback = SpiceTensorBoardCallback(circuit_manager, verbose=1)
    callbacks.append(tb_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,  # Save every 20k steps
        save_path=log_dir,
        name_prefix="clara_spice_checkpoint"
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10000,  # Evaluate every 10k steps
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1
    )
    callbacks.append(eval_callback)
    
    print(f"\nStarting training...")
    print(f"   Total timesteps: {config['total_timesteps']:,}")
    print(f"   Log directory: {log_dir}")
    print(f"   Using real circuits from: {len(stats['circuit_names'])} SPICE files")
    
    # Start training
    try:
        model.learn(
            total_timesteps=config['total_timesteps'],
            callback=callbacks,
            progress_bar=False  # Disable progress bar to avoid dependency issues
        )
        
        # Save final model
        final_model_path = f"{log_dir}/clara_spice_final_model"
        model.save(final_model_path)
        
        print(f"\nTraining completed successfully!")
        print(f"   Final model saved: {final_model_path}.zip")
        print(f"   Training logs: {log_dir}")
        
        # Test the trained model
        print(f"\nTesting trained model...")
        test_model_on_spice_circuits(model, circuit_manager, config)
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user")
        # Save current model
        interrupted_model_path = f"{log_dir}/clara_spice_interrupted_model"
        model.save(interrupted_model_path)
        print(f"   Model saved: {interrupted_model_path}.zip")
    
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        env.close()
        eval_env.close()
        
        if use_wandb:
            wandb.finish()


def test_model_on_spice_circuits(model, circuit_manager: SpiceCircuitManager, config: Dict[str, Any]):
    """Test the trained model on different SPICE circuits."""
    print(f"Testing model on {len(circuit_manager.suitable_circuits)} different circuits...")
    
    results = {}
    
    for circuit_name, circuit_data in circuit_manager.suitable_circuits.items():
        # Create test environment
        test_env = AnalogLayoutSpiceEnvWrapper(
            circuit_manager=circuit_manager,
            grid_size=config['grid_size'],
            max_components=config['max_components']
        )
        
        # Force use of specific circuit
        circuit_graph = circuit_manager.convert_to_networkx_graph(circuit_data)
        
        # Run test episodes
        episode_rewards = []
        success_count = 0
        
        for episode in range(5):  # Test 5 episodes per circuit
            result = test_env.reset(circuit_graph=circuit_graph)
            obs = result[0] if isinstance(result, tuple) else result
            
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 50:
                action, _ = model.predict(obs, deterministic=True)
                result = test_env.step(action)
                
                if len(result) == 5:  # Gymnasium format
                    obs, reward, terminated, truncated, info = result
                    done = terminated or truncated
                else:  # Gym format
                    obs, reward, done, info = result
                
                episode_reward += reward
                steps += 1
            
            episode_rewards.append(episode_reward)
            
            # Check success (all components placed)
            if np.sum(test_env.placed_mask[:test_env.num_components]) == test_env.num_components:
                success_count += 1
        
        avg_reward = np.mean(episode_rewards)
        success_rate = success_count / 5
        
        results[circuit_name] = {
            'avg_reward': avg_reward,
            'success_rate': success_rate,
            'num_components': circuit_data['num_components']
        }
        
        print(f"   {circuit_name}: {avg_reward:.2f} reward, {success_rate:.1%} success")
        
        test_env.close()
    
    # Summary
    avg_success = np.mean([r['success_rate'] for r in results.values()])
    avg_reward = np.mean([r['avg_reward'] for r in results.values()])
    
    print(f"\nOverall Test Results:")
    print(f"   Average success rate: {avg_success:.1%}")
    print(f"   Average reward: {avg_reward:.2f}")
    
    return results


if __name__ == "__main__":
    main()