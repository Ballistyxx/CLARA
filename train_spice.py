#!/usr/bin/env python3
"""
SPICE-integrated training script for CLARA.
Trains the model on circuits parsed from SPICE files.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecEnv
from datetime import datetime

from analog_layout_env import AnalogLayoutEnv
from reward import AdaptiveRewardCalculator
from spice_parser import SpiceParser
import networkx as nx


class SpiceLayoutEnvWrapper(AnalogLayoutEnv):
    """Wrapper that integrates SPICE circuit loading with adaptive rewards."""
    
    def __init__(self, *args, spice_circuits: List[nx.Graph] = None, **kwargs):
        self.spice_circuits = spice_circuits or []
        self.reward_calculator = AdaptiveRewardCalculator()
        self.episode_count = 0
        super().__init__(*args, **kwargs)
    
    def reset(self, **kwargs):
        """Reset with a SPICE circuit if available."""
        self.episode_count += 1
        if hasattr(self, 'reward_calculator'):
            self.reward_calculator.update_weights_for_episode(self.episode_count)
        
        # Use SPICE circuit if available
        if self.spice_circuits:
            # Choose circuit based on episode count for curriculum learning
            circuit_idx = self.episode_count % len(self.spice_circuits)
            selected_circuit = self.spice_circuits[circuit_idx].copy()
            
            # Filter to manageable size for training
            if len(selected_circuit.nodes) > self.max_components:
                # Take subset of nodes (keeping connectivity)
                nodes_to_keep = list(selected_circuit.nodes())[:self.max_components]
                selected_circuit = selected_circuit.subgraph(nodes_to_keep).copy()
                
                # Renumber nodes to be sequential
                mapping = {old_id: new_id for new_id, old_id in enumerate(nodes_to_keep)}
                selected_circuit = nx.relabel_nodes(selected_circuit, mapping)
                
                # Update matched_component references
                for node_id in selected_circuit.nodes():
                    matched = selected_circuit.nodes[node_id].get('matched_component', -1)
                    if matched in mapping:
                        selected_circuit.nodes[node_id]['matched_component'] = mapping[matched]
                    else:
                        selected_circuit.nodes[node_id]['matched_component'] = -1
            
            return super().reset(circuit_graph=selected_circuit, **kwargs)
        else:
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


def load_spice_circuits(spice_dir: str = "./schematics") -> List[nx.Graph]:
    """Load all SPICE files from a directory."""
    spice_dir = Path(spice_dir)
    parser = SpiceParser()
    circuits = []
    
    print(f"📁 Loading SPICE files from {spice_dir}...")
    
    for spice_file in spice_dir.glob("*.spice"):
        try:
            print(f"  Parsing {spice_file.name}...")
            circuit = parser.parse_spice_file(str(spice_file))
            
            # Filter out circuits that are too large or too small
            if 3 <= len(circuit.nodes) <= 20:
                circuits.append(circuit)
                print(f"    ✅ Added circuit with {len(circuit.nodes)} components")
            else:
                print(f"    ⚠️  Skipped circuit ({len(circuit.nodes)} components - outside range 3-20)")
                
        except Exception as e:
            print(f"    ❌ Failed to parse {spice_file.name}: {e}")
    
    print(f"📊 Loaded {len(circuits)} SPICE circuits")
    return circuits


def setup_spice_training_environment(config: Dict[str, Any], spice_circuits: List[nx.Graph]) -> VecEnv:
    """Setup training environment with SPICE circuits."""
    
    def make_env():
        return SpiceLayoutEnvWrapper(
            grid_size=config['grid_size'],
            max_components=config['max_components'],
            spice_circuits=spice_circuits
        )
    
    # Create vectorized environment
    env = make_vec_env(make_env, n_envs=config['n_envs'])
    
    return env


class SpiceTrainingCallback(BaseCallback):
    """Custom callback for SPICE-based training."""
    
    def __init__(self, spice_circuits: List[nx.Graph], verbose=0):
        super().__init__(verbose)
        self.spice_circuits = spice_circuits
        self.episode_rewards = []
        self.circuit_performance = {}  # Track performance per circuit type
        
    def _on_step(self) -> bool:
        # Log SPICE-specific metrics
        for env_idx, done in enumerate(self.locals.get('dones', [])):
            if done:
                env = self.training_env.envs[env_idx]
                if hasattr(env, 'episode_count'):
                    info = self.locals.get('infos', [{}])[env_idx]
                    
                    # Track circuit-specific performance
                    if hasattr(env, 'circuit') and env.circuit:
                        circuit_size = len(env.circuit.nodes)
                        placed = len(env.component_positions)
                        success_rate = placed / circuit_size if circuit_size > 0 else 0
                        
                        if circuit_size not in self.circuit_performance:
                            self.circuit_performance[circuit_size] = []
                        self.circuit_performance[circuit_size].append(success_rate)
                        
                        # Log metrics
                        self.logger.record(f"spice/circuit_size_{circuit_size}_success", success_rate)
                        
                    # Log reward components
                    if 'reward_components' in info:
                        for component, value in info['reward_components'].items():
                            self.logger.record(f"reward/{component}", value)
        
        return True


def main():
    """Main training function for SPICE-integrated CLARA."""
    
    print("🚀 CLARA SPICE Training")
    print("="*50)
    
    # Configuration
    config = {
        'grid_size': 20,           # Larger grid for complex circuits
        'max_components': 15,      # Allow more components for real circuits
        'n_envs': 4,               # Parallel environments
        'total_timesteps': 200000, # Longer training for complex circuits
        'learning_rate': 5e-5,     # Lower learning rate for stability
        'batch_size': 128,         # Larger batch size
        'n_steps': 2048,           # Longer rollouts
        'n_epochs': 4,             # Fewer epochs per update
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.1,         # Tighter clipping
        'ent_coef': 0.01,
        'vf_coef': 0.25,
        'max_grad_norm': 0.5,
        'seed': 42
    }
    
    # Set random seeds
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    # Load SPICE circuits
    spice_circuits = load_spice_circuits("./schematics")
    
    if not spice_circuits:
        print("❌ No SPICE circuits found! Please add .spice files to ./schematics/")
        return 1
    
    print(f"\n📋 Circuit Summary:")
    circuit_sizes = {}
    for circuit in spice_circuits:
        size = len(circuit.nodes)
        circuit_sizes[size] = circuit_sizes.get(size, 0) + 1
    
    for size, count in sorted(circuit_sizes.items()):
        print(f"  {count} circuits with {size} components")
    
    # Create training environment
    print(f"\n🏗️  Setting up training environment...")
    env = setup_spice_training_environment(config, spice_circuits)
    
    # Create evaluation environment
    eval_env = setup_spice_training_environment({**config, 'n_envs': 1}, spice_circuits)
    
    # Initialize PPO with standard policy
    print(f"🤖 Initializing PPO model...")
    model = PPO(
        "MultiInputPolicy",  # Using standard policy for now
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
        device='auto',
        seed=config['seed']
    )
    
    # Setup logging
    log_dir = f"./logs/clara_spice_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logger
    model.set_logger(configure(log_dir, ["stdout", "tensorboard"]))
    
    # Setup callbacks
    callbacks = [
        SpiceTrainingCallback(spice_circuits, verbose=1),
        CheckpointCallback(
            save_freq=20000,
            save_path=log_dir,
            name_prefix="clara_spice_checkpoint"
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=log_dir,
            log_path=log_dir,
            eval_freq=10000,
            deterministic=True,
            render=False,
            n_eval_episodes=3
        )
    ]
    
    print(f"\n🎓 Starting training...")
    print(f"Configuration: {config}")
    print(f"Logging to: {log_dir}")
    
    # Train the model
    try:
        model.learn(
            total_timesteps=config['total_timesteps'],
            callback=callbacks,
            progress_bar=False
        )
        
        # Save final model
        final_model_path = os.path.join(log_dir, "clara_spice_final_model")
        model.save(final_model_path)
        print(f"\n✅ Training completed! Final model saved to: {final_model_path}")
        
        # Save circuit information
        circuit_info_path = os.path.join(log_dir, "spice_circuits_info.txt")
        with open(circuit_info_path, 'w') as f:
            f.write("SPICE Circuits Used in Training:\n")
            f.write("="*40 + "\n")
            for i, circuit in enumerate(spice_circuits):
                f.write(f"\nCircuit {i}:\n")
                f.write(f"  Components: {len(circuit.nodes)}\n")
                f.write(f"  Connections: {len(circuit.edges)}\n")
                
                # List component types
                component_types = {}
                for node in circuit.nodes():
                    comp_type = circuit.nodes[node].get('component_type', 'unknown')
                    type_names = {0: 'NMOS', 1: 'PMOS', 2: 'R', 3: 'C', 4: 'L', 5: 'I', 6: 'V', 7: 'SUB'}
                    type_name = type_names.get(comp_type, f'Type{comp_type}')
                    component_types[type_name] = component_types.get(type_name, 0) + 1
                
                f.write(f"  Component types: {component_types}\n")
        
        print(f"📋 Circuit information saved to: {circuit_info_path}")
        
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        model.save(os.path.join(log_dir, "clara_spice_interrupted_model"))
        
    finally:
        env.close()
        eval_env.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())