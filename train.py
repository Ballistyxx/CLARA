import os
import numpy as np
import torch
from typing import Dict, Any
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
import wandb
from datetime import datetime

from analog_layout_env import AnalogLayoutEnv
from reward import AdaptiveRewardCalculator
import networkx as nx


class AnalogLayoutEnvWrapper(AnalogLayoutEnv):
    """Wrapper to integrate adaptive reward calculator."""
    
    def __init__(self, *args, **kwargs):
        self.reward_calculator = AdaptiveRewardCalculator()
        self.episode_count = 0
        super().__init__(*args, **kwargs)
    
    def step(self, action):
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
    
    def reset(self, **kwargs):
        self.episode_count += 1
        if hasattr(self, 'reward_calculator'):
            self.reward_calculator.update_weights_for_episode(self.episode_count)
        return super().reset(**kwargs)


class TensorBoardCallback(BaseCallback):
    """Custom callback for logging training metrics."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = []
        
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
                    
                    # Calculate success rate (all components placed)
                    success = np.sum(env.placed_mask[:env.num_components]) == env.num_components
                    self.success_rate.append(success)
                    
                    if len(self.success_rate) >= 100:
                        recent_success_rate = np.mean(self.success_rate[-100:])
                        self.logger.record("metrics/success_rate_100", recent_success_rate)
        
        return True


class WandbCallback(BaseCallback):
    """Callback for Weights & Biases logging."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        if self.num_timesteps % 1000 == 0:  # Log every 1000 steps
            metrics = {}
            
            # Add training metrics
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                for key in self.model.logger.name_to_value:
                    metrics[key] = self.model.logger.name_to_value[key]
            
            wandb.log(metrics, step=self.num_timesteps)
        
        return True


class CurriculumCallback(BaseCallback):
    """Callback to implement curriculum learning."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.curriculum_stage = 0
        self.stage_transitions = [10000, 50000, 100000]  # Timesteps for stage transitions
        
    def _on_step(self) -> bool:
        # Check for curriculum stage transitions
        current_timesteps = self.num_timesteps
        
        for i, transition in enumerate(self.stage_transitions):
            if current_timesteps >= transition and self.curriculum_stage <= i:
                self.curriculum_stage = i + 1
                self._update_curriculum_stage()
                if self.verbose > 0:
                    print(f"Advancing to curriculum stage {self.curriculum_stage}")
        
        return True
    
    def _update_curriculum_stage(self):
        """Update environment parameters based on curriculum stage."""
        for env in self.training_env.envs:
            if hasattr(env, 'set_difficulty'):
                env.set_difficulty(self.curriculum_stage)


def create_sample_circuits(num_circuits: int = 50) -> list:
    """Create a set of sample analog circuits for training."""
    circuits = []
    
    for i in range(num_circuits):
        if i < 15:  # Simple circuits
            num_components = np.random.randint(3, 5)
        elif i < 35:  # Medium circuits
            num_components = np.random.randint(5, 8)
        else:  # Complex circuits
            num_components = np.random.randint(8, 12)
        
        G = nx.Graph()
        
        # Add components with different types
        component_types = [0, 1, 2]  # MOSFET, Resistor, Capacitor
        for j in range(num_components):
            comp_type = np.random.choice(component_types)
            width = 1 if comp_type == 1 else np.random.randint(1, 3)  # Resistors are usually thin
            height = 1 if comp_type == 1 else np.random.randint(1, 3)
            
            G.add_node(j, 
                      component_type=comp_type,
                      width=width,
                      height=height,
                      matched_component=-1)
        
        # Add edges based on circuit patterns
        if num_components >= 4:
            # Create some common analog patterns
            if np.random.random() < 0.3:  # Differential pair pattern
                G.add_edge(0, 1)  # Matched transistors
                G.add_edge(0, 2)  # Common connection
                G.add_edge(1, 2)
                G.nodes[0]['matched_component'] = 1
                G.nodes[1]['matched_component'] = 0
            
            # Add some random connections
            for _ in range(min(num_components, np.random.randint(2, num_components + 1))):
                u, v = np.random.choice(num_components, 2, replace=False)
                G.add_edge(u, v)
        
        circuits.append(G)
    
    return circuits


def setup_training_environment(config: Dict[str, Any]) -> VecEnv:
    """Setup the training environment with curriculum learning."""
    
    def make_env():
        return AnalogLayoutEnvWrapper(
            grid_size=config['grid_size'],
            max_components=config['max_components']
        )
    
    # Create vectorized environment
    env = make_vec_env(make_env, n_envs=config['n_envs'])
    
    return env


def main():
    """Main training function."""
    
    # Configuration
    config = {
        'grid_size': 15,      # Smaller grid for faster training
        'max_components': 6,   # Fewer components initially
        'n_envs': 4,          # Fewer parallel envs to reduce CPU load
        'total_timesteps': 100000,  # Shorter training for testing
        'learning_rate': 1e-4,     # Lower learning rate for stability
        'batch_size': 128,         # Larger batch for stability
        'n_steps': 1024,          # Smaller rollout buffer
        'n_epochs': 4,            # Fewer epochs per update
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.1,        # Tighter clipping
        'ent_coef': 0.01,
        'vf_coef': 0.25,         # Lower value function coefficient
        'max_grad_norm': 0.5,
        'seed': 42
    }
    
    # Set random seeds
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    # Initialize Weights & Biases (optional)
    use_wandb = os.getenv('USE_WANDB', 'false').lower() == 'true'
    if use_wandb:
        wandb.init(
            project="clara-analog-layout",
            config=config,
            name=f"clara_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Create training environment
    env = setup_training_environment(config)
    
    # Create evaluation environment
    eval_env = setup_training_environment({**config, 'n_envs': 1})
    
    # Initialize PPO - use standard policy first, custom policy can be added later
    policy = "MultiInputPolicy"  # Use standard policy for now
    print("Using standard MultiInputPolicy (custom policy can be integrated later)")
    
    model = PPO(
        policy,
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
    log_dir = f"./logs/clara_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logger
    model.set_logger(configure(log_dir, ["stdout", "tensorboard"]))
    
    # Setup callbacks
    callbacks = [
        TensorBoardCallback(verbose=1),
        CurriculumCallback(verbose=1),
        CheckpointCallback(
            save_freq=10000,
            save_path=log_dir,
            name_prefix="clara_checkpoint"
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=log_dir,
            log_path=log_dir,
            eval_freq=5000,
            deterministic=True,
            render=False,
            n_eval_episodes=5
        )
    ]
    
    if use_wandb:
        callbacks.append(WandbCallback(verbose=1))
    
    print("Starting CLARA training...")
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
        final_model_path = os.path.join(log_dir, "clara_final_model")
        model.save(final_model_path)
        print(f"Training completed! Final model saved to: {final_model_path}")
        
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        model.save(os.path.join(log_dir, "clara_interrupted_model"))
        
    finally:
        env.close()
        eval_env.close()
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()