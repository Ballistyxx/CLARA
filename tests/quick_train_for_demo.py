#!/usr/bin/env python3
"""
Quick training script to create a model for demonstrating the updated run_model.py
"""

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from analog_layout_env import AnalogLayoutEnv
import os


def quick_train():
    """Train a simple model quickly for demonstration."""
    print("Quick training for grid-agnostic model demonstration...")
    
    # Create environment
    def make_env():
        return AnalogLayoutEnv(grid_size=16, max_components=6)
    
    env = make_vec_env(make_env, n_envs=2)
    
    # Create PPO model
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=1e-3,
        n_steps=128,
        batch_size=32,
        n_epochs=4,
        verbose=1
    )
    
    print("Training for 2K timesteps...")
    model.learn(total_timesteps=2000)
    
    # Save model
    os.makedirs("./logs", exist_ok=True)
    model_path = "./logs/clara_final_model"
    model.save(model_path)
    
    print(f"Model saved to {model_path}.zip")
    print("Now you can run: python run_model.py")
    
    env.close()


if __name__ == "__main__":
    quick_train()