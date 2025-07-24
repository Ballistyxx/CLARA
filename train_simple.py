#!/usr/bin/env python3
"""
Simplified training script for CLARA that handles dependency issues gracefully.
"""

import os
import sys
import numpy as np
from typing import Dict, Any


def check_dependencies():
    """Check if required dependencies are available."""
    required = {
        'torch': 'PyTorch',
        'stable_baselines3': 'Stable-Baselines3',
        'networkx': 'NetworkX',
        'matplotlib': 'Matplotlib'
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"{name} available")
        except ImportError:
            missing.append(name)
            print(f"{name} missing")
    
    return len(missing) == 0, missing


def test_environment():
    """Test environment creation."""
    try:
        from analog_layout_env import AnalogLayoutEnv
        
        # Create simple environment
        env = AnalogLayoutEnv(grid_size=10, max_components=5)
        print("Environment created successfully")
        
        # Test basic functionality
        obs = env.reset()
        print(f"Reset successful. Observation keys: {list(obs.keys())}")
        
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step successful. Reward: {reward:.3f}, Done: {done}")
        
        return True
        
    except Exception as e:
        print(f"Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_dummy_training():
    """Create a dummy training loop for testing."""
    try:
        from analog_layout_env import AnalogLayoutEnv
        
        print("\nRunning dummy training loop...")
        
        env = AnalogLayoutEnv(grid_size=8, max_components=4)
        
        for episode in range(5):
            obs = env.reset()
            total_reward = 0
            steps = 0
            
            print(f"\nEpisode {episode + 1}:")
            
            while steps < 20:  # Max steps per episode
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            print(f"Total reward: {total_reward:.3f}, Steps: {steps}")
            
            if hasattr(env, 'component_positions'):
                print(f"Components placed: {len(env.component_positions)}")
        
        print("Dummy training completed successfully!")
        return True
        
    except Exception as e:
        print(f"Dummy training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def try_full_training():
    """Attempt full SB3 training if possible."""
    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from analog_layout_env import AnalogLayoutEnv
        
        print("\nAttempting full SB3 training...")
        
        # Simple environment creation without custom policy for now
        def make_env():
            return AnalogLayoutEnv(grid_size=10, max_components=5)
        
        # Create vectorized environment
        env = make_vec_env(make_env, n_envs=2)
        print("Vectorized environment created")
        
        # Create simple PPO model (without custom policy)
        model = PPO("MultiInputPolicy", env, verbose=1, 
                   learning_rate=1e-4, n_steps=512, batch_size=64)
        print("PPO model created")
        
        # Short training run
        print("Training for 5000 steps...")
        model.learn(total_timesteps=5000)
        
        # Save model
        model.save("clara_simple_model")
        print("Model saved as 'clara_simple_model'")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"Full training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    print("CLARA Simple Training Test")
    print("="*40)
    
    # Check dependencies
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Please install them first using:")
        print("pip install torch stable-baselines3 networkx matplotlib")
        return 1
    
    # Test environment
    if not test_environment():
        print("\nEnvironment test failed. Cannot proceed with training.")
        return 1
    
    # Run dummy training
    if not create_dummy_training():
        print("\nDummy training failed.")
        return 1
    
    # Try full training if possible
    try_full_training()
    
    print("\nTraining test completed!")
    print("\nTo run the full CLARA training:")
    print("1. Ensure all dependencies are installed")
    print("2. Run: python3 train.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())