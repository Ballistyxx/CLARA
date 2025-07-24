#!/usr/bin/env python3
"""
Quick test script to validate training setup before running full training.
"""

import numpy as np
from analog_layout_env import AnalogLayoutEnv
from train import AnalogLayoutEnvWrapper


def test_single_episode():
    """Test a single episode to check for issues."""
    print("Testing single episode...")
    
    env = AnalogLayoutEnvWrapper(grid_size=8, max_components=4)
    
    result = env.reset()
    if isinstance(result, tuple):
        obs, info = result
    else:
        obs = result
    print(f"Reset successful. Num components: {env.num_components}")
    print(f"Observation keys: {list(obs.keys())}")
    
    total_reward = 0
    step = 0
    max_steps = 20
    
    while step < max_steps:
        action = env.action_space.sample()
        result = env.step(action)
        
        if len(result) == 5:  # Gymnasium format
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:  # Gym format
            obs, reward, done, info = result
        
        total_reward += reward
        step += 1
        
        print(f"Step {step}: reward={reward:.3f}, done={done}, "
              f"valid_action={info.get('valid_action', True)}, "
              f"placed={len(env.component_positions)}/{env.num_components}")
        
        if done:
            print(f"Episode finished in {step} steps")
            break
    
    if step >= max_steps:
        print(f"Episode reached max steps ({max_steps})")
    
    print(f"Total reward: {total_reward:.3f}")
    print(f"Components placed: {len(env.component_positions)}/{env.num_components}")
    
    # Check reward breakdown
    if 'reward_components' in info:
        print("Reward breakdown:")
        for component, value in info['reward_components'].items():
            if value != 0:
                print(f"  {component}: {value:.3f}")
    
    return total_reward, step, len(env.component_positions)


def test_multiple_episodes(num_episodes=5):
    """Test multiple episodes for consistency."""
    print(f"\nTesting {num_episodes} episodes...")
    
    env = AnalogLayoutEnvWrapper(grid_size=8, max_components=4)
    
    episode_stats = []
    
    for episode in range(num_episodes):
        result = env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
        total_reward = 0
        step = 0
        max_steps = 20
        
        while step < max_steps:
            action = env.action_space.sample()
            result = env.step(action)
            
            if len(result) == 5:  # Gymnasium format
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:  # Gym format
                obs, reward, done, info = result
            
            total_reward += reward
            step += 1
            
            if done:
                break
        
        components_placed = len(env.component_positions)
        episode_stats.append((total_reward, step, components_placed))
        
        print(f"Episode {episode + 1}: reward={total_reward:.3f}, "
              f"steps={step}, placed={components_placed}/{env.num_components}")
    
    # Summary statistics
    rewards = [stat[0] for stat in episode_stats]
    steps = [stat[1] for stat in episode_stats]
    placements = [stat[2] for stat in episode_stats]
    
    print(f"\nSummary over {num_episodes} episodes:")
    print(f"Avg reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"Avg steps: {np.mean(steps):.1f} ± {np.std(steps):.1f}")
    print(f"Avg components placed: {np.mean(placements):.1f} ± {np.std(placements):.1f}")
    print(f"Success rate (all components): {sum(p >= env.num_components for p in placements) / num_episodes * 100:.1f}%")


def test_action_space():
    """Test action space validity."""
    print("\nTesting action space...")
    
    env = AnalogLayoutEnvWrapper(grid_size=8, max_components=4)
    result = env.reset()
    if isinstance(result, tuple):
        obs, info = result
    else:
        obs = result
    
    print(f"Action space: {env.action_space}")
    print(f"Action space shape: {env.action_space.nvec}")
    
    # Generate some sample actions
    for i in range(5):
        action = env.action_space.sample()
        print(f"Sample action {i + 1}: {action} "
              f"(target_comp={action[0]}, relation={action[1]}, orient={action[2]})")


def main():
    """Run all tests."""
    print("CLARA Training Test Suite")
    print("=" * 50)
    
    try:
        test_action_space()
        test_single_episode()
        test_multiple_episodes(3)
        
        print("\nAll tests completed successfully!")
        print("The environment appears to be working correctly.")
        print("You can now run the full training with: python3 train.py")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())