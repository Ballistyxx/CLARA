#!/usr/bin/env python3
"""
Script to load and run the trained CLARA model for inference.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from analog_layout_env import AnalogLayoutEnv
from visualize import AnalogLayoutVisualizer
from data.circuit_generator import AnalogCircuitGenerator
import argparse


def load_model(model_path: str):
    """Load the trained CLARA model."""
    if not os.path.exists(model_path + '.zip'):
        raise FileNotFoundError(f"Model not found at {model_path}.zip")
    
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    print("Model loaded successfully!")
    return model


def run_single_episode(model, env, deterministic=True, render=False):
    """Run a single episode with the trained model."""
    result = env.reset()
    if isinstance(result, tuple):
        obs, info = result
    else:
        obs = result
    
    total_reward = 0
    step = 0
    episode_data = {
        'positions': [],
        'rewards': [],
        'actions': [],
        'circuit': env.circuit
    }
    
    if render:
        print(f"\nRunning episode with {env.num_components} components")
        print(f"Circuit nodes: {list(env.circuit.nodes())}")
        print(f"Circuit edges: {list(env.circuit.edges())}")
    
    while step < env.max_steps:
        # Get action from model
        action, _states = model.predict(obs, deterministic=deterministic)
        
        # Store episode data
        episode_data['positions'].append(env.component_positions.copy())
        episode_data['actions'].append(action.copy())
        
        # Take step
        result = env.step(action)
        if len(result) == 5:  # Gymnasium format
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:  # Gym format
            obs, reward, done, info = result
        
        total_reward += reward
        step += 1
        
        episode_data['rewards'].append(reward)
        
        if render:
            print(f"Step {step}: action={action}, reward={reward:.3f}, "
                  f"valid_action={info.get('valid_action', True)}, "
                  f"placed={len(env.component_positions)}/{env.num_components}")
            
            if 'reward_components' in info:
                breakdown = info['reward_components']
                non_zero = {k: v for k, v in breakdown.items() if abs(v) > 0.001}
                if non_zero:
                    print(f"Reward breakdown: {non_zero}")
        
        if done:
            break
    
    # Final positions
    episode_data['positions'].append(env.component_positions.copy())
    
    success = len(env.component_positions) == env.num_components
    
    if render:
        print(f"\nEpisode Summary:")
        print(f"Total reward: {total_reward:.3f}")
        print(f"Steps taken: {step}")
        print(f"Components placed: {len(env.component_positions)}/{env.num_components}")
        print(f"Success: {'Yes' if success else 'No'}")

    return episode_data, total_reward, step, success


def run_multiple_episodes(model, env, num_episodes=5, deterministic=True):
    """Run multiple episodes and collect statistics."""
    print(f"\n  Running {num_episodes} episodes...")
    
    episode_results = []
    total_rewards = []
    steps_taken = []
    success_count = 0
    
    for episode in range(num_episodes):
        episode_data, reward, steps, success = run_single_episode(
            model, env, deterministic=deterministic, render=False
        )
        
        episode_results.append(episode_data)
        total_rewards.append(reward)
        steps_taken.append(steps)
        if success:
            success_count += 1
        
        print(f"Episode {episode + 1}: reward={reward:.3f}, steps={steps}, "
              f"placed={len(episode_data['positions'][-1])}/{env.num_components}, "
              f"success={'Y' if success else 'N'}")
    
    # Statistics
    print(f"\nStatistics over {num_episodes} episodes:")
    print(f"Average reward: {np.mean(total_rewards):.3f} ± {np.std(total_rewards):.3f}")
    print(f"Average steps: {np.mean(steps_taken):.1f} ± {np.std(steps_taken):.1f}")
    print(f"Success rate: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    
    return episode_results, {
        'rewards': total_rewards,
        'steps': steps_taken,
        'success_rate': success_count / num_episodes
    }


def visualize_episode(episode_data, save_path=None):
    """Visualize the final layout from an episode."""
    visualizer = AnalogLayoutVisualizer()
    
    circuit = episode_data['circuit']
    final_positions = episode_data['positions'][-1]
    total_reward = sum(episode_data['rewards'])
    
    title = f"CLARA Layout (Reward: {total_reward:.1f})"
    
    fig = visualizer.visualize_layout(
        circuit=circuit,
        component_positions=final_positions,
        title=title,
        show_connections=True,
        save_path=save_path
    )
    
    return fig


def test_different_circuits(model):
    """Test the model on different circuit types."""
    print("\nTesting model on different circuit types...")
    
    generator = AnalogCircuitGenerator()
    visualizer = AnalogLayoutVisualizer()
    
    # Test different circuit types
    circuit_types = [
        ("Differential Pair", generator.generate_differential_pair),
        ("Current Mirror", generator.generate_current_mirror),
        ("Common Source", generator.generate_common_source_amplifier),
        ("Random Circuit", lambda: generator.generate_random_circuit(3, 6)),
    ]
    
    results = []
    
    for circuit_name, circuit_func in circuit_types:
        print(f"\nTesting {circuit_name}...")
        
        try:
            circuit = circuit_func()
            
            # Skip if too many components
            if len(circuit.nodes) > 6:
                print(f"Skipping - too many components ({len(circuit.nodes)})")
                continue
            
            # Create environment with this specific circuit
            env = AnalogLayoutEnv(grid_size=15, max_components=6)
            env.reset(circuit_graph=circuit)
            
            # Run episode
            episode_data, reward, steps, success = run_single_episode(
                model, env, deterministic=True, render=False
            )
            
            print(f"Reward: {reward:.3f}, Steps: {steps}, Success: {'Yes' if success else 'No'}")
            
            # Visualize
            fig = visualize_episode(episode_data, save_path=f"layout_{circuit_name.lower().replace(' ', '_')}.png")
            plt.close(fig)
            
            results.append({
                'name': circuit_name,
                'reward': reward,
                'steps': steps,
                'success': success,
                'components': len(circuit.nodes)
            })
            
        except Exception as e:
            print(f"Error: {e}")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run trained CLARA model')
    parser.add_argument('--model', '-m', type=str, 
                       default='./logs/clara_20250722_185325/clara_checkpoint_80000_steps',
                       help='Path to trained model (without .zip extension)')
    parser.add_argument('--episodes', '-e', type=int, default=3,
                       help='Number of episodes to run')
    parser.add_argument('--deterministic', '-d', action='store_true',
                       help='Use deterministic actions')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--test-circuits', '-t', action='store_true',
                       help='Test on different circuit types')
    parser.add_argument('--render', '-r', action='store_true',
                       help='Print detailed step-by-step output')
    
    args = parser.parse_args()
    
    try:
        # Load model
        model = load_model(args.model)
        
        # Create environment (same config as training)
        env = AnalogLayoutEnv(grid_size=15, max_components=6)
        
        print(f"Environment: {env.grid_size}×{env.grid_size} grid, max {env.max_components} components")
        
        # Run episodes
        if args.episodes > 0:
            if args.episodes == 1:
                episode_data, reward, steps, success = run_single_episode(
                    model, env, deterministic=args.deterministic, render=args.render
                )
                
                if args.visualize:
                    fig = visualize_episode(episode_data, save_path="clara_layout.png")
                    print("Layout saved as 'clara_layout.png'")
                    plt.show()
            else:
                episode_results, stats = run_multiple_episodes(
                    model, env, num_episodes=args.episodes, 
                    deterministic=args.deterministic
                )
                
                if args.visualize:
                    # Visualize the best episode
                    best_idx = np.argmax(stats['rewards'])
                    best_episode = episode_results[best_idx]
                    fig = visualize_episode(best_episode, save_path="clara_best_layout.png")
                    print(f"Best layout (episode {best_idx + 1}) saved as 'clara_best_layout.png'")
                    plt.show()
        
        # Test different circuits
        if args.test_circuits:
            circuit_results = test_different_circuits(model)
            
            print(f"\nCircuit Type Performance Summary:")
            for result in circuit_results:
                print(f"{result['name']}: {result['reward']:.1f} reward, "
                      f"{result['steps']} steps, {result['components']} components")
        
        print("\nModel evaluation completed!")
        
    except FileNotFoundError as e:
        print(f" Error: {e}")
        print("Make sure you've run training first: python3 train.py")
        return 1
    except Exception as e:
        print(f" Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
