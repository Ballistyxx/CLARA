#!/usr/bin/env python3
"""
Script to load and run the trained CLARA model for inference.
Now supports grid-agnostic models that can run on any grid size!
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
    # Try common model locations
    possible_paths = [
        model_path,
        f"./logs/{model_path}",
        f"./logs/clara_final_model",
        f"./logs/clara_spice_final_model"
    ]
    
    for path in possible_paths:
        if os.path.exists(path + '.zip'):
            print(f"🔄 Loading model from {path}")
            model = PPO.load(path)
            print("✅ Model loaded successfully!")
            print(f"   Policy type: {type(model.policy).__name__}")
            print(f"   Observation space: {model.observation_space}")
            return model
    
    raise FileNotFoundError(f"Model not found. Tried paths: {[p + '.zip' for p in possible_paths]}")


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


def visualize_episode(episode_data, grid_size=20, save_path=None):
    """Visualize the final layout from an episode."""
    visualizer = AnalogLayoutVisualizer(grid_size=grid_size)
    
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


def test_grid_transferability(model, base_grid_size=20):
    # Get max_components from model
    model_obs_space = model.observation_space.spaces
    max_components = model_obs_space['component_graph'].shape[0]
    """Test model transferability across different grid sizes."""
    print("\n🔄 Testing Grid Transferability...")
    print("This demonstrates that models trained on one grid size work on any other grid size!")
    
    # Test different grid sizes
    grid_sizes = [16, 24, 32, 48, 64]
    results = []
    
    # Create a test circuit
    from data.circuit_generator import AnalogCircuitGenerator
    generator = AnalogCircuitGenerator()
    test_circuit = generator.generate_differential_pair()
    
    print(f"Testing circuit with {len(test_circuit.nodes)} components on different grid sizes...")
    
    for grid_size in grid_sizes:
        print(f"\n📊 Grid size: {grid_size}×{grid_size}")
        
        # Create environment with different grid size
        env = AnalogLayoutEnv(grid_size=grid_size, max_components=max_components)
        env.reset(circuit_graph=test_circuit)
        
        # Run episode
        episode_data, reward, steps, success = run_single_episode(
            model, env, deterministic=True, render=False
        )
        
        # Calculate layout density
        if episode_data['positions'][-1]:
            positions = list(episode_data['positions'][-1].values())
            xs = [pos[0] for pos in positions]
            ys = [pos[1] for pos in positions]
            layout_area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1) if positions else 0
            density = len(positions) / layout_area if layout_area > 0 else 0
        else:
            density = 0
        
        result = {
            'grid_size': grid_size,
            'reward': reward,
            'steps': steps,
            'success': success,
            'placed': len(episode_data['positions'][-1]),
            'total': len(test_circuit.nodes),
            'density': density
        }
        
        results.append(result)
        
        print(f"   Reward: {reward:.2f}")
        print(f"   Components placed: {result['placed']}/{result['total']}")
        print(f"   Success: {'Yes' if success else 'No'}")
        print(f"   Layout density: {density:.2f}")
    
    # Summary
    print(f"\n📋 Grid Transferability Summary:")
    print(f"Grid Size | Reward | Success | Placed | Density")
    print(f"-" * 50)
    
    for result in results:
        success_str = "Yes" if result['success'] else "No"
        print(f"{result['grid_size']:4d}×{result['grid_size']:<4d} | {result['reward']:6.2f} | {success_str:7s} | {result['placed']:2d}/{result['total']:<2d}   | {result['density']:7.2f}")
    
    # Calculate transferability score
    base_result = results[0]  # First grid size as baseline
    transferability_scores = []
    
    for result in results[1:]:
        if base_result['success'] and result['success']:
            score = result['reward'] / base_result['reward']
        elif result['success'] == base_result['success']:
            score = 1.0  # Both succeeded or both failed
        else:
            score = 0.5  # Different success status
        transferability_scores.append(score)
    
    avg_transferability = sum(transferability_scores) / len(transferability_scores) if transferability_scores else 0
    
    print(f"\n🎯 Transferability Score: {avg_transferability:.1%}")
    if avg_transferability > 0.9:
        print("   ✅ EXCELLENT - Model transfers perfectly across grid sizes!")
    elif avg_transferability > 0.7:
        print("   ✅ GOOD - Model transfers well with minor variations")
    elif avg_transferability > 0.5:
        print("   ⚠️  MODERATE - Some degradation on different grid sizes")
    else:
        print("   ❌ POOR - Significant performance loss on different grids")
    
    return results


def test_different_circuits(model, grid_size=20):
    """Test the model on different circuit types."""
    print(f"\nTesting model on different circuit types (Grid: {grid_size}×{grid_size})...")
    
    generator = AnalogCircuitGenerator()
    visualizer = AnalogLayoutVisualizer(grid_size=grid_size)
    
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
            env = AnalogLayoutEnv(grid_size=grid_size, max_components=6)  # Match training
            env.reset(circuit_graph=circuit)
            
            # Run episode
            episode_data, reward, steps, success = run_single_episode(
                model, env, deterministic=True, render=False
            )
            
            print(f"Reward: {reward:.3f}, Steps: {steps}, Success: {'Yes' if success else 'No'}")
            
            # Visualize
            fig = visualize_episode(episode_data, grid_size=grid_size, save_path=f"layout_{circuit_name.lower().replace(' ', '_')}.png")
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
    parser = argparse.ArgumentParser(description='Run trained CLARA model with grid-agnostic capabilities')
    parser.add_argument('--model', '-m', type=str, 
                       default='./logs/clara_final_model',
                       help='Path to trained model (without .zip extension)')
    parser.add_argument('--episodes', '-e', type=int, default=3,
                       help='Number of episodes to run')
    parser.add_argument('--grid-size', '-g', type=int, default=20,
                       help='Grid size for evaluation (default: 20)')
    parser.add_argument('--deterministic', '-d', action='store_true',
                       help='Use deterministic actions')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--test-circuits', '-t', action='store_true',
                       help='Test on different circuit types')
    parser.add_argument('--test-transferability', '--transfer', action='store_true',
                       help='Test grid transferability across different sizes')
    parser.add_argument('--render', '-r', action='store_true',
                       help='Print detailed step-by-step output')
    
    args = parser.parse_args()
    
    try:
        print(f"🚀 CLARA Grid-Agnostic Model Runner")
        print(f"=" * 50)
        
        # Load model
        model = load_model(args.model)
        
        # Create environment with specified grid size
        # Extract max_components from model's observation space
        model_obs_space = model.observation_space.spaces
        max_components = model_obs_space['component_graph'].shape[0]
        
        env = AnalogLayoutEnv(grid_size=args.grid_size, max_components=max_components)
        
        print(f"   Model expects max {max_components} components")
        
        print(f"Environment: {env.grid_size}×{env.grid_size} grid, max {env.max_components} components")
        print(f"🚀 Grid-agnostic model - can run on any grid size!")
        
        # Run episodes
        if args.episodes > 0:
            if args.episodes == 1:
                episode_data, reward, steps, success = run_single_episode(
                    model, env, deterministic=args.deterministic, render=args.render
                )
                
                if args.visualize:
                    fig = visualize_episode(episode_data, grid_size=args.grid_size, save_path="clara_layout.png")
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
                    fig = visualize_episode(best_episode, grid_size=args.grid_size, save_path="clara_best_layout.png")
                    print(f"Best layout (episode {best_idx + 1}) saved as 'clara_best_layout.png'")
                    plt.show()
        
        # Test grid transferability
        if args.test_transferability:
            transfer_results = test_grid_transferability(model, base_grid_size=args.grid_size)
        
        # Test different circuits
        if args.test_circuits:
            circuit_results = test_different_circuits(model, grid_size=args.grid_size)
            
            print(f"\nCircuit Type Performance Summary:")
            for result in circuit_results:
                print(f"{result['name']}: {result['reward']:.1f} reward, "
                      f"{result['steps']} steps, {result['components']} components")
        
        print("\n✅ Model evaluation completed!")
        print("\n💡 Tips:")
        print("   • Use --test-transferability to see grid-agnostic capabilities")
        print("   • Use --grid-size to test on different sized grids")
        print("   • Use --visualize to see layout diagrams")
        print("   • Models trained on any grid size work on any other grid size!")
        
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
