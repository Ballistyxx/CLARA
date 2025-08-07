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
from enhanced_spice_parser import EnhancedSpiceParser, parse_multiple_spice_files
from train_spice_real import SpiceCircuitManager
import argparse
import networkx as nx


def sample_large_circuit(circuit: nx.Graph, max_components: int, strategy='diverse') -> nx.Graph:
    """
    Create a representative subset of a large circuit.
    
    Args:
        circuit: Original NetworkX graph
        max_components: Maximum number of components to keep
        strategy: Sampling strategy ('diverse', 'connected', 'random')
    
    Returns:
        Subgraph with at most max_components nodes
    """
    if len(circuit.nodes) <= max_components:
        return circuit
    
    print(f"   Circuit has {len(circuit.nodes)} components, sampling {max_components} for model compatibility")
    
    if strategy == 'diverse':
        # Sample to get diverse component types and good connectivity
        nodes_to_keep = []
        
        # Get component type distribution
        type_counts = {}
        for node in circuit.nodes():
            comp_type = circuit.nodes[node].get('component_type', 0)
            if comp_type not in type_counts:
                type_counts[comp_type] = []
            type_counts[comp_type].append(node)
        
        # Sample proportionally from each type
        components_per_type = max_components // len(type_counts)
        remainder = max_components % len(type_counts)
        
        for comp_type, nodes in type_counts.items():
            sample_count = components_per_type + (1 if remainder > 0 else 0)
            if remainder > 0:
                remainder -= 1
            
            # Sample nodes of this type, preferring highly connected ones
            node_degrees = [(node, circuit.degree(node)) for node in nodes]
            node_degrees.sort(key=lambda x: x[1], reverse=True)
            
            selected = [node for node, _ in node_degrees[:sample_count]]
            nodes_to_keep.extend(selected)
        
        # Ensure we don't exceed max_components
        nodes_to_keep = nodes_to_keep[:max_components]
        
    elif strategy == 'connected':
        # Select the most connected components
        node_degrees = [(node, circuit.degree(node)) for node in circuit.nodes()]
        node_degrees.sort(key=lambda x: x[1], reverse=True)
        nodes_to_keep = [node for node, _ in node_degrees[:max_components]]
        
    else:  # random
        import random
        nodes_to_keep = random.sample(list(circuit.nodes()), max_components)
    
    # Create subgraph and remap node IDs to be sequential
    subgraph = circuit.subgraph(nodes_to_keep).copy()
    
    # Remap nodes to sequential IDs (0, 1, 2, ...)
    mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(nodes_to_keep))}
    remapped_circuit = nx.relabel_nodes(subgraph, mapping)
    
    # Update matched_component references in node attributes
    for node in remapped_circuit.nodes():
        attrs = remapped_circuit.nodes[node]
        if 'matched_component' in attrs and attrs['matched_component'] != -1:
            old_matched = attrs['matched_component']
            if old_matched in mapping:
                attrs['matched_component'] = mapping[old_matched]
            else:
                attrs['matched_component'] = -1  # Referenced component not in subset
    
    print(f"   Sampled circuit: {len(remapped_circuit.nodes)} components, {len(remapped_circuit.edges)} connections")
    print(f"   Component types in sample: {[remapped_circuit.nodes[n].get('component_type', 0) for n in remapped_circuit.nodes()]}")
    
    return remapped_circuit


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
            print(f"Loading model from {path}")
            model = PPO.load(path)
            print("Model loaded successfully!")
            print(f"   Policy type: {type(model.policy).__name__}")
            print(f"   Observation space: {model.observation_space}")
            return model
    
    raise FileNotFoundError(f"Model not found. Tried paths: {[p + '.zip' for p in possible_paths]}")


def run_single_episode(model, env, deterministic=True, render=False, reset_env=True):
    """Run a single episode with the trained model."""
    if reset_env:
        result = env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
    else:
        # Environment already reset with specific circuit, just get current observation
        obs = env._get_observation()
    
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


def run_multiple_episodes(model, env, num_episodes=5, deterministic=True, specific_circuit=None):
    """Run multiple episodes and collect statistics."""
    print(f"\n  Running {num_episodes} episodes...")
    
    episode_results = []
    total_rewards = []
    steps_taken = []
    success_count = 0
    
    for episode in range(num_episodes):
        if specific_circuit is not None:
            # Reset environment with the specific circuit for each episode
            env.reset(circuit_graph=specific_circuit)
            episode_data, reward, steps, success = run_single_episode(
                model, env, deterministic=deterministic, render=False, reset_env=False
            )
        else:
            # Use default reset behavior
            episode_data, reward, steps, success = run_single_episode(
                model, env, deterministic=deterministic, render=False, reset_env=True
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


def test_grid_transferability(model, circuit_manager, base_grid_size=20):
    """Test model transferability across different grid sizes using real SPICE circuits."""
    # Get max_components from model
    model_obs_space = model.observation_space.spaces
    max_components = model_obs_space['component_graph'].shape[0]
    
    print("\nTesting Grid Transferability with Real SPICE Circuits...")
    print("This demonstrates that models trained on one grid size work on any other grid size!")
    
    # Test different grid sizes
    grid_sizes = [16, 24, 32, 48, 64]
    results = []
    
    # Get a test circuit from SPICE data
    test_circuit_name = list(circuit_manager.suitable_circuits.keys())[0]
    test_circuit_data = circuit_manager.suitable_circuits[test_circuit_name]
    
    test_circuit = test_circuit_data['circuit_graph']
    #test_circuit = circuit_manager.convert_to_networkx_graph(test_circuit_data)
    
    print(f"Testing {test_circuit_name} with {len(test_circuit.nodes)} components on different grid sizes...")
    
    for grid_size in grid_sizes:
        print(f"\nGrid size: {grid_size}×{grid_size}")
        
        # Create environment with different grid size
        env = AnalogLayoutEnv(grid_size=grid_size, max_components=max_components)
        env.reset(circuit_graph=test_circuit)
        
        # Run episode
        episode_data, reward, steps, success = run_single_episode(
            model, env, deterministic=True, render=False, reset_env=False
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
    print(f"\nGrid Transferability Summary:")
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
    
    print(f"\nTransferability Score: {avg_transferability:.1%}")
    if avg_transferability > 0.9:
        print("   EXCELLENT - Model transfers perfectly across grid sizes!")
    elif avg_transferability > 0.7:
        print("   GOOD - Model transfers well with minor variations")
    elif avg_transferability > 0.5:
        print("    MODERATE - Some degradation on different grid sizes")
    else:
        print("   POOR - Significant performance loss on different grids")
    
    return results


def test_different_circuits(model, circuit_manager, grid_size=20, sampling_strategy='diverse'):
    """Test the model on different real SPICE circuit types."""
    print(f"\nTesting model on different SPICE circuits (Grid: {grid_size}×{grid_size})...")
    
    # Get max_components from model
    model_obs_space = model.observation_space.spaces
    max_components = model_obs_space['component_graph'].shape[0]
    
    visualizer = AnalogLayoutVisualizer(grid_size=grid_size)
    
    results = []
    
    # Test all available SPICE circuits
    for circuit_name, circuit_data in circuit_manager.suitable_circuits.items():
        print(f"\nTesting {circuit_name} ({circuit_data['subcircuit_name']})...")
        
        try:
            # Convert to NetworkX graph
            # circuit = circuit_manager.convert_to_networkx_graph(circuit_data)
            circuit = circuit_data['circuit_graph']
            original_size = len(circuit.nodes)
            
            # Handle large circuits by sampling
            if original_size > max_components:
                print(f"Sampling - circuit has {original_size} components, model limit is {max_components}")
                circuit = sample_large_circuit(circuit, max_components, strategy=sampling_strategy)
            
            # Create environment with this specific circuit
            env = AnalogLayoutEnv(grid_size=grid_size, max_components=max_components)
            env.reset(circuit_graph=circuit)
            
            # Run episode
            episode_data, reward, steps, success = run_single_episode(
                model, env, deterministic=True, render=False, reset_env=False
            )
            
            print(f"Reward: {reward:.3f}, Steps: {steps}, Success: {'Yes' if success else 'No'}")
            print(f"Components: {len(circuit.nodes)} (original: {original_size}, total: {circuit_data['num_components']})")
            
            # Visualize
            safe_name = circuit_name.replace('.spice', '').replace(' ', '_').replace('/', '_')
            fig = visualize_episode(episode_data, grid_size=grid_size, 
                                  save_path=f"layout_spice_{safe_name}.png")
            plt.close(fig)
            
            results.append({
                'name': circuit_name,
                'subcircuit': circuit_data['subcircuit_name'],
                'reward': reward,
                'steps': steps,
                'success': success,
                'components': len(circuit.nodes),
                'original_components': original_size,
                'total_components': circuit_data['num_components']
            })
            
        except Exception as e:
            print(f"Error testing {circuit_name}: {e}")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run trained CLARA model with real SPICE circuits')
    parser.add_argument('--model', '-m', type=str, 
                       default='./logs/best_model',
                       help='Path to trained model (without .zip extension)')
    parser.add_argument('--episodes', '-e', type=int, default=3,
                       help='Number of episodes to run')
    parser.add_argument('--grid-size', '-g', type=int, default=20,
                       help='Grid size for evaluation (default: 20)')
    parser.add_argument('--circuit', '-c', type=str, default=None,
                       help='Specific SPICE circuit name to test (e.g., "NAND.spice")')
    parser.add_argument('--spice-dir', type=str, 
                       default='/home/eli/Documents/Internship/CLARA/data/netlists/programmable_pll_subcircuits',
                       help='Directory containing SPICE files')
    parser.add_argument('--sampling-strategy', type=str, default='diverse',
                       choices=['diverse', 'connected', 'random'],
                       help='Strategy for sampling large circuits (diverse, connected, random)')
    parser.add_argument('--deterministic', '-d', action='store_true',
                       help='Use deterministic actions')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--test-circuits', '-t', action='store_true',
                       help='Test on all available SPICE circuits')
    parser.add_argument('--test-transferability', '--transfer', action='store_true',
                       help='Test grid transferability across different sizes')
    parser.add_argument('--render', '-r', action='store_true',
                       help='Print detailed step-by-step output')
    
    args = parser.parse_args()
    
    try:
        print(f"CLARA Model with Real SPICE Circuits")
        print(f"=" * 50)
        
        # Load model
        model = load_model(args.model)
        
        # Initialize SPICE circuit manager
        print(f"Loading SPICE circuits from {args.spice_dir}")
        circuit_manager = SpiceCircuitManager(args.spice_dir)
        
        # Print circuit statistics
        stats = circuit_manager.get_circuit_stats()
        print(f"Loaded {stats['total_circuits']} SPICE circuits for evaluation")
        print(f"   Component range: {stats['component_range'][0]}-{stats['component_range'][1]}")
        print(f"   Available circuits: {', '.join(list(stats['circuit_names'])[:3])}...")
        
        # Extract max_components from model's observation space
        model_obs_space = model.observation_space.spaces
        max_components = model_obs_space['component_graph'].shape[0]
        
        print(f"   Model expects max {max_components} components")
        print(f"   Grid: {args.grid_size}×{args.grid_size}")
        
        # Run episodes on SPICE circuits
        if args.episodes > 0:
            # Select circuit to test
            if args.circuit:
                if args.circuit in circuit_manager.suitable_circuits:
                    circuit_data = circuit_manager.suitable_circuits[args.circuit]
                    # test_circuit = circuit_manager.convert_to_networkx_graph(circuit_data)
                    test_circuit = circuit_data['circuit_graph']
                    circuit_name = args.circuit
                else:
                    print(f"Circuit {args.circuit} not found. Available circuits:")
                    for name in circuit_manager.suitable_circuits.keys():
                        print(f"   - {name}")
                    return 1
            else:
                # Use random circuit from SPICE data
                test_circuit = circuit_manager.get_random_circuit()
                circuit_name = "random SPICE circuit"
            
            # Handle large circuits by sampling
            actual_components = len(test_circuit.nodes)
            
            print(f"\nTesting on {circuit_name} ({actual_components} components)")
            
            if actual_components > max_components:
                print(f"   Model trained for {max_components} components, circuit has {actual_components}")
                test_circuit = sample_large_circuit(test_circuit, max_components, strategy=args.sampling_strategy)
                print(f"   Using sampled circuit with {len(test_circuit.nodes)} components")
            
            # Create environment with model's max_components
            env = AnalogLayoutEnv(grid_size=args.grid_size, max_components=max_components)
            env.reset(circuit_graph=test_circuit)
            
            if args.episodes == 1:
                episode_data, reward, steps, success = run_single_episode(
                    model, env, deterministic=args.deterministic, render=args.render, reset_env=False
                )
                
                if args.visualize:
                    safe_name = circuit_name.replace('.spice', '').replace(' ', '_').replace('/', '_')
                    fig = visualize_episode(episode_data, grid_size=args.grid_size, 
                                          save_path=f"clara_layout_{safe_name}.png")
                    print(f"Layout saved as 'clara_layout_{safe_name}.png'")
                    plt.show()
            else:
                # For multiple episodes with the same circuit, we can reset each time
                episode_results, stats = run_multiple_episodes(
                    model, env, num_episodes=args.episodes, 
                    deterministic=args.deterministic, specific_circuit=test_circuit
                )
                
                if args.visualize:
                    # Visualize the best episode
                    best_idx = np.argmax(stats['rewards'])
                    best_episode = episode_results[best_idx]
                    safe_name = circuit_name.replace('.spice', '').replace(' ', '_').replace('/', '_')
                    fig = visualize_episode(best_episode, grid_size=args.grid_size, 
                                          save_path=f"clara_best_layout_{safe_name}.png")
                    print(f"Best layout (episode {best_idx + 1}) saved as 'clara_best_layout_{safe_name}.png'")
                    plt.show()
        
        # Test grid transferability
        if args.test_transferability:
            transfer_results = test_grid_transferability(model, circuit_manager, base_grid_size=args.grid_size)
        
        # Test different circuits
        if args.test_circuits:
            circuit_results = test_different_circuits(model, circuit_manager, 
                                                     grid_size=args.grid_size, 
                                                     sampling_strategy=args.sampling_strategy)
            
            print(f"\nSPICE Circuit Performance Summary:")
            for result in circuit_results:
                components_info = f"{result['components']} components"
                if result.get('original_components', 0) != result['components']:
                    components_info += f" (sampled from {result['original_components']})"
                print(f"{result['name']} ({result['subcircuit']}): {result['reward']:.1f} reward, "
                      f"{result['steps']} steps, {components_info}")
        
        print("\nModel evaluation completed!")
        print("\nTips:")
        print("   - Use --circuit to test specific SPICE circuits")
        print("   - Use --test-circuits to test all available circuits")
        print("   - Use --sampling-strategy to control large circuit sampling (diverse/connected/random)")
        print("   - Use --test-transferability to see grid-agnostic capabilities")
        print("   - Use --visualize to see layout diagrams")
        print("   - Large circuits are automatically sampled to fit model constraints!")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you've run training first: python3 train_spice_real.py")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
