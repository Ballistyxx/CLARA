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


def _capture_visualization_step(live_viz_data, visualizer, env, step, 
                               save_frames, create_gif, gif_frames, viz_delay, 
                               components_placed, placement_step):
    """Handle visualization for a single step - only when component placement changes."""
    
    # Real-time visualization
    if live_viz_data:
        _update_live_visualization(live_viz_data, env, step, placement_step)
        
        # Handle user input for stepping
        if not live_viz_data['auto_mode']:
            user_input = input(f"\nPlacement {placement_step} (Step {step}) - Press ENTER for next placement (q=quit, f=fast): ").strip().lower()
            if user_input == 'q':
                raise KeyboardInterrupt("User requested quit")
            elif user_input == 'f':
                live_viz_data['auto_mode'] = True
                print("Switched to auto mode (fast forward)")
        else:
            import time
            time.sleep(viz_delay)
    
    # Save individual frames (only for placement changes)
    if save_frames:
        fig = visualizer.visualize_layout(
            circuit=env.circuit,
            component_positions=env.component_positions,
            title=f"Placement {placement_step}: {components_placed}/{env.num_components} components placed (Step {step})",
            show_connections=True,
            save_path=f"placement_{placement_step:03d}.png"
        )
        plt.close(fig)
        print(f"Saved placement_{placement_step:03d}.png")
    
    # Capture for GIF (only for placement changes)
    if create_gif:
        fig = visualizer.visualize_layout(
            circuit=env.circuit,
            component_positions=env.component_positions,
            title=f"Placement {placement_step}: {components_placed}/{env.num_components} components placed",
            show_connections=True
        )
        
        # Convert matplotlib figure to numpy array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        gif_frames.append(buf)
        plt.close(fig)


def _update_live_visualization(live_viz_data, env, step, placement_step):
    """Update the live visualization display."""
    ax = live_viz_data['ax']
    fig = live_viz_data['fig']
    
    # Clear and set up the plot
    ax.clear()
    ax.set_xlim(-0.5, env.grid_size - 0.5)
    ax.set_ylim(-0.5, env.grid_size - 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Placement {placement_step}: {len(env.component_positions)}/{env.num_components} components placed\n"
                f"Real-time CLARA Layout Visualization (Step {step})", fontsize=14, fontweight='bold')
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    
    # Draw placed components
    for comp_id, (x, y, orientation) in env.component_positions.items():
        component_attrs = env.circuit.nodes[comp_id]
        width = float(component_attrs.get('width', 1))
        height = float(component_attrs.get('height', 1))
        comp_type = component_attrs.get('component_type', 0)
        
        # Rotate dimensions based on orientation
        if orientation in [1, 3]:  # 90° or 270° rotation
            width, height = height, width
        
        # Component type colors
        colors = {
            0: 'lightblue',   # NMOS
            1: 'lightcoral',  # PMOS  
            2: 'lightgreen',  # Resistor
            3: 'yellow',      # Capacitor
            4: 'orange',      # Inductor
        }
        color = colors.get(comp_type, 'lightgray')
        
        # Draw component rectangle
        from matplotlib.patches import Rectangle
        rect = Rectangle((x, y), width, height, 
                        facecolor=color, edgecolor='black', 
                        linewidth=1.5, alpha=0.8)
        ax.add_patch(rect)
        
        # Add component label
        center_x = x + width / 2
        center_y = y + height / 2
        ax.text(center_x, center_y, str(comp_id), 
               ha='center', va='center', fontweight='bold', fontsize=8)
        
        # Add type indicator
        type_names = ['N', 'P', 'R', 'C', 'L']
        if comp_type < len(type_names):
            ax.text(x + 0.1, y + height - 0.1, type_names[comp_type], 
                   ha='left', va='top', fontsize=6, fontweight='bold')
    
    # Draw connections between placed components
    for edge in env.circuit.edges():
        comp1, comp2 = edge
        if comp1 in env.component_positions and comp2 in env.component_positions:
            pos1 = env.component_positions[comp1]
            pos2 = env.component_positions[comp2]
            
            # Get component dimensions for center calculation
            attrs1 = env.circuit.nodes[comp1]
            attrs2 = env.circuit.nodes[comp2]
            w1, h1 = float(attrs1.get('width', 1)), float(attrs1.get('height', 1))
            w2, h2 = float(attrs2.get('width', 1)), float(attrs2.get('height', 1))
            
            # Adjust for orientation
            if pos1[2] in [1, 3]:  # 90° or 270°
                w1, h1 = h1, w1
            if pos2[2] in [1, 3]:  # 90° or 270°
                w2, h2 = h2, w2
            
            center1 = (pos1[0] + w1/2, pos1[1] + h1/2)
            center2 = (pos2[0] + w2/2, pos2[1] + h2/2)
            
            ax.plot([center1[0], center2[0]], [center1[1], center2[1]], 
                   'k--', alpha=0.4, linewidth=1)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='lightblue', edgecolor='black', label='NMOS'),
        plt.Rectangle((0,0),1,1, facecolor='lightcoral', edgecolor='black', label='PMOS'),
        plt.Rectangle((0,0),1,1, facecolor='lightgreen', edgecolor='black', label='Resistor'),
        plt.Rectangle((0,0),1,1, facecolor='yellow', edgecolor='black', label='Capacitor'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Update display
    plt.draw()
    plt.tight_layout()


def _create_gif_from_frames(frames, filename):
    """Create an animated GIF from captured frames."""
    try:
        import imageio
        imageio.mimsave(filename, frames, duration=0.5)
        print(f"Animation saved as {filename}")
    except ImportError:
        print("Warning: imageio not available, cannot create GIF")
        print("Install with: pip install imageio")


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


def run_single_episode(model, env, deterministic=True, render=False, reset_env=True, 
                      live_viz=False, save_frames=False, create_gif=False, viz_delay=0.5,
                      show_action_efficiency=False):
    """Run a single episode with the trained model and optional visualization."""
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
    
    # Action efficiency tracking
    if show_action_efficiency:
        action_stats = {
            'total_steps': 0,
            'valid_steps': 0,
            'placement_steps': 0,
            'invalid_actions': 0,
            'invalid_placements': 0,
            'no_placement_steps': 0
        }
    
    episode_data = {
        'positions': [],
        'rewards': [],
        'actions': [],
        'circuit': env.circuit
    }
    
    # Initialize visualization if requested
    live_viz_data = None
    gif_frames = []
    placement_step = 0  # Track placement events, not just steps
    
    if live_viz or save_frames or create_gif:
        visualizer = AnalogLayoutVisualizer(grid_size=env.grid_size)
        
        if live_viz:
            print("\n" + "="*60)
            print("REAL-TIME VISUALIZATION MODE")
            print("Controls:")
            print("  - Press ENTER to advance to next component placement")
            print("  - Type 'q' + ENTER to quit")
            print("  - Type 'f' + ENTER to fast-forward (auto mode)")
            print("  - Only shows when components are actually placed")
            print("="*60)
            
            # Set up interactive plotting
            plt.ion()
            fig, ax = plt.subplots(figsize=(12, 10))
            
            live_viz_data = {
                'fig': fig,
                'ax': ax,
                'auto_mode': False,
                'visualizer': visualizer
            }
    
    if render:
        print(f"\nRunning episode with {env.num_components} components")
        print(f"Circuit nodes: {list(env.circuit.nodes())}")
        print(f"Circuit edges: {list(env.circuit.edges())}")
    
    # Initial visualization (empty grid)
    if live_viz or save_frames or create_gif:
        placement_step += 1
        _capture_visualization_step(
            live_viz_data, visualizer, env, 0, 
            save_frames, create_gif, gif_frames, viz_delay, 
            len(env.component_positions), placement_step
        )
    
    while step < env.max_steps:
        # Track components before action
        components_before = len(env.component_positions)
        
        # Get action from model with action masking for better efficiency
        if hasattr(env, 'action_masks') and env.enable_action_masking:
            # Use action masking to avoid invalid actions
            action_mask = env.action_masks()
            action, _states = model.predict(obs, deterministic=deterministic, action_mask=action_mask)
        else:
            # Fallback to standard prediction
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
        
        # Track action efficiency
        if show_action_efficiency:
            action_stats['total_steps'] += 1
            valid_action = info.get('valid_action', True)
            placement_successful = info.get('placement_successful', False)
            
            if not valid_action:
                action_stats['invalid_actions'] += 1
            elif placement_successful:
                action_stats['placement_steps'] += 1
                action_stats['valid_steps'] += 1
            else:
                action_stats['valid_steps'] += 1
                action_stats['no_placement_steps'] += 1
        
        # Check if a component was actually placed
        components_after = len(env.component_positions)
        component_placed = components_after > components_before
        
        # Visualization only when component is placed
        if component_placed and (live_viz or save_frames or create_gif):
            placement_step += 1
            _capture_visualization_step(
                live_viz_data, visualizer, env, step, 
                save_frames, create_gif, gif_frames, viz_delay,
                components_after, placement_step
            )
        
        if render:
            print(f"Step {step}: action={action}, reward={reward:.3f}, "
                  f"valid_action={info.get('valid_action', True)}, "
                  f"placed={len(env.component_positions)}/{env.num_components}")
            
            if component_placed:
                print(f"  ✓ Component placed! Total: {components_after}/{env.num_components}")
            elif not info.get('valid_action', True):
                print(f"  ✗ Invalid action")
            else:
                print(f"  - No placement this step")
            
            if 'reward_components' in info:
                breakdown = info['reward_components']
                non_zero = {k: v for k, v in breakdown.items() if abs(v) > 0.001}
                if non_zero:
                    print(f"    Reward breakdown: {non_zero}")
        
        if done:
            break
    
    # Final positions
    episode_data['positions'].append(env.component_positions.copy())
    
    # Clean up visualization
    if live_viz_data:
        plt.ioff()
        plt.close(live_viz_data['fig'])
        print("\nVisualization completed!")
    
    # Create GIF if requested
    if create_gif and gif_frames:
        _create_gif_from_frames(gif_frames, "component_placement_animation.gif")
    
    success = len(env.component_positions) == env.num_components
    
    # Print action efficiency analysis
    if show_action_efficiency and action_stats['total_steps'] > 0:
        print(f"\n📊 Action Efficiency Analysis:")
        print(f"   Total RL steps: {action_stats['total_steps']}")
        print(f"   Valid actions: {action_stats['valid_steps']} ({action_stats['valid_steps']/action_stats['total_steps']*100:.1f}%)")
        print(f"   Successful placements: {action_stats['placement_steps']} ({action_stats['placement_steps']/action_stats['total_steps']*100:.1f}%)")
        print(f"   Invalid actions: {action_stats['invalid_actions']} ({action_stats['invalid_actions']/action_stats['total_steps']*100:.1f}%)")
        print(f"   Valid but no placement: {action_stats['no_placement_steps']} ({action_stats['no_placement_steps']/action_stats['total_steps']*100:.1f}%)")
        
        if action_stats['placement_steps'] > 0:
            efficiency = action_stats['placement_steps'] / action_stats['valid_steps'] * 100
            print(f"   Placement efficiency: {efficiency:.1f}% (placements per valid action)")
            
            if action_stats['invalid_actions'] > action_stats['placement_steps']:
                print(f"   ⚠️  High invalid action rate - consider enabling action masking")
            elif efficiency > 80:
                print(f"   ✅ Excellent placement efficiency!")
            elif efficiency > 60:
                print(f"   👍 Good placement efficiency")
            else:
                print(f"   📈 Room for improvement in placement efficiency")
    
    if render:
        print(f"\nEpisode Summary:")
        print(f"Total reward: {total_reward:.3f}")
        print(f"Steps taken: {step}")
        print(f"Components placed: {len(env.component_positions)}/{env.num_components}")
        print(f"Success: {'Yes' if success else 'No'}")

    return episode_data, total_reward, step, success


def run_multiple_episodes(model, env, num_episodes=5, deterministic=True, specific_circuit=None,
                         live_viz=False, save_frames=False, create_gif=False, viz_delay=0.5,
                         show_action_efficiency=False):
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
                model, env, deterministic=deterministic, render=False, reset_env=False,
                live_viz=live_viz, save_frames=save_frames, create_gif=create_gif, 
                viz_delay=viz_delay, show_action_efficiency=show_action_efficiency
            )
        else:
            # Use default reset behavior
            episode_data, reward, steps, success = run_single_episode(
                model, env, deterministic=deterministic, render=False, reset_env=True,
                live_viz=live_viz, save_frames=save_frames, create_gif=create_gif, 
                viz_delay=viz_delay, show_action_efficiency=show_action_efficiency
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
                       default='data/netlists/programmable_pll_subcircuits',
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
    parser.add_argument('--live-viz', action='store_true',
                       help='Show real-time component placement visualization (interactive, advances only when components are placed)')
    parser.add_argument('--save-frames', action='store_true',
                       help='Save individual frame images for each component placement')
    parser.add_argument('--create-gif', action='store_true',
                       help='Create animated GIF showing component placement progression')
    parser.add_argument('--viz-delay', type=float, default=0.5,
                       help='Delay between placements in auto mode (seconds, default: 0.5)')
    parser.add_argument('--show-efficiency', action='store_true',
                       help='Show detailed action efficiency analysis (invalid actions, placement rates, etc.)')
    
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
                    model, env, deterministic=args.deterministic, render=args.render, reset_env=False,
                    live_viz=args.live_viz, save_frames=args.save_frames, 
                    create_gif=args.create_gif, viz_delay=args.viz_delay,
                    show_action_efficiency=args.show_efficiency
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
                    deterministic=args.deterministic, specific_circuit=test_circuit,
                    live_viz=args.live_viz, save_frames=args.save_frames,
                    create_gif=args.create_gif, viz_delay=args.viz_delay,
                    show_action_efficiency=args.show_efficiency
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
        print("\nVisualization Options:")
        print("   - Use --live-viz for real-time component placement visualization (interactive)")
        print("     * Only advances when components are actually placed (skips invalid actions)")
        print("     * Press ENTER to see next placement, 'f' for auto mode, 'q' to quit")
        print("   - Use --save-frames to save individual PNG files for each component placement")
        print("   - Use --create-gif to generate an animated GIF of component placement progression")
        print("   - Use --viz-delay to control animation speed (default: 0.5 seconds)")
        print("   - Use --show-efficiency to analyze action efficiency and placement rates")
        print("\nAnalysis Options:")
        print("   - Action masking is automatically enabled to improve step efficiency")
        print("   - Use --show-efficiency to see detailed breakdown of action types")
        print("\nTips:")
        print("   - Use --circuit to test specific SPICE circuits")
        print("   - Use --test-circuits to test all available circuits")
        print("   - Use --sampling-strategy to control large circuit sampling (diverse/connected/random)")
        print("   - Use --test-transferability to see grid-agnostic capabilities")
        print("   - Use --visualize to see layout diagrams")
        print("   - Large circuits are automatically sampled to fit model constraints!")
        print("\nExample commands:")
        print("   python run_model.py --live-viz                     # Interactive real-time viewing")
        print("   python run_model.py --create-gif --save-frames     # Create animation + frames")
        print("   python run_model.py --live-viz --show-efficiency   # Real-time view + action analysis")
        print("   python run_model.py --circuit LDO.spice --show-efficiency  # Analyze specific circuit efficiency")

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
