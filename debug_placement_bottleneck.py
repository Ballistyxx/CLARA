#!/usr/bin/env python3
"""
Debug why placement gets stuck at exactly 11 components.
"""

import numpy as np
import sys
from pathlib import Path
import networkx as nx
from enhanced_spice_parser import EnhancedSpiceParser
from analog_layout_env import AnalogLayoutEnv

def convert_parsed_circuit_to_networkx(circuit_data):
    """Convert parsed SPICE data to NetworkX graph."""
    G = nx.Graph()
    
    components = circuit_data['components']
    
    # Add nodes with attributes
    for i, comp in enumerate(components):
        clara_type = comp['type_value']
        width = max(1, int(comp['width']))
        height = max(1, int(comp['length']))
        
        # Simple matched component detection
        matched_component = -1
        for j, other_comp in enumerate(components):
            if (j != i and 
                other_comp['type_value'] == comp['type_value'] and
                abs(other_comp['width'] - comp['width']) < 0.1 and
                other_comp['device_model'] == comp['device_model']):
                matched_component = j
                break
        
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

def debug_placement_step_by_step():
    """Step through placement process to find bottleneck."""
    print("PLACEMENT BOTTLENECK DEBUG")
    print("=" * 50)
    
    ldo_file = "/home/eli/Documents/Internship/CLARA/data/netlists/sky130_am_ip__ldo_01v8.spice"
    
    if not Path(ldo_file).exists():
        print(f"File not found: {ldo_file}")
        return
    
    # Parse and create small test circuit
    parser = EnhancedSpiceParser()
    circuit_data = parser.parse_spice_file(ldo_file)
    
    full_circuit = convert_parsed_circuit_to_networkx(circuit_data)
    nodes = list(full_circuit.nodes())[:20]  # Test with 20 components
    sampled_circuit = full_circuit.subgraph(nodes).copy()
    sampled_circuit = nx.relabel_nodes(sampled_circuit, {node: i for i, node in enumerate(nodes)})
    
    # Create environment
    env = AnalogLayoutEnv(
        grid_size=25,
        max_components=100,
        enable_action_masking=True
    )
    
    result = env.reset(circuit_graph=sampled_circuit)
    obs = result[0] if isinstance(result, tuple) else result
    
    print(f"Circuit: {env.num_components} components")
    print(f"Grid size: {env.grid_size}x{env.grid_size}")
    print(f"Max steps: {env.max_steps}")
    
    # Manual step-by-step placement
    step = 0
    placement_failures = 0
    consecutive_failures = 0
    
    while step < 50:
        placed_count = np.sum(env.placed_mask[:env.num_components])
        
        if placed_count == env.num_components:
            print(f"All components placed!")
            break
        
        print(f"\n--- Step {step + 1} ---")
        print(f"Placed: {placed_count}/{env.num_components}")
        
        # Get action mask
        if 'action_mask' in obs:
            action_mask = obs['action_mask']
            valid_targets = np.where(action_mask == 1)[0]
            print(f"Valid targets: {len(valid_targets)} {list(valid_targets)}")
        else:
            valid_targets = []
            print(f"No action mask available")
        
        if len(valid_targets) == 0:
            print(f"No valid targets - breaking")
            break
        
        # Try multiple actions to understand failure patterns
        action_attempts = []
        
        for attempt in range(5):  # Try 5 different actions
            target = np.random.choice(valid_targets)
            relation = np.random.randint(0, 7)
            orientation = np.random.randint(0, 4)
            action = np.array([target, relation, orientation])
            
            # Save state before attempting
            saved_positions = env.component_positions.copy()
            saved_occupied = env.occupied_cells.copy()
            saved_mask = env.placed_mask.copy()
            saved_step = env.current_step
            
            # Try the action
            step_result = env.step(action)
            if len(step_result) == 5:
                obs_new, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs_new, reward, done, info = step_result
            
            action_valid = info.get('valid_action', True)
            new_placed_count = np.sum(env.placed_mask[:env.num_components])
            
            action_attempts.append({
                'action': action,
                'valid': action_valid,
                'reward': reward,
                'placed_change': new_placed_count - placed_count,
                'done': done
            })
            
            # Restore state for next attempt
            env.component_positions = saved_positions
            env.occupied_cells = saved_occupied
            env.placed_mask = saved_mask
            env.current_step = saved_step
        
        # Analyze attempts
        valid_attempts = [a for a in action_attempts if a['valid']]
        successful_attempts = [a for a in action_attempts if a['placed_change'] > 0]
        
        print(f"Action attempts: {len(valid_attempts)}/5 valid, {len(successful_attempts)}/5 successful")
        
        if len(valid_attempts) == 0:
            print(f"No valid actions possible")
            break
        
        if len(successful_attempts) == 0:
            print(f" Valid actions but no placement progress")
            placement_failures += 1
            consecutive_failures += 1
            
            if consecutive_failures >= 10:
                print(f"Too many consecutive placement failures - breaking")
                break
            
            # Show details of failed attempts
            for i, attempt in enumerate(valid_attempts):
                print(f"     Attempt {i+1}: action={attempt['action']}, reward={attempt['reward']:.2f}")
        else:
            consecutive_failures = 0
            # Use the best successful attempt
            best_attempt = max(successful_attempts, key=lambda x: x['reward'])
            
            # Actually perform the best action
            step_result = env.step(best_attempt['action'])
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            
            new_placed_count = np.sum(env.placed_mask[:env.num_components])
            print(f"Placed component: {placed_count} → {new_placed_count}")
            print(f"     Action: {best_attempt['action']}, Reward: {reward:.2f}")
        
        step += 1
        
        if done:
            print(f"Episode ended (done=True)")
            break
    
    final_placed = np.sum(env.placed_mask[:env.num_components])
    print(f"\nFINAL RESULTS:")
    print(f"   Placed: {final_placed}/{env.num_components} ({100*final_placed/env.num_components:.1f}%)")
    print(f"   Steps: {step}")
    print(f"   Placement failures: {placement_failures}")
    
    # Analyze grid occupancy
    print(f"\nGRID ANALYSIS:")
    occupied_count = len(env.occupied_cells)
    total_cells = env.grid_size * env.grid_size
    print(f"   Occupied cells: {occupied_count}/{total_cells} ({100*occupied_count/total_cells:.1f}%)")
    
    # Show component positions
    print(f"\nCOMPONENT POSITIONS:")
    for comp_id in range(min(env.num_components, 15)):  # Show first 15
        if comp_id in env.component_positions:
            x, y, orient = env.component_positions[comp_id]
            print(f"     Component {comp_id}: ({x}, {y}) orient={orient}")
        else:
            print(f"     Component {comp_id}: NOT PLACED")

def analyze_grid_space_utilization():
    """Analyze if grid space is the limiting factor."""
    print(f"\nGRID SPACE UTILIZATION ANALYSIS")
    print("=" * 50)
    
    ldo_file = "/home/eli/Documents/Internship/CLARA/data/netlists/sky130_am_ip__ldo_01v8.spice"
    parser = EnhancedSpiceParser()
    circuit_data = parser.parse_spice_file(ldo_file)
    
    full_circuit = convert_parsed_circuit_to_networkx(circuit_data)
    nodes = list(full_circuit.nodes())[:20]
    sampled_circuit = full_circuit.subgraph(nodes).copy()
    sampled_circuit = nx.relabel_nodes(sampled_circuit, {node: i for i, node in enumerate(nodes)})
    
    # Test different grid sizes
    grid_sizes = [15, 20, 25, 30, 40]
    
    for grid_size in grid_sizes:
        print(f"\nGrid size {grid_size}x{grid_size}:")
        
        env = AnalogLayoutEnv(
            grid_size=grid_size,
            max_components=100,
            enable_action_masking=True
        )
        
        result = env.reset(circuit_graph=sampled_circuit)
        obs = result[0] if isinstance(result, tuple) else result
        
        # Calculate theoretical maximum components that could fit
        total_cells = grid_size * grid_size
        
        # Estimate component sizes from circuit
        component_areas = []
        for i in range(env.num_components):
            attrs = sampled_circuit.nodes[i]
            width = attrs.get('width', 1)
            height = attrs.get('height', 1)
            component_areas.append(width * height)
        
        total_component_area = sum(component_areas)
        theoretical_fit = total_cells >= total_component_area
        
        print(f"     Total cells: {total_cells}")
        print(f"     Component area needed: {total_component_area}")
        print(f"     Theoretical fit: {'Yes' if theoretical_fit else 'No'}")
        print(f"     Utilization if all placed: {100*total_component_area/total_cells:.1f}%")
        
        # Quick test
        placed_after_10_steps = simulate_quick_placement(env, sampled_circuit, 10)
        print(f"     Placed after 10 steps: {placed_after_10_steps}/{env.num_components}")

def simulate_quick_placement(env, circuit, max_steps):
    """Quick placement simulation."""
    result = env.reset(circuit_graph=circuit)
    obs = result[0] if isinstance(result, tuple) else result
    
    for step in range(max_steps):
        placed_count = np.sum(env.placed_mask[:env.num_components])
        if placed_count == env.num_components:
            break
        
        if 'action_mask' in obs:
            action_mask = obs['action_mask']
            valid_targets = np.where(action_mask == 1)[0]
            
            if len(valid_targets) > 0:
                target = np.random.choice(valid_targets)
                relation = np.random.randint(0, 7)
                orientation = np.random.randint(0, 4)
                action = np.array([target, relation, orientation])
                
                step_result = env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_result
                
                if done:
                    break
            else:
                break
        else:
            break
    
    return np.sum(env.placed_mask[:env.num_components])

if __name__ == "__main__":
    debug_placement_step_by_step()
    analyze_grid_space_utilization()