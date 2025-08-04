#!/usr/bin/env python3
"""
Direct test of action space compatibility issues.
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
        # Map SPICE types to CLARA types
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

def test_ldo_action_space():
    """Test LDO circuit with action space analysis."""
    print("LDO ACTION SPACE DEBUG TEST")
    print("=" * 50)
    
    ldo_file = "/home/eli/Documents/Internship/CLARA/data/netlists/sky130_am_ip__ldo_01v8.spice"
    
    if not Path(ldo_file).exists():
        print(f"File not found: {ldo_file}")
        return
    
    # Parse LDO circuit
    parser = EnhancedSpiceParser()
    circuit_data = parser.parse_spice_file(ldo_file)
    
    print(f"Circuit Statistics:")
    print(f"   Total components: {circuit_data['num_components']}")
    print(f"   Total connections: {circuit_data['circuit_graph'].number_of_edges()}")
    
    # Convert to NetworkX
    full_circuit = convert_parsed_circuit_to_networkx(circuit_data)
    print(f"   NetworkX nodes: {len(full_circuit.nodes)}")
    print(f"   NetworkX edges: {len(full_circuit.edges)}")
    
    # Test different circuit sampling strategies
    test_configs = [
        {"name": "Full Circuit", "max_nodes": len(full_circuit.nodes), "max_components": 200},
        {"name": "Sampled 49", "max_nodes": 49, "max_components": 100},
        {"name": "Sampled 20", "max_nodes": 20, "max_components": 50},
        {"name": "Sampled 10", "max_nodes": 10, "max_components": 20}
    ]
    
    for config in test_configs:
        print(f"\nTesting: {config['name']}")
        
        # Create sampled circuit
        nodes = list(full_circuit.nodes())[:config['max_nodes']]
        if len(nodes) < len(full_circuit.nodes()):
            sampled_circuit = full_circuit.subgraph(nodes).copy()
            # Relabel to sequential IDs
            sampled_circuit = nx.relabel_nodes(sampled_circuit, {node: i for i, node in enumerate(nodes)})
        else:
            sampled_circuit = full_circuit
        
        print(f"   Circuit: {len(sampled_circuit.nodes)} nodes, {len(sampled_circuit.edges)} edges")
        
        # Create environment
        env = AnalogLayoutEnv(
            grid_size=20,
            max_components=config['max_components']
        )
        
        # Test reset
        try:
            result = env.reset(circuit_graph=sampled_circuit)
            obs = result[0] if isinstance(result, tuple) else result
            print(f"   Reset successful")
            
            # Analyze initial state
            placed_count = np.sum(env.placed_mask[:env.num_components])
            print(f"   Initial placed: {placed_count}/{env.num_components}")
            
            # Test action validity over many samples
            valid_actions = 0
            invalid_target = 0
            invalid_placement = 0
            total_tests = 100  # Sample 100 actions
            
            for _ in range(total_tests):
                action = env.action_space.sample()
                target_id, relation, orientation = action
                
                # Check target validity
                if target_id >= env.num_components or not env.placed_mask[target_id]:
                    invalid_target += 1
                    continue
                
                # Test the action (but reset state after each test)
                saved_positions = env.component_positions.copy()
                saved_occupied = env.occupied_cells.copy()
                saved_mask = env.placed_mask.copy()
                saved_step = env.current_step
                
                step_result = env.step(action)
                if len(step_result) == 5:
                    _, _, _, _, info = step_result
                else:
                    _, _, _, info = step_result
                
                if info.get('valid_action', True):
                    valid_actions += 1
                else:
                    invalid_placement += 1
                
                # Restore state
                env.component_positions = saved_positions
                env.occupied_cells = saved_occupied
                env.placed_mask = saved_mask
                env.current_step = saved_step
            
            print(f"   Action validity ({total_tests} samples):")
            print(f"     Valid: {valid_actions}/{total_tests} ({100*valid_actions/total_tests:.1f}%)")
            print(f"     Invalid target: {invalid_target}/{total_tests} ({100*invalid_target/total_tests:.1f}%)")
            print(f"     Invalid placement: {invalid_placement}/{total_tests} ({100*invalid_placement/total_tests:.1f}%)")
            
            # Simulate sequential placement
            print(f"   Sequential placement test:")
            episode_result = simulate_placement_episode(env, sampled_circuit, max_steps=50)
            print(f"     Final placed: {episode_result['placed']}/{episode_result['total']} ({100*episode_result['success_rate']:.1f}%)")
            print(f"     Steps taken: {episode_result['steps']}")
            print(f"     Invalid actions: {episode_result['invalid_actions']}")
            
        except Exception as e:
            print(f"   Test failed: {e}")
            import traceback
            traceback.print_exc()

def simulate_placement_episode(env, circuit_graph, max_steps=50):
    """Simulate a placement episode to understand why models get stuck."""
    result = env.reset(circuit_graph=circuit_graph)
    obs = result[0] if isinstance(result, tuple) else result
    
    steps = 0
    invalid_actions = 0
    
    while steps < max_steps:
        # Check if done
        placed_count = np.sum(env.placed_mask[:env.num_components])
        if placed_count == env.num_components:
            break  # All placed successfully
        
        # Sample action
        action = env.action_space.sample()
        
        # Step
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
        
        steps += 1
        
        if not info.get('valid_action', True):
            invalid_actions += 1
        
        if done:
            break
    
    final_placed = np.sum(env.placed_mask[:env.num_components])
    
    return {
        'placed': final_placed,
        'total': env.num_components,
        'success_rate': final_placed / env.num_components,
        'steps': steps,
        'invalid_actions': invalid_actions
    }

def analyze_action_masking_potential():
    """Analyze how action masking could improve performance."""
    print(f"\nACTION MASKING ANALYSIS")
    print("=" * 50)
    
    ldo_file = "/home/eli/Documents/Internship/CLARA/data/netlists/sky130_am_ip__ldo_01v8.spice"
    parser = EnhancedSpiceParser()
    circuit_data = parser.parse_spice_file(ldo_file)
    
    # Test with smaller circuit
    full_circuit = convert_parsed_circuit_to_networkx(circuit_data)
    nodes = list(full_circuit.nodes())[:20]  # Use 20 components
    sampled_circuit = full_circuit.subgraph(nodes).copy()
    sampled_circuit = nx.relabel_nodes(sampled_circuit, {node: i for i, node in enumerate(nodes)})
    
    env = AnalogLayoutEnv(grid_size=20, max_components=50)
    result = env.reset(circuit_graph=sampled_circuit)
    obs = result[0] if isinstance(result, tuple) else result
    
    print(f"Circuit: {env.num_components} components")
    
    # Simulate with and without action masking
    for use_masking in [False, True]:
        env_copy = AnalogLayoutEnv(grid_size=20, max_components=50)
        result = env_copy.reset(circuit_graph=sampled_circuit)
        
        valid_actions = 0
        total_actions = 0
        
        for step in range(20):  # Test 20 steps
            if use_masking:
                # Get valid targets (placed components)
                valid_targets = [i for i in range(env_copy.num_components) if env_copy.placed_mask[i]]
                if not valid_targets:
                    break
                
                # Create masked action
                target = np.random.choice(valid_targets)
                relation = np.random.randint(0, 7)  # 7 spatial relations
                orientation = np.random.randint(0, 4)  # 4 orientations
                action = np.array([target, relation, orientation])
            else:
                action = env_copy.action_space.sample()
            
            step_result = env_copy.step(action)
            if len(step_result) == 5:
                _, _, _, _, info = step_result
            else:
                _, _, _, info = step_result
            
            total_actions += 1
            if info.get('valid_action', True):
                valid_actions += 1
        
        final_placed = np.sum(env_copy.placed_mask[:env_copy.num_components])
        print(f"{'With' if use_masking else 'Without'} action masking:")
        print(f"   Valid actions: {valid_actions}/{total_actions} ({100*valid_actions/total_actions:.1f}%)")
        print(f"   Final placed: {final_placed}/{env_copy.num_components} ({100*final_placed/env_copy.num_components:.1f}%)")

if __name__ == "__main__":
    test_ldo_action_space()
    analyze_action_masking_potential()