#!/usr/bin/env python3
"""
Test LDO circuit with action masking enabled.
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

def intelligent_action_selection(env, obs, temperature=1.0):
    """More intelligent action selection using action mask and heuristics."""
    if 'action_mask' not in obs:
        return env.action_space.sample()
    
    action_mask = obs['action_mask']
    valid_targets = np.where(action_mask == 1)[0]
    
    if len(valid_targets) == 0:
        return env.action_space.sample()
    
    # Prefer targets that are already well-connected or central
    target_scores = []
    for target in valid_targets:
        # Score based on connectivity (components with more connections are better references)
        connections = np.sum(obs['component_graph'][target])
        # Add some randomness
        score = connections + np.random.exponential(temperature)
        target_scores.append(score)
    
    # Select target probabilistically based on scores
    if len(target_scores) > 0:
        target_probs = np.array(target_scores)
        target_probs = target_probs / np.sum(target_probs)
        target = np.random.choice(valid_targets, p=target_probs)
    else:
        target = np.random.choice(valid_targets)
    
    # Select relation and orientation
    relation = np.random.randint(0, 7)
    orientation = np.random.randint(0, 4)
    
    return np.array([target, relation, orientation])

def test_ldo_with_intelligent_masking():
    """Test LDO circuit with intelligent action selection."""
    print("LDO CIRCUIT TEST WITH INTELLIGENT ACTION MASKING")
    print("=" * 60)
    
    ldo_file = "data/netlists/sky130_am_ip__ldo_01v8.spice"
    
    if not Path(ldo_file).exists():
        print(f"File not found: {ldo_file}")
        return
    
    # Parse LDO circuit
    parser = EnhancedSpiceParser()
    circuit_data = parser.parse_spice_file(ldo_file)
    
    print(f"Full LDO Circuit: {circuit_data['num_components']} components, {circuit_data['circuit_graph'].number_of_edges()} edges")
    
    # Test different circuit sizes
    test_sizes = [20, 30, 49, 60]
    
    full_circuit = convert_parsed_circuit_to_networkx(circuit_data)
    
    for target_size in test_sizes:
        if target_size > len(full_circuit.nodes):
            continue
            
        print(f"\nTesting with {target_size} components:")
        
        # Sample circuit
        nodes = list(full_circuit.nodes())[:target_size]
        sampled_circuit = full_circuit.subgraph(nodes).copy()
        sampled_circuit = nx.relabel_nodes(sampled_circuit, {node: i for i, node in enumerate(nodes)})
        
        # Create environment with action masking
        env = AnalogLayoutEnv(
            grid_size=25,  # Larger grid for bigger circuits
            max_components=100,
            enable_action_masking=True
        )
        
        # Run multiple test episodes
        success_rates = []
        
        for episode in range(5):
            result = env.reset(circuit_graph=sampled_circuit)
            obs = result[0] if isinstance(result, tuple) else result
            
            placed_history = [1]  # Start with 1 component placed
            steps = 0
            max_steps = target_size * 3  # Allow more steps for larger circuits
            
            while steps < max_steps:
                # Check if done
                placed_count = np.sum(env.placed_mask[:env.num_components])
                if placed_count == env.num_components:
                    break
                
                # Use intelligent action selection
                action = intelligent_action_selection(env, obs, temperature=0.5)
                
                # Step
                step_result = env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_result
                
                steps += 1
                placed_count = np.sum(env.placed_mask[:env.num_components])
                placed_history.append(placed_count)
                
                if done:
                    break
            
            final_placed = np.sum(env.placed_mask[:env.num_components])
            success_rate = final_placed / env.num_components
            success_rates.append(success_rate)
            
            print(f"     Episode {episode + 1}: {final_placed}/{env.num_components} placed ({100*success_rate:.1f}%) in {steps} steps")
        
        avg_success = np.mean(success_rates)
        std_success = np.std(success_rates)
        
        print(f"   Average success: {100*avg_success:.1f}% ± {100*std_success:.1f}%")
        
        # Check if we meet the original target
        if target_size == 49 and avg_success >= 0.8:
            print(f"   MEETS TARGET: >80% success on 49-component circuit!")
        elif target_size == 49:
            print(f"    Below target: {100*avg_success:.1f}% < 80% on 49-component circuit")

def analyze_placement_strategies():
    """Analyze different placement strategies."""
    print(f"\nPLACEMENT STRATEGY ANALYSIS")
    print("=" * 60)
    
    ldo_file = "data/netlists/sky130_am_ip__ldo_01v8.spice"
    parser = EnhancedSpiceParser()
    circuit_data = parser.parse_spice_file(ldo_file)
    
    full_circuit = convert_parsed_circuit_to_networkx(circuit_data)
    nodes = list(full_circuit.nodes())[:30]  # Test with 30 components
    sampled_circuit = full_circuit.subgraph(nodes).copy()
    sampled_circuit = nx.relabel_nodes(sampled_circuit, {node: i for i, node in enumerate(nodes)})
    
    strategies = [
        {"name": "Random Selection", "temperature": 10.0},
        {"name": "Moderate Intelligence", "temperature": 1.0},
        {"name": "High Intelligence", "temperature": 0.3}
    ]
    
    for strategy in strategies:
        print(f"\n{strategy['name']}:")
        
        success_rates = []
        
        for episode in range(3):
            env = AnalogLayoutEnv(
                grid_size=25,
                max_components=100,
                enable_action_masking=True
            )
            
            result = env.reset(circuit_graph=sampled_circuit)
            obs = result[0] if isinstance(result, tuple) else result
            
            steps = 0
            max_steps = 60
            
            while steps < max_steps:
                placed_count = np.sum(env.placed_mask[:env.num_components])
                if placed_count == env.num_components:
                    break
                
                action = intelligent_action_selection(env, obs, temperature=strategy['temperature'])
                
                step_result = env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_result
                
                steps += 1
                
                if done:
                    break
            
            final_placed = np.sum(env.placed_mask[:env.num_components])
            success_rate = final_placed / env.num_components
            success_rates.append(success_rate)
        
        avg_success = np.mean(success_rates)
        print(f"     Average success: {100*avg_success:.1f}%")

if __name__ == "__main__":
    test_ldo_with_intelligent_masking()
    analyze_placement_strategies()