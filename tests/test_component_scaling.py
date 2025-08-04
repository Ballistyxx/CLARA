#!/usr/bin/env python3
"""
Test component scaling fix.
"""

import numpy as np
import sys
from pathlib import Path
import networkx as nx
from enhanced_spice_parser import EnhancedSpiceParser
from analog_layout_env import AnalogLayoutEnv

def convert_parsed_circuit_to_networkx_with_scaling(circuit_data):
    """Convert parsed SPICE data to NetworkX graph with scaling."""
    G = nx.Graph()
    
    components = circuit_data['components']
    
    # Add nodes with attributes
    for i, comp in enumerate(components):
        clara_type = comp['type_value']
        
        # Use normalized component dimensions to prevent grid overflow
        # Apply square root scaling to reduce extreme sizes
        raw_width = max(0.1, comp['width'])
        raw_length = max(0.1, comp['length'])
        
        # Square root scaling with reasonable bounds
        width = max(1, min(8, int(np.sqrt(raw_width) + 0.5)))
        height = max(1, min(8, int(np.sqrt(raw_length) + 0.5)))
        
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

def test_scaling_improvements():
    """Test the component scaling improvements."""
    print("COMPONENT SCALING TEST")
    print("=" * 50)
    
    ldo_file = "/home/eli/Documents/Internship/CLARA/data/netlists/sky130_am_ip__ldo_01v8.spice"
    
    if not Path(ldo_file).exists():
        print(f"File not found: {ldo_file}")
        return
    
    # Parse LDO circuit
    parser = EnhancedSpiceParser()
    circuit_data = parser.parse_spice_file(ldo_file)
    
    print(f"Original Circuit: {circuit_data['num_components']} components")
    
    # Compare original vs scaled component sizes
    components = circuit_data['components'][:30]  # First 30 for comparison
    
    print(f"\nCOMPONENT SIZE COMPARISON (first 30):")
    print(f"{'Name':<20} {'Original W×L':<15} {'Scaled W×H':<12} {'Area Reduction'}")
    print("-" * 70)
    
    total_original_area = 0
    total_scaled_area = 0
    
    for comp in components:
        original_width = comp['width']
        original_length = comp['length']
        original_area = original_width * original_length
        
        # Apply scaling
        raw_width = max(0.1, original_width)
        raw_length = max(0.1, original_length)
        scaled_width = max(1, min(8, int(np.sqrt(raw_width) + 0.5)))
        scaled_height = max(1, min(8, int(np.sqrt(raw_length) + 0.5)))
        scaled_area = scaled_width * scaled_height
        
        reduction = (original_area - scaled_area) / max(original_area, 0.001) * 100
        
        total_original_area += original_area
        total_scaled_area += scaled_area
        
        print(f"{comp['name'][:18]:<20} {original_width:.1f}×{original_length:.1f} ({original_area:.0f}){'':<4} {scaled_width}×{scaled_height} ({scaled_area}){'':<6} {reduction:>6.1f}%")
    
    total_reduction = (total_original_area - total_scaled_area) / max(total_original_area, 0.001) * 100
    
    print(f"\nSCALING SUMMARY:")
    print(f"   Original total area: {total_original_area:.0f}")
    print(f"   Scaled total area: {total_scaled_area:.0f}")
    print(f"   Area reduction: {total_reduction:.1f}%")
    
    # Test placement with scaled components
    full_circuit = convert_parsed_circuit_to_networkx_with_scaling(circuit_data)
    nodes = list(full_circuit.nodes())[:30]  # Test with 30 components
    sampled_circuit = full_circuit.subgraph(nodes).copy()
    sampled_circuit = nx.relabel_nodes(sampled_circuit, {node: i for i, node in enumerate(nodes)})
    
    print(f"\nPLACEMENT TEST WITH SCALED COMPONENTS:")
    
    # Test placement
    env = AnalogLayoutEnv(
        grid_size=30,  # Should be plenty now
        max_components=100,
        enable_action_masking=True
    )
    
    result = env.reset(circuit_graph=sampled_circuit)
    obs = result[0] if isinstance(result, tuple) else result
    
    # Calculate theoretical space usage
    component_areas = []
    for i in range(env.num_components):
        attrs = sampled_circuit.nodes[i]
        width = attrs.get('width', 1)
        height = attrs.get('height', 1)
        component_areas.append(width * height)
    
    total_component_area = sum(component_areas)
    total_grid_area = env.grid_size * env.grid_size
    utilization = total_component_area / total_grid_area
    
    print(f"   Circuit: {env.num_components} components")
    print(f"   Grid: {env.grid_size}x{env.grid_size} = {total_grid_area} cells")
    print(f"   Component area: {total_component_area} cells")
    print(f"   Theoretical utilization: {100*utilization:.1f}%")
    
    # Run placement test
    steps = 0
    max_steps = 60
    
    while steps < max_steps:
        placed_count = np.sum(env.placed_mask[:env.num_components])
        if placed_count == env.num_components:
            break
        
        # Use intelligent action selection
        if 'action_mask' in obs:
            action_mask = obs['action_mask']
            valid_targets = np.where(action_mask == 1)[0]
            
            if len(valid_targets) > 0:
                target = np.random.choice(valid_targets)
                relation = np.random.randint(0, 7)
                orientation = np.random.randint(0, 4)
                action = np.array([target, relation, orientation])
            else:
                print(f"   No valid targets at step {steps}")
                break
        else:
            action = env.action_space.sample()
        
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
    
    print(f"   Results: {final_placed}/{env.num_components} placed ({100*success_rate:.1f}%)")
    print(f"   Steps: {steps}")
    
    if success_rate > 0.8:
        print(f"   EXCELLENT: >80% success rate achieved!")
    elif success_rate > 0.6:
        print(f"   GOOD: >60% success rate")
    else:
        print(f"    Still needs improvement")

if __name__ == "__main__":
    test_scaling_improvements()