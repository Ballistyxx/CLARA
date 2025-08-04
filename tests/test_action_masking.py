#!/usr/bin/env python3
"""
Test the new action masking functionality.
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

def masked_action_sampling(env, num_samples=100):
    """Sample actions using action mask."""
    valid_actions = 0
    invalid_actions = 0
    
    for _ in range(num_samples):
        obs = env._get_observation()
        
        if 'action_mask' in obs:
            # Use action mask for target selection
            action_mask = obs['action_mask']
            valid_targets = np.where(action_mask == 1)[0]
            
            if len(valid_targets) > 0:
                # Sample valid target
                target = np.random.choice(valid_targets)
                relation = np.random.randint(0, 7)
                orientation = np.random.randint(0, 4)
                action = np.array([target, relation, orientation])
                
                # Test this action
                saved_state = save_env_state(env)
                step_result = env.step(action)
                if len(step_result) == 5:
                    _, _, _, _, info = step_result
                else:
                    _, _, _, info = step_result
                
                if info.get('valid_action', True):
                    valid_actions += 1
                else:
                    invalid_actions += 1
                
                restore_env_state(env, saved_state)
            else:
                # No valid targets
                invalid_actions += 1
        else:
            # No action masking - sample randomly
            action = env.action_space.sample()
            
            saved_state = save_env_state(env)
            step_result = env.step(action)
            if len(step_result) == 5:
                _, _, _, _, info = step_result
            else:
                _, _, _, info = step_result
            
            if info.get('valid_action', True):
                valid_actions += 1
            else:
                invalid_actions += 1
            
            restore_env_state(env, saved_state)
    
    return valid_actions, invalid_actions

def save_env_state(env):
    """Save environment state for restoration."""
    return {
        'component_positions': env.component_positions.copy(),
        'occupied_cells': env.occupied_cells.copy(),
        'placed_mask': env.placed_mask.copy(),
        'current_step': env.current_step
    }

def restore_env_state(env, state):
    """Restore environment state."""
    env.component_positions = state['component_positions']
    env.occupied_cells = state['occupied_cells']
    env.placed_mask = state['placed_mask']
    env.current_step = state['current_step']

def test_action_masking():
    """Test action masking with real circuit."""
    print("ACTION MASKING TEST")
    print("=" * 50)
    
    ldo_file = "/home/eli/Documents/Internship/CLARA/data/netlists/sky130_am_ip__ldo_01v8.spice"
    
    if not Path(ldo_file).exists():
        print(f"File not found: {ldo_file}")
        return
    
    # Parse LDO circuit
    parser = EnhancedSpiceParser()
    circuit_data = parser.parse_spice_file(ldo_file)
    
    # Convert to NetworkX and sample subset
    full_circuit = convert_parsed_circuit_to_networkx(circuit_data)
    nodes = list(full_circuit.nodes())[:20]  # Use 20 components for testing
    sampled_circuit = full_circuit.subgraph(nodes).copy()
    sampled_circuit = nx.relabel_nodes(sampled_circuit, {node: i for i, node in enumerate(nodes)})
    
    print(f"Test circuit: {len(sampled_circuit.nodes)} nodes, {len(sampled_circuit.edges)} edges")
    
    # Test with and without action masking
    configs = [
        {"name": "Without Action Masking", "enable_masking": False},
        {"name": "With Action Masking", "enable_masking": True}
    ]
    
    for config in configs:
        print(f"\n{config['name']}:")
        
        # Create environment
        env = AnalogLayoutEnv(
            grid_size=20,
            max_components=50,
            enable_action_masking=config['enable_masking']
        )
        
        # Reset with test circuit
        result = env.reset(circuit_graph=sampled_circuit)
        obs = result[0] if isinstance(result, tuple) else result
        
        print(f"   Initial placed: {np.sum(env.placed_mask[:env.num_components])}/{env.num_components}")
        
        # Check observation space
        if 'action_mask' in obs:
            action_mask = obs['action_mask']
            valid_targets = np.sum(action_mask)
            print(f"   Valid targets: {valid_targets}/{env.max_components}")
        else:
            print(f"   No action mask in observation")
        
        # Test action sampling
        valid, invalid = masked_action_sampling(env, num_samples=50)
        total = valid + invalid
        print(f"   Action validity: {valid}/{total} ({100*valid/total:.1f}%)")
        
        # Test full placement episode
        episode_result = test_placement_episode(env, sampled_circuit)
        print(f"   Episode result: {episode_result['placed']}/{episode_result['total']} placed")
        print(f"                  {episode_result['steps']} steps, {episode_result['invalid_actions']} invalid actions")

def test_placement_episode(env, circuit_graph, max_steps=30):
    """Test a full placement episode."""
    result = env.reset(circuit_graph=circuit_graph)
    obs = result[0] if isinstance(result, tuple) else result
    
    steps = 0
    invalid_actions = 0
    
    while steps < max_steps:
        # Check if done
        placed_count = np.sum(env.placed_mask[:env.num_components])
        if placed_count == env.num_components:
            break
        
        # Sample action using mask if available
        if 'action_mask' in obs:
            action_mask = obs['action_mask']
            valid_targets = np.where(action_mask == 1)[0]
            
            if len(valid_targets) > 0:
                target = np.random.choice(valid_targets)
                relation = np.random.randint(0, 7)
                orientation = np.random.randint(0, 4)
                action = np.array([target, relation, orientation])
            else:
                action = env.action_space.sample()
        else:
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
        'steps': steps,
        'invalid_actions': invalid_actions
    }

if __name__ == "__main__":
    test_action_masking()