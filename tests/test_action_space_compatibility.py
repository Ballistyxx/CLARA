#!/usr/bin/env python3
"""
Test script to debug action space compatibility issues between 
trained models and actual SPICE circuits.
"""

import numpy as np
import sys
from pathlib import Path
from enhanced_spice_parser import EnhancedSpiceParser
from analog_layout_env import AnalogLayoutEnv
from train_spice_real import SpiceCircuitManager, AnalogLayoutSpiceEnvWrapper

def analyze_action_space_mismatch():
    """Debug action space compatibility between model and circuits."""
    print("ACTION SPACE COMPATIBILITY ANALYSIS")
    print("=" * 60)
    
    # Test circuits
    test_files = [
        "data/netlists/sky130_am_ip__ldo_01v8.spice",
        "data/programmable_pll_subcircuits/NAND.spice"
    ]
    
    # Environment configurations
    configs = [
        {"name": "Standard", "max_components": 10, "grid_size": 15},
        {"name": "Trained Model", "max_components": 100, "grid_size": 20},
        {"name": "Large Circuit", "max_components": 200, "grid_size": 25}
    ]
    
    parser = EnhancedSpiceParser()
    
    for test_file in test_files:
        if not Path(test_file).exists():
            print(f" File not found: {test_file}")
            continue
            
        print(f"\nAnalyzing: {Path(test_file).name}")
        
        # Parse circuit
        try:
            circuit_data = parser.parse_spice_file(test_file)
            print(f"   Raw components: {circuit_data['num_components']}")
            print(f"   Connections: {circuit_data['circuit_graph'].number_of_edges()}")
        except Exception as e:
            print(f"   Parse error: {e}")
            continue
        
        # Test different environment configurations
        for config in configs:
            print(f"\n   Testing with {config['name']} config:")
            print(f"      max_components: {config['max_components']}")
            
            try:
                # Create environment
                env = AnalogLayoutEnv(
                    grid_size=config['grid_size'],
                    max_components=config['max_components']
                )
                
                # Create circuit manager to convert to NetworkX
                circuit_manager = SpiceCircuitManager("/tmp")  # Dummy directory
                circuit_graph = circuit_manager.convert_to_networkx_graph(circuit_data)
                
                print(f"      Circuit nodes: {len(circuit_graph.nodes)}")
                print(f"      Circuit edges: {len(circuit_graph.edges)}")
                
                # Test reset
                result = env.reset(circuit_graph=circuit_graph)
                obs = result[0] if isinstance(result, tuple) else result
                
                print(f"      Reset successful")
                print(f"      Observation shapes:")
                for key, value in obs.items():
                    print(f"         {key}: {value.shape}")
                
                # Test action space compatibility
                action_space = env.action_space
                print(f"      Action space: {action_space}")
                
                # Test several actions to identify issues
                invalid_actions = 0
                valid_actions = 0
                
                for test_step in range(10):
                    # Sample random action
                    action = action_space.sample()
                    
                    # Check action validity
                    target_component_id = action[0]
                    
                    # Check if target component exists and is placed
                    if (target_component_id >= len(circuit_graph.nodes) or
                        not env.placed_mask[target_component_id]):
                        invalid_actions += 1
                        reason = "target not placed" if target_component_id < len(circuit_graph.nodes) else "target out of bounds"
                        print(f"         Action {action}: INVALID ({reason})")
                    else:
                        valid_actions += 1
                        print(f"         Action {action}: valid")
                
                print(f"      Action validity: {valid_actions}/{valid_actions + invalid_actions} valid")
                
                # Test actual step execution
                action = action_space.sample()
                result = env.step(action)
                
                if len(result) == 5:  # Gymnasium
                    obs, reward, terminated, truncated, info = result
                    done = terminated or truncated
                else:  # Gym
                    obs, reward, done, info = result
                
                print(f"      Step test: reward={reward:.3f}, done={done}, valid={info.get('valid_action', 'N/A')}")
                
            except Exception as e:
                print(f"      Environment error: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\nRECOMMENDATIONS:")
    print(f"   1. Model trained for max_components=100 but circuits have varying sizes")
    print(f"   2. Action masking needed to prevent invalid target component references")
    print(f"   3. Environment should handle variable circuit sizes gracefully")
    print(f"   4. Consider dynamic action space sizing based on actual circuit size")


def test_specific_model_on_ldo():
    """Test a specific trained model on the LDO circuit."""
    print(f"\nSPECIFIC MODEL TESTING ON LDO")
    print("=" * 60)
    
    # Use the connectivity-fixed LDO circuit
    ldo_file = "data/netlists/sky130_am_ip__ldo_01v8.spice"
    
    if not Path(ldo_file).exists():
        print(f"LDO file not found: {ldo_file}")
        return
    
    try:
        # Check for existing trained model
        model_paths = [
            "./logs/clara_20250722_185325/clara_final_model.zip",
            "./clara_final_model.zip",
            "./models/clara_best_model.zip"
        ]
        
        model_path = None
        for path in model_paths:
            if Path(path).exists():
                model_path = path
                break
        
        if not model_path:
            print(f" No trained model found. Paths checked:")
            for path in model_paths:
                print(f"      {path}")
            print(f"   Creating simple test without trained model...")
        
        # Parse LDO circuit with fixed connectivity
        parser = EnhancedSpiceParser()
        circuit_data = parser.parse_spice_file(ldo_file)
        
        print(f"LDO Circuit After Connectivity Fix:")
        print(f"   Components: {circuit_data['num_components']}")
        print(f"   Connections: {circuit_data['circuit_graph'].number_of_edges()}")
        
        # Create environment that matches training config
        env = AnalogLayoutEnv(
            grid_size=20,
            max_components=100  # Training config from train_spice_real.py
        )
        
        # Convert to NetworkX format
        circuit_manager = SpiceCircuitManager("/tmp")
        circuit_graph = circuit_manager.convert_to_networkx_graph(circuit_data)
        
        # Sample a smaller subset for testing (simulate what training would see)
        nodes = list(circuit_graph.nodes())
        sampled_nodes = nodes[:min(49, len(nodes))]  # Take up to 49 components
        
        print(f"   Testing with {len(sampled_nodes)} components (sampled from {len(nodes)})")
        
        # Create subgraph
        test_graph = circuit_graph.subgraph(sampled_nodes).copy()
        # Relabel nodes to be sequential starting from 0
        test_graph = nx.relabel_nodes(test_graph, {node: i for i, node in enumerate(sampled_nodes)})
        
        print(f"   Test graph: {len(test_graph.nodes)} nodes, {len(test_graph.edges)} edges")
        
        # Test environment with this circuit
        result = env.reset(circuit_graph=test_graph)
        obs = result[0] if isinstance(result, tuple) else result
        
        print(f"Environment reset successful")
        print(f"   Placed components: {np.sum(env.placed_mask[:env.num_components])}/{env.num_components}")
        
        # Simulate model behavior - test action validity patterns
        print(f"\nSimulating model behavior:")
        
        action_stats = {"valid": 0, "invalid_target": 0, "invalid_placement": 0, "total": 0}
        
        for step in range(20):  # Test 20 steps
            action = env.action_space.sample()
            action_stats["total"] += 1
            
            # Predict what would happen
            target_component_id = action[0]
            
            if (target_component_id >= env.num_components or 
                not env.placed_mask[target_component_id]):
                action_stats["invalid_target"] += 1
                print(f"   Step {step}: Action {action} -> INVALID TARGET")
                continue
            
            # Try the action
            result = env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
            else:
                obs, reward, done, info = result
            
            if info.get('valid_action', True):
                action_stats["valid"] += 1
                placed = np.sum(env.placed_mask[:env.num_components])
                print(f"   Step {step}: Action {action} -> VALID (reward={reward:.2f}, placed={placed}/{env.num_components})")
            else:
                action_stats["invalid_placement"] += 1
                print(f"   Step {step}: Action {action} -> INVALID PLACEMENT")
        
        print(f"\nACTION VALIDITY ANALYSIS:")
        total = action_stats["total"]
        print(f"   Valid actions: {action_stats['valid']}/{total} ({100*action_stats['valid']/total:.1f}%)")
        print(f"   Invalid targets: {action_stats['invalid_target']}/{total} ({100*action_stats['invalid_target']/total:.1f}%)")
        print(f"   Invalid placements: {action_stats['invalid_placement']}/{total} ({100*action_stats['invalid_placement']/total:.1f}%)")
        
        final_placed = np.sum(env.placed_mask[:env.num_components])
        success_rate = final_placed / env.num_components
        print(f"\nFINAL RESULTS:")
        print(f"   Components placed: {final_placed}/{env.num_components} ({100*success_rate:.1f}%)")
        
        if success_rate < 0.8:  # Less than 80% success
            print(f"    Success rate below target (80%)")
            print(f"   SUGGESTED FIXES:")
            print(f"      1. Implement action masking for target component selection")
            print(f"      2. Add curriculum learning starting with smaller circuits")
            print(f"      3. Improve placement position calculation for invalid placements")
            print(f"      4. Increase exploration during training to handle diverse circuits")
        else:
            print(f"   Success rate meets target!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Import required for circuit conversion
    import networkx as nx
    
    analyze_action_space_mismatch()
    test_specific_model_on_ldo()