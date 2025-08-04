#!/usr/bin/env python3
"""
Test script for grid-agnostic environment conversion.
Tests the new component list observation format.
"""

import numpy as np
import networkx as nx
from analog_layout_env import AnalogLayoutEnv


def test_environment_basic():
    """Test basic environment functionality with new observation format."""
    print("Testing grid-agnostic environment...")
    
    try:
        # Create environment
        env = AnalogLayoutEnv(grid_size=16, max_components=8)
        
        # Test reset
        result = env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
        
        print(f"Environment reset successful")
        print(f"   Observation keys: {list(obs.keys())}")
        
        # Check new observation format
        assert "placed_components_list" in obs, "Missing placed_components_list key"
        assert "placed_components" not in obs, "Old placed_components key still present"
        
        # Check shapes
        comp_list_shape = obs["placed_components_list"].shape
        expected_shape = (env.max_components, 4)
        assert comp_list_shape == expected_shape, f"Wrong shape: {comp_list_shape} vs {expected_shape}"
        
        print(f"New observation format correct")
        print(f"   placed_components_list shape: {comp_list_shape}")
        
        # Check initial placement (first component should be placed)
        comp_list = obs["placed_components_list"]
        first_comp = comp_list[0]
        if first_comp[3] == 1.0:  # Valid flag
            print(f"First component placed: [{first_comp[0]:.2f}, {first_comp[1]:.2f}, {first_comp[2]:.2f}]")
        
        # Test a few steps
        total_reward = 0
        for step in range(5):
            action = env.action_space.sample()
            result = env.step(action)
            
            if len(result) == 5:  # Gymnasium format
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:  # Gym format
                obs, reward, done, info = result
            
            total_reward += reward
            
            # Check observation format
            assert "placed_components_list" in obs
            comp_list = obs["placed_components_list"]
            
            placed_count = np.sum(comp_list[:, 3] == 1.0)  # Count valid components
            
            print(f"   Step {step + 1}: reward={reward:.2f}, done={done}, placed={placed_count}")
            
            if done:
                break
        
        print(f"Environment test completed. Total reward: {total_reward:.2f}")
        return True
        
    except Exception as e:
        print(f"Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_grid_transferability():
    """Test that the same circuit works on different grid sizes."""
    print("\nTesting grid transferability...")
    
    try:
        # Create a small test circuit
        circuit = nx.Graph()
        circuit.add_node(0, component_type=0, width=2, height=1, matched_component=-1)
        circuit.add_node(1, component_type=1, width=2, height=1, matched_component=-1)
        circuit.add_node(2, component_type=2, width=1, height=2, matched_component=-1)
        circuit.add_edge(0, 1)
        circuit.add_edge(1, 2)
        
        # Test on different grid sizes
        grid_sizes = [16, 32, 64]
        results = {}
        
        for grid_size in grid_sizes:
            print(f"   Testing grid size {grid_size}x{grid_size}...")
            
            env = AnalogLayoutEnv(grid_size=grid_size, max_components=8)
            result = env.reset(circuit_graph=circuit)
            
            if isinstance(result, tuple):
                obs, info = result
            else:
                obs = result
            
            # Check that observations are normalized consistently
            comp_list = obs["placed_components_list"]
            first_comp = comp_list[0]
            
            if first_comp[3] == 1.0:  # Component is placed
                # Positions should be normalized [0,1] regardless of grid size
                assert 0 <= first_comp[0] <= 1, f"X position not normalized: {first_comp[0]}"
                assert 0 <= first_comp[1] <= 1, f"Y position not normalized: {first_comp[1]}"
                
                results[grid_size] = {
                    'norm_x': first_comp[0],
                    'norm_y': first_comp[1],
                    'actual_x': first_comp[0] * grid_size,
                    'actual_y': first_comp[1] * grid_size
                }
        
        print(f"Grid transferability test passed")
        for grid_size, data in results.items():
            print(f"   Grid {grid_size}: norm=({data['norm_x']:.3f}, {data['norm_y']:.3f}) "
                  f"actual=({data['actual_x']:.1f}, {data['actual_y']:.1f})")
        
        return True
        
    except Exception as e:
        print(f"Grid transferability test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_efficiency():
    """Test that memory usage is reduced for large grids."""
    print("\nTesting memory efficiency...")
    
    try:
        import sys
        
        # Compare memory usage of observations
        env_small = AnalogLayoutEnv(grid_size=16, max_components=8)
        env_large = AnalogLayoutEnv(grid_size=64, max_components=8)
        
        result_small = env_small.reset()
        obs_small = result_small[0] if isinstance(result_small, tuple) else result_small
        
        result_large = env_large.reset()
        obs_large = result_large[0] if isinstance(result_large, tuple) else result_large
        
        # Calculate sizes
        comp_list_size_small = obs_small["placed_components_list"].nbytes
        comp_list_size_large = obs_large["placed_components_list"].nbytes
        
        # Old grid format would be (grid_size, grid_size, max_components)
        old_format_small = 16 * 16 * 8  # bytes (int8)
        old_format_large = 64 * 64 * 8  # bytes (int8)
        
        print(f"Memory efficiency test results:")
        print(f"   New format - 16x16: {comp_list_size_small} bytes")
        print(f"   New format - 64x64: {comp_list_size_large} bytes")
        print(f"   Old format - 16x16: {old_format_small} bytes")
        print(f"   Old format - 64x64: {old_format_large} bytes")
        
        # New format should be constant size regardless of grid
        assert comp_list_size_small == comp_list_size_large, "Component list size should be constant"
        
        # New format should be much smaller for large grids
        large_grid_savings = (old_format_large - comp_list_size_large) / old_format_large
        print(f"   Memory savings for 64x64 grid: {large_grid_savings:.1%}")
        
        assert large_grid_savings > 0.8, f"Expected >80% savings, got {large_grid_savings:.1%}"
        
        return True
        
    except Exception as e:
        print(f"Memory efficiency test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("CLARA Grid-Agnostic Environment Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Environment", test_environment_basic),
        ("Grid Transferability", test_grid_transferability),
        ("Memory Efficiency", test_memory_efficiency),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"{test_name} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    for i, (test_name, _) in enumerate(tests):
        status = "PASS" if results[i] else "FAIL"
        print(f"{test_name}: {status}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! Grid-agnostic environment is ready.")
        return 0
    else:
        print(" Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())