#!/usr/bin/env python3
"""
Test script to verify CLARA environment setup and basic functionality.
"""

import sys
import traceback
import numpy as np


def test_environment():
    """Test the basic environment functionality."""
    print("Testing CLARA Environment...")
    
    try:
        from analog_layout_env import AnalogLayoutEnv
        print("Successfully imported AnalogLayoutEnv")
        
        # Create environment
        env = AnalogLayoutEnv(grid_size=10, max_components=5)
        print("Environment created successfully")
        
        # Test reset
        obs = env.reset()
        print("Environment reset successful")
        print(f"Observation keys: {list(obs.keys())}")
        print(f"Component graph shape: {obs['component_graph'].shape}")
        print(f"Placed components shape: {obs['placed_components'].shape}")
        print(f"Netlist features shape: {obs['netlist_features'].shape}")
        
        # Test step
        action = env.action_space.sample()
        print(f"   Sample action: {action}")
        
        obs, reward, done, info = env.step(action)
        print("Environment step successful")
        print(f"Reward: {reward:.3f}")
        print(f"Done: {done}")
        print(f"Info keys: {list(info.keys())}")
        
        env.render()
        
        return True
        
    except Exception as e:
        print(f"Environment test failed: {e}")
        traceback.print_exc()
        return False


def test_circuit_generator():
    """Test the circuit generator."""
    print("\nTesting Circuit Generator...")
    
    try:
        from data.circuit_generator import AnalogCircuitGenerator
        print("Successfully imported AnalogCircuitGenerator")
        
        generator = AnalogCircuitGenerator()
        
        # Test differential pair
        circuit = generator.generate_differential_pair()
        print("Generated differential pair")
        print(f"Nodes: {len(circuit.nodes)}, Edges: {len(circuit.edges)}")
        
        # Test random circuit
        random_circuit = generator.generate_random_circuit(4, 6)
        print("Generated random circuit")
        print(f"Nodes: {len(random_circuit.nodes)}, Edges: {len(random_circuit.edges)}")
        
        return True
        
    except Exception as e:
        print(f"Circuit generator test failed: {e}")
        traceback.print_exc()
        return False


def test_reward_system():
    """Test the reward calculation system."""
    print("\nTesting Reward System...")
    
    try:
        from reward import RewardCalculator
        from data.circuit_generator import AnalogCircuitGenerator
        print("Successfully imported reward components")
        
        # Create sample data
        generator = AnalogCircuitGenerator()
        circuit = generator.generate_differential_pair()
        
        positions = {
            0: (5, 5, 0),
            1: (7, 5, 0),
            2: (6, 7, 0)
        }
        
        # Test reward calculation
        calculator = RewardCalculator()
        total_reward, components = calculator.calculate_total_reward(
            circuit=circuit,
            component_positions=positions,
            grid_size=20,
            num_components=len(circuit.nodes),
            placed_count=len(positions)
        )
        
        print("Reward calculation successful")
        print(f"Total reward: {total_reward:.3f}")
        print(f"Symmetry: {components.symmetry:.3f}")
        print(f"Compactness: {components.compactness:.3f}")
        print(f"Connectivity: {components.connectivity:.3f}")

        return True
        
    except Exception as e:
        print(f"Reward system test failed: {e}")
        traceback.print_exc()
        return False


def test_visualization():
    """Test the visualization system."""
    print("\nTesting Visualization System...")
    
    try:
        from visualize import AnalogLayoutVisualizer
        from data.circuit_generator import AnalogCircuitGenerator
        print("Successfully imported visualization components")
        
        # Create sample data
        generator = AnalogCircuitGenerator()
        circuit = generator.generate_differential_pair()
        
        positions = {
            0: (5, 5, 0),
            1: (7, 5, 0),
            2: (6, 7, 0),
            3: (5, 3, 0),
            4: (7, 3, 0)
        }
        
        # Test visualization
        visualizer = AnalogLayoutVisualizer(grid_size=12, figsize=(8, 6))
        fig = visualizer.visualize_layout(
            circuit, positions, 
            title="Test Layout",
            show_connections=True,
            save_path="test_layout.png"
        )
        
        print("Visualization successful")
        print("Saved test layout to 'test_layout.png'")
        
        # Clean up
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        return True
        
    except Exception as e:
        print(f"Visualization test failed: {e}")
        traceback.print_exc()
        return False


def test_imports():
    """Test all critical imports."""
    print("\nTesting Critical Imports...")
    
    imports_to_test = [
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("gym", "OpenAI Gym"),
        ("networkx", "NetworkX"),
        ("matplotlib", "Matplotlib"),
        ("stable_baselines3", "Stable-Baselines3"),
    ]
    
    optional_imports = [
        ("torch_geometric", "PyTorch Geometric"),
        ("wandb", "Weights & Biases"),
    ]
    
    success_count = 0
    
    for module_name, display_name in imports_to_test:
        try:
            __import__(module_name)
            print(f"{display_name} imported successfully")
            success_count += 1
        except ImportError as e:
            print(f"{display_name} import failed: {e}")
    
    for module_name, display_name in optional_imports:
        try:
            __import__(module_name)
            print(f"{display_name} (optional) imported successfully")
        except ImportError:
            print(f"{display_name} (optional) not available")
    
    return success_count == len(imports_to_test)


def main():
    """Run all tests."""
    print("CLARA System Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Environment Test", test_environment),
        ("Circuit Generator Test", test_circuit_generator),
        ("Reward System Test", test_reward_system),
        ("Visualization Test", test_visualization),
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
        print("All tests passed! CLARA is ready for training.")
        return 0
    else:
        print("Some tests failed. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())