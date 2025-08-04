#!/usr/bin/env python3
"""
Test script for SPICE-based training pipeline.
"""

import sys
from pathlib import Path
from spice_parser import SpiceParser
from train_spice import SpiceLayoutEnvWrapper, load_spice_circuits


def test_spice_parsing():
    """Test SPICE file parsing."""
    print("Testing SPICE parsing...")
    
    parser = SpiceParser()
    spice_file = "./schematics/sky130_ef_ip__simple_por.spice"
    
    if not Path(spice_file).exists():
        print(f"SPICE file not found: {spice_file}")
        return False
    
    try:
        circuit = parser.parse_spice_file(spice_file)
        print(f"Parsed circuit with {len(circuit.nodes)} components")
        return True
    except Exception as e:
        print(f"Parsing failed: {e}")
        return False


def test_spice_environment():
    """Test SPICE environment wrapper."""
    print("\nTesting SPICE environment...")
    
    try:
        # Load circuits
        circuits = load_spice_circuits("./schematics")
        if not circuits:
            print("No circuits loaded")
            return False
        
        print(f"Loaded {len(circuits)} circuits")
        
        # Create environment
        env = SpiceLayoutEnvWrapper(
            grid_size=20,
            max_components=15,
            spice_circuits=circuits
        )
        
        # Test reset
        result = env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
        
        print(f"Environment reset successful")
        print(f"   Circuit components: {env.num_components}")
        print(f"   Observation keys: {list(obs.keys())}")
        
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
            
            print(f"   Step {step + 1}: reward={reward:.2f}, done={done}, "
                  f"placed={len(env.component_positions)}")
            
            if done:
                break
        
        print(f"Environment test completed. Total reward: {total_reward:.2f}")
        return True
        
    except Exception as e:
        print(f"Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_circuit_filtering():
    """Test circuit filtering and size management."""
    print("\nTesting circuit filtering...")
    
    try:
        circuits = load_spice_circuits("./schematics")
        
        if not circuits:
            print("No circuits available for filtering test")
            return False
        
        # Test with different max_components settings
        for max_comp in [5, 10, 15]:
            env = SpiceLayoutEnvWrapper(
                grid_size=15,
                max_components=max_comp,
                spice_circuits=circuits
            )
            
            result = env.reset()
            if isinstance(result, tuple):
                obs, info = result
            else:
                obs = result
            
            print(f"   Max {max_comp} components: got {env.num_components} components")
            
            if env.num_components > max_comp:
                print(f"Circuit not properly filtered! {env.num_components} > {max_comp}")
                return False
        
        print("Circuit filtering works correctly")
        return True
        
    except Exception as e:
        print(f"Circuit filtering test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("CLARA SPICE Training Test Suite")
    print("="*50)
    
    tests = [
        ("SPICE Parsing", test_spice_parsing),
        ("SPICE Environment", test_spice_environment),
        ("Circuit Filtering", test_circuit_filtering),
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
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    
    for i, (test_name, _) in enumerate(tests):
        status = "PASS" if results[i] else "FAIL"
        print(f"{test_name}: {status}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! SPICE training is ready.")
        print("\nTo start SPICE-based training, run:")
        print("  python3 train_spice.py")
        return 0
    else:
        print(" Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())