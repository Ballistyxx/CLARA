#!/usr/bin/env python3
"""
Test script for grid-agnostic policy conversion.
Tests the new ComponentListEncoder and policy integration.
"""

import torch
import numpy as np
import networkx as nx
from analog_layout_env import AnalogLayoutEnv
from policy import AnalogLayoutPolicy, ComponentListEncoder
from stable_baselines3.common.type_aliases import Schedule


def test_component_list_encoder():
    """Test the ComponentListEncoder directly."""
    print("Testing ComponentListEncoder...")
    
    try:
        max_components = 8
        batch_size = 4
        
        # Create encoder
        encoder = ComponentListEncoder(max_components=max_components, output_dim=64)
        
        # Create test data - component list format
        component_list = torch.randn(batch_size, max_components, 4)
        
        # Set valid flags for first few components
        component_list[:, :3, 3] = 1.0  # First 3 components are valid
        component_list[:, 3:, 3] = -1.0  # Rest are invalid
        
        # Forward pass
        output = encoder(component_list)
        
        expected_shape = (batch_size, 64)
        assert output.shape == expected_shape, f"Wrong output shape: {output.shape} vs {expected_shape}"
        
        print(f"ComponentListEncoder test passed")
        print(f"   Input shape: {component_list.shape}")
        print(f"   Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"ComponentListEncoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_policy_creation():
    """Test creating the policy with new observation space."""
    print("\nTesting policy creation...")
    
    try:
        # Create environment to get observation/action spaces
        env = AnalogLayoutEnv(grid_size=16, max_components=8)
        
        # Create learning rate schedule
        def lr_schedule(progress_remaining: float) -> float:
            return 1e-4
        
        # Create policy
        policy = AnalogLayoutPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lr_schedule
        )
        
        print(f"Policy creation successful")
        print(f"   Policy type: {type(policy).__name__}")
        print(f"   Max components: {policy.max_components}")
        print(f"   Node features: {policy.node_features}")
        
        # Check that it uses ComponentListEncoder
        assert isinstance(policy.placement_encoder, ComponentListEncoder), "Should use ComponentListEncoder"
        
        return True
        
    except Exception as e:
        print(f"Policy creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_policy_forward_pass():
    """Test policy forward pass with new observation format."""
    print("\nTesting policy forward pass...")
    
    try:
        # Create environment and policy
        env = AnalogLayoutEnv(grid_size=16, max_components=8)
        
        def lr_schedule(progress_remaining: float) -> float:
            return 1e-4
        
        policy = AnalogLayoutPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lr_schedule
        )
        
        # Get observation from environment
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        
        # Convert to batch format for policy
        batch_obs = {}
        for key, value in obs.items():
            batch_obs[key] = torch.tensor(value).unsqueeze(0).float()
        
        print(f"   Batch observation keys: {list(batch_obs.keys())}")
        print(f"   placed_components_list shape: {batch_obs['placed_components_list'].shape}")
        
        # Test forward pass
        with torch.no_grad():
            actions, values, log_probs = policy.forward(batch_obs, deterministic=True)
        
        print(f"Policy forward pass successful")
        print(f"   Actions shape: {actions.shape}")
        print(f"   Values shape: {values.shape}")
        print(f"   Log probs shape: {log_probs.shape}")
        print(f"   Sample action: {actions[0].numpy()}")
        
        return True
        
    except Exception as e:
        print(f"Policy forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_policy_evaluate_actions():
    """Test policy action evaluation for training."""
    print("\nTesting policy action evaluation...")
    
    try:
        # Create environment and policy
        env = AnalogLayoutEnv(grid_size=16, max_components=8)
        
        def lr_schedule(progress_remaining: float) -> float:
            return 1e-4
        
        policy = AnalogLayoutPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lr_schedule
        )
        
        # Get observation
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        
        # Convert to batch format
        batch_obs = {}
        for key, value in obs.items():
            batch_obs[key] = torch.tensor(value).unsqueeze(0).float()
        
        # Create sample actions
        actions = torch.tensor([[0, 2, 1]])  # target=0, relation=2, orientation=1
        
        # Test evaluate_actions
        with torch.no_grad():
            values, log_probs, entropy = policy.evaluate_actions(batch_obs, actions)
        
        print(f"Policy action evaluation successful")
        print(f"   Values shape: {values.shape}")
        print(f"   Log probs shape: {log_probs.shape}")
        print(f"   Entropy shape: {entropy.shape}")
        
        return True
        
    except Exception as e:
        print(f"Policy action evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_grid_size_independence():
    """Test that policy works with different grid sizes."""
    print("\nTesting grid size independence...")
    
    try:
        def lr_schedule(progress_remaining: float) -> float:
            return 1e-4
        
        results = {}
        
        # Test different grid sizes
        for grid_size in [16, 32, 64]:
            print(f"   Testing grid size {grid_size}x{grid_size}...")
            
            # Create environment
            env = AnalogLayoutEnv(grid_size=grid_size, max_components=8)
            
            # Create policy
            policy = AnalogLayoutPolicy(
                observation_space=env.observation_space,
                action_space=env.action_space,
                lr_schedule=lr_schedule
            )
            
            # Get observation
            result = env.reset()
            obs = result[0] if isinstance(result, tuple) else result
            
            # Convert to batch format
            batch_obs = {}
            for key, value in obs.items():
                batch_obs[key] = torch.tensor(value).unsqueeze(0).float()
            
            # Test forward pass
            with torch.no_grad():
                actions, values, log_probs = policy.forward(batch_obs, deterministic=True)
            
            results[grid_size] = {
                'actions': actions[0].numpy(),
                'values': values[0].item(),
                'component_list_shape': batch_obs['placed_components_list'].shape
            }
        
        print(f"Grid size independence test passed")
        for grid_size, data in results.items():
            print(f"   Grid {grid_size}: actions={data['actions']}, value={data['values']:.3f}")
            print(f"                comp_list_shape={data['component_list_shape']}")
        
        # Check that component list shapes are the same (grid-agnostic)
        shapes = [data['component_list_shape'] for data in results.values()]
        assert all(shape == shapes[0] for shape in shapes), "Component list shapes should be identical"
        
        return True
        
    except Exception as e:
        print(f"Grid size independence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all policy tests."""
    print("CLARA Grid-Agnostic Policy Test Suite")
    print("=" * 50)
    
    tests = [
        ("ComponentListEncoder", test_component_list_encoder),
        ("Policy Creation", test_policy_creation),
        ("Policy Forward Pass", test_policy_forward_pass),
        ("Policy Action Evaluation", test_policy_evaluate_actions),
        ("Grid Size Independence", test_grid_size_independence),
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
        print("All tests passed! Grid-agnostic policy is ready.")
        return 0
    else:
        print(" Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())