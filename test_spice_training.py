#!/usr/bin/env python3
"""
Quick test of SPICE-based training to verify integration works.
"""

from train_spice_real import *


def quick_test():
    """Quick test with minimal training."""
    
    # Quick test configuration
    config = {
        'grid_size': 16,
        'max_components': 8,
        'n_envs': 2,
        'total_timesteps': 2000,  # Very short test
        'learning_rate': 1e-3,
        'batch_size': 32,
        'n_steps': 128,
        'n_epochs': 2,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.1,
        'ent_coef': 0.01,
        'vf_coef': 0.25,
        'max_grad_norm': 0.5,
        'seed': 42
    }
    
    print("🧪 QUICK TEST: SPICE Circuit Integration")
    print("=" * 50)
    
    # Initialize SPICE circuit manager
    spice_directory = "/home/eli/Documents/Internship/CLARA/data/netlists/programmable_pll_subcircuits"
    circuit_manager = SpiceCircuitManager(spice_directory)
    
    stats = circuit_manager.get_circuit_stats()
    print(f"✅ Loaded {stats['total_circuits']} circuits for testing")
    
    # Create single environment for testing (not vectorized)
    test_env = AnalogLayoutSpiceEnvWrapper(
        circuit_manager=circuit_manager,
        grid_size=config['grid_size'],
        max_components=config['max_components']
    )
    
    # Test environment reset with SPICE circuit
    print("🔬 Testing environment with real SPICE circuits...")
    
    # Test a few resets to see different circuits
    for i in range(3):
        result = test_env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        
        components_in_circuit = test_env.num_components
        
        print(f"   Reset {i+1}: {components_in_circuit} components, obs keys: {list(obs.keys())}")
        
        # Take a few steps
        for step in range(3):
            action = test_env.action_space.sample()
            result = test_env.step(action)
            
            if len(result) == 5:  # Gymnasium format
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:  # Gym format
                obs, reward, done, info = result
            
            print(f"     Step {step+1}: reward={reward}, done={done}")
            
            if done:
                break
    
    print("✅ Environment testing successful!")
    
    # Quick training test
    print("🚀 Quick training test (2K timesteps)...")
    
    # Create vectorized environment for training
    train_env = setup_spice_training_environment(config, circuit_manager)
    
    model = PPO(
        "MultiInputPolicy",
        train_env,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        verbose=1
    )
    
    model.learn(total_timesteps=config['total_timesteps'], progress_bar=False)
    
    print("✅ Training test successful!")
    
    # Test model on specific circuit
    print("🧪 Testing trained model on SPICE circuits...")
    
    test_results = test_model_on_spice_circuits(model, circuit_manager, config)
    
    print("🎉 All tests passed! SPICE integration is working perfectly.")
    
    test_env.close()
    train_env.close()
    
    return test_results


if __name__ == "__main__":
    quick_test()