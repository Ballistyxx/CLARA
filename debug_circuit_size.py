#!/usr/bin/env python3
"""
Debug script to isolate the circuit size truncation issue.
"""

from run_model import *
from train_spice_real import SpiceCircuitManager
from analog_layout_env import AnalogLayoutEnv
from stable_baselines3 import PPO

def debug_circuit_flow():
    """Debug the complete flow to see where components get lost."""
    
    print("DEBUGGING CIRCUIT SIZE ISSUE")
    print("=" * 50)
    
    # 1. Load circuit manager
    print("1. Loading SPICE circuit manager...")
    circuit_manager = SpiceCircuitManager('/home/eli/Documents/Internship/CLARA/data/netlists')
    
    # 2. Get specific circuit
    print("2. Getting LDO circuit...")
    circuit_data = circuit_manager.suitable_circuits['sky130_am_ip__ldo_01v8.spice']
    original_circuit = circuit_manager.convert_to_networkx_graph(circuit_data)
    print(f"   Original circuit: {len(original_circuit.nodes)} nodes")
    
    # 3. Sample circuit
    print("3. Sampling circuit to 12 components...")
    sampled_circuit = sample_large_circuit(original_circuit, 12, strategy='diverse')
    print(f"   Sampled circuit: {len(sampled_circuit.nodes)} nodes")
    print(f"   Sampled node IDs: {list(sampled_circuit.nodes())}")
    
    # 4. Create environment
    print("4. Creating environment with max_components=12...")
    env = AnalogLayoutEnv(grid_size=20, max_components=12)
    print(f"   Environment max_components: {env.max_components}")
    
    # 5. Reset environment with sampled circuit
    print("5. Resetting environment with sampled circuit...")
    result = env.reset(circuit_graph=sampled_circuit)
    obs = result[0] if isinstance(result, tuple) else result
    
    print(f"   Environment num_components: {env.num_components}")
    print(f"   Environment circuit nodes: {list(env.circuit.nodes())}")
    print(f"   Observation shapes:")
    for key, value in obs.items():
        print(f"     {key}: {value.shape}")
    
    # 6. Check observation content
    print("6. Checking observation content...")
    print(f"   Netlist features (first 5 rows):")
    for i in range(min(5, env.num_components)):
        print(f"     Component {i}: {obs['netlist_features'][i]}")
    
    # 7. Load model and test prediction
    print("7. Loading model...")
    model = PPO.load('./logs/best_model')
    
    print("8. Testing model prediction...")
    action, _states = model.predict(obs, deterministic=True)
    print(f"   Predicted action: {action}")
    
    # 8. Manually run a few steps
    print("9. Running manual steps...")
    for step in range(3):
        print(f"   Step {step + 1}:")
        print(f"     Action: {action}")
        
        result = env.step(action)
        if len(result) == 5:  # Gymnasium format
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:  # Gym format
            obs, reward, done, info = result
        
        print(f"     Reward: {reward}")
        print(f"     Valid action: {info.get('valid_action', True)}")
        print(f"     Components placed: {len(env.component_positions)}/{env.num_components}")
        print(f"     Environment still has {env.num_components} components")
        
        if done:
            print(f"     Episode done!")
            break
            
        # Get next action
        action, _states = model.predict(obs, deterministic=True)
    
    print("\nDebug complete!")

if __name__ == "__main__":
    debug_circuit_flow()