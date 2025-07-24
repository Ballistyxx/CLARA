#!/usr/bin/env python3
"""
Simple test of the trained model without visualization.
"""

from stable_baselines3 import PPO
from analog_layout_env import AnalogLayoutEnv
from data.circuit_generator import AnalogCircuitGenerator


def simple_test():
    """Simple test without graphics."""
    
    print("Testing CLARA trained model")
    print("=" * 40)
    
    # Load model
    try:
        model = PPO.load("./logs/clara_20250722_185325/clara_final_model")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Create test circuit
    generator = AnalogCircuitGenerator()
    circuit = generator.generate_differential_pair()
    
    print(f"Test circuit: {len(circuit.nodes)} components")
    for i, node in enumerate(circuit.nodes()):
        attrs = circuit.nodes[node]
        comp_type = attrs.get('component_type', 0)
        matched = attrs.get('matched_component', -1)
        match_str = f" (matched with {matched})" if matched != -1 else ""
        print(f"Component {node}: type {comp_type}{match_str}")

    print(f"Connections: {list(circuit.edges())}")

    # Create environment
    env = AnalogLayoutEnv(grid_size=15, max_components=6)
    result = env.reset(circuit_graph=circuit)
    if isinstance(result, tuple):
        obs, info = result
    else:
        obs = result
    
    print(f"\nRunning inference...")
    print(f"Environment: {env.grid_size}×{env.grid_size} grid")
    
    # Run episode
    total_reward = 0
    step = 0
    
    while step < env.max_steps:
        # Get action from model
        action, _states = model.predict(obs, deterministic=True)
        
        # Take step
        result = env.step(action)
        if len(result) == 5:  # Gymnasium format
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:  # Gym format
            obs, reward, done, info = result
        
        total_reward += reward
        step += 1
        
        # Show progress every few steps
        if step % 5 == 0 or done:
            print(f"Step {step}: {len(env.component_positions)}/{env.num_components} placed, "
                  f"reward={reward:.2f}, total={total_reward:.2f}")
        
        if done:
            break
    
    # Results
    print(f"\nFinal Results:")
    print(f"Components placed: {len(env.component_positions)}/{env.num_components}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Steps taken: {step}")
    success = len(env.component_positions) == env.num_components
    print(f"Success: {'Yes' if success else 'No'}")

    # Show final layout in text
    print(f"\nFinal Layout:")
    if env.component_positions:
        for comp_id, (x, y, orient) in env.component_positions.items():
            comp_type = circuit.nodes[comp_id].get('component_type', 0)
            type_names = {0: 'NMOS', 1: 'PMOS', 2: 'R', 3: 'C', 4: 'L', 5: 'I', 6: 'V'}
            type_name = type_names.get(comp_type, f'Type{comp_type}')
            print(f"  {type_name}{comp_id}: ({x:2d}, {y:2d}) orientation {orient*90:3d}°")
    else:
        print("No components placed")
    
    # Test different circuit sizes
    print(f"\nTesting different circuit sizes:")
    for size in [3, 4, 5, 6]:
        try:
            small_circuit = generator.generate_random_circuit(size, size)
            env.reset(circuit_graph=small_circuit)
            
            # Quick run
            obs = env._get_observation()
            quick_reward = 0
            for i in range(size * 2):  # Limited steps
                action, _states = model.predict(obs, deterministic=True)
                result = env.step(action)
                if len(result) == 5:
                    obs, reward, terminated, truncated, info = result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = result
                quick_reward += reward
                if done:
                    break
            
            placed = len(env.component_positions)
            print(f"  {size} components: {placed}/{size} placed, reward={quick_reward:.1f}")
            
        except Exception as e:
            print(f"  {size} components: Error - {e}")


if __name__ == "__main__":
    simple_test()