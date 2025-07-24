#!/usr/bin/env python3
"""
Quick demo to visualize the trained CLARA model output.
"""

import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from analog_layout_env import AnalogLayoutEnv
from visualize import AnalogLayoutVisualizer
from data.circuit_generator import AnalogCircuitGenerator


def demo_model():
    """Quick demo of the trained model."""
    
    # Load model
    print("Loading trained model...")
    model = PPO.load("./logs/clara_20250722_185325/clara_final_model")
    
    # Create a simple test circuit
    generator = AnalogCircuitGenerator()
    circuit = generator.generate_differential_pair()
    
    print(f"Test circuit: {len(circuit.nodes)} components, {len(circuit.edges)} connections")
    
    # Create environment
    env = AnalogLayoutEnv(grid_size=15, max_components=6)
    result = env.reset(circuit_graph=circuit)
    if isinstance(result, tuple):
        obs, info = result
    else:
        obs = result
    
    print("Running model on differential pair circuit...")
    
    # Run episode
    positions_history = []
    step = 0
    
    while step < env.max_steps:
        # Record current positions
        positions_history.append(env.component_positions.copy())
        
        # Get action from model
        action, _states = model.predict(obs, deterministic=True)
        
        # Take step
        result = env.step(action)
        if len(result) == 5:  # Gymnasium format
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:  # Gym format
            obs, reward, done, info = result
        
        step += 1
        
        print(f"Step {step}: action={action}, reward={reward:.2f}, "
              f"placed={len(env.component_positions)}/{env.num_components}")
        
        if done:
            break
    
    # Final positions
    final_positions = env.component_positions
    
    print(f"\nFinal result: {len(final_positions)}/{env.num_components} components placed")
    
    # Create visualization
    visualizer = AnalogLayoutVisualizer(grid_size=15, figsize=(10, 8))
    
    fig = visualizer.visualize_layout(
        circuit=circuit,
        component_positions=final_positions,
        title=f"CLARA Layout Demo ({len(final_positions)}/{env.num_components} components)",
        show_connections=True,
        save_path="clara_demo_layout.png"
    )
    
    print("Layout saved as 'clara_demo_layout.png'")
    
    # Show component details
    print("\nComponent details:")
    for comp_id, (x, y, orient) in final_positions.items():
        comp_type = circuit.nodes[comp_id].get('component_type', 0)
        type_names = {0: 'NMOS', 1: 'PMOS', 2: 'R', 3: 'C', 4: 'L', 5: 'I', 6: 'V'}
        type_name = type_names.get(comp_type, f'Type{comp_type}')
        print(f"  Component {comp_id} ({type_name}): position ({x}, {y}), orientation {orient*90}°")
    
    plt.show()
    
    return fig


if __name__ == "__main__":
    try:
        demo_model()
        print("\nDemo completed successfully!")
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()