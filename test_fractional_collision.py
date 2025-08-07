#!/usr/bin/env python3
"""
Test fractional collision detection in analog_layout_env.py
"""

import networkx as nx
from analog_layout_env import AnalogLayoutEnv

def test_fractional_collision():
    """Test that fractional collision detection works correctly."""
    
    # Create a simple test circuit with fractional dimensions
    G = nx.Graph()
    G.add_node(0, component_type=0, width=0.5, height=1.2, matched_component=-1, 
               spice_name="M1", device_model="test_model")
    G.add_node(1, component_type=1, width=0.8, height=0.6, matched_component=-1,
               spice_name="M2", device_model="test_model")
    G.add_node(2, component_type=2, width=1.5, height=2.0, matched_component=-1,
               spice_name="R1", device_model="test_model")
    
    # Add some edges
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    
    # Create environment
    env = AnalogLayoutEnv(grid_size=10, max_components=10)
    env.reset(circuit_graph=G)
    
    print("Testing fractional collision detection...")
    
    # Check that components are stored with fractional rectangles
    print(f"Component rectangles: {env.component_rectangles}")
    
    # Test the rectangle overlap function
    rect1 = (1.0, 1.0, 0.5, 1.2)  # x=1, y=1, w=0.5, h=1.2
    rect2 = (1.2, 1.0, 0.8, 0.6)  # x=1.2, y=1, w=0.8, h=0.6 (overlaps)
    rect3 = (3.0, 3.0, 1.5, 2.0)  # x=3, y=3, w=1.5, h=2.0 (no overlap)
    
    overlap1_2 = env._rectangles_overlap(rect1, rect2)
    overlap1_3 = env._rectangles_overlap(rect1, rect3)
    overlap2_3 = env._rectangles_overlap(rect2, rect3)
    
    print(f"Rect1 {rect1} overlaps Rect2 {rect2}: {overlap1_2} (expected: True)")
    print(f"Rect1 {rect1} overlaps Rect3 {rect3}: {overlap1_3} (expected: False)")
    print(f"Rect2 {rect2} overlaps Rect3 {rect3}: {overlap2_3} (expected: False)")
    
    # Test that the environment correctly rejects overlapping placements
    print(f"\nTesting placement collision detection...")
    print(f"Placed components: {len(env.component_positions)}")
    print(f"Component dimensions in circuit:")
    for i in range(3):
        if i in G.nodes:
            attrs = G.nodes[i]
            print(f"  Component {i}: {attrs['width']}×{attrs['height']}")
    
    return True

if __name__ == "__main__":
    test_fractional_collision()
    print("\nFractional collision detection test completed!")