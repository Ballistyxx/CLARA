#!/usr/bin/env python3
"""
Quick demo to show fractional component dimensions working in CLARA.
"""

import networkx as nx
from visualize import AnalogLayoutVisualizer

def create_demo_circuit():
    """Create a demo circuit with components of different fractional sizes."""
    G = nx.Graph()
    
    # Add components with different fractional sizes
    components = [
        {"id": 0, "type": 0, "width": 0.3, "height": 1.5, "name": "M1_small"},
        {"id": 1, "type": 1, "width": 0.6, "height": 2.0, "name": "M2_medium"},
        {"id": 2, "type": 0, "width": 1.2, "height": 0.4, "name": "M3_wide"},
        {"id": 3, "type": 2, "width": 0.1, "height": 8.0, "name": "R1_tall"},
        {"id": 4, "type": 3, "width": 2.5, "height": 1.8, "name": "C1_large"},
    ]
    
    for comp in components:
        G.add_node(comp["id"],
                   component_type=comp["type"],
                   width=comp["width"],
                   height=comp["height"],
                   matched_component=-1,
                   spice_name=comp["name"],
                   device_model=f"demo_model_{comp['type']}")
    
    # Add some connections
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(3, 4)
    
    return G

def main():
    """Demonstrate fractional component visualization."""
    
    # Create demo circuit
    circuit = create_demo_circuit()
    
    # Place components manually to show different sizes clearly
    positions = {
        0: (2, 3, 0),    # Small: 0.3x1.5
        1: (5, 3, 0),    # Medium: 0.6x2.0  
        2: (8, 3, 0),    # Wide: 1.2x0.4
        3: (11, 2, 0),   # Tall: 0.1x8.0
        4: (14, 2, 0),   # Large: 2.5x1.8
    }
    
    # Create visualizer
    visualizer = AnalogLayoutVisualizer(grid_size=20)
    
    # Generate visualization
    fig = visualizer.visualize_layout(
        circuit=circuit,
        component_positions=positions,
        title="CLARA Fractional Dimensions Demo",
        show_connections=True,
        save_path="fractional_dimensions_demo.png"
    )
    
    print("Created fractional dimensions demonstration!")
    print("Before: All components appeared as 1x1 squares")
    print("After: Components show their actual fractional sizes:")
    print("  - M1_small: 0.3×1.5 (thin tall)")
    print("  - M2_medium: 0.6×2.0 (medium)")  
    print("  - M3_wide: 1.2×0.4 (wide short)")
    print("  - R1_tall: 0.1×8.0 (very thin tall)")
    print("  - C1_large: 2.5×1.8 (large)")
    print("Visualization saved as 'fractional_dimensions_demo.png'")

if __name__ == "__main__":
    main()