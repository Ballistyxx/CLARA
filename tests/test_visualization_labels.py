#!/usr/bin/env python3
"""
Test script to verify that device model labels are working in visualization.
"""

import networkx as nx
from visualize import AnalogLayoutVisualizer
import matplotlib.pyplot as plt

def test_device_model_labels():
    """Test that visualization shows device models instead of generic labels."""
    
    print("TESTING VISUALIZATION DEVICE MODEL LABELS")
    print("=" * 50)
    
    # Create a test circuit with various device models
    circuit = nx.Graph()
    
    # Add nodes with different device models (matching what we get from SPICE parser)
    test_components = [
        {
            'id': 0,
            'component_type': 1,  # PMOS
            'device_model': 'sky130_fd_pr__pfet_g5v0d10v5',
            'spice_name': 'XM1',
            'width': 2, 'height': 1, 'matched_component': -1
        },
        {
            'id': 1,
            'component_type': 0,  # NMOS
            'device_model': 'sky130_fd_pr__nfet_01v8',
            'spice_name': 'XM2',
            'width': 1, 'height': 2, 'matched_component': -1
        },
        {
            'id': 2,
            'component_type': 2,  # Resistor
            'device_model': 'sky130_fd_pr__res_xhigh_po_0p35',
            'spice_name': 'XR1',
            'width': 1, 'height': 3, 'matched_component': -1
        },
        {
            'id': 3,
            'component_type': 3,  # Capacitor
            'device_model': 'sky130_fd_pr__cap_mim_m3_1',
            'spice_name': 'XC1',
            'width': 2, 'height': 2, 'matched_component': -1
        },
        {
            'id': 4,
            'component_type': 7,  # Subcircuit
            'device_model': 'sky130_fd_sc_hvl__inv_1',
            'spice_name': 'X1',
            'width': 1, 'height': 1, 'matched_component': -1
        }
    ]
    
    # Add nodes to circuit
    for comp in test_components:
        circuit.add_node(comp['id'], **{k: v for k, v in comp.items() if k != 'id'})
    
    # Add some connections
    circuit.add_edge(0, 1)  # PMOS to NMOS
    circuit.add_edge(1, 2)  # NMOS to Resistor
    circuit.add_edge(2, 3)  # Resistor to Capacitor
    circuit.add_edge(3, 4)  # Capacitor to Subcircuit
    
    # Create some test positions
    component_positions = {
        0: (2, 3, 0),   # PMOS at (2,3) orientation 0
        1: (4, 2, 0),   # NMOS at (4,2) orientation 0  
        2: (6, 1, 0),   # Resistor at (6,1) orientation 0
        3: (8, 3, 0),   # Capacitor at (8,3) orientation 0
        4: (10, 2, 0),  # Subcircuit at (10,2) orientation 0
    }
    
    print(f"Created test circuit with {len(test_components)} components:")
    for comp in test_components:
        print(f"  {comp['spice_name']}: {comp['device_model']}")
    
    # Create visualizer and generate layout
    visualizer = AnalogLayoutVisualizer(grid_size=15)
    
    fig = visualizer.visualize_layout(
        circuit=circuit,
        component_positions=component_positions,
        title="Device Model Label Test",
        show_connections=True,
        save_path="test_device_labels.png"
    )
    
    print(f"\nVisualization created and saved as 'test_device_labels.png'")
    print(f"   Expected labels (full device model names):")
    print(f"   - sky130_fd_pr__pfet_g5v0d10v5 (instead of PMOS1)")
    print(f"   - sky130_fd_pr__nfet_01v8 (instead of NMOS2)")  
    print(f"   - sky130_fd_pr__res_xhigh_po_0p35 (instead of R3)")
    print(f"   - sky130_fd_pr__cap_mim_m3_1 (instead of C4)")
    print(f"   - sky130_fd_sc_hvl__inv_1 (instead of X5)")
    
    print(f"\nFull device model names will be displayed directly on components")
    
    plt.show()
    
    return fig

if __name__ == "__main__":
    test_device_model_labels()