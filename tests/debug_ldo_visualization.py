#!/usr/bin/env python3
"""
Debug why all components appear the same size in the LDO visualization.
"""

import numpy as np
from pathlib import Path
from enhanced_spice_parser import EnhancedSpiceParser
from train_spice_real import SpiceCircuitManager
import networkx as nx

def debug_ldo_component_sizes():
    """Debug component sizes in the LDO circuit."""
    print("🔍 DEBUGGING LDO COMPONENT SIZES")
    print("=" * 50)
    
    ldo_file = "/home/eli/Documents/Internship/CLARA/data/netlists/sky130_am_ip__ldo_01v8.spice"
    
    if not Path(ldo_file).exists():
        print(f"❌ File not found: {ldo_file}")
        return
    
    # Parse with enhanced parser
    parser = EnhancedSpiceParser()
    circuit_data = parser.parse_spice_file(ldo_file)
    
    print(f"📊 Raw Components (first 20):")
    components = circuit_data['components'][:20]  # Show first 20
    
    for i, comp in enumerate(components):
        clara_override = comp.get('clara_override', {})
        print(f"   {i:2d}. {comp['name'][:15]:<15}: L={comp['length']:<5.1f} W={comp['width']:<5.1f} mx={comp.get('mx', 'N/A')} my={comp.get('my', 'N/A')} CLARA={clara_override}")
    
    # Test circuit manager conversion
    print(f"\n🔄 Circuit Manager Conversion:")
    
    # Create temporary directory for circuit manager  
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy LDO to temp directory
        temp_ldo = os.path.join(temp_dir, "ldo_test.spice")
        with open(ldo_file, 'r') as src, open(temp_ldo, 'w') as dst:
            dst.write(src.read())
        
        # Test circuit manager
        circuit_manager = SpiceCircuitManager(temp_dir)
        
        if len(circuit_manager.suitable_circuits) > 0:
            circuit_name = list(circuit_manager.suitable_circuits.keys())[0]
            circuit_data = circuit_manager.suitable_circuits[circuit_name]
            
            # Convert to NetworkX
            graph = circuit_manager.convert_to_networkx_graph(circuit_data)
            
            print(f"NetworkX Graph Component Sizes (first 20):")
            for i, node_id in enumerate(list(graph.nodes())[:20]):
                attrs = graph.nodes[node_id]
                comp_type = attrs.get('component_type', 0)
                width = attrs.get('width', 1)
                height = attrs.get('height', 1)
                spice_name = attrs.get('spice_name', f'comp_{node_id}')
                
                print(f"   {i:2d}. {spice_name[:15]:<15}: type={comp_type} width={width} height={height}")
            
            # Analyze size distribution
            sizes = {}
            for node_id in graph.nodes():
                attrs = graph.nodes[node_id]
                width = attrs.get('width', 1)
                height = attrs.get('height', 1)
                size_key = f"{width}x{height}"
                sizes[size_key] = sizes.get(size_key, 0) + 1
            
            print(f"\n📈 Size Distribution:")
            for size_key, count in sorted(sizes.items()):
                print(f"   {size_key}: {count} components")
            
            # Check if all components are the same size
            unique_sizes = len(sizes)
            total_components = len(graph.nodes())
            
            if unique_sizes == 1:
                print(f"\n⚠️  ISSUE FOUND: All {total_components} components have the same size!")
                print(f"   This is why they all appear as squares in the visualization.")
                print(f"   Common size: {list(sizes.keys())[0]}")
            else:
                print(f"\n✅ Size variety: {unique_sizes} different sizes for {total_components} components")

def suggest_fixes():
    """Suggest how to fix the visualization issue."""
    print(f"\n💡 SOLUTIONS:")
    print("=" * 30)
    print("1. Add CLARA comments to the LDO circuit with mx/my values")
    print("   Example: XM1 D G S B model L=1.0 W=5.0 ;CLARA override-size mx=2 my=3")
    print()
    print("2. Improve the scaling algorithm to preserve more size variation")
    print("   - Use wider scaling ranges")
    print("   - Preserve relative ratios better")
    print()
    print("3. Create a demo circuit with explicit CLARA sizing for visualization")
    print("   - This would show different component sizes properly")

def create_demo_circuit_with_sizing():
    """Create a demo circuit that will show different component sizes."""
    demo_circuit = """
* Demo circuit with explicit CLARA sizing for visualization
.subckt demo_sized_circuit VDD VSS VIN VOUT

* Small components (1x1)
XM1 N1 VIN VSS VSS nmos L=0.5 W=1.0 ;CLARA override-size mx=1 my=1
XM2 N2 VIN VSS VSS nmos L=0.5 W=1.0 ;CLARA override-size mx=1 my=1

* Medium components (2x2)  
XM3 N3 N1 VDD VDD pmos L=1.0 W=2.0 ;CLARA override-size mx=2 my=2
XM4 N4 N2 VDD VDD pmos L=1.0 W=2.0 ;CLARA override-size mx=2 my=2

* Large components (3x4)
XM5 VOUT N3 VDD VDD pmos L=2.0 W=10.0 ;CLARA override-size mx=3 my=4
XM6 VOUT N4 VSS VSS nmos L=2.0 W=8.0 ;CLARA override-size mx=3 my=4

* Very large component (4x2)
XM7 N5 VOUT VDD VDD pmos L=1.0 W=15.0 ;CLARA override-size mx=4 my=2

* Resistors with different sizes
XR1 VIN N1 resistor ;CLARA override-size mx=1 my=3  
XR2 N5 VOUT resistor ;CLARA override-size mx=2 my=1

.ends demo_sized_circuit
"""
    
    with open("/home/eli/Documents/Internship/CLARA/demo_sized_circuit.spice", 'w') as f:
        f.write(demo_circuit)
    
    print(f"\n📄 Created demo_sized_circuit.spice with explicit component sizing")
    print("   This circuit should show different component sizes when visualized")
    print("   Run: python run_model.py --test-circuits --visualize --circuit demo_sized_circuit.spice")

if __name__ == "__main__":
    debug_ldo_component_sizes()
    suggest_fixes()
    create_demo_circuit_with_sizing()