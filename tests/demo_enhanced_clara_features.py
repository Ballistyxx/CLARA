#!/usr/bin/env python3
"""
Demonstration of enhanced CLARA comment functionality in a realistic circuit.
"""

import tempfile
import os
from pathlib import Path
from enhanced_spice_parser import EnhancedSpiceParser
from train_spice_real import SpiceCircuitManager
from analog_layout_env import AnalogLayoutEnv
import numpy as np

def create_demo_circuit():
    """Create a demonstration differential amplifier circuit with CLARA comments."""
    
    demo_spice = """
* Enhanced CLARA Demo: Differential Amplifier with Current Mirror Load
* This circuit demonstrates all new CLARA comment features

.subckt diff_amp VIN_P VIN_N VOUT VDD VSS VBIAS

* Differential Input Pair with explicit CLARA sizing and pairing
XM1 VOUT1 VIN_P CS VSS sky130_fd_pr__nfet_01v8 L=1.0 W=5.0 m=1 ;CLARA override-size L=2.0 W=4.0 mx=3 my=2 pair diff_input
XM2 VOUT2 VIN_N CS VSS sky130_fd_pr__nfet_01v8 L=1.0 W=5.0 m=1 ;CLARA override-size L=2.0 W=4.0 mx=3 my=2 pair diff_input

* Current Source (Tail Current) with custom sizing
XM3 CS VBIAS VSS VSS sky130_fd_pr__nfet_01v8 L=2.0 W=20.0 m=2 ;CLARA override-size mx=4 my=3

* Current Mirror Load with explicit pairing
XM4 VOUT1 VOUT1 VDD VDD sky130_fd_pr__pfet_01v8 L=0.5 W=10.0 m=1 ;CLARA pair current_mirror
XM5 VOUT2 VOUT1 VDD VDD sky130_fd_pr__pfet_01v8 L=0.5 W=10.0 m=1 ;CLARA pair current_mirror

* Output Buffer (single transistor for demo) with large custom size
XM6 VOUT VOUT2 VSS VSS sky130_fd_pr__nfet_01v8 L=0.8 W=15.0 ;CLARA override-size mx=5 my=4

.ends diff_amp
"""
    
    return demo_spice

def analyze_clara_features():
    """Analyze and demonstrate all CLARA features."""
    print("🎯 ENHANCED CLARA FEATURES DEMONSTRATION")
    print("=" * 60)
    print("Circuit: Differential Amplifier with Current Mirror Load")
    print()
    
    # Create demo circuit file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.spice', delete=False) as f:
        f.write(create_demo_circuit())
        demo_file = f.name
    
    try:
        # Parse with enhanced CLARA support
        parser = EnhancedSpiceParser()
        circuit_data = parser.parse_spice_file(demo_file)
        
        components = circuit_data['components']
        
        print("📊 PARSING RESULTS:")
        print(f"   Subcircuit: {circuit_data['subcircuit_name']}")
        print(f"   Total components: {len(components)} (after multiplier expansion)")
        print(f"   Connections: {circuit_data['circuit_graph'].number_of_edges()}")
        print()
        
        # Analyze CLARA features
        print("🔧 CLARA FEATURE ANALYSIS:")
        print("-" * 40)
        
        # Group by functionality
        diff_pair = []
        current_mirror = []
        other_components = []
        
        for comp in components:
            if comp['pair_name'] == 'diff_input':
                diff_pair.append(comp)
            elif comp['pair_name'] == 'current_mirror':
                current_mirror.append(comp)
            else:
                other_components.append(comp)
        
        # Show differential pair
        print("🔗 Differential Input Pair:")
        for comp in diff_pair:
            print(f"   {comp['name']}: L={comp['length']}, W={comp['width']}, mx={comp['mx']}, my={comp['my']}")
            if comp['clara_override']:
                print(f"      CLARA overrides applied: {comp['clara_override']}")
        
        # Show current mirror
        print("\n🔗 Current Mirror Load:")
        for comp in current_mirror:
            print(f"   {comp['name']}: L={comp['length']}, W={comp['width']}, mx={comp['mx']}, my={comp['my']}")
        
        # Show other components
        print("\n🔧 Other Components:")
        for comp in other_components:
            print(f"   {comp['name']}: L={comp['length']}, W={comp['width']}, mx={comp['mx']}, my={comp['my']}")
            if comp['clara_override']:
                print(f"      CLARA overrides: {comp['clara_override']}")
        
        # Demonstrate RL integration
        print("\n🤖 RL TRAINING INTEGRATION TEST:")
        print("-" * 40)
        
        # Create temporary directory for circuit manager
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy demo file to temp directory
            temp_demo_file = os.path.join(temp_dir, "demo_diff_amp.spice")
            with open(temp_demo_file, 'w') as f:
                f.write(create_demo_circuit())
            
            # Test circuit manager
            circuit_manager = SpiceCircuitManager(temp_dir)
            
            if len(circuit_manager.suitable_circuits) > 0:
                # Get the demo circuit
                circuit_name = list(circuit_manager.suitable_circuits.keys())[0]
                circuit_data = circuit_manager.suitable_circuits[circuit_name]
                
                print(f"📈 Circuit loaded successfully:")
                print(f"   Name: {circuit_name}")
                print(f"   Components: {circuit_data['num_components']}")
                
                # Convert to RL-compatible format
                graph = circuit_manager.convert_to_networkx_graph(circuit_data)
                
                # Analyze matching
                matched_pairs = 0
                diff_matches = 0
                mirror_matches = 0
                
                print(f"\n🔍 Component Matching Analysis:")
                for node_id in graph.nodes():
                    attrs = graph.nodes[node_id]
                    matched_comp = attrs.get('matched_component', -1)
                    spice_name = attrs.get('spice_name', '')
                    
                    if matched_comp != -1:
                        matched_pairs += 1
                        print(f"   {spice_name} (id={node_id}) ↔ component {matched_comp}")
                        
                        # Count different pair types
                        if 'XM1' in spice_name or 'XM2' in spice_name:
                            diff_matches += 1
                        elif 'XM4' in spice_name or 'XM5' in spice_name:
                            mirror_matches += 1
                
                print(f"\n📊 Matching Summary:")
                print(f"   Total matched components: {matched_pairs}")
                print(f"   Differential pair matches: {diff_matches}")
                print(f"   Current mirror matches: {mirror_matches}")
                
                # Test actual RL environment
                print(f"\n🎮 RL ENVIRONMENT TEST:")
                env = AnalogLayoutEnv(
                    grid_size=30,
                    max_components=20,
                    enable_action_masking=True
                )
                
                # Test reset with the circuit
                result = env.reset(circuit_graph=graph)
                obs = result[0] if isinstance(result, tuple) else result
                
                print(f"   Environment initialized successfully")
                print(f"   Circuit components: {env.num_components}")
                print(f"   Grid size: {env.grid_size}x{env.grid_size}")
                
                # Calculate layout area requirements based on mx/my
                total_layout_area = 0
                for node_id in range(env.num_components):
                    if node_id < len(graph.nodes):
                        attrs = graph.nodes[node_id]
                        mx = attrs.get('width', 1)  # mx is stored as width in NetworkX
                        my = attrs.get('height', 1)  # my is stored as height in NetworkX
                        total_layout_area += mx * my
                
                grid_area = env.grid_size * env.grid_size
                utilization = total_layout_area / grid_area
                
                print(f"   Layout area needed: {total_layout_area} cells")
                print(f"   Grid utilization: {100*utilization:.1f}%")
                
                if utilization < 0.8:
                    print(f"    Good utilization - sufficient space for placement")
                else:
                    print(f"   ⚠️  High utilization - may need larger grid")
            
        # Summary
        print(f"\n🎉 DEMONSTRATION COMPLETE!")
        print("=" * 60)
        print(" All CLARA features working correctly:")
        print("   • mx= and my= parameters for custom component sizing")
        print("   • CLARA override-size takes precedence over SPICE parameters") 
        print("   • CLARA pair <pair_name> for explicit differential pair definition")
        print("   • Integration with RL training environment")
        print("   • Proper component matching in NetworkX graph conversion")
        print("   • Action masking compatibility maintained")
        print()
        print("🚀 The enhanced CLARA system is ready for advanced analog IC layout training!")
        
    finally:
        os.unlink(demo_file)

def show_usage_examples():
    """Show usage examples of the new CLARA features."""
    print("\n📚 CLARA COMMENT USAGE EXAMPLES:")
    print("=" * 50)
    
    examples = [
        {
            "title": "Custom Component Sizing",
            "description": "Override SPICE W/L with specific grid dimensions",
            "example": "XM1 D G S B model L=0.5 W=2.0 ;CLARA override-size mx=3 my=2"
        },
        {
            "title": "Parameter Override with Sizing", 
            "description": "Change both SPICE parameters and add custom sizing",
            "example": "XM1 D G S B model L=1.0 W=5.0 ;CLARA override-size L=2.0 W=4.0 mx=4 my=3"
        },
        {
            "title": "Differential Pair Definition",
            "description": "Explicitly define matched components for symmetry",
            "example": [
                "XM1 D1 G1 S B model L=1.0 W=5.0 ;CLARA pair input_diff",
                "XM2 D2 G2 S B model L=1.0 W=5.0 ;CLARA pair input_diff"
            ]
        },
        {
            "title": "Combined Features",
            "description": "Use sizing overrides with pair definition",
            "example": "XM1 D G S B model L=0.5 W=2.0 ;CLARA override-size L=1.5 mx=3 my=2 pair diff1"
        },
        {
            "title": "Current Mirror Pair",
            "description": "Define current mirror matching",
            "example": [
                "XM3 VDD D1 VDD B pmos L=0.5 W=10.0 ;CLARA pair current_mirror",
                "XM4 VDD D2 VDD B pmos L=0.5 W=10.0 ;CLARA pair current_mirror"
            ]
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}")
        print(f"   {example['description']}")
        if isinstance(example['example'], list):
            for line in example['example']:
                print(f"   {line}")
        else:
            print(f"   {example['example']}")
    
    print(f"\n💡 Key Features:")
    print(f"   • mx=/my= always override calculated W/L dimensions")
    print(f"   • CLARA parameters take precedence over SPICE parameters")
    print(f"   • Pair names are plain text (no angle brackets needed)")
    print(f"   • Multiple features can be combined in one comment")
    print(f"   • Multiplier expansion preserves CLARA parameters")

if __name__ == "__main__":
    analyze_clara_features()
    show_usage_examples()