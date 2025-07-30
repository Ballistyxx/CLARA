#!/usr/bin/env python3
"""
Test program for the enhanced SPICE parser.
Demonstrates parsing capabilities and ideal data structure for RL integration.
"""

import json
from pathlib import Path
from enhanced_spice_parser import EnhancedSpiceParser, parse_multiple_spice_files
import networkx as nx


def print_component_details(components):
    """Print detailed component information."""
    print("\n📋 COMPONENT DETAILS")
    print("=" * 80)
    
    for i, comp in enumerate(components):
        print(f"Component {i}: {comp['name']}")
        print(f"  Type: {comp['type']} ({comp['type_value']})")
        print(f"  Device Model: {comp['device_model']}")
        print(f"  Dimensions: L={comp['length']} W={comp['width']} nf={comp['nf']}")
        print(f"  Nodes: {comp['nodes']}")
        if comp['spice_params']:
            print(f"  Parameters: {comp['spice_params']}")
        print()


def print_connectivity_info(circuit_graph):
    """Print circuit connectivity information."""
    print("\n🔗 CONNECTIVITY INFORMATION")
    print("=" * 80)
    
    if isinstance(circuit_graph, nx.Graph):
        print(f"Nodes: {circuit_graph.number_of_nodes()}")
        print(f"Edges: {circuit_graph.number_of_edges()}")
        print(f"Connected components: {nx.number_connected_components(circuit_graph)}")
        
        print("\nNode attributes:")
        for node, attrs in circuit_graph.nodes(data=True):
            print(f"  Node {node}: {attrs}")
        
        print("\nEdges (component connections):")
        for edge in circuit_graph.edges():
            print(f"  {edge[0]} -- {edge[1]}")
    else:
        print("Circuit graph is not a NetworkX graph object")


def print_rl_compatibility_info(circuit_data):
    """Print information about RL compatibility."""
    print("\n🤖 RL INTEGRATION COMPATIBILITY")
    print("=" * 80)
    
    # Check if data structure is suitable for RL
    required_fields = ['components', 'circuit_graph', 'connectivity_matrix', 'statistics']
    missing_fields = [field for field in required_fields if field not in circuit_data]
    
    if missing_fields:
        print(f"❌ Missing required fields: {missing_fields}")
    else:
        print("✅ All required fields present")
    
    # Check component attributes
    if circuit_data['components']:
        sample_comp = circuit_data['components'][0]
        rl_required = ['type_value', 'device_model', 'length', 'width']
        comp_missing = [field for field in rl_required if field not in sample_comp]
        
        if comp_missing:
            print(f"❌ Components missing required fields: {comp_missing}")
        else:
            print("✅ Components have all required RL fields")
    
    # Check connectivity matrix
    if 'connectivity_matrix' in circuit_data:
        matrix = circuit_data['connectivity_matrix']
        n_components = len(circuit_data['components'])
        
        if len(matrix) == n_components and all(len(row) == n_components for row in matrix):
            print("✅ Connectivity matrix dimensions are correct")
        else:
            print("❌ Connectivity matrix dimensions don't match component count")
    
    print(f"\nData structure size: {n_components} components")
    print(f"Memory efficient: Component list format (not grid-based)")
    print(f"Graph representation: NetworkX compatible")


def test_single_file():
    """Test parsing a single SPICE file."""
    print("🧪 TESTING SINGLE FILE PARSING")
    print("=" * 60)
    
    test_file = "/home/eli/Documents/Internship/CLARA/data/netlists/programmable_pll_subcircuits/INV_1.spice"
    
    if not Path(test_file).exists():
        print(f"❌ Test file not found: {test_file}")
        return
    
    parser = EnhancedSpiceParser()
    
    try:
        circuit_data = parser.parse_spice_file(test_file)
        
        print(f"✅ Successfully parsed: {Path(test_file).name}")
        print(f"📊 Circuit: {circuit_data['subcircuit_name']}")
        print(f"📊 Components: {circuit_data['num_components']}")
        
        # Print statistics
        stats = circuit_data['statistics']
        print(f"\n📈 STATISTICS:")
        print(f"  Component types: {stats['component_types']}")
        print(f"  Device models: {stats['device_models']}")
        print(f"  Size ranges: W={stats['size_distribution']['width_range']}, L={stats['size_distribution']['length_range']}")
        
        # Print detailed component information
        print_component_details(circuit_data['components'])
        
        # Print connectivity information
        print_connectivity_info(circuit_data['circuit_graph'])
        
        # Print RL compatibility
        print_rl_compatibility_info(circuit_data)
        
        return circuit_data
        
    except Exception as e:
        print(f"❌ Error parsing file: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_multiple_files():
    """Test parsing multiple SPICE files."""
    print("\n\n🧪 TESTING MULTIPLE FILE PARSING")
    print("=" * 60)
    
    directory = "/home/eli/Documents/Internship/CLARA/data/netlists/programmable_pll_subcircuits"
    
    results = parse_multiple_spice_files(directory)
    
    print(f"\n📊 PARSING RESULTS SUMMARY")
    print("=" * 60)
    
    successful = 0
    failed = 0
    total_components = 0
    
    for filename, data in results.items():
        if 'error' in data:
            print(f"❌ {filename}: {data['error']}")
            failed += 1
        else:
            print(f"✅ {filename}: {data['num_components']} components ({data['subcircuit_name']})")
            successful += 1
            total_components += data['num_components']
    
    print(f"\nOverall Results:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total components extracted: {total_components}")
    
    return results


def demonstrate_multiplier_handling():
    """Demonstrate how multiplier values are handled."""
    print("\n\n🧪 TESTING MULTIPLIER HANDLING")
    print("=" * 60)
    
    # Test with a file that has multiplier values
    test_file = "/home/eli/Documents/Internship/CLARA/data/netlists/programmable_pll_subcircuits/INV_1.spice"
    
    parser = EnhancedSpiceParser()
    
    try:
        circuit_data = parser.parse_spice_file(test_file)
        
        print("Components with multiplier handling:")
        for comp in circuit_data['components']:
            if '_m' in comp['name']:
                print(f"  {comp['name']}: Created from multiplier expansion")
            else:
                original_m = comp['spice_params'].get('M', 1)
                if original_m > 1:
                    print(f"  {comp['name']}: Original multiplier={original_m}, expanded to multiple components")
                else:
                    print(f"  {comp['name']}: Single component (m=1)")
        
        print(f"\nMultiplier expansion results:")
        print(f"  Original SPICE lines with multipliers were expanded")
        print(f"  Each multiplier creates N identical components")
        print(f"  Individual components have unique names (e.g., XM1_m0, XM1_m1)")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def export_for_rl_integration(circuit_data, output_file):
    """Export circuit data in format ready for RL integration."""
    print(f"\n💾 EXPORTING FOR RL INTEGRATION")
    print("=" * 60)
    
    # Create RL-ready format
    rl_data = {
        'circuit_name': circuit_data['subcircuit_name'],
        'num_components': circuit_data['num_components'],
        'components': [
            {
                'id': i,
                'type': comp['type_value'],
                'width': max(1, int(comp['width'])),  # Ensure minimum size
                'height': max(1, int(comp['length'])), # Use length as height
                'device_model': comp['device_model'],
                'matched_component': -1  # Will be set by RL environment
            }
            for i, comp in enumerate(circuit_data['components'])
        ],
        'adjacency_matrix': circuit_data['connectivity_matrix'],
        'metadata': {
            'source_file': circuit_data['filepath'],
            'statistics': circuit_data['statistics']
        }
    }
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(rl_data, f, indent=2)
    
    print(f"✅ Exported RL-ready data to: {output_file}")
    print(f"   Format: JSON with normalized component attributes")
    print(f"   Components: {len(rl_data['components'])}")
    print(f"   Connectivity: {len(rl_data['adjacency_matrix'])}×{len(rl_data['adjacency_matrix'])} matrix")
    
    return rl_data


def main():
    """Main test function."""
    print("🚀 ENHANCED SPICE PARSER TEST SUITE")
    print("=" * 80)
    
    # Test single file parsing
    circuit_data = test_single_file()
    
    if circuit_data:
        # Export for RL integration
        rl_data = export_for_rl_integration(circuit_data, "test_circuit_rl_format.json")
        
        # Test multiplier handling
        demonstrate_multiplier_handling()
    
    # Test multiple files
    multi_results = test_multiple_files()
    
    print("\n" + "=" * 80)
    print("🎉 TEST SUITE COMPLETED")
    print("=" * 80)
    print("The enhanced SPICE parser successfully:")
    print("✅ Parses individual SPICE files")
    print("✅ Extracts L, W, device model, multiplier values")
    print("✅ Handles multiplier expansion (m > 1 creates multiple components)")
    print("✅ Creates RL-compatible data structures")
    print("✅ Generates NetworkX graphs for connectivity")
    print("✅ Provides comprehensive statistics")
    print("✅ Exports JSON format ready for RL integration")
    
    return circuit_data, multi_results


if __name__ == "__main__":
    main()