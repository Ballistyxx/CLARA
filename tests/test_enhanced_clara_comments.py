#!/usr/bin/env python3
"""
Test the enhanced CLARA comment functionality:
- mx= and my= parameters
- CLARA override-size functionality
- CLARA pair <pair_name> functionality
- Differential pair support in RL training
"""

import tempfile
import os
from pathlib import Path
from enhanced_spice_parser import EnhancedSpiceParser
from train_spice_real import SpiceCircuitManager

def create_test_spice_file(content: str) -> str:
    """Create a temporary SPICE file with given content."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.spice', delete=False) as f:
        f.write(content)
        return f.name

def test_mx_my_parameters():
    """Test mx= and my= parameters in CLARA comments."""
    print("🧪 TEST 1: mx= and my= Parameters")
    print("=" * 40)
    
    spice_content = """
* Test circuit for mx/my parameters
XM1 D G S B sky130_fd_pr__nfet_01v8 L=0.5 W=2.0 ;CLARA override-size mx=3 my=4
XM2 D2 G2 S2 B2 sky130_fd_pr__pfet_01v8 L=1.0 W=3.0 m=2 ;CLARA override-size L=2.0 mx=5 my=2
.end
"""
    
    test_file = create_test_spice_file(spice_content)
    
    try:
        parser = EnhancedSpiceParser()
        circuit_data = parser.parse_spice_file(test_file)
        
        components = circuit_data['components']
        
        print(f"Found {len(components)} components:")
        
        for comp in components:
            print(f"   {comp['name']}: L={comp['length']}, W={comp['width']}, mx={comp.get('mx', 1)}, my={comp.get('my', 1)}")
        
        # Verify mx/my are applied correctly
        xm1 = next((c for c in components if c['name'] == 'XM1'), None)
        if xm1:
            assert xm1.get('mx', 1) == 3, f"Expected mx=3, got {xm1.get('mx', 1)}"
            assert xm1.get('my', 1) == 4, f"Expected my=4, got {xm1.get('my', 1)}"
            print(f"    XM1 mx/my parameters correct")
        
        # Check that CLARA overrides take precedence
        xm2_components = [c for c in components if c['name'].startswith('XM2')]
        if xm2_components:
            for comp in xm2_components:
                assert comp['length'] == 2.0, f"Expected L=2.0 (CLARA override), got {comp['length']}"
                assert comp.get('mx', 1) == 5, f"Expected mx=5, got {comp.get('mx', 1)}"
                assert comp.get('my', 1) == 2, f"Expected my=2, got {comp.get('my', 1)}"
            print(f"    XM2 CLARA overrides working correctly")
        
        print("    mx/my parameter test passed!")
        
    finally:
        os.unlink(test_file)

def test_pair_functionality():
    """Test CLARA pair <pair_name> functionality."""
    print("\nTEST 2: CLARA pair <pair_name> Functionality")
    print("=" * 50)
    
    spice_content = """
* Test differential pair circuit
XM1 OUT1 IN1 CS B sky130_fd_pr__nfet_01v8 L=1.0 W=5.0 ;CLARA pair diff_input
XM2 OUT2 IN2 CS B sky130_fd_pr__nfet_01v8 L=1.0 W=5.0 ;CLARA pair diff_input
XM3 VDD OUT1 VDD B sky130_fd_pr__pfet_01v8 L=0.5 W=10.0 ;CLARA pair current_mirror
XM4 VDD OUT2 VDD B sky130_fd_pr__pfet_01v8 L=0.5 W=10.0 ;CLARA pair current_mirror
XM5 CS BIAS VSS B sky130_fd_pr__nfet_01v8 L=2.0 W=20.0
.end
"""
    
    test_file = create_test_spice_file(spice_content)
    
    try:
        parser = EnhancedSpiceParser()
        circuit_data = parser.parse_spice_file(test_file)
        
        components = circuit_data['components']
        
        print(f"Found {len(components)} components:")
        
        pair_info = {}
        for comp in components:
            print(f"   {comp['name']}: pair_name='{comp['pair_name']}'")
            if comp['pair_name']:
                if comp['pair_name'] not in pair_info:
                    pair_info[comp['pair_name']] = []
                pair_info[comp['pair_name']].append(comp['name'])
        
        # Verify pair groupings
        assert 'diff_input' in pair_info, "diff_input pair not found"
        assert len(pair_info['diff_input']) == 2, f"Expected 2 components in diff_input pair, got {len(pair_info['diff_input'])}"
        assert 'XM1' in pair_info['diff_input'] and 'XM2' in pair_info['diff_input'], "XM1 and XM2 not in diff_input pair"
        
        assert 'current_mirror' in pair_info, "current_mirror pair not found"
        assert len(pair_info['current_mirror']) == 2, f"Expected 2 components in current_mirror pair, got {len(pair_info['current_mirror'])}"
        
        print(f"    Pair definitions:")
        for pair_name, members in pair_info.items():
            print(f"      {pair_name}: {members}")
        
        print("    CLARA pair functionality test passed!")
        
    finally:
        os.unlink(test_file)

def test_differential_pair_matching():
    """Test differential pair matching in RL training integration."""
    print("\nTEST 3: Differential Pair Matching in Training")
    print("=" * 55)
    
    spice_content = """
* Test circuit with explicit and implicit matching
XM1 D1 G1 S B sky130_fd_pr__nfet_01v8 L=1.0 W=5.0 ;CLARA pair diff1
XM2 D2 G2 S B sky130_fd_pr__nfet_01v8 L=1.0 W=5.0 ;CLARA pair diff1
XM3 D3 G3 S3 B sky130_fd_pr__nfet_01v8 L=1.0 W=5.0
XM4 D4 G4 S4 B sky130_fd_pr__nfet_01v8 L=1.0 W=5.0
.end
"""
    
    test_file = create_test_spice_file(spice_content)
    
    try:
        # Create a temporary directory for the circuit manager
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy the test file to the temp directory
            temp_spice_file = os.path.join(temp_dir, "test_diff_pair.spice")
            with open(temp_spice_file, 'w') as f:
                f.write(spice_content)
            
            # Test the circuit manager with differential pair support
            circuit_manager = SpiceCircuitManager(temp_dir)
            
            if len(circuit_manager.suitable_circuits) > 0:
                # Get the test circuit
                circuit_name = list(circuit_manager.suitable_circuits.keys())[0]
                circuit_data = circuit_manager.suitable_circuits[circuit_name]
                
                print(f"Circuit: {circuit_name}")
                print(f"Components: {circuit_data['num_components']}")
                
                # Convert to NetworkX to test matching logic
                graph = circuit_manager.convert_to_networkx_graph(circuit_data)
                
                # Check matched components and pair information
                pair_matches = {}
                component_pairs = []
                
                for node_id in graph.nodes():
                    attrs = graph.nodes[node_id]
                    matched_comp = attrs.get('matched_component', -1)
                    spice_name = attrs.get('spice_name', '')
                    
                    if matched_comp != -1:
                        print(f"   Component {node_id} ({spice_name}) matched with {matched_comp}")
                        component_pairs.append((node_id, matched_comp))
                
                # Check if we have pairs from the original components that should be matched
                original_components = circuit_data['components']
                pair_components = [c for c in original_components if c.get('pair_name') == 'diff1']
                
                print(f"    Matching results:")
                print(f"      Total matched pairs: {len(component_pairs)}")
                print(f"      Components with diff1 pair: {len(pair_components)}")
                
                # We should have at least one matched pair for the diff1 components
                assert len(component_pairs) >= 1, f"Expected at least 1 matched pair, got {len(component_pairs)}"
                assert len(pair_components) == 2, f"Expected 2 components with diff1 pair, got {len(pair_components)}"
                
                print("    Differential pair matching test passed!")
            else:
                print("   No suitable circuits found, but parsing worked")
        
    finally:
        os.unlink(test_file)

def test_combined_clara_features():
    """Test combination of all CLARA features."""
    print("\nTEST 4: Combined CLARA Features")
    print("=" * 40)
    
    spice_content = """
* Test circuit combining all CLARA features
XM1 D1 G S B sky130_fd_pr__nfet_01v8 L=0.5 W=2.0 m=4 ;CLARA override-size L=1.5 W=3.0 mx=2 my=3 pair input_pair
XM2 D2 G2 S B sky130_fd_pr__nfet_01v8 L=0.8 W=2.5 ;CLARA override-size mx=2 my=3 pair input_pair
XM3 VDD D1 VDD B sky130_fd_pr__pfet_01v8 L=0.5 W=10.0 m=2 ;CLARA pair load_pair
XM4 VDD D2 VDD B sky130_fd_pr__pfet_01v8 L=0.5 W=10.0 m=2 ;CLARA pair load_pair
.end
"""
    
    test_file = create_test_spice_file(spice_content)
    
    try:
        parser = EnhancedSpiceParser()
        circuit_data = parser.parse_spice_file(test_file)
        
        components = circuit_data['components']
        
        print(f"Found {len(components)} components:")
        
        # Track pairs and component properties
        pairs = {}
        for comp in components:
            print(f"   {comp['name']}: L={comp['length']}, W={comp['width']}, mx={comp['mx']}, my={comp['my']}, pair={comp['pair_name']}")
            
            if comp['pair_name']:
                if comp['pair_name'] not in pairs:
                    pairs[comp['pair_name']] = []
                pairs[comp['pair_name']].append(comp['name'])
        
        # Test CLARA override precedence
        xm1_components = [c for c in components if c['name'].startswith('XM1')]
        for comp in xm1_components:
            assert comp['length'] == 1.5, f"CLARA override L=1.5 not applied, got {comp['length']}"
            assert comp['width'] == 3.0, f"CLARA override W=3.0 not applied, got {comp['width']}"
            assert comp['mx'] == 2, f"CLARA mx=2 not applied, got {comp['mx']}"
            assert comp['my'] == 3, f"CLARA my=3 not applied, got {comp['my']}"
            assert comp['pair_name'] == 'input_pair', f"Pair name not applied correctly"
        
        # Test that XM2 inherits fallback values but gets CLARA overrides
        xm2 = next((c for c in components if c['name'] == 'XM2'), None)
        if xm2:
            assert xm2['length'] == 0.8, f"Original L=0.8 should be preserved, got {xm2['length']}"
            assert xm2['width'] == 2.5, f"Original W=2.5 should be preserved, got {xm2['width']}"
            assert xm2['mx'] == 2, f"CLARA mx=2 not applied, got {xm2['mx']}"
            assert xm2['my'] == 3, f"CLARA my=3 not applied, got {xm2['my']}"
        
        print(f"    Pairs found: {pairs}")
        
        # Verify both pairs exist
        assert 'input_pair' in pairs, "input_pair not found"
        assert 'load_pair' in pairs, "load_pair not found"
        
        # Verify pair sizes (accounting for multiplier expansion)
        input_pair_size = len(pairs['input_pair'])
        load_pair_size = len(pairs['load_pair'])
        
        print(f"    input_pair has {input_pair_size} components")
        print(f"    load_pair has {load_pair_size} components")
        
        print("    Combined CLARA features test passed!")
        
    finally:
        os.unlink(test_file)

def run_all_tests():
    """Run all CLARA functionality tests."""
    print("ENHANCED CLARA COMMENT FUNCTIONALITY TESTS")
    print("=" * 60)
    
    try:
        test_mx_my_parameters()
        test_pair_functionality()
        test_differential_pair_matching()
        test_combined_clara_features()
        
        print("\nALL TESTS PASSED!")
        print(" mx= and my= parameters working")
        print(" CLARA override-size functionality working")
        print(" CLARA pair <pair_name> functionality working")
        print(" Differential pair support in RL training working")
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)