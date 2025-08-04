#!/usr/bin/env python3
"""
Test script for the enhanced SPICE parser with all X-components and CLARA overrides.
"""

import tempfile
import os
from enhanced_spice_parser import EnhancedSpiceParser

def test_enhanced_parser():
    """Test the enhanced SPICE parser with various component types and CLARA overrides."""
    
    # Create a test SPICE file with various components and CLARA overrides
    test_spice_content = """
* Test SPICE file for enhanced parser
.subckt test_circuit IN OUT VDD VSS

* Transistors (should be parsed)
XM1 OUT IN VDD VDD sky130_fd_pr__pfet_g5v0d10v5 L=0.5 W=2 nf=1 m=1
XM2 OUT IN VSS VSS sky130_fd_pr__nfet_g5v0d10v5 L=0.5 W=1 nf=1 m=4
XM3 net1 IN VDD VDD sky130_fd_pr__pfet_g5v0d10v5 L=0.5 W=5 nf=1 m=1000 ;CLARA override-size L=20 W=50 nf=1 m=1

* Resistors (should be parsed)
XR1 OUT net1 sky130_fd_pr__res_xhigh_po_0p35 L=100 mult=1 m=1
XR2 net2 net3 sky130_fd_pr__res_generic_po L=50 W=2 m=2 ;CLARA override-size L=10 W=1 m=1

* Capacitors (should be parsed)
XC1 OUT VSS sky130_fd_pr__cap_mim_m3_1 W=10 L=10 m=1
XC2 net1 VSS sky130_fd_pr__cap_mim_m3_2 W=5 L=5 m=3

* Generic subcircuit instances (should be parsed)  
X1 IN net1 VDD VSS sky130_fd_sc_hvl__inv_1
X2 net1 net2 VDD VSS sky130_fd_sc_hvl__nand2_1
Xinv1 net2 OUT VDD VSS custom_inverter W=4 L=0.35

* Non-X components (should be ignored)
R1 net1 net2 1k
C1 OUT VSS 1p
M1 drain gate source bulk nmos W=1 L=0.5

.ends test_circuit
"""

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.spice', delete=False) as f:
        f.write(test_spice_content)
        temp_file = f.name

    try:
        print("TESTING ENHANCED SPICE PARSER")
        print("=" * 50)
        
        # Parse the test file
        parser = EnhancedSpiceParser()
        result = parser.parse_spice_file(temp_file)
        
        print(f"\nParsing Results:")
        print(f"Subcircuit name: {result['subcircuit_name']}")
        print(f"Total components: {result['num_components']}")
        
        print(f"\nComponent breakdown:")
        stats = result['statistics']
        for comp_type, count in stats['component_types'].items():
            print(f"  {comp_type}: {count}")
        
        print(f"\nDetailed components:")
        for i, comp in enumerate(result['components']):
            clara_info = ""
            if comp['clara_override']:
                clara_info = f" [CLARA: {comp['clara_override']}]"
            
            print(f"  {i+1}. {comp['name']} ({comp['type']}) - "
                  f"Model: {comp['device_model']}, "
                  f"L={comp['length']}, W={comp['width']}, "
                  f"Nodes: {len(comp['nodes'])}{clara_info}")
        
        print(f"\nDevice models found:")
        for model, count in stats['device_models'].items():
            print(f"  {model}: {count}")
        
        # Test specific expectations
        print(f"\nVALIDATION TESTS:")
        
        # Should find all X-components
        component_names = [comp['name'] for comp in result['components']]
        expected_base_names = ['XM1', 'XM2', 'XM3', 'XR1', 'XR2', 'XC1', 'XC2', 'X1', 'X2', 'Xinv1']
        
        print(f"Expected base components to find: {len(expected_base_names)}")
        print(f"Actual components found: {len([name for name in component_names if not '_m' in name])}")
        
        # Check for CLARA overrides
        clara_components = [comp for comp in result['components'] if comp['clara_override']]
        print(f"Components with CLARA overrides: {len(clara_components)}")
        
        for comp in clara_components:
            print(f"  {comp['name']}: {comp['clara_override']}")
        
        # Check multiplier expansion
        multiplied_components = [comp for comp in result['components'] if '_m' in comp['name']]
        print(f"Multiplied components (m > 1): {len(multiplied_components)}")
        
        # Should not find non-X components
        non_x_names = ['R1', 'C1', 'M1']
        found_non_x = [name for name in component_names if any(non_x in name for non_x in non_x_names)]
        print(f"Non-X components incorrectly found: {len(found_non_x)} (should be 0)")
        
        print("\nEnhanced parser test completed!")
        
        return result
        
    finally:
        # Clean up temp file
        os.unlink(temp_file)

if __name__ == "__main__":
    test_enhanced_parser()