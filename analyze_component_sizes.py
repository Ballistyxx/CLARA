#!/usr/bin/env python3
"""
Analyze actual component sizes from LDO circuit.
"""

import numpy as np
from pathlib import Path
from enhanced_spice_parser import EnhancedSpiceParser

def analyze_ldo_component_sizes():
    """Analyze component sizes in LDO circuit."""
    print("COMPONENT SIZE ANALYSIS")
    print("=" * 50)
    
    ldo_file = "/home/eli/Documents/Internship/CLARA/data/netlists/sky130_am_ip__ldo_01v8.spice"
    
    if not Path(ldo_file).exists():
        print(f"File not found: {ldo_file}")
        return
    
    parser = EnhancedSpiceParser()
    circuit_data = parser.parse_spice_file(ldo_file)
    
    components = circuit_data['components']
    
    print(f"Total components: {len(components)}")
    
    # Analyze component dimensions
    widths = []
    lengths = []
    areas = []
    types = {}
    
    print(f"\nComponent Details (first 30):")
    print(f"{'ID':<3} {'Name':<20} {'Type':<10} {'W':<8} {'L':<8} {'Area':<6}")
    print("-" * 60)
    
    for i, comp in enumerate(components[:30]):  # Show first 30
        width = comp['width']
        length = comp['length']
        area = width * length
        comp_type = comp['type']
        
        widths.append(width)
        lengths.append(length)
        areas.append(area)
        
        types[comp_type] = types.get(comp_type, 0) + 1
        
        print(f"{i:<3} {comp['name'][:18]:<20} {comp_type:<10} {width:<8.1f} {length:<8.1f} {area:<6.1f}")
    
    print(f"\nSize Statistics:")
    print(f"   Width range: {min(widths):.1f} - {max(widths):.1f}")
    print(f"   Length range: {min(lengths):.1f} - {max(lengths):.1f}")
    print(f"   Area range: {min(areas):.1f} - {max(areas):.1f}")
    print(f"   Average area: {np.mean(areas):.1f}")
    print(f"   Total area: {sum(areas):.1f}")
    
    print(f"\nComponent Types:")
    for comp_type, count in sorted(types.items()):
        type_components = [comp for comp in components if comp['type'] == comp_type]
        type_areas = [comp['width'] * comp['length'] for comp in type_components]
        avg_area = np.mean(type_areas) if type_areas else 0
        print(f"   {comp_type}: {count} components, avg area {avg_area:.1f}")
    
    # Find the largest components
    print(f"\nLargest Components:")
    sorted_components = sorted(enumerate(components), key=lambda x: x[1]['width'] * x[1]['length'], reverse=True)
    
    for i, (idx, comp) in enumerate(sorted_components[:10]):
        area = comp['width'] * comp['length']
        print(f"   {i+1}. {comp['name']}: {comp['width']:.1f} x {comp['length']:.1f} = {area:.1f} cells")
    
    # Recommend optimal grid sizes
    print(f"\nGRID SIZE RECOMMENDATIONS:")
    total_area = sum(areas)
    
    # Account for placement inefficiency (assume 70% efficiency)
    recommended_area = total_area / 0.7
    recommended_size = int(np.sqrt(recommended_area)) + 5
    
    print(f"   Component area: {total_area:.0f} cells")
    print(f"   With 70% efficiency: {recommended_area:.0f} cells needed")
    print(f"   Recommended grid size: {recommended_size}x{recommended_size}")
    
    # Test different grid sizes
    grid_sizes = [30, 40, 50, 64]
    for size in grid_sizes:
        total_cells = size * size
        utilization = total_area / total_cells
        status = "Good" if utilization < 0.6 else "Tight" if utilization < 0.8 else "Too small"
        print(f"   {size}x{size} ({total_cells} cells): {100*utilization:.1f}% utilization {status}")

def analyze_component_scaling_strategies():
    """Analyze strategies to handle large component sizes."""
    print(f"\nCOMPONENT SCALING STRATEGIES")
    print("=" * 50)
    
    ldo_file = "/home/eli/Documents/Internship/CLARA/data/netlists/sky130_am_ip__ldo_01v8.spice"
    parser = EnhancedSpiceParser()
    circuit_data = parser.parse_spice_file(ldo_file)
    
    components = circuit_data['components']
    areas = [comp['width'] * comp['length'] for comp in components]
    
    print(f"Original total area: {sum(areas):.0f}")
    
    # Strategy 1: Normalize to unit dimensions
    print(f"\n1. Unit Normalization (all components 1x1):")
    unit_area = len(components)
    grid_size = int(np.sqrt(unit_area / 0.6)) + 2  # 60% efficiency
    print(f"   Total area: {unit_area}")
    print(f"   Recommended grid: {grid_size}x{grid_size}")
    
    # Strategy 2: Log scaling
    print(f"\n2. Logarithmic Scaling:")
    log_widths = [max(1, int(np.log10(max(comp['width'], 1)) + 1)) for comp in components]
    log_lengths = [max(1, int(np.log10(max(comp['length'], 1)) + 1)) for comp in components]
    log_areas = [w * l for w, l in zip(log_widths, log_lengths)]
    log_total = sum(log_areas)
    grid_size = int(np.sqrt(log_total / 0.6)) + 2
    print(f"   Width range: {min(log_widths)} - {max(log_widths)}")
    print(f"   Length range: {min(log_lengths)} - {max(log_lengths)}")
    print(f"   Total area: {log_total}")
    print(f"   Recommended grid: {grid_size}x{grid_size}")
    
    # Strategy 3: Square root scaling
    print(f"\n3. Square Root Scaling:")
    sqrt_widths = [max(1, int(np.sqrt(comp['width']))) for comp in components]
    sqrt_lengths = [max(1, int(np.sqrt(comp['length']))) for comp in components]
    sqrt_areas = [w * l for w, l in zip(sqrt_widths, sqrt_lengths)]
    sqrt_total = sum(sqrt_areas)
    grid_size = int(np.sqrt(sqrt_total / 0.6)) + 2
    print(f"   Width range: {min(sqrt_widths)} - {max(sqrt_widths)}")
    print(f"   Length range: {min(sqrt_lengths)} - {max(sqrt_lengths)}")
    print(f"   Total area: {sqrt_total}")
    print(f"   Recommended grid: {grid_size}x{grid_size}")
    
    # Strategy 4: Capped scaling
    print(f"\n4. Capped Scaling (max 5x5):")
    cap_widths = [min(5, max(1, int(comp['width']))) for comp in components]
    cap_lengths = [min(5, max(1, int(comp['length']))) for comp in components]
    cap_areas = [w * l for w, l in zip(cap_widths, cap_lengths)]
    cap_total = sum(cap_areas)
    grid_size = int(np.sqrt(cap_total / 0.6)) + 2
    print(f"   Width range: {min(cap_widths)} - {max(cap_widths)}")
    print(f"   Length range: {min(cap_lengths)} - {max(cap_lengths)}")
    print(f"   Total area: {cap_total}")
    print(f"   Recommended grid: {grid_size}x{grid_size}")

if __name__ == "__main__":
    analyze_ldo_component_sizes()
    analyze_component_scaling_strategies()