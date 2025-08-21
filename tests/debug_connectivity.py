#!/usr/bin/env python3
"""
Debug utility to analyze connectivity patterns in SPICE circuits.
Identifies problematic nodes causing connectivity explosion.
"""

import sys
import os
import networkx as nx
from collections import Counter, defaultdict
from enhanced_spice_parser import EnhancedSpiceParser
import matplotlib.pyplot as plt


def analyze_connectivity_patterns(spice_file: str):
    """Analyze connectivity patterns in a SPICE circuit."""
    print(f"CONNECTIVITY ANALYSIS: {spice_file}")
    print("=" * 60)
    
    parser = EnhancedSpiceParser()
    circuit_data = parser.parse_spice_file(spice_file)
    
    # Basic stats
    print(f"BASIC STATISTICS:")
    print(f"   Raw components: {len(circuit_data['components'])}")
    print(f"   Total edges: {circuit_data['circuit_graph'].number_of_edges()}")
    print(f"   Avg degree: {2 * circuit_data['circuit_graph'].number_of_edges() / max(circuit_data['circuit_graph'].number_of_nodes(), 1):.1f}")
    print()
    
    # Analyze node sharing patterns
    print(f"NODE SHARING ANALYSIS:")
    node_to_components = defaultdict(list)
    
    # Parse components again to get node info
    with open(spice_file, 'r') as f:
        lines = f.readlines()
    
    component_nodes = []
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('*') or line.startswith('.'):
            continue
            
        parts = line.split()
        if len(parts) < 3:
            continue
            
        name = parts[0]
        if not name.upper().startswith('X'):
            continue
            
        # Extract nodes (varies by component type)
        if name.upper().startswith('XM'):
            nodes = parts[1:5] if len(parts) >= 5 else parts[1:4]
        elif name.upper().startswith('XR'):
            nodes = parts[1:3] if len(parts) >= 3 else []
        elif name.upper().startswith('XC'):
            nodes = parts[1:3] if len(parts) >= 3 else []
        else:
            # Find device model position and extract nodes before it
            device_model_idx = None
            for i in range(len(parts) - 1, 0, -1):
                if '=' not in parts[i]:
                    device_model_idx = i
                    break
            
            if device_model_idx:
                nodes = parts[1:device_model_idx]
            else:
                nodes = parts[1:3] if len(parts) > 2 else []
        
        component_nodes.append((name, nodes))
        
        # Track node sharing
        for node in nodes:
            node_to_components[node].append(name)
    
    # Analyze node sharing patterns
    sharing_stats = Counter()
    problematic_nodes = []
    
    for node, components in node_to_components.items():
        count = len(components)
        sharing_stats[count] += 1
        
        if count > 10:  # Nodes shared by many components
            problematic_nodes.append((node, count, components[:5]))  # Show first 5
    
    print(f"   Node sharing distribution:")
    for shared_count in sorted(sharing_stats.keys()):
        print(f"     {shared_count} components: {sharing_stats[shared_count]} nodes")
    
    print(f"\nPROBLEMATIC NODES (shared by >10 components):")
    for node, count, sample_components in sorted(problematic_nodes, key=lambda x: x[1], reverse=True)[:10]:
        print(f"   '{node}': {count} components (e.g., {', '.join(sample_components)}...)")
    
    # Analyze component types sharing nodes
    print(f"\nCOMPONENT TYPE ANALYSIS:")
    type_sharing = defaultdict(lambda: defaultdict(int))
    
    for name, nodes in component_nodes:
        comp_type = name[:2].upper()  # XM, XR, XC, etc.
        for node in nodes:
            type_sharing[node][comp_type] += 1
    
    multi_type_nodes = []
    for node, type_counts in type_sharing.items():
        if len(type_counts) > 1:  # Node shared by multiple component types
            total = sum(type_counts.values())
            multi_type_nodes.append((node, total, dict(type_counts)))
    
    print(f"   Nodes shared by multiple component types: {len(multi_type_nodes)}")
    for node, total, type_counts in sorted(multi_type_nodes, key=lambda x: x[1], reverse=True)[:5]:
        types_str = ", ".join([f"{t}:{c}" for t, c in type_counts.items()])
        print(f"     '{node}': {total} total ({types_str})")
    
    # Identify likely bias/reference/clock nodes
    print(f"\nSUSPECTED GLOBAL NODES:")
    suspected_globals = []
    
    for node, count, _ in problematic_nodes:
        node_lower = node.lower()
        if any(keyword in node_lower for keyword in ['vdd', 'vss', 'gnd', 'vcc', 'vee', 'avdd', 'avss', 
                                                     'bias', 'ref', 'clk', 'clock', 'en', 'enable',
                                                     'vbg', 'vout', 'vin', 'sub', 'bulk']):
            suspected_globals.append((node, count))
    
    for node, count in suspected_globals:
        print(f"   '{node}': {count} connections")
    
    print(f"\nRECOMMENDATIONS:")
    total_problematic = sum(count for _, count, _ in problematic_nodes)
    print(f"   - Filter out {len(suspected_globals)} global nodes")
    print(f"   - This could reduce {total_problematic} connections")
    print(f"   - Focus on signal nodes with 2-4 connections")
    print(f"   - Implement max connections per node limit (e.g., 5-8)")
    
    return {
        'total_components': len(component_nodes),
        'total_edges': circuit_data['circuit_graph'].number_of_edges(),
        'problematic_nodes': problematic_nodes,
        'suspected_globals': suspected_globals,
        'node_sharing_stats': dict(sharing_stats)
    }


def visualize_connectivity_distribution(analysis_data):
    """Create visualization of connectivity distribution."""
    sharing_stats = analysis_data['node_sharing_stats']
    
    plt.figure(figsize=(12, 8))
    
    # Main distribution plot
    plt.subplot(2, 2, 1)
    shared_counts = list(sharing_stats.keys())
    node_counts = list(sharing_stats.values())
    
    plt.bar(shared_counts, node_counts, alpha=0.7)
    plt.xlabel('Components sharing node')
    plt.ylabel('Number of nodes')
    plt.title('Node Sharing Distribution')
    plt.yscale('log')
    
    # Cumulative connections
    plt.subplot(2, 2, 2)
    cumulative_connections = []
    total_connections = 0
    
    for shared_count in sorted(shared_counts):
        connections_from_this_level = shared_count * (shared_count - 1) // 2 * sharing_stats[shared_count]
        total_connections += connections_from_this_level
        cumulative_connections.append(total_connections)
    
    plt.plot(sorted(shared_counts), cumulative_connections, 'ro-')
    plt.xlabel('Max components per node')
    plt.ylabel('Total connections')
    plt.title('Cumulative Connection Count')
    
    # Problem nodes
    plt.subplot(2, 2, 3)
    problematic = analysis_data['problematic_nodes'][:10]
    if problematic:
        nodes, counts, _ = zip(*problematic)
        nodes = [n[:8] + '...' if len(n) > 8 else n for n in nodes]  # Truncate long names
        
        plt.barh(range(len(nodes)), counts, alpha=0.7)
        plt.yticks(range(len(nodes)), nodes)
        plt.xlabel('Number of connections')
        plt.title('Top Problematic Nodes')
    
    # Summary stats
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.8, f"Total Components: {analysis_data['total_components']}", fontsize=12)
    plt.text(0.1, 0.7, f"Total Edges: {analysis_data['total_edges']}", fontsize=12)
    plt.text(0.1, 0.6, f"Problematic Nodes: {len(analysis_data['problematic_nodes'])}", fontsize=12)
    plt.text(0.1, 0.5, f"Suspected Globals: {len(analysis_data['suspected_globals'])}", fontsize=12)
    
    avg_degree = 2 * analysis_data['total_edges'] / max(analysis_data['total_components'], 1)
    plt.text(0.1, 0.4, f"Avg Degree: {avg_degree:.1f}", fontsize=12)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Summary Statistics')
    
    plt.tight_layout()
    plt.savefig('connectivity_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Test files
    test_files = [
        "data/netlists/sky130_am_ip__ldo_01v8.spice",
        "data/programmable_pll_subcircuits/NAND.spice"
    ]
    
    for spice_file in test_files:
        if os.path.exists(spice_file):
            print(f"\n" + "="*80)
            analysis = analyze_connectivity_patterns(spice_file)
            
            if spice_file.endswith("ldo_01v8.spice"):
                print(f"\nCreating visualization...")
                visualize_connectivity_distribution(analysis)
            
            print(f"\n" + "="*80)
        else:
            print(f"File not found: {spice_file}")