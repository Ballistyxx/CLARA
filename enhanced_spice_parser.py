#!/usr/bin/env python3
"""
Enhanced SPICE parser for programmable PLL circuits.
Focuses on extracting L, W, device model, and multiplier values for RL integration.
"""

import re
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from pathlib import Path
from enum import Enum
import json
from dataclasses import dataclass


class ComponentType(Enum):
    """Component types for CLARA RL environment."""
    NMOS = 0
    PMOS = 1
    RESISTOR = 2
    CAPACITOR = 3
    INDUCTOR = 4
    CURRENT_SOURCE = 5
    VOLTAGE_SOURCE = 6
    SUBCIRCUIT = 7


@dataclass
class ComponentInfo:
    """Data structure for parsed SPICE component."""
    name: str
    component_type: ComponentType
    nodes: List[str]
    device_model: str
    length: float  # L parameter
    width: float   # W parameter
    multiplier: int  # m parameter (creates multiple identical components)
    nf: int = 1     # Number of fingers
    spice_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.spice_params is None:
            self.spice_params = {}


class EnhancedSpiceParser:
    """Enhanced SPICE parser for programmable PLL circuits."""
    
    def __init__(self):
        self.component_counter = 0
        self.node_map = {}  # Maps node names to integers
        
        # Device model patterns for component type identification
        self.device_patterns = {
            ComponentType.NMOS: [
                r'nfet', r'nmos', r'n_fet', r'n_mos',
                r'sky130_fd_pr__nfet'
            ],
            ComponentType.PMOS: [
                r'pfet', r'pmos', r'p_fet', r'p_mos', 
                r'sky130_fd_pr__pfet'
            ],
            ComponentType.RESISTOR: [
                r'res', r'resistor', r'sky130_fd_pr__res'
            ],
            ComponentType.CAPACITOR: [
                r'cap', r'capacitor', r'sky130_fd_pr__cap'
            ]
        }
        
        # Power/ground nodes to exclude from connectivity
        self.power_nodes = {
            'vdd', 'vdd3v3', 'vdd1v8', 'vss', 'vss3v3', 'vss1v8', 
            '0', 'gnd', 'ground', 'vcc', 'vee'
        }
    
    def parse_spice_file(self, filepath: str) -> Dict[str, Any]:
        """
        Parse a SPICE file and return structured data for RL integration.
        
        Args:
            filepath: Path to the SPICE file
            
        Returns:
            Dictionary containing parsed circuit data
        """
        print(f"📄 Parsing SPICE file: {filepath}")
        
        self.component_counter = 0
        self.node_map = {}
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Parse all components
        raw_components = []
        subcircuit_name = ""
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Extract subcircuit name
            if line.startswith('.subckt'):
                parts = line.split()
                if len(parts) > 1:
                    subcircuit_name = parts[1]
                continue
            
            # Skip comments, empty lines, and directives
            if (not line or line.startswith('*') or 
                line.startswith('.') or line.startswith('+')):
                continue
            
            # Parse component line
            component = self._parse_component_line(line, line_num)
            if component:
                raw_components.append(component)
        
        print(f"   Found {len(raw_components)} raw components")
        
        # Expand components based on multiplier values
        expanded_components = self._expand_multiplied_components(raw_components)
        
        print(f"   Expanded to {len(expanded_components)} individual components")
        
        # Create NetworkX graph for RL integration
        circuit_graph = self._create_circuit_graph(expanded_components)
        
        # Create structured data
        circuit_data = {
            'subcircuit_name': subcircuit_name,
            'filepath': str(filepath),
            'num_components': len(expanded_components),
            'components': [self._component_to_dict(comp) for comp in expanded_components],
            'circuit_graph': circuit_graph,
            'connectivity_matrix': self._create_connectivity_matrix(circuit_graph),
            'statistics': self._calculate_statistics(expanded_components)
        }
        
        return circuit_data
    
    def _parse_component_line(self, line: str, line_num: int) -> Optional[ComponentInfo]:
        """Parse a single SPICE component line."""
        parts = line.split()
        if len(parts) < 3:
            return None
        
        name = parts[0]
        
        # Only parse transistor components (XM prefix)
        if not name.startswith('XM'):
            return None
        
        try:
            # Standard format: XMx node1 node2 node3 node4 device_model L=x W=y nf=z m=w
            if len(parts) < 5:
                return None
            
            nodes = parts[1:5]  # Typically drain, gate, source, bulk
            device_model = parts[5] if len(parts) > 5 else ""
            
            # Parse parameters
            params = {}
            for part in parts[6:]:
                if '=' in part:
                    key, value = part.split('=', 1)
                    try:
                        # Try to convert to float for numeric parameters
                        params[key.upper()] = float(value)
                    except ValueError:
                        params[key.upper()] = value
            
            # Extract key parameters
            length = params.get('L', 0.0)
            width = params.get('W', 0.0)
            multiplier = int(params.get('M', 1))
            nf = int(params.get('NF', 1))
            
            # Determine component type from device model
            component_type = self._identify_component_type(device_model)
            
            if component_type is None:
                print(f"   Warning: Unknown device model '{device_model}' at line {line_num}")
                return None
            
            return ComponentInfo(
                name=name,
                component_type=component_type,
                nodes=nodes,
                device_model=device_model,
                length=length,
                width=width,
                multiplier=multiplier,
                nf=nf,
                spice_params=params
            )
            
        except (ValueError, IndexError) as e:
            print(f"   Warning: Failed to parse line {line_num}: {line} - {e}")
            return None
    
    def _identify_component_type(self, device_model: str) -> Optional[ComponentType]:
        """Identify component type from device model string."""
        model_lower = device_model.lower()
        
        for comp_type, patterns in self.device_patterns.items():
            for pattern in patterns:
                if pattern in model_lower:
                    return comp_type
        
        return None
    
    def _expand_multiplied_components(self, raw_components: List[ComponentInfo]) -> List[ComponentInfo]:
        """Expand components with multiplier > 1 into multiple identical components."""
        expanded = []
        
        for comp in raw_components:
            if comp.multiplier <= 1:
                expanded.append(comp)
            else:
                # Create multiple identical components
                for i in range(comp.multiplier):
                    new_comp = ComponentInfo(
                        name=f"{comp.name}_m{i}",
                        component_type=comp.component_type,
                        nodes=comp.nodes.copy(),
                        device_model=comp.device_model,
                        length=comp.length,
                        width=comp.width,
                        multiplier=1,  # Individual components have multiplier 1
                        nf=comp.nf,
                        spice_params=comp.spice_params.copy()
                    )
                    expanded.append(new_comp)
        
        return expanded
    
    def _create_circuit_graph(self, components: List[ComponentInfo]) -> nx.Graph:
        """Create NetworkX graph for RL integration."""
        G = nx.Graph()
        
        # Add components as nodes with attributes for RL
        for i, comp in enumerate(components):
            # Calculate component dimensions for RL (normalized)
            # Use W and L to determine relative size
            width_units = max(1, int(comp.width))
            height_units = max(1, int(comp.length))
            
            # Find matched components (components with similar characteristics)
            matched_component = self._find_matched_component(comp, components, i)
            
            G.add_node(i,
                      component_type=comp.component_type.value,
                      width=width_units,
                      height=height_units,
                      matched_component=matched_component,
                      spice_name=comp.name,
                      device_model=comp.device_model,
                      length=comp.length,
                      width_param=comp.width,
                      nf=comp.nf)
        
        # Add edges based on node connectivity
        self._add_connectivity_edges(G, components)
        
        return G
    
    def _find_matched_component(self, comp: ComponentInfo, all_components: List[ComponentInfo], 
                               current_index: int) -> int:
        """Find matched component for symmetry requirements."""
        # Look for components with same type, similar dimensions
        for j, other_comp in enumerate(all_components):
            if (j != current_index and 
                other_comp.component_type == comp.component_type and
                abs(other_comp.width - comp.width) < 0.1 and
                abs(other_comp.length - comp.length) < 0.1 and
                other_comp.device_model == comp.device_model):
                return j
        
        return -1  # No match found
    
    def _add_connectivity_edges(self, G: nx.Graph, components: List[ComponentInfo]):
        """Add edges based on shared nodes (excluding power/ground)."""
        # Create node-to-component mapping
        node_to_components = {}
        
        for i, comp in enumerate(components):
            for node in comp.nodes:
                node_lower = node.lower()
                if node_lower not in self.power_nodes:
                    if node not in node_to_components:
                        node_to_components[node] = []
                    node_to_components[node].append(i)
        
        # Add edges between components sharing nodes
        for node, comp_list in node_to_components.items():
            if len(comp_list) > 1:
                # Connect all components sharing this node
                for i in range(len(comp_list)):
                    for j in range(i + 1, len(comp_list)):
                        if not G.has_edge(comp_list[i], comp_list[j]):
                            G.add_edge(comp_list[i], comp_list[j])
    
    def _create_connectivity_matrix(self, G: nx.Graph) -> List[List[int]]:
        """Create adjacency matrix representation."""
        n = G.number_of_nodes()
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        
        for edge in G.edges():
            i, j = edge
            if i < n and j < n:
                matrix[i][j] = 1
                matrix[j][i] = 1
        
        return matrix
    
    def _component_to_dict(self, comp: ComponentInfo) -> Dict[str, Any]:
        """Convert ComponentInfo to dictionary for serialization."""
        return {
            'name': comp.name,
            'type': comp.component_type.name,
            'type_value': comp.component_type.value,
            'nodes': comp.nodes,
            'device_model': comp.device_model,
            'length': comp.length,
            'width': comp.width,
            'nf': comp.nf,
            'spice_params': comp.spice_params
        }
    
    def _calculate_statistics(self, components: List[ComponentInfo]) -> Dict[str, Any]:
        """Calculate statistics about the parsed circuit."""
        stats = {
            'total_components': len(components),
            'component_types': {},
            'device_models': {},
            'size_distribution': {
                'width_range': [0, 0],
                'length_range': [0, 0],
                'avg_width': 0,
                'avg_length': 0
            }
        }
        
        # Count by type
        for comp in components:
            type_name = comp.component_type.name
            stats['component_types'][type_name] = stats['component_types'].get(type_name, 0) + 1
            
            model = comp.device_model
            stats['device_models'][model] = stats['device_models'].get(model, 0) + 1
        
        # Size statistics
        if components:
            widths = [comp.width for comp in components if comp.width > 0]
            lengths = [comp.length for comp in components if comp.length > 0]
            
            if widths:
                stats['size_distribution']['width_range'] = [min(widths), max(widths)]
                stats['size_distribution']['avg_width'] = sum(widths) / len(widths)
            
            if lengths:
                stats['size_distribution']['length_range'] = [min(lengths), max(lengths)]
                stats['size_distribution']['avg_length'] = sum(lengths) / len(lengths)
        
        return stats


def parse_multiple_spice_files(directory: str) -> Dict[str, Any]:
    """Parse multiple SPICE files from a directory."""
    parser = EnhancedSpiceParser()
    results = {}
    
    spice_dir = Path(directory)
    spice_files = list(spice_dir.glob("*.spice"))
    
    print(f"🔍 Found {len(spice_files)} SPICE files in {directory}")
    
    for spice_file in spice_files:
        try:
            circuit_data = parser.parse_spice_file(str(spice_file))
            results[spice_file.name] = circuit_data
            
            print(f"✅ {spice_file.name}: {circuit_data['num_components']} components")
            
        except Exception as e:
            print(f"❌ Failed to parse {spice_file.name}: {e}")
            results[spice_file.name] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    # Test the parser
    test_file = "/home/eli/Documents/Internship/CLARA/data/netlists/sky130_am_ip__ldo_01v8.spice"
    
    parser = EnhancedSpiceParser()
    result = parser.parse_spice_file(test_file)
    
    # Save JSON dump to file
    output_file = "parsed_circuit_data.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print("\\n" + "="*60)
    print("PARSED CIRCUIT DATA")
    print("="*60)
    print(f"✅ Circuit data saved to {output_file}")
    print(f"📊 Summary: {result['num_components']} components in '{result['subcircuit_name']}'")
    
    # Also print a brief summary
    stats = result['statistics']
    print(f"📈 Component types: {stats['component_types']}")
    print(f"📏 Size ranges - Width: {stats['size_distribution']['width_range']}, Length: {stats['size_distribution']['length_range']}")
