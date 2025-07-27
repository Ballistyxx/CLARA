#!/usr/bin/env python3
"""
SPICE netlist parser for CLARA analog layout system.
Converts SPICE files to NetworkX graphs compatible with CLARA's training pipeline.
"""

import re
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
from enum import Enum
import json


class ComponentType(Enum):
    """Component types for CLARA environment."""
    MOSFET_N = 0
    MOSFET_P = 1
    RESISTOR = 2
    CAPACITOR = 3
    INDUCTOR = 4
    CURRENT_SOURCE = 5
    VOLTAGE_SOURCE = 6
    SUBCIRCUIT = 7


class SpiceParser:
    """Parser for SPICE netlist files."""
    
    def __init__(self):
        self.component_counter = 0
        self.node_map = {}  # Maps node names to integers
        self.component_types = {
            'M': ComponentType.MOSFET_N,  # Will be refined based on device type
            'R': ComponentType.RESISTOR,
            'C': ComponentType.CAPACITOR,
            'L': ComponentType.INDUCTOR,
            'I': ComponentType.CURRENT_SOURCE,
            'V': ComponentType.VOLTAGE_SOURCE,
            'X': ComponentType.SUBCIRCUIT
        }
        
        # Device type patterns for MOSFETs
        self.nmos_patterns = [
            r'nfet', r'nmos', r'n_fet', r'n_mos',
            r'sky130_fd_pr__nfet', r'sky130_fd_pr__special_nfet'
        ]
        self.pmos_patterns = [
            r'pfet', r'pmos', r'p_fet', r'p_mos',
            r'sky130_fd_pr__pfet', r'sky130_fd_pr__special_pfet'
        ]
    
    def parse_spice_file(self, filepath: str) -> nx.Graph:
        """
        Parse a SPICE file and return a NetworkX graph.
        
        Args:
            filepath: Path to the SPICE file
            
        Returns:
            NetworkX graph representing the circuit
        """
        self.component_counter = 0
        self.node_map = {}
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Parse the netlist
        components = []
        nets = set()
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            
            # Skip comments, empty lines, and most directives
            if not line or line.startswith('*') or line.startswith('.subckt') or line.startswith('.ends') or line.startswith('.end'):
                continue
            
            # Skip PININFO comments
            if line.startswith('*.PININFO'):
                continue
                
            # Parse component line
            component = self._parse_component_line(line, line_num)
            if component:
                components.append(component)
                # Add nodes to nets
                for node in component['nodes']:
                    if node not in ['vdd', 'vdd3v3', 'vdd1v8', 'vss', 'vss3v3', 'vss1v8', '0']:
                        nets.add(node)
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add components as nodes
        component_id = 0
        for i, component in enumerate(components):
            # For now, include all components except digital subcircuits
            if (component['type'] == ComponentType.SUBCIRCUIT and 
                any(pattern in component['params'].get('model', '').lower() 
                    for pattern in ['buf', 'inv', 'schmitt'])):
                continue  # Skip digital logic gates
                
            # Determine component dimensions based on type and parameters
            width, height = self._get_component_dimensions(component)
            
            # Find matched components (for differential pairs, current mirrors, etc.)
            matched_component = self._find_matched_component(component, components, i)
            
            G.add_node(component_id,
                      component_type=component['type'].value,
                      width=width,
                      height=height,
                      matched_component=matched_component,
                      spice_name=component['name'],
                      spice_params=component['params'],
                      original_index=i)
            
            component_id += 1
        
        # Add edges based on net connectivity
        self._add_connectivity_edges(G, components, nets)
        
        return G
    
    def _parse_component_line(self, line: str, line_num: int) -> Optional[Dict]:
        """Parse a single component line from SPICE netlist."""
        parts = line.split()
        if len(parts) < 2:
            return None
        
        name = parts[0]
        component_type = self._identify_component_type(name, parts)
        
        if component_type is None:
            return None
        
        # Extract nodes and parameters based on component type
        if component_type == ComponentType.SUBCIRCUIT:
            # Subcircuit: X<name> <nodes...> <subckt_name> <params...>
            nodes = []
            params = {}
            subckt_name = ""
            
            # Find where subcircuit name starts (usually after nodes)
            for i, part in enumerate(parts[1:], 1):
                if not part.startswith('net') and not part.startswith('vdd') and not part.startswith('vss') and '=' not in part:
                    if i > 1:  # First non-node part could be subckt name
                        subckt_name = part
                        # Everything before this are nodes
                        nodes = parts[1:i]
                        # Everything after are parameters
                        param_parts = parts[i+1:]
                        break
                elif '=' in part:
                    # This is a parameter, so everything before are nodes
                    nodes = parts[1:i]
                    param_parts = parts[i:]
                    break
            
            # Parse parameters
            for part in param_parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    params[key] = value
            
        elif component_type in [ComponentType.MOSFET_N, ComponentType.MOSFET_P]:
            # MOSFET: M<name> <drain> <gate> <source> <bulk> <model> <params...>
            if len(parts) >= 6:
                nodes = parts[1:5]  # drain, gate, source, bulk
                model = parts[5] if len(parts) > 5 else ""
                params = {'model': model}
                
                # Parse remaining parameters
                for part in parts[6:]:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        params[key] = value
            else:
                return None
                
        elif component_type == ComponentType.RESISTOR:
            # Resistor: R<name> <node1> <node2> <model> <params...>
            if len(parts) >= 4:
                nodes = parts[1:3]  # Two nodes
                model = parts[3] if len(parts) > 3 else ""
                params = {'model': model}
                
                for part in parts[4:]:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        params[key] = value
            else:
                return None
                
        elif component_type == ComponentType.CAPACITOR:
            # Capacitor: C<name> <node1> <node2> <model> <params...>
            if len(parts) >= 4:
                nodes = parts[1:3]  # Two nodes
                model = parts[3] if len(parts) > 3 else ""
                params = {'model': model}
                
                for part in parts[4:]:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        params[key] = value
            else:
                return None
        else:
            # Generic two-terminal device
            nodes = parts[1:3] if len(parts) >= 3 else []
            params = {}
        
        return {
            'name': name,
            'type': component_type,
            'nodes': nodes,
            'params': params,
            'line_num': line_num
        }
    
    def _identify_component_type(self, name: str, parts: List[str]) -> Optional[ComponentType]:
        """Identify component type from name and context."""
        if not name:
            return None
            
        first_char = name[0].upper()
        
        if first_char == 'M':
            # Determine if NMOS or PMOS from model name
            model_info = ' '.join(parts).lower()
            
            for pattern in self.pmos_patterns:
                if re.search(pattern, model_info):
                    return ComponentType.MOSFET_P
            
            for pattern in self.nmos_patterns:
                if re.search(pattern, model_info):
                    return ComponentType.MOSFET_N
            
            # Default to NMOS if unclear
            return ComponentType.MOSFET_N
            
        elif first_char == 'X':
            # Subcircuit - try to determine the actual component type
            model_info = ' '.join(parts).lower()
            
            # Check for MOSFETs
            for pattern in self.pmos_patterns:
                if re.search(pattern, model_info):
                    return ComponentType.MOSFET_P
            
            for pattern in self.nmos_patterns:
                if re.search(pattern, model_info):
                    return ComponentType.MOSFET_N
            
            # Check for other components
            if 'cap' in model_info or 'mim' in model_info:
                return ComponentType.CAPACITOR
            elif 'res' in model_info:
                return ComponentType.RESISTOR
            elif 'ind' in model_info:
                return ComponentType.INDUCTOR
            
            # Check component name prefix
            name_lower = name.lower()
            if name_lower.startswith('xm'):
                # Likely a MOSFET
                if 'pfet' in model_info or 'pmos' in model_info:
                    return ComponentType.MOSFET_P
                else:
                    return ComponentType.MOSFET_N
            elif name_lower.startswith('xr'):
                return ComponentType.RESISTOR
            elif name_lower.startswith('xc'):
                return ComponentType.CAPACITOR
            elif name_lower.startswith('xl'):
                return ComponentType.INDUCTOR
            
            # Default to subcircuit
            return ComponentType.SUBCIRCUIT
            
        elif first_char in self.component_types:
            return self.component_types[first_char]
        
        return None
    
    def _get_component_dimensions(self, component: Dict) -> Tuple[int, int]:
        """Estimate component dimensions based on type and parameters."""
        comp_type = component['type']
        params = component['params']
        
        if comp_type in [ComponentType.MOSFET_N, ComponentType.MOSFET_P]:
            # Extract width and length from parameters
            try:
                width = float(params.get('W', 1))
                length = float(params.get('L', 1))
                nf = int(params.get('nf', 1))  # Number of fingers
                
                # Convert to grid units (rough approximation)
                grid_width = max(1, min(4, int(width / 0.5)))  # Scale factor
                grid_height = max(1, min(4, int(length / 0.5)))
                
                # Account for number of fingers
                if nf > 1:
                    grid_width = min(6, grid_width + int(nf / 2))
                
                return grid_width, grid_height
                
            except (ValueError, TypeError):
                return 2, 2  # Default MOSFET size
                
        elif comp_type == ComponentType.RESISTOR:
            try:
                length = float(params.get('L', 1))
                mult = int(params.get('mult', 1))
                
                # Resistors are typically long and thin
                grid_length = max(1, min(5, int(length / 100)))  # Scale factor
                grid_width = 1
                
                if mult > 1:
                    grid_width = min(3, int(mult / 2))
                
                return grid_width, grid_length
                
            except (ValueError, TypeError):
                return 1, 2  # Default resistor size
                
        elif comp_type == ComponentType.CAPACITOR:
            try:
                width = float(params.get('W', 1))
                length = float(params.get('L', 1))
                
                # Capacitors are typically square-ish
                grid_width = max(1, min(3, int(width / 10)))
                grid_height = max(1, min(3, int(length / 10)))
                
                return grid_width, grid_height
                
            except (ValueError, TypeError):
                return 2, 2  # Default capacitor size
        
        # Default size for other components
        return 1, 1
    
    def _find_matched_component(self, component: Dict, all_components: List[Dict], 
                               current_index: int) -> int:
        """Find matched component for differential pairs, current mirrors, etc."""
        comp_type = component['type']
        comp_params = component['params']
        
        # Only look for matches in MOSFETs
        if comp_type not in [ComponentType.MOSFET_N, ComponentType.MOSFET_P]:
            return -1
        
        # Look for components with similar parameters
        target_width = comp_params.get('W', '1')
        target_length = comp_params.get('L', '1')
        target_model = comp_params.get('model', '')
        
        for i, other_comp in enumerate(all_components):
            if i == current_index:
                continue
                
            if other_comp['type'] != comp_type:
                continue
            
            other_params = other_comp['params']
            
            # Check if parameters match (indicating a matched pair)
            if (other_params.get('W', '1') == target_width and
                other_params.get('L', '1') == target_length and
                other_params.get('model', '') == target_model):
                
                # Additional check: similar naming pattern
                comp_name = component['name'].lower()
                other_name = other_comp['name'].lower()
                
                # Look for patterns like M1/M2, M3/M4, etc.
                if self._are_matched_names(comp_name, other_name):
                    return i
        
        return -1
    
    def _are_matched_names(self, name1: str, name2: str) -> bool:
        """Check if two component names suggest they are matched."""
        # Extract numeric parts
        num1_match = re.search(r'(\d+)$', name1)
        num2_match = re.search(r'(\d+)$', name2)
        
        if num1_match and num2_match:
            num1 = int(num1_match.group(1))
            num2 = int(num2_match.group(1))
            
            # Check for consecutive numbers or common differential pair patterns
            if abs(num1 - num2) == 1:  # Consecutive numbers
                return True
            
            # Common patterns: 1&2, 3&4, 5&6, etc.
            if num1 % 2 == 1 and num2 == num1 + 1:
                return True
            if num2 % 2 == 1 and num1 == num2 + 1:
                return True
        
        return False
    
    def _add_connectivity_edges(self, G: nx.Graph, components: List[Dict], nets: Set[str]):
        """Add edges to graph based on net connectivity."""
        # Create mapping from original component index to graph node ID
        original_to_graph = {}
        for node_id in G.nodes():
            original_idx = G.nodes[node_id].get('original_index', node_id)
            original_to_graph[original_idx] = node_id
        
        # Group components by the nets they connect to
        net_to_components = {}
        
        for i, component in enumerate(components):
            # Skip if this component is not in the graph
            if i not in original_to_graph:
                continue
                
            graph_node_id = original_to_graph[i]
                
            for node in component['nodes']:
                # Skip power and ground nets
                if node in ['vdd', 'vdd3v3', 'vdd1v8', 'vss', 'vss3v3', 'vss1v8', '0']:
                    continue
                    
                if node not in net_to_components:
                    net_to_components[node] = []
                net_to_components[node].append(graph_node_id)
        
        # Add edges between components connected to the same net
        for net, component_list in net_to_components.items():
            if len(component_list) > 1:
                # Connect all components on this net
                for i in range(len(component_list)):
                    for j in range(i + 1, len(component_list)):
                        comp1, comp2 = component_list[i], component_list[j]
                        if G.has_node(comp1) and G.has_node(comp2):
                            G.add_edge(comp1, comp2, net=net)
    
    def save_circuit_to_json(self, circuit: nx.Graph, filepath: str, 
                           source_spice: str = ""):
        """Save parsed circuit to JSON format compatible with CLARA."""
        data = {
            'source_spice': source_spice,
            'nodes': [],
            'edges': []
        }
        
        # Save nodes
        for node_id in circuit.nodes():
            node_data = {'id': node_id}
            node_data.update(circuit.nodes[node_id])
            
            # Convert any non-serializable values
            for key, value in node_data.items():
                if isinstance(value, np.integer):
                    node_data[key] = int(value)
                elif isinstance(value, np.floating):
                    node_data[key] = float(value)
            
            data['nodes'].append(node_data)
        
        # Save edges
        for edge in circuit.edges(data=True):
            edge_data = {
                'source': int(edge[0]),
                'target': int(edge[1])
            }
            if len(edge) > 2 and edge[2]:  # Edge attributes
                edge_data.update(edge[2])
            data['edges'].append(edge_data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_circuit_from_json(self, filepath: str) -> nx.Graph:
        """Load circuit from JSON format."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        G = nx.Graph()
        
        # Load nodes
        for node_data in data['nodes']:
            node_id = node_data.pop('id')
            G.add_node(node_id, **node_data)
        
        # Load edges
        for edge_data in data['edges']:
            source = edge_data.pop('source')
            target = edge_data.pop('target')
            G.add_edge(source, target, **edge_data)
        
        return G


def main():
    """Test the SPICE parser with the example schematic."""
    parser = SpiceParser()
    
    # Parse the example SPICE file
    spice_file = "/home/eli/Documents/Internship/CLARA/schematics/sky130_ef_ip__simple_por.spice"
    
    if not Path(spice_file).exists():
        print(f"❌ SPICE file not found: {spice_file}")
        return
    
    print("🔧 Parsing SPICE file...")
    try:
        circuit = parser.parse_spice_file(spice_file)
        
        print(f"✅ Successfully parsed SPICE file!")
        print(f"   Components: {len(circuit.nodes)}")
        print(f"   Connections: {len(circuit.edges)}")
        
        # Show component details
        print("\n📋 Component Details:")
        for node_id in sorted(circuit.nodes()):
            attrs = circuit.nodes[node_id]
            comp_type = attrs.get('component_type', 0)
            type_names = {0: 'NMOS', 1: 'PMOS', 2: 'R', 3: 'C', 4: 'L', 5: 'I', 6: 'V', 7: 'SUB'}
            type_name = type_names.get(comp_type, f'Type{comp_type}')
            matched = attrs.get('matched_component', -1)
            match_str = f" (matched with {matched})" if matched != -1 else ""
            
            print(f"  {node_id}: {type_name} {attrs.get('width', 1)}×{attrs.get('height', 1)} "
                  f"[{attrs.get('spice_name', 'N/A')}]{match_str}")
        
        # Show connections
        print(f"\n🔗 Connections:")
        for edge in circuit.edges(data=True):
            net_info = f" (net: {edge[2].get('net', 'unknown')})" if len(edge) > 2 else ""
            print(f"  {edge[0]} ↔ {edge[1]}{net_info}")
        
        # Save to JSON
        output_file = "/home/eli/Documents/Internship/CLARA/data/circuits/spice_parsed_por.json"
        parser.save_circuit_to_json(circuit, output_file, source_spice=spice_file)
        print(f"\n💾 Saved circuit to: {output_file}")
        
        # Test compatibility with CLARA environment
        print(f"\n🧪 Testing CLARA compatibility...")
        from analog_layout_env import AnalogLayoutEnv
        
        env = AnalogLayoutEnv(grid_size=15, max_components=len(circuit.nodes))
        result = env.reset(circuit_graph=circuit)
        
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
        
        print(f"✅ Circuit loaded successfully in CLARA environment!")
        print(f"   Grid size: {env.grid_size}×{env.grid_size}")
        print(f"   Components to place: {env.num_components}")
        
    except Exception as e:
        print(f"❌ Error parsing SPICE file: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()