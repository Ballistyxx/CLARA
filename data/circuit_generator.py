import json
import networkx as nx
import numpy as np
from typing import List, Dict, Any
from enum import Enum


class ComponentType(Enum):
    MOSFET_N = 0
    MOSFET_P = 1
    RESISTOR = 2
    CAPACITOR = 3
    INDUCTOR = 4
    CURRENT_SOURCE = 5
    VOLTAGE_SOURCE = 6


class AnalogCircuitGenerator:
    """Generator for common analog circuit topologies."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.component_id_counter = 0
    
    def reset_counter(self):
        """Reset component ID counter."""
        self.component_id_counter = 0
    
    def _add_component(self, G: nx.Graph, comp_type: ComponentType, 
                      width: int = 1, height: int = 1, 
                      matched_component: int = -1) -> int:
        """Add a component to the graph and return its ID."""
        comp_id = self.component_id_counter
        self.component_id_counter += 1
        
        G.add_node(comp_id,
                   component_type=comp_type.value,
                   width=width,
                   height=height,
                   matched_component=matched_component)
        
        return comp_id
    
    def generate_differential_pair(self) -> nx.Graph:
        """Generate a differential pair circuit."""
        self.reset_counter()
        G = nx.Graph()
        
        # Two matched NMOS transistors
        m1 = self._add_component(G, ComponentType.MOSFET_N, 2, 2)
        m2 = self._add_component(G, ComponentType.MOSFET_N, 2, 2, matched_component=m1)
        G.nodes[m1]['matched_component'] = m2
        
        # Current source (tail current)
        i_tail = self._add_component(G, ComponentType.CURRENT_SOURCE, 1, 2)
        
        # Load resistors (matched)
        r1 = self._add_component(G, ComponentType.RESISTOR, 1, 1)
        r2 = self._add_component(G, ComponentType.RESISTOR, 1, 1, matched_component=r1)
        G.nodes[r1]['matched_component'] = r2
        
        # Connections
        G.add_edge(m1, i_tail)  # M1 source to current source
        G.add_edge(m2, i_tail)  # M2 source to current source
        G.add_edge(m1, r1)      # M1 drain to R1
        G.add_edge(m2, r2)      # M2 drain to R2
        
        return G
    
    def generate_current_mirror(self) -> nx.Graph:
        """Generate a current mirror circuit."""
        self.reset_counter()
        G = nx.Graph()
        
        # Reference transistor (diode-connected)
        m_ref = self._add_component(G, ComponentType.MOSFET_N, 2, 2)
        
        # Mirror transistors (matched to reference)
        m1 = self._add_component(G, ComponentType.MOSFET_N, 2, 2, matched_component=m_ref)
        m2 = self._add_component(G, ComponentType.MOSFET_N, 2, 2, matched_component=m_ref)
        
        G.nodes[m_ref]['matched_component'] = m1  # Set primary match
        
        # Current source (input current)
        i_in = self._add_component(G, ComponentType.CURRENT_SOURCE, 1, 2)
        
        # Connections
        G.add_edge(m_ref, i_in)  # Reference transistor to input current
        G.add_edge(m_ref, m1)    # Gate connections (common gate)
        G.add_edge(m_ref, m2)    # Gate connections (common gate)
        G.add_edge(m1, m2)       # Sources connected
        
        return G
    
    def generate_common_source_amplifier(self) -> nx.Graph:
        """Generate a common source amplifier."""
        self.reset_counter()
        G = nx.Graph()
        
        # Main amplifier transistor
        m_amp = self._add_component(G, ComponentType.MOSFET_N, 3, 3)
        
        # Load (could be resistor or current source)
        load = self._add_component(G, ComponentType.RESISTOR, 1, 2)
        
        # Bias current source
        i_bias = self._add_component(G, ComponentType.CURRENT_SOURCE, 1, 2)
        
        # Coupling capacitors
        c_in = self._add_component(G, ComponentType.CAPACITOR, 1, 1)
        c_out = self._add_component(G, ComponentType.CAPACITOR, 1, 1)
        
        # Connections
        G.add_edge(c_in, m_amp)   # Input coupling
        G.add_edge(m_amp, load)   # Drain to load
        G.add_edge(m_amp, i_bias) # Source to bias
        G.add_edge(load, c_out)   # Output coupling
        
        return G
    
    def generate_operational_transconductance_amplifier(self) -> nx.Graph:
        """Generate a simple OTA (differential pair + current mirror load)."""
        self.reset_counter()
        G = nx.Graph()
        
        # Input differential pair
        m1 = self._add_component(G, ComponentType.MOSFET_N, 2, 2)
        m2 = self._add_component(G, ComponentType.MOSFET_N, 2, 2, matched_component=m1)
        G.nodes[m1]['matched_component'] = m2
        
        # Current mirror load (PMOS)
        m3 = self._add_component(G, ComponentType.MOSFET_P, 2, 2)
        m4 = self._add_component(G, ComponentType.MOSFET_P, 2, 2, matched_component=m3)
        G.nodes[m3]['matched_component'] = m4
        
        # Tail current source
        i_tail = self._add_component(G, ComponentType.CURRENT_SOURCE, 1, 2)
        
        # Bias voltage source for PMOS gates
        v_bias = self._add_component(G, ComponentType.VOLTAGE_SOURCE, 1, 1)
        
        # Connections
        # Differential pair
        G.add_edge(m1, i_tail)
        G.add_edge(m2, i_tail)
        
        # Current mirror connections
        G.add_edge(m1, m3)  # M1 drain to M3 drain
        G.add_edge(m2, m4)  # M2 drain to M4 drain
        G.add_edge(m3, m4)  # PMOS gates connected
        G.add_edge(m3, v_bias)  # Bias connection
        
        return G
    
    def generate_bandgap_reference(self) -> nx.Graph:
        """Generate a simplified bandgap reference circuit."""
        self.reset_counter()
        G = nx.Graph()
        
        # Bipolar transistors (represented as special MOSFETs)
        q1 = self._add_component(G, ComponentType.MOSFET_N, 2, 3)  # Larger for different current density
        q2 = self._add_component(G, ComponentType.MOSFET_N, 1, 3, matched_component=q1)
        
        # Operational amplifier (simplified as single component)
        opamp = self._add_component(G, ComponentType.MOSFET_N, 3, 2)
        
        # Resistors
        r1 = self._add_component(G, ComponentType.RESISTOR, 1, 2)
        r2 = self._add_component(G, ComponentType.RESISTOR, 1, 1, matched_component=r1)
        G.nodes[r1]['matched_component'] = r2
        
        # PMOS current sources
        m_curr1 = self._add_component(G, ComponentType.MOSFET_P, 2, 2)
        m_curr2 = self._add_component(G, ComponentType.MOSFET_P, 2, 2, matched_component=m_curr1)
        G.nodes[m_curr1]['matched_component'] = m_curr2
        
        # Connections (simplified topology)
        G.add_edge(q1, r1)
        G.add_edge(q2, r2)
        G.add_edge(r1, opamp)
        G.add_edge(r2, opamp)
        G.add_edge(q1, m_curr1)
        G.add_edge(q2, m_curr2)
        G.add_edge(m_curr1, m_curr2)  # Current mirror
        
        return G
    
    def generate_random_circuit(self, min_components: int = 3, max_components: int = 10) -> nx.Graph:
        """Generate a random circuit with specified component range."""
        self.reset_counter()
        num_components = np.random.randint(min_components, max_components + 1)
        
        G = nx.Graph()
        component_types = list(ComponentType)
        
        # Add random components
        for _ in range(num_components):
            comp_type = np.random.choice(component_types)
            
            # Size based on component type
            if comp_type in [ComponentType.MOSFET_N, ComponentType.MOSFET_P]:
                width, height = np.random.randint(1, 4), np.random.randint(2, 4)
            elif comp_type == ComponentType.RESISTOR:
                width, height = 1, np.random.randint(1, 3)
            elif comp_type == ComponentType.CAPACITOR:
                width, height = np.random.randint(1, 3), np.random.randint(1, 3)
            else:
                width, height = np.random.randint(1, 3), np.random.randint(1, 3)
            
            self._add_component(G, comp_type, width, height)
        
        # Add random connections
        nodes = list(G.nodes())
        num_edges = np.random.randint(max(1, num_components - 1), min(num_components * 2, len(nodes) * (len(nodes) - 1) // 2))
        
        for _ in range(num_edges):
            if len(nodes) > 1:
                u, v = np.random.choice(nodes, 2, replace=False)
                G.add_edge(u, v)
        
        # Add some matched pairs for symmetry
        if num_components >= 4:
            nodes_list = list(G.nodes())
            num_pairs = min(2, num_components // 2)
            
            for _ in range(num_pairs):
                if len(nodes_list) >= 2:
                    pair = np.random.choice(nodes_list, 2, replace=False)
                    nodes_list.remove(pair[0])
                    nodes_list.remove(pair[1])
                    
                    G.nodes[pair[0]]['matched_component'] = pair[1]
                    G.nodes[pair[1]]['matched_component'] = pair[0]
        
        return G
    
    def save_circuit_to_json(self, circuit: nx.Graph, filename: str):
        """Save circuit to JSON file."""
        data = {
            'nodes': [],
            'edges': []
        }
        
        # Save nodes
        for node_id in circuit.nodes():
            node_data = {'id': node_id}
            node_data.update(circuit.nodes[node_id])
            data['nodes'].append(node_data)
        
        # Save edges
        for edge in circuit.edges():
            data['edges'].append({'source': edge[0], 'target': edge[1]})
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_circuit_from_json(self, filename: str) -> nx.Graph:
        """Load circuit from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        G = nx.Graph()
        
        # Load nodes
        for node_data in data['nodes']:
            node_id = node_data.pop('id')
            G.add_node(node_id, **node_data)
        
        # Load edges
        for edge_data in data['edges']:
            G.add_edge(edge_data['source'], edge_data['target'])
        
        return G


def generate_circuit_dataset(output_dir: str = "data/circuits", num_circuits: int = 100):
    """Generate a dataset of analog circuits."""
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    generator = AnalogCircuitGenerator()
    
    circuit_info = []
    
    # Generate specific circuit types
    specific_circuits = [
        ("differential_pair", generator.generate_differential_pair),
        ("current_mirror", generator.generate_current_mirror),
        ("common_source", generator.generate_common_source_amplifier),
        ("ota", generator.generate_operational_transconductance_amplifier),
        ("bandgap", generator.generate_bandgap_reference),
    ]
    
    circuit_count = 0
    
    # Generate multiple instances of each specific circuit type
    for circuit_name, circuit_func in specific_circuits:
        for i in range(5):  # 5 instances of each type
            try:
                circuit = circuit_func()
                filename = os.path.join(output_dir, f"{circuit_name}_{i:02d}.json")
                generator.save_circuit_to_json(circuit, filename)
                
                circuit_info.append({
                    'filename': f"{circuit_name}_{i:02d}.json",
                    'type': circuit_name,
                    'num_components': len(circuit.nodes),
                    'num_connections': len(circuit.edges),
                    'difficulty': 'easy' if len(circuit.nodes) <= 5 else 'medium' if len(circuit.nodes) <= 8 else 'hard'
                })
                
                circuit_count += 1
                
            except Exception as e:
                print(f"Error generating {circuit_name}_{i}: {e}")
    
    # Generate random circuits
    remaining_circuits = num_circuits - circuit_count
    
    for i in range(remaining_circuits):
        try:
            # Generate circuits of varying complexity
            if i < remaining_circuits // 3:
                circuit = generator.generate_random_circuit(3, 5)  # Easy
                difficulty = 'easy'
            elif i < 2 * remaining_circuits // 3:
                circuit = generator.generate_random_circuit(5, 8)  # Medium
                difficulty = 'medium'
            else:
                circuit = generator.generate_random_circuit(8, 10)  # Hard
                difficulty = 'hard'
            
            filename = os.path.join(output_dir, f"random_{i:03d}.json")
            generator.save_circuit_to_json(circuit, filename)
            
            circuit_info.append({
                'filename': f"random_{i:03d}.json",
                'type': 'random',
                'num_components': len(circuit.nodes),
                'num_connections': len(circuit.edges),
                'difficulty': difficulty
            })
            
        except Exception as e:
            print(f"Error generating random circuit {i}: {e}")
    
    # Save circuit metadata
    metadata_file = os.path.join(output_dir, "circuit_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(circuit_info, f, indent=2)
    
    print(f"Generated {len(circuit_info)} circuits in {output_dir}")
    print(f"Circuit distribution: {len([c for c in circuit_info if c['difficulty'] == 'easy'])} easy, "
          f"{len([c for c in circuit_info if c['difficulty'] == 'medium'])} medium, "
          f"{len([c for c in circuit_info if c['difficulty'] == 'hard'])} hard")
    
    return circuit_info


if __name__ == "__main__":
    # Generate the circuit dataset
    print("Generating analog circuit dataset...")
    circuit_info = generate_circuit_dataset(num_circuits=50)
    
    # Print some examples
    print("\nExample circuits generated:")
    for circuit in circuit_info[:10]:
        print(f"- {circuit['filename']}: {circuit['num_components']} components, "
              f"{circuit['num_connections']} connections, {circuit['difficulty']} difficulty")