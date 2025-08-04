import numpy as np
import networkx as nx
from typing import Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class RewardComponents:
    """Container for different reward components."""
    symmetry: float = 0.0
    compactness: float = 0.0
    connectivity: float = 0.0
    completion: float = 0.0
    placement_step: float = 0.0
    invalid_action: float = 0.0
    invalid_placement: float = 0.0


class RewardCalculator:
    """Modular reward calculation for analog layout placement."""
    
    def __init__(self, weights: Dict[str, float] = None):
        """Initialize reward calculator with component weights."""
        self.weights = weights or {
            'symmetry': 0.2,
            'compactness': 0.45,
            'connectivity': 0.0,
            'completion': 1.0,
            'placement_step': 0.1,
            'invalid_action': -1.0,
            'invalid_placement': -0.5
        }
    
    def calculate_total_reward(self, 
                             circuit: nx.Graph,
                             component_positions: Dict[int, Tuple[int, int, int]],
                             grid_size: int,
                             num_components: int,
                             placed_count: int,
                             is_valid_action: bool = True,
                             is_valid_placement: bool = True) -> Tuple[float, RewardComponents]:
        """Calculate total reward and return breakdown."""
        
        components = RewardComponents()
        
        # Penalty for invalid actions
        if not is_valid_action:
            components.invalid_action = self.weights['invalid_action']
            return sum([getattr(components, attr) * self.weights.get(attr, 1.0) 
                       for attr in components.__dict__.keys()]), components
        
        if not is_valid_placement:
            components.invalid_placement = self.weights['invalid_placement']
            return sum([getattr(components, attr) * self.weights.get(attr, 1.0) 
                       for attr in components.__dict__.keys()]), components
        
        # Positive rewards
        components.placement_step = self.weights['placement_step']
        components.symmetry = self._calculate_symmetry_reward(circuit, component_positions)
        components.compactness = self._calculate_compactness_reward(component_positions, grid_size)
        components.connectivity = self._calculate_connectivity_reward(circuit, component_positions)
        
        # Completion bonus
        if placed_count == num_components:
            components.completion = self.weights['completion']
        
        # Calculate weighted total
        total_reward = sum([getattr(components, attr) * self.weights.get(attr, 1.0) 
                           for attr in components.__dict__.keys()])
        
        return total_reward, components
    
    def _calculate_symmetry_reward(self, 
                                 circuit: nx.Graph, 
                                 component_positions: Dict[int, Tuple[int, int, int]]) -> float:
        """
        Calculate symmetry reward based on matched component placement.
        Rewards symmetric placement of matched pairs.
        """
        if len(component_positions) < 2:
            return 0.0
        
        symmetry_score = 0.0
        matched_pairs_found = 0
        
        for node_id in circuit.nodes():
            node_attrs = circuit.nodes[node_id]
            matched_comp = node_attrs.get('matched_component', -1)
            
            if (matched_comp != -1 and 
                node_id in component_positions and 
                matched_comp in component_positions):
                
                pos1 = component_positions[node_id]
                pos2 = component_positions[matched_comp]
                
                # Check for various symmetry patterns
                x1, y1, orient1 = pos1
                x2, y2, orient2 = pos2
                
                matched_pairs_found += 1
                
                # Horizontal symmetry (mirrored across vertical axis)
                if abs(y1 - y2) <= 1:  # Same or adjacent y-coordinates
                    if abs(x1 + x2) <= 2:  # Symmetric about center or close
                        symmetry_score += 5.0
                        # Bonus for matching orientations in symmetric pairs
                        if (orient1 + orient2) % 4 == 0 or abs(orient1 - orient2) == 2:
                            symmetry_score += 2.0
                
                # Vertical symmetry (mirrored across horizontal axis)
                elif abs(x1 - x2) <= 1:  # Same or adjacent x-coordinates
                    if abs(y1 + y2) <= 2:  # Symmetric about center or close
                        symmetry_score += 5.0
                        if (orient1 + orient2) % 4 == 0 or abs(orient1 - orient2) == 2:
                            symmetry_score += 2.0
                
                # Diagonal symmetry
                elif abs(abs(x1 - x2) - abs(y1 - y2)) <= 1:
                    symmetry_score += 3.0
        
        # Normalize by number of matched pairs to avoid unfair advantage
        if matched_pairs_found > 0:
            return symmetry_score / matched_pairs_found
        
        return 0.0
    
    def _calculate_compactness_reward(self, 
                                    component_positions: Dict[int, Tuple[int, int, int]], 
                                    grid_size: int) -> float:
        """
        Calculate compactness reward based on bounding box of placed components.
        Smaller bounding box = higher reward.
        """
        if len(component_positions) <= 1:
            return 0.0
        
        positions = list(component_positions.values())
        xs = [pos[0] for pos in positions]
        ys = [pos[1] for pos in positions]
        
        bbox_width = max(xs) - min(xs) + 1
        bbox_height = max(ys) - min(ys) + 1
        bbox_area = bbox_width * bbox_height
        
        max_area = grid_size * grid_size
        
        # Compactness score: higher for smaller bounding boxes
        compactness = 1.0 - (bbox_area / max_area)
        
        # Additional bonus for square-like layouts (aspect ratio close to 1)
        aspect_ratio = max(bbox_width, bbox_height) / min(bbox_width, bbox_height)
        aspect_bonus = max(0, 2.0 - aspect_ratio) * 0.5
        
        return max(0, compactness * 1.0 + aspect_bonus)
    
    def _calculate_connectivity_reward(self, 
                                     circuit: nx.Graph, 
                                     component_positions: Dict[int, Tuple[int, int, int]]) -> float:
        """
        Calculate connectivity reward based on proximity of connected components.
        Connected components should be placed close to each other.
        """
        if len(component_positions) < 2:
            return 0.0
        
        connectivity_score = 0.0
        total_edges = 0
        
        for edge in circuit.edges():
            comp1, comp2 = edge
            
            if comp1 in component_positions and comp2 in component_positions:
                pos1 = component_positions[comp1]
                pos2 = component_positions[comp2]
                
                # Manhattan distance between components
                distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                
                # Higher reward for closer components
                edge_reward = max(0, 1.0 - distance * 0.1)  # Max reward when distance = 0
                
                # Additional bonus for adjacent placement
                if distance == 1:
                    edge_reward += 0.5
                
                connectivity_score += edge_reward
                total_edges += 1
        
        # Normalize by number of edges
        if total_edges > 0:
            return connectivity_score / total_edges
        
        return 0.0
    
    def _calculate_routing_efficiency_reward(self, 
                                           circuit: nx.Graph, 
                                           component_positions: Dict[int, Tuple[int, int, int]]) -> float:
        """
        Calculate routing efficiency reward (simplified HPWL approximation).
        Note: This is kept simple as routing is not the main focus.
        """
        if len(component_positions) < 2:
            return 0.0
        
        total_wirelength = 0.0
        total_nets = 0
        
        # For each net (edge), calculate the minimum bounding box
        for edge in circuit.edges():
            comp1, comp2 = edge
            
            if comp1 in component_positions and comp2 in component_positions:
                pos1 = component_positions[comp1]
                pos2 = component_positions[comp2]
                
                # Half-perimeter wirelength for this net
                hpwl = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                total_wirelength += hpwl
                total_nets += 1
        
        if total_nets > 0:
            avg_wirelength = total_wirelength / total_nets
            # Return inverse of average wirelength (lower is better)
            return max(0, 20.0 - avg_wirelength)
        
        return 0.0


class AdaptiveRewardCalculator(RewardCalculator):
    """Adaptive reward calculator that adjusts weights during training."""
    
    def __init__(self, initial_weights: Dict[str, float] = None):
        super().__init__(initial_weights)
        self.episode_count = 0
        self.initial_weights = self.weights.copy()
    
    def update_weights_for_episode(self, episode: int):
        """Update reward weights based on training progress."""
        self.episode_count = episode
        
        # Curriculum learning: gradually increase importance of complex rewards
        progress = min(1.0, episode / 1000.0)  # Full curriculum after 1000 episodes
        
        # Early training: focus on basic placement and compactness
        if progress < 0.3:
            self.weights['placement_step'] = 2.0
            self.weights['compactness'] = 2.0
            self.weights['connectivity'] = 0.5
            self.weights['symmetry'] = 0.5
        
        # Mid training: balance all components
        elif progress < 0.7:
            self.weights['placement_step'] = 1.0
            self.weights['compactness'] = 1.5
            self.weights['connectivity'] = 1.0
            self.weights['symmetry'] = 1.5
        
        # Late training: emphasize quality metrics
        else:
            self.weights['placement_step'] = 0.5
            self.weights['compactness'] = 1.0
            self.weights['connectivity'] = 2.0
            self.weights['symmetry'] = 3.0
    
    def get_reward_summary(self) -> str:
        """Get a summary of current reward weights."""
        summary = f"Episode {self.episode_count} Reward Weights:\n"
        for component, weight in self.weights.items():
            summary += f"  {component}: {weight:.2f}\n"
        return summary