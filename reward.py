import numpy as np
import networkx as nx
from typing import Dict, Tuple, List
from dataclasses import dataclass
import yaml
import os
from pathlib import Path


def load_reward_config(config_path: str = "configs/rewards.yaml") -> Dict[str, any]:
    """Load reward configuration from YAML file. Raises error if file not found or invalid."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Reward config file {config_path} not found. "
                              f"Please ensure {config_path} exists in the project root.")
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            raise ValueError(f"Reward config file {config_path} is empty or invalid YAML")
        
        # Validate required sections
        required_sections = ['base_weights']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Required section '{section}' not found in {config_path}")
        
        print(f"Successfully loaded reward configuration from {config_path}")
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {config_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading reward config: {e}")


@dataclass
class RewardComponents:
    """Container for different reward components."""
    # Positive rewards
    symmetry: float = 0.0
    compactness: float = 0.0
    connectivity: float = 0.0
    completion: float = 0.0
    placement_step: float = 0.0
    device_grouping: float = 0.0
    
    # Graduated penalties (replace binary invalid_action/invalid_placement)
    overlap_severity: float = 0.0      # Proportional to overlap area
    bounds_violation: float = 0.0      # Proportional to out-of-bounds distance
    spacing_violation: float = 0.0     # Proportional to spacing violations
    
    # Learning signals
    near_miss_bonus: float = 0.0       # Reward for almost-valid placements


class RewardCalculator:
    """Modular reward calculation for analog layout placement."""
    
    def __init__(self, weights: Dict[str, float] = None, config_path: str = None):
        """Initialize reward calculator with component weights from YAML config."""
        
        # Load from YAML if no weights provided
        if weights is None:
            config = load_reward_config(config_path or "configs/rewards.yaml")
            weights = config['base_weights']
        
        self.weights = weights
        print(f"RewardCalculator initialized with weights: {self.weights}")
    
    def calculate_total_reward(self, 
                             circuit: nx.Graph,
                             component_positions: Dict[int, Tuple[int, int, int]],
                             grid_size: int,
                             num_components: int,
                             placed_count: int,
                             attempted_position: Tuple[int, int, int] = None,
                             is_valid_action: bool = True,
                             is_valid_placement: bool = True) -> Tuple[float, RewardComponents]:
        """Calculate total reward with graduated penalties."""
        
        components = RewardComponents()
        
        # Always calculate positive rewards (encourage good aspects even with violations)
        components.placement_step = self.weights.get('placement_step', 0.0)
        components.symmetry = self._calculate_symmetry_reward(circuit, component_positions)
        components.compactness = self._calculate_compactness_reward(component_positions, grid_size)
        components.connectivity = self._calculate_connectivity_reward(circuit, component_positions)
        components.device_grouping = self._calculate_device_grouping_reward(circuit, component_positions)
        
        # Calculate graduated penalties
        components.overlap_severity = self._calculate_overlap_penalty(component_positions, circuit)
        components.bounds_violation = self._calculate_bounds_violation_penalty(
            component_positions, circuit, grid_size)
        components.spacing_violation = self._calculate_spacing_violation_penalty(component_positions)
        
        # Learning signals
        if attempted_position:
            components.near_miss_bonus = self._calculate_near_miss_bonus(
                attempted_position, component_positions, circuit, grid_size)
        
        # Completion bonus
        if placed_count == num_components:
            components.completion = self.weights.get('completion', 0.0)
        
        # Calculate weighted total
        total_reward = sum([getattr(components, attr) * self.weights.get(attr, 0.0) 
                           for attr in components.__dict__.keys()])
        
        # Enhanced diagnostic printing
        print(f"\n=== GRADUATED REWARD CALCULATION ===")
        print(f"Positive Rewards:")
        for attr in ['symmetry', 'compactness', 'connectivity', 'device_grouping', 'placement_step', 'completion']:
            raw_score = getattr(components, attr)
            weight = self.weights.get(attr, 0.0)
            weighted = raw_score * weight
            if raw_score != 0.0:
                print(f"  {attr:15s}: {raw_score:8.3f} * {weight:6.2f} = {weighted:8.3f}")
        
        print(f"Constraint Penalties:")
        for attr in ['overlap_severity', 'bounds_violation', 'spacing_violation']:
            raw_score = getattr(components, attr)
            weight = self.weights.get(attr, 0.0)
            weighted = raw_score * weight
            if raw_score != 0.0:
                print(f"  {attr:15s}: {raw_score:8.3f} * {weight:6.2f} = {weighted:8.3f}")
        
        print(f"Learning Signals:")
        if components.near_miss_bonus != 0.0:
            weight = self.weights.get('near_miss_bonus', 0.0)
            weighted = components.near_miss_bonus * weight
            print(f"  near_miss_bonus  : {components.near_miss_bonus:8.3f} * {weight:6.2f} = {weighted:8.3f}")
        
        print(f"FINAL TOTAL REWARD: {total_reward:.3f}")
        print(f"=== END GRADUATED REWARD DEBUG ===\n")
        
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
    
    def _calculate_device_grouping_reward(self, 
                                        circuit: nx.Graph, 
                                        component_positions: Dict[int, Tuple[int, int, int]]) -> float:
        """
        Calculate device grouping reward based on proximity of components with same device model.
        Components with the same device_model (e.g., 'sky130_fd_pr__pfet_g5v0d10v5') 
        should be placed close to each other for better analog layout practices.
        """
        if len(component_positions) < 2:
            return 0.0
        
        # Group components by device model
        device_groups = {}
        for node_id in component_positions.keys():
            if node_id in circuit.nodes():
                device_model = circuit.nodes[node_id].get('device_model', '')
                if device_model:  # Only consider components with device models
                    if device_model not in device_groups:
                        device_groups[device_model] = []
                    device_groups[device_model].append(node_id)
        
        total_grouping_score = 0.0
        total_comparisons = 0
        
        # Calculate grouping score for each device model
        for device_model, component_ids in device_groups.items():
            if len(component_ids) < 2:
                continue  # Skip groups with only one component
            
            # Calculate average distance within this device group
            group_distance_sum = 0.0
            group_comparisons = 0
            
            for i, comp1 in enumerate(component_ids):
                for comp2 in component_ids[i+1:]:
                    pos1 = component_positions[comp1]
                    pos2 = component_positions[comp2]
                    
                    # Manhattan distance between components of same device model
                    distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                    
                    # Higher reward for closer components (inverse distance)
                    # Use exponential decay to strongly favor very close placement
                    distance_reward = max(0, 2.0 * np.exp(-distance * 0.3))
                    
                    # Extra bonus for adjacent placement
                    if distance == 1:
                        distance_reward += 1.0
                    # Bonus for being in same row/column
                    elif distance == abs(pos1[0] - pos2[0]) or distance == abs(pos1[1] - pos2[1]):
                        distance_reward += 0.5
                    
                    group_distance_sum += distance_reward
                    group_comparisons += 1
            
            if group_comparisons > 0:
                # Average grouping score for this device model
                avg_group_score = group_distance_sum / group_comparisons
                
                # Weight by group size (larger groups get more importance)
                group_weight = min(3.0, len(component_ids) / 2.0)  # Cap at 3x weight
                
                total_grouping_score += avg_group_score * group_weight
                total_comparisons += group_comparisons
        
        # Normalize by total number of comparisons
        if total_comparisons > 0:
            return total_grouping_score / len(device_groups)  # Average per device type
        
        return 0.0
    
    def _get_component_dimensions(self, comp_id: int, circuit: nx.Graph, orientation: int) -> Tuple[int, int]:
        """Get component width and height based on orientation."""
        if comp_id not in circuit.nodes():
            return (1, 1)  # Default size
        
        node_attrs = circuit.nodes[comp_id]
        base_width = int(node_attrs.get('width', 1))
        base_height = int(node_attrs.get('height', 1))
        
        # Swap dimensions for rotated orientations
        if orientation in [1, 3]:  # 90° or 270° rotation
            return (base_height, base_width)
        return (base_width, base_height)
    
    def _calculate_rectangle_overlap(self, rect1: Tuple[int, int, int, int], 
                                   rect2: Tuple[int, int, int, int]) -> float:
        """Calculate overlap area between two rectangles."""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        # Calculate intersection
        left = max(x1, x2)
        right = min(x1 + w1, x2 + w2)
        top = max(y1, y2)
        bottom = min(y1 + h1, y2 + h2)
        
        if left < right and top < bottom:
            return (right - left) * (bottom - top)
        return 0.0
    
    def _calculate_overlap_penalty(self, 
                                  component_positions: Dict[int, Tuple[int, int, int]],
                                  circuit: nx.Graph) -> float:
        """Calculate penalty based on component overlap severity."""
        if len(component_positions) < 2:
            return 0.0
        
        total_overlap_area = 0.0
        components = list(component_positions.items())
        
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                comp1_id, (x1, y1, o1) = components[i]
                comp2_id, (x2, y2, o2) = components[j]
                
                w1, h1 = self._get_component_dimensions(comp1_id, circuit, o1)
                w2, h2 = self._get_component_dimensions(comp2_id, circuit, o2)
                
                overlap_area = self._calculate_rectangle_overlap(
                    (x1, y1, w1, h1), (x2, y2, w2, h2)
                )
                total_overlap_area += overlap_area
        
        return total_overlap_area
    
    def _calculate_bounds_violation_penalty(self,
                                           component_positions: Dict[int, Tuple[int, int, int]],
                                           circuit: nx.Graph,
                                           grid_size: int) -> float:
        """Calculate penalty for components placed out of bounds."""
        total_violation = 0.0
        
        for comp_id, (x, y, orientation) in component_positions.items():
            w, h = self._get_component_dimensions(comp_id, circuit, orientation)
            
            # Calculate how far out of bounds (if any)
            violation_x = max(0, (x + w) - grid_size) + max(0, -x)
            violation_y = max(0, (y + h) - grid_size) + max(0, -y)
            
            total_violation += violation_x + violation_y
        
        return total_violation
    
    def _calculate_spacing_violation_penalty(self,
                                           component_positions: Dict[int, Tuple[int, int, int]]) -> float:
        """Calculate penalty for components placed too close together."""
        if len(component_positions) < 2:
            return 0.0
        
        violation_count = 0.0
        components = list(component_positions.items())
        
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                comp1_id, (x1, y1, o1) = components[i]
                comp2_id, (x2, y2, o2) = components[j]
                
                # Manhattan distance between component centers
                distance = abs(x1 - x2) + abs(y1 - y2)
                
                # Violation if too close (less than minimum spacing)
                min_spacing = 1  # Minimum 1 grid unit spacing
                if distance < min_spacing:
                    violation_count += (min_spacing - distance)
        
        return violation_count
    
    def _calculate_near_miss_bonus(self,
                                  attempted_position: Tuple[int, int, int],
                                  component_positions: Dict[int, Tuple[int, int, int]],
                                  circuit: nx.Graph,
                                  grid_size: int) -> float:
        """Reward attempts that are close to being valid."""
        if attempted_position is None:
            return 0.0
            
        x, y, orientation = attempted_position
        
        # Check if small adjustments would reduce violations
        current_violations = (
            self._calculate_overlap_penalty({-1: attempted_position}, circuit) +
            self._calculate_bounds_violation_penalty({-1: attempted_position}, circuit, grid_size)
        )
        
        # If current position has violations, check if nearby positions are better
        if current_violations > 0:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    test_pos = (x + dx, y + dy, orientation)
                    test_violations = (
                        self._calculate_overlap_penalty({-1: test_pos}, circuit) +
                        self._calculate_bounds_violation_penalty({-1: test_pos}, circuit, grid_size)
                    )
                    
                    if test_violations < current_violations:
                        return 1.0  # Near miss found
        
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
    
    def __init__(self, initial_weights: Dict[str, float] = None, config_path: str = None):
        # Load curriculum config from YAML
        self.config = load_reward_config(config_path or "configs/rewards.yaml")
        self.curriculum_config = self.config.get('curriculum_config', {})
        self.curriculum_weights = self.config.get('curriculum_weights', {})
        
        # Use base_weights as initial weights if none provided
        if initial_weights is None:
            initial_weights = self.config['base_weights']
        
        super().__init__(initial_weights, config_path)
        self.episode_count = 0
        self.initial_weights = self.weights.copy()
        
        # Get curriculum episodes from config
        self.full_curriculum_episodes = self.curriculum_config.get('full_curriculum_episodes', 1000)
        
        print(f"AdaptiveRewardCalculator initialized:")
        print(f"  Full curriculum episodes: {self.full_curriculum_episodes}")
        print(f"  Curriculum stages: {list(self.curriculum_weights.keys())}")
    
    def update_weights_for_episode(self, episode: int):
        """Update reward weights based on training progress using YAML configuration."""
        self.episode_count = episode
        
        # Calculate progress based on config
        progress = min(1.0, episode / self.full_curriculum_episodes)
        
        # Determine which stage weights to use
        if progress < 0.3 and 'early_training' in self.curriculum_weights:
            stage_weights = self.curriculum_weights['early_training']
        elif progress < 0.7 and 'mid_training' in self.curriculum_weights:
            stage_weights = self.curriculum_weights['mid_training']
        elif 'late_training' in self.curriculum_weights:
            stage_weights = self.curriculum_weights['late_training']
        else:
            raise ValueError("Missing curriculum weight stages in YAML configuration. "
                           "Expected 'early_training', 'mid_training', and 'late_training' sections.")
        
        # Update weights from YAML configuration
        for key, value in stage_weights.items():
            if key in self.weights:
                self.weights[key] = value
            else:
                print(f"Warning: Curriculum weight '{key}' not found in base weights, skipping")
    
    def get_reward_summary(self) -> str:
        """Get a summary of current reward weights."""
        summary = f"Episode {self.episode_count} Reward Weights:\n"
        for component, weight in self.weights.items():
            summary += f"  {component}: {weight:.2f}\n"
        return summary