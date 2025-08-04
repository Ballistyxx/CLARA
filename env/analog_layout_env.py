#!/usr/bin/env python3
"""
Analog-friendly layout environment with PFET/NFET row discipline, 
relational actions, advanced symmetry patterns, and analog-specific rewards.
"""

try:
    import gymnasium as gym
    from gymnasium import spaces
    USING_GYMNASIUM = True
except ImportError:
    import gym
    from gym import spaces
    USING_GYMNASIUM = False

import numpy as np
import networkx as nx
from typing import Dict, Tuple, Optional, List, Any, Union
from enum import Enum
import random

from env.layout_grid import LayoutGrid, ComponentType, RowType, ComponentPlacement


class SpatialRelation(Enum):
    """Extended spatial relations including analog patterns."""
    LEFT_OF = 0
    RIGHT_OF = 1
    ABOVE = 2
    BELOW = 3
    MIRRORED_X = 4      # Mirror about X-axis  
    MIRRORED_Y = 5      # Mirror about Y-axis
    ADJACENT = 6        # Adjacent placement
    MIRROR_ABOUT = 7    # Mirror about arbitrary axis
    COMMON_CENTROID = 8 # Common-centroid pattern
    INTERDIGITATE = 9   # Interdigitated pattern


class PatternType(Enum):
    """Symmetry pattern types."""
    NONE = 0
    MIRROR = 1
    COMMON_CENTROID = 2
    INTERDIGITATED = 3


class AnalogLayoutEnv(gym.Env):
    """
    Analog IC layout environment with row discipline and advanced placement patterns.
    """
    
    def __init__(self,
                 grid_size: int = 64,
                 max_components: int = 16,
                 pmos_rows: int = 3,
                 nmos_rows: int = 3,
                 mixed_rows: int = 2,
                 enable_action_masking: bool = True,
                 reward_weights: Optional[Dict[str, float]] = None):
        """
        Initialize analog layout environment.
        
        Args:
            grid_size: Grid dimensions (square)
            max_components: Maximum components per circuit
            pmos_rows: Number of PMOS-only rows
            nmos_rows: Number of NMOS-only rows  
            mixed_rows: Number of mixed rows for passives
            enable_action_masking: Enable invalid action masking
            reward_weights: Custom reward weights
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.max_components = max_components
        self.max_steps = max_components * 3
        self.enable_action_masking = enable_action_masking
        
        # Initialize layout grid
        self.layout_grid = LayoutGrid(
            grid_width=grid_size,
            grid_height=grid_size,
            pmos_rows=pmos_rows,
            nmos_rows=nmos_rows,
            mixed_rows=mixed_rows
        )
        
        # Action space: [target_component_or_group, relation, orientation, region]
        self.action_space = spaces.MultiDiscrete([
            max_components + 10,  # target (components + groups)
            len(SpatialRelation),  # spatial relation
            4,                     # orientation (0°, 90°, 180°, 270°)
            4                      # coarse region (optional)
        ])
        
        # Observation space
        self.observation_space = spaces.Dict({
            'component_graph': spaces.Box(
                low=0, high=1, 
                shape=(max_components, max_components), 
                dtype=np.float32
            ),
            'component_features': spaces.Box(
                low=-1, high=10,
                shape=(max_components, 8),  # Extended features
                dtype=np.float32
            ),
            'placement_state': spaces.Box(
                low=0, high=1,
                shape=(grid_size, grid_size),
                dtype=np.float32
            ),
            'placed_mask': spaces.Box(
                low=0, high=1,
                shape=(max_components,),
                dtype=np.float32
            ),
            'row_constraints': spaces.Box(
                low=0, high=3,  # Row types
                shape=(grid_size,),
                dtype=np.float32
            ),
            'locked_groups': spaces.Box(
                low=0, high=1,
                shape=(10, max_components),  # Group membership matrix
                dtype=np.float32
            )
        })
        
        # Default reward weights
        default_weights = {
            'completion': 4.0,
            'symmetry': 3.0,
            'row_consistency': 2.0,
            'pattern_validity': 2.0,
            'abutment_alignment': 1.5,
            'crossings': -1.5,
            'congestion_variance': -1.0,
            'perimeter_area_ratio': -1.0,
            'step_penalty': -0.01
        }
        self.reward_weights = reward_weights or default_weights
        
        # Environment state
        self.current_circuit = None
        self.component_to_place = 0
        self.placed_components = set()
        self.step_count = 0
        self.episode_ended = False
        
        # Pattern tracking
        self.active_patterns = {}  # pattern_id -> pattern_info
        self.next_pattern_id = 0
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Union[Dict[str, np.ndarray], Tuple[Dict[str, np.ndarray], Dict]]:
        """Reset environment to initial state."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Reset grid and state
        self.layout_grid.reset()
        self.placed_components.clear()
        self.active_patterns.clear()
        self.next_pattern_id = 0
        self.component_to_place = 0
        self.step_count = 0
        self.episode_ended = False
        
        # Set circuit (from options or generate random)
        if options and 'circuit' in options:
            self.current_circuit = options['circuit']
        else:
            self.current_circuit = self._generate_random_circuit()
        
        obs = self._get_observation()
        
        if USING_GYMNASIUM:
            return obs, {}
        else:
            return obs
    
    def step(self, action: np.ndarray) -> Union[Tuple[Dict[str, np.ndarray], float, bool, Dict], 
                                                   Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]]:
        """Execute one environment step."""
        if self.episode_ended:
            obs = self._get_observation()
            if USING_GYMNASIUM:
                return obs, 0.0, True, False, {}
            else:
                return obs, 0.0, True, {}
        
        self.step_count += 1
        
        # Parse action
        target_id, relation, orientation, region = action
        target_id = int(target_id)
        relation = SpatialRelation(int(relation))
        orientation = int(orientation) * 90  # Convert to degrees
        region = int(region)
        
        # Execute placement action
        reward, placement_successful = self._execute_placement_action(
            target_id, relation, orientation, region
        )
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.step_count >= self.max_steps
        
        if terminated or truncated:
            self.episode_ended = True
            # Add completion bonus
            if len(self.placed_components) == len(self.current_circuit.nodes):
                reward += self.reward_weights['completion']
        
        obs = self._get_observation()
        info = {
            'placement_successful': placement_successful,
            'components_placed': len(self.placed_components),
            'total_components': len(self.current_circuit.nodes),
            'step_count': self.step_count
        }
        
        if USING_GYMNASIUM:
            return obs, reward, terminated, truncated, info
        else:
            return obs, reward, terminated or truncated, info
    
    def get_action_mask(self, obs: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """
        Get action mask for invalid actions.
        Returns boolean array aligned with flattened action space.
        """
        if not self.enable_action_masking:
            return np.ones(np.prod(self.action_space.nvec), dtype=bool)
        
        if obs is None:
            obs = self._get_observation()
        
        mask = np.zeros(np.prod(self.action_space.nvec), dtype=bool)
        
        # If no components placed, only allow initial placement
        if len(self.placed_components) == 0:
            # Allow placement anywhere valid for first component
            for region in range(4):
                action_idx = self._flatten_action([0, SpatialRelation.ADJACENT.value, 0, region])
                mask[action_idx] = True
            return mask
        
        # Get next component to place
        next_comp_id = self._get_next_component_to_place()
        if next_comp_id is None:
            return mask  # All zeros - no valid actions
        
        next_comp_attrs = self.current_circuit.nodes[next_comp_id]
        next_comp_type = ComponentType(next_comp_attrs['component_type'])
        
        # Check each possible action
        for target_id in range(self.max_components + 10):
            for relation in SpatialRelation:
                for orientation in range(4):
                    for region in range(4):
                        if self._is_valid_action(
                            next_comp_id, target_id, relation, 
                            orientation * 90, region, next_comp_type
                        ):
                            action_idx = self._flatten_action([
                                target_id, relation.value, orientation, region
                            ])
                            mask[action_idx] = True
        
        return mask
    
    def _execute_placement_action(self,
                                 target_id: int,
                                 relation: SpatialRelation,
                                 orientation: int,
                                 region: int) -> Tuple[float, bool]:
        """Execute a placement action and return reward."""
        # Get next component to place
        next_comp_id = self._get_next_component_to_place()
        if next_comp_id is None:
            return self.reward_weights['step_penalty'], False
        
        next_comp_attrs = self.current_circuit.nodes[next_comp_id]
        comp_type = ComponentType(next_comp_attrs['component_type'])
        width = next_comp_attrs.get('width', 1)
        height = next_comp_attrs.get('height', 1)
        
        # Determine placement position based on relation and target
        placement_pos = self._calculate_placement_position(
            target_id, relation, next_comp_id, width, height, orientation, region
        )
        
        if placement_pos is None:
            return self.reward_weights['step_penalty'], False
        
        x, y = placement_pos
        
        # Attempt placement
        success = self.layout_grid.place_component(
            next_comp_id, x, y, width, height, orientation, comp_type
        )
        
        if success:
            self.placed_components.add(next_comp_id)
            
            # Handle pattern creation for special relations
            if relation in [SpatialRelation.MIRROR_ABOUT, SpatialRelation.COMMON_CENTROID, 
                          SpatialRelation.INTERDIGITATE]:
                self._create_pattern(next_comp_id, target_id, relation)
            
            # Calculate reward
            reward = self._calculate_placement_reward()
            return reward, True
        else:
            return self.reward_weights['step_penalty'], False
    
    def _calculate_placement_position(self,
                                    target_id: int,
                                    relation: SpatialRelation,
                                    comp_id: int,
                                    width: int, height: int,
                                    orientation: int,
                                    region: int) -> Optional[Tuple[int, int]]:
        """Calculate placement position based on relational action."""
        # Adjust dimensions for orientation
        if orientation in [90, 270]:
            width, height = height, width
        
        # Handle initial placement (no target)
        if len(self.placed_components) == 0 or target_id not in self.placed_components:
            return self._place_in_region(region, width, height, 
                                       ComponentType(self.current_circuit.nodes[comp_id]['component_type']))
        
        # Get target placement
        target_placement = self.layout_grid.placements[target_id]
        target_x, target_y = target_placement.x, target_placement.y
        target_w, target_h = target_placement.width, target_placement.height
        
        # Calculate position based on relation
        if relation == SpatialRelation.LEFT_OF:
            x = target_x - width - 1
            y = target_y
        elif relation == SpatialRelation.RIGHT_OF:
            x = target_x + target_w + 1
            y = target_y
        elif relation == SpatialRelation.ABOVE:
            x = target_x
            y = target_y - height - 1
        elif relation == SpatialRelation.BELOW:
            x = target_x
            y = target_y + target_h + 1
        elif relation == SpatialRelation.ADJACENT:
            # Find adjacent empty space
            candidates = [
                (target_x - width, target_y),
                (target_x + target_w, target_y),
                (target_x, target_y - height),
                (target_x, target_y + target_h)
            ]
            for cx, cy in candidates:
                if (0 <= cx < self.grid_size - width and 
                    0 <= cy < self.grid_size - height):
                    comp_type = ComponentType(self.current_circuit.nodes[comp_id]['component_type'])
                    valid, _ = self.layout_grid.is_valid_placement(comp_id, cx, cy, width, height, comp_type)
                    if valid:
                        return cx, cy
            return None
        elif relation == SpatialRelation.MIRRORED_X:
            # Mirror across X-axis (horizontal flip)
            grid_center_y = self.grid_size // 2
            y = 2 * grid_center_y - target_y - target_h
            x = target_x
        elif relation == SpatialRelation.MIRRORED_Y:
            # Mirror across Y-axis (vertical flip)
            grid_center_x = self.grid_size // 2
            x = 2 * grid_center_x - target_x - target_w
            y = target_y
        else:
            # Default to right placement
            x = target_x + target_w + 1
            y = target_y
        
        # Bounds check
        if (0 <= x < self.grid_size - width and 
            0 <= y < self.grid_size - height):
            return x, y
        
        return None
    
    def _place_in_region(self, region: int, width: int, height: int, comp_type: ComponentType) -> Optional[Tuple[int, int]]:
        """Place component in specified coarse region."""
        region_size = self.grid_size // 2
        
        if region == 0:  # Top-left
            start_x, start_y = 0, 0
        elif region == 1:  # Top-right
            start_x, start_y = region_size, 0
        elif region == 2:  # Bottom-left
            start_x, start_y = 0, region_size
        else:  # Bottom-right
            start_x, start_y = region_size, region_size
        
        # Try positions in region
        for y in range(start_y, min(start_y + region_size, self.grid_size - height)):
            for x in range(start_x, min(start_x + region_size, self.grid_size - width)):
                valid, _ = self.layout_grid.is_valid_placement(-1, x, y, width, height, comp_type)
                if valid:
                    return x, y
        
        return None
    
    def _calculate_placement_reward(self) -> float:
        """Calculate reward for current placement state."""
        if not self.placed_components:
            return 0.0
        
        reward = 0.0
        
        # Row consistency reward
        row_consistency = self.layout_grid.get_row_consistency_score()
        reward += self.reward_weights['row_consistency'] * row_consistency
        
        # Symmetry reward (for matched components)
        symmetry_score = self._calculate_symmetry_score()
        reward += self.reward_weights['symmetry'] * symmetry_score
        
        # Pattern validity reward (for locked groups)
        pattern_validity = self._calculate_pattern_validity_score()
        reward += self.reward_weights['pattern_validity'] * pattern_validity
        
        # Crossing penalty
        connections = list(self.current_circuit.edges())
        crossings = self.layout_grid.count_crossings(connections)
        max_crossings = len(connections) * (len(connections) - 1) // 2
        crossing_penalty = crossings / max(max_crossings, 1) if max_crossings > 0 else 0
        reward += self.reward_weights['crossings'] * crossing_penalty
        
        # Congestion variance penalty
        congestion_var = self.layout_grid.get_congestion_variance(connections)
        normalized_congestion = min(congestion_var / 10.0, 1.0)  # Normalize
        reward += self.reward_weights['congestion_variance'] * normalized_congestion
        
        # Perimeter/area ratio penalty (encourage rectangular layouts)
        perimeter_ratio = self.layout_grid.get_perimeter_area_ratio()
        normalized_ratio = min(perimeter_ratio / 2.0, 1.0)  # Normalize to [0,1]
        reward += self.reward_weights['perimeter_area_ratio'] * normalized_ratio
        
        # Step penalty
        reward += self.reward_weights['step_penalty']
        
        return reward
    
    def _calculate_symmetry_score(self) -> float:
        """Calculate symmetry score for matched components."""
        if not self.current_circuit:
            return 0.0
        
        total_pairs = 0
        symmetric_pairs = 0
        
        for node_id in self.current_circuit.nodes():
            if node_id not in self.placed_components:
                continue
                
            matched_comp = self.current_circuit.nodes[node_id].get('matched_component', -1)
            if matched_comp != -1 and matched_comp in self.placed_components and node_id < matched_comp:
                total_pairs += 1
                
                # Check if pair is placed symmetrically
                if self._check_pair_symmetry(node_id, matched_comp):
                    symmetric_pairs += 1
        
        return symmetric_pairs / max(total_pairs, 1)
    
    def _check_pair_symmetry(self, comp1_id: int, comp2_id: int) -> bool:
        """Check if two components are placed symmetrically."""
        if comp1_id not in self.layout_grid.placements or comp2_id not in self.layout_grid.placements:
            return False
        
        p1 = self.layout_grid.placements[comp1_id]
        p2 = self.layout_grid.placements[comp2_id]
        
        # Simple symmetry checks
        # Horizontal symmetry (same Y, different X)
        if abs(p1.y - p2.y) <= 1 and abs(p1.x - p2.x) > 2:
            return True
        
        # Vertical symmetry (same X, different Y)  
        if abs(p1.x - p2.x) <= 1 and abs(p1.y - p2.y) > 2:
            return True
        
        return False
    
    def _calculate_pattern_validity_score(self) -> float:
        """Calculate score for pattern validity (mirrors, common-centroid, etc.)."""
        if not self.active_patterns:
            return 1.0
        
        valid_patterns = 0
        total_patterns = len(self.active_patterns)
        
        for pattern_id, pattern_info in self.active_patterns.items():
            if self._validate_pattern(pattern_info):
                valid_patterns += 1
        
        return valid_patterns / max(total_patterns, 1)
    
    def _validate_pattern(self, pattern_info: Dict[str, Any]) -> bool:
        """Validate that a pattern is still preserved."""
        # TODO: Implement specific pattern validation
        # For now, return True (patterns are considered valid)
        return True
    
    def _create_pattern(self, comp_id: int, target_id: int, relation: SpatialRelation):
        """Create a new pattern group."""
        pattern_id = self.next_pattern_id
        self.next_pattern_id += 1
        
        if relation == SpatialRelation.MIRROR_ABOUT:
            pattern_type = PatternType.MIRROR
        elif relation == SpatialRelation.COMMON_CENTROID:
            pattern_type = PatternType.COMMON_CENTROID
        elif relation == SpatialRelation.INTERDIGITATE:
            pattern_type = PatternType.INTERDIGITATED
        else:
            pattern_type = PatternType.NONE
        
        pattern_data = {
            'components': [comp_id, target_id],
            'relation': relation,
            'created_step': self.step_count
        }
        
        self.active_patterns[pattern_id] = {
            'pattern_type': pattern_type,
            'pattern_data': pattern_data
        }
        
        # Create locked group in layout grid
        self.layout_grid.create_locked_group(
            pattern_id, [comp_id, target_id], 
            pattern_type.name.lower(), pattern_data
        )
    
    def _get_next_component_to_place(self) -> Optional[int]:
        """Get the next component ID to place."""
        for node_id in self.current_circuit.nodes():
            if node_id not in self.placed_components:
                return node_id
        return None
    
    def _is_valid_action(self,
                        comp_id: int,
                        target_id: int, 
                        relation: SpatialRelation,
                        orientation: int,
                        region: int,
                        comp_type: ComponentType) -> bool:
        """Check if action is valid."""
        # If no components placed, only allow regional placement
        if len(self.placed_components) == 0:
            return relation == SpatialRelation.ADJACENT
        
        # Check if target exists and is placed
        if target_id >= len(self.current_circuit.nodes) or target_id not in self.placed_components:
            return False
        
        # Check component type constraints (row discipline)
        comp_attrs = self.current_circuit.nodes[comp_id]
        width = comp_attrs.get('width', 1)
        height = comp_attrs.get('height', 1)
        
        # Test placement position
        pos = self._calculate_placement_position(target_id, relation, comp_id, width, height, orientation, region)
        if pos is None:
            return False
        
        x, y = pos
        if orientation in [90, 270]:
            width, height = height, width
        
        valid, _ = self.layout_grid.is_valid_placement(comp_id, x, y, width, height, comp_type)
        return valid
    
    def _flatten_action(self, action: List[int]) -> int:
        """Convert multi-discrete action to flat index."""
        flat_idx = 0
        multiplier = 1
        
        for i in reversed(range(len(action))):
            flat_idx += action[i] * multiplier
            multiplier *= self.action_space.nvec[i]
        
        return flat_idx
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if all components are placed
        return len(self.placed_components) == len(self.current_circuit.nodes)
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        if self.current_circuit is None:
            # Return zero observation
            return {
                'component_graph': np.zeros((self.max_components, self.max_components), dtype=np.float32),
                'component_features': np.zeros((self.max_components, 8), dtype=np.float32),
                'placement_state': np.zeros((self.grid_size, self.grid_size), dtype=np.float32),
                'placed_mask': np.zeros((self.max_components,), dtype=np.float32),
                'row_constraints': np.zeros((self.grid_size,), dtype=np.float32),
                'locked_groups': np.zeros((10, self.max_components), dtype=np.float32)
            }
        
        # Component adjacency matrix
        adj_matrix = np.zeros((self.max_components, self.max_components), dtype=np.float32)
        for i, j in self.current_circuit.edges():
            if i < self.max_components and j < self.max_components:
                adj_matrix[i, j] = 1.0
                adj_matrix[j, i] = 1.0
        
        # Component features
        features = np.zeros((self.max_components, 8), dtype=np.float32)
        for i, (node_id, attrs) in enumerate(self.current_circuit.nodes(data=True)):
            if i < self.max_components:
                features[i] = [
                    attrs.get('component_type', 0),
                    attrs.get('width', 1),
                    attrs.get('height', 1),
                    attrs.get('matched_component', -1),
                    1.0 if node_id in self.placed_components else 0.0,
                    attrs.get('device_model_hash', 0) % 100 / 100.0,  # Normalized hash
                    attrs.get('length', 0),
                    attrs.get('width_param', 0)
                ]
        
        # Placement state (component occupancy grid)
        placement_state = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                comp_id = self.layout_grid.grid[y, x]
                if comp_id > 0:
                    placement_state[y, x] = comp_id / self.max_components
        
        # Placed mask
        placed_mask = np.zeros((self.max_components,), dtype=np.float32)
        for comp_id in self.placed_components:
            if comp_id < self.max_components:
                placed_mask[comp_id] = 1.0
        
        # Row constraints
        row_constraints = np.zeros((self.grid_size,), dtype=np.float32)
        for y in range(self.grid_size):
            row = self.layout_grid.get_row_for_position(y)
            if row:
                row_constraints[y] = row.row_type.value
        
        # Locked groups matrix
        locked_groups = np.zeros((10, self.max_components), dtype=np.float32)
        for group_id, group_info in self.layout_grid.locked_groups.items():
            if group_id < 10:
                for comp_id in group_info['component_ids']:
                    if comp_id < self.max_components:
                        locked_groups[group_id, comp_id] = 1.0
        
        return {
            'component_graph': adj_matrix,
            'component_features': features,
            'placement_state': placement_state,
            'placed_mask': placed_mask,
            'row_constraints': row_constraints,
            'locked_groups': locked_groups
        }
    
    def _generate_random_circuit(self) -> nx.Graph:
        """Generate a random test circuit."""
        num_components = random.randint(3, min(self.max_components, 8))
        circuit = nx.Graph()
        
        # Add components with random types and attributes
        for i in range(num_components):
            comp_type = random.choice([ComponentType.NMOS, ComponentType.PMOS, ComponentType.RESISTOR])
            
            circuit.add_node(i, 
                component_type=comp_type.value,
                width=random.randint(1, 3),
                height=random.randint(1, 3),
                matched_component=-1,
                device_model_hash=hash(f"device_{comp_type.name}_{i}") % 1000
            )
        
        # Add random connections
        num_edges = random.randint(num_components - 1, num_components + 2)
        for _ in range(num_edges):
            i, j = random.sample(range(num_components), 2)
            circuit.add_edge(i, j)
        
        return circuit
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment (placeholder)."""
        if mode == 'human':
            print(f"Step {self.step_count}: {len(self.placed_components)}/{len(self.current_circuit.nodes)} components placed")
            return None
        elif mode == 'rgb_array':
            # Return a simple visualization as RGB array
            img = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    comp_id = self.layout_grid.grid[y, x]
                    if comp_id > 0:
                        img[y, x] = [255, 0, 0]  # Red for occupied
            return img
        
        return None