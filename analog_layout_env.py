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
from typing import Dict, Tuple, Optional, List, Any
from enum import Enum


class SpatialRelation(Enum):
    LEFT_OF = 0
    RIGHT_OF = 1
    ABOVE = 2
    BELOW = 3
    MIRRORED_X = 4
    MIRRORED_Y = 5
    ADJACENT = 6


class ComponentType(Enum):
    MOSFET = 0
    RESISTOR = 1
    CAPACITOR = 2
    INDUCTOR = 3


class AnalogLayoutEnv(gym.Env):
    """
    Custom Gym environment for analog IC component placement using relational actions.
    """
    
    def __init__(self, grid_size: int = 64, max_components: int = 10):
        super().__init__()
        
        self.grid_size = grid_size
        self.max_components = max_components
        self.max_steps = max_components * 3  # Limit episode length
        
        # Action space: [target_component_id, spatial_relation_type, orientation]
        self.action_space = spaces.MultiDiscrete([
            max_components,  # target component (reference for placement)
            len(SpatialRelation),  # spatial relation type
            4  # orientation (0°, 90°, 180°, 270°)
        ])
        
        # Observation space
        self.observation_space = spaces.Dict({
            "component_graph": spaces.Box(
                low=0, high=1, 
                shape=(max_components, max_components), 
                dtype=np.float32
            ),
            "placed_components_list": spaces.Box(
                low=-1.0, high=1.0,  # Normalized coordinates + -1 for unplaced
                shape=(max_components, 4),  # [norm_x, norm_y, norm_orientation, valid]
                dtype=np.float32
            ),
            "netlist_features": spaces.Box(
                low=0, high=10,
                shape=(max_components, 4),  # [component_type, width, height, matched_component]
                dtype=np.float32
            ),
            "placement_mask": spaces.Box(
                low=0, high=1,
                shape=(max_components,),
                dtype=np.int8
            )
        })
        
        # Environment state
        self.reset()
    
    def reset(self, circuit_graph: Optional[nx.Graph] = None, seed: Optional[int] = None, **kwargs) -> Dict[str, np.ndarray]:
        """Reset environment with a new circuit."""
        if seed is not None:
            np.random.seed(seed)
            
        if circuit_graph is None:
            circuit_graph = self._generate_random_circuit()
        
        self.circuit = circuit_graph
        self.num_components = len(self.circuit.nodes)
        self.occupied_cells = set()  # Track (x,y) coordinates only
        self.component_positions = {}  # component_id -> (x, y, orientation)
        self.placed_mask = np.zeros(self.max_components, dtype=bool)
        self.current_step = 0
        
        # Place the first component at the center to start
        if self.num_components > 0:
            first_component = list(self.circuit.nodes())[0]
            center_x, center_y = self.grid_size // 2, self.grid_size // 2
            self._place_component(first_component, center_x, center_y, 0)
        
        obs = self._get_observation()
        info = {}
        if USING_GYMNASIUM:
            return obs, info
        else:
            return obs
    
    def _format_step_return(self, obs, reward, terminated, info):
        """Format step return for both gym and gymnasium compatibility."""
        if USING_GYMNASIUM:
            truncated = False  # We don't use truncation
            return obs, reward, terminated, truncated, info
        else:
            return obs, reward, terminated, info
    
    def step(self, action: np.ndarray):
        """Execute one environment step."""
        target_component_id, spatial_relation, orientation = action
        
        info = {"valid_action": True, "reward_breakdown": {}}
        
        # Get the next unplaced component
        next_component = self._get_next_unplaced_component()
        if next_component is None:
            # All components placed
            reward = self._calculate_reward()
            info["reward_breakdown"] = reward
            return self._format_step_return(self._get_observation(), sum(reward.values()), True, info)
        
        # Check if target component is placed
        if (target_component_id >= min(self.num_components, len(self.placed_mask)) or 
            not self.placed_mask[target_component_id]):
            # Invalid action - referencing unplaced component
            info["valid_action"] = False
            reward = {"invalid_action": -1.0}  # Reduced penalty
            info["reward_breakdown"] = reward
            self.current_step += 1
            # Don't terminate, just give penalty and continue
            done = self.current_step >= self.max_steps
            return self._format_step_return(self._get_observation(), sum(reward.values()), done, info)
        
        # Calculate proposed position based on spatial relation
        target_pos = self.component_positions[target_component_id]
        proposed_pos = self._calculate_relational_position(
            target_pos, spatial_relation, next_component
        )
        
        if proposed_pos is None:
            # Invalid placement (out of bounds or overlap)
            info["valid_action"] = False
            reward = {"invalid_placement": -0.5}  # Reduced penalty
            info["reward_breakdown"] = reward
            self.current_step += 1
            done = self.current_step >= self.max_steps
            return self._format_step_return(self._get_observation(), sum(reward.values()), done, info)
        
        # Place the component
        x, y = proposed_pos
        self._place_component(next_component, x, y, orientation)
        self.current_step += 1
        
        # Calculate reward
        reward = self._calculate_reward()
        info["reward_breakdown"] = reward
        
        # Check if episode is done
        all_placed = np.sum(self.placed_mask[:self.num_components]) == self.num_components
        max_steps_reached = self.current_step >= self.max_steps
        done = all_placed or max_steps_reached
        
        return self._format_step_return(self._get_observation(), sum(reward.values()), done, info)
    
    def _get_next_unplaced_component(self) -> Optional[int]:
        """Get the next component that hasn't been placed yet."""
        for i in range(min(self.num_components, len(self.placed_mask))):
            if not self.placed_mask[i]:
                return i
        return None
    
    def _place_component(self, component_id: int, x: int, y: int, orientation: int):
        """Place a component on the grid."""
        component_attrs = self.circuit.nodes[component_id]
        width = component_attrs.get('width', 1)
        height = component_attrs.get('height', 1)
        
        # Adjust dimensions based on orientation
        if orientation in [1, 3]:  # 90° or 270°
            width, height = height, width
        
        # Mark grid positions as occupied using set
        for dx in range(width):
            for dy in range(height):
                if 0 <= x + dx < self.grid_size and 0 <= y + dy < self.grid_size:
                    self.occupied_cells.add((x + dx, y + dy))
        
        self.component_positions[component_id] = (x, y, orientation)
        self.placed_mask[component_id] = True
    
    def _calculate_relational_position(self, target_pos: Tuple[int, int, int], 
                                     relation: int, component_id: int) -> Optional[Tuple[int, int]]:
        """Calculate position based on spatial relation to target component."""
        target_x, target_y, target_orient = target_pos
        component_attrs = self.circuit.nodes[component_id]
        comp_width = component_attrs.get('width', 1)
        comp_height = component_attrs.get('height', 1)
        
        # Calculate offset based on relation
        offset_map = {
            SpatialRelation.LEFT_OF.value: (-comp_width, 0),
            SpatialRelation.RIGHT_OF.value: (2, 0),  # Assuming target width ~1-2
            SpatialRelation.ABOVE.value: (0, -comp_height),
            SpatialRelation.BELOW.value: (0, 2),  # Assuming target height ~1-2
            SpatialRelation.MIRRORED_X.value: (-comp_width - 1, 0),
            SpatialRelation.MIRRORED_Y.value: (0, -comp_height - 1),
            SpatialRelation.ADJACENT.value: (1, 1)
        }
        
        offset_x, offset_y = offset_map.get(relation, (1, 1))
        new_x = target_x + offset_x
        new_y = target_y + offset_y
        
        # Check bounds and overlap
        if (new_x < 0 or new_y < 0 or 
            new_x + comp_width > self.grid_size or 
            new_y + comp_height > self.grid_size):
            return None
        
        # Check for overlap using set
        for dx in range(comp_width):
            for dy in range(comp_height):
                if (new_x + dx, new_y + dy) in self.occupied_cells:
                    return None
        
        return (new_x, new_y)
    
    def _calculate_reward(self) -> Dict[str, float]:
        """Calculate reward based on current layout."""
        rewards = {
            "placement_step": 1.0,  # Reward for each successful placement
            "compactness": 0.0,
            "connectivity": 0.0,
            "symmetry": 0.0,
            "completion": 0.0
        }
        
        if len(self.component_positions) == 0:
            return rewards
        
        # Compactness reward
        positions = list(self.component_positions.values())
        if len(positions) > 1:
            xs = [pos[0] for pos in positions]
            ys = [pos[1] for pos in positions]
            bbox_area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
            max_area = self.grid_size * self.grid_size
            compactness = 1.0 - (bbox_area / max_area)
            rewards["compactness"] = compactness * 5.0
        
        # Connectivity reward
        connectivity_score = 0.0
        for edge in self.circuit.edges():
            comp1, comp2 = edge
            if comp1 in self.component_positions and comp2 in self.component_positions:
                pos1 = self.component_positions[comp1]
                pos2 = self.component_positions[comp2]
                distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])  # Manhattan distance
                connectivity_score += max(0, 10 - distance)  # Closer = better
        rewards["connectivity"] = connectivity_score * 0.1
        
        # Symmetry reward (simplified - check for mirrored components)
        symmetry_score = 0.0
        for node_id in self.circuit.nodes():
            node_attrs = self.circuit.nodes[node_id]
            matched_comp = node_attrs.get('matched_component', -1)
            if (matched_comp != -1 and 
                node_id in self.component_positions and 
                matched_comp in self.component_positions):
                pos1 = self.component_positions[node_id]
                pos2 = self.component_positions[matched_comp]
                # Simple symmetry check (could be enhanced)
                if abs(pos1[1] - pos2[1]) < 2:  # Roughly same y-coordinate
                    symmetry_score += 5.0
        rewards["symmetry"] = symmetry_score
        
        # Completion bonus
        if np.sum(self.placed_mask[:self.num_components]) == self.num_components:
            rewards["completion"] = 20.0
        
        return rewards
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        # Component graph as adjacency matrix
        adj_matrix = np.zeros((self.max_components, self.max_components))
        if self.circuit is not None:
            for i in range(min(self.num_components, self.max_components)):
                for j in range(min(self.num_components, self.max_components)):
                    if self.circuit.has_edge(i, j):
                        adj_matrix[i, j] = 1.0
        
        # Placed components list with normalized positions
        placed_components_list = np.full((self.max_components, 4), -1.0)
        for comp_id, (x, y, orientation) in self.component_positions.items():
            if comp_id < self.max_components:
                placed_components_list[comp_id] = [
                    x / self.grid_size,      # Normalized x [0,1]
                    y / self.grid_size,      # Normalized y [0,1]  
                    orientation / 4.0,       # Normalized orientation [0,1]
                    1.0                      # Valid flag
                ]
        
        # Netlist features
        netlist_features = np.zeros((self.max_components, 4))
        if self.circuit is not None:
            for i, node_id in enumerate(list(self.circuit.nodes())[:self.max_components]):
                attrs = self.circuit.nodes[node_id]
                netlist_features[i] = [
                    attrs.get('component_type', 0),
                    attrs.get('width', 1),
                    attrs.get('height', 1),
                    attrs.get('matched_component', -1)
                ]
        
        return {
            "component_graph": adj_matrix.astype(np.float32),
            "placed_components_list": placed_components_list.astype(np.float32),
            "netlist_features": netlist_features.astype(np.float32),
            "placement_mask": self.placed_mask.astype(np.int8)
        }
    # NOTE: Random circuit generation commented out - now using real SPICE circuits
    # See train_spice_real.py for SPICE-based training with actual analog circuits
    def _generate_random_circuit(self) -> nx.Graph:
        """
        Generate a random small analog circuit for testing.
        DEPRECATED: Now using real SPICE circuits for training.
        This method kept for backward compatibility with existing code.
        """
        # Simple fallback circuit for compatibility
        G = nx.Graph()
        G.add_node(0, component_type=0, width=2, height=1, matched_component=-1)
        G.add_node(1, component_type=1, width=2, height=1, matched_component=-1) 
        G.add_node(2, component_type=0, width=1, height=2, matched_component=-1)
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        return G
        
        # COMMENTED OUT: Original random circuit generation
        # Now transitioning to exclusively use real SPICE circuits
        """
        num_components = np.random.randint(3, min(8, self.max_components + 1))
        G = nx.Graph()
        
        # Add components with attributes
        component_types = list(ComponentType)
        for i in range(num_components):
            comp_type = np.random.choice(component_types)
            G.add_node(i, 
                      component_type=comp_type.value,
                      width=np.random.randint(1, 3),
                      height=np.random.randint(1, 3),
                      matched_component=-1)  # Will be set for matched pairs
        
        # Add some random connections
        num_edges = min(num_components * 2, num_components * (num_components - 1) // 2)
        for _ in range(num_edges):
            i, j = np.random.choice(num_components, 2, replace=False)
            G.add_edge(i, j)
        
        # Add some matched component pairs (for symmetry)
        if num_components >= 4:
            pairs = [(0, 1), (2, 3)]
            for comp1, comp2 in pairs:
                if comp1 < num_components and comp2 < num_components:
                    G.nodes[comp1]['matched_component'] = comp2
                    G.nodes[comp2]['matched_component'] = comp1
        
        return G
        """

    def render(self, mode='human'):
        """Render the current state (basic text representation)."""
        print(f"Grid {self.grid_size}x{self.grid_size}:")
        print(f"Placed components: {np.sum(self.placed_mask[:self.num_components])}/{self.num_components}")
        for i, (comp_id, (x, y, orient)) in enumerate(self.component_positions.items()):
            print(f"Component {comp_id}: ({x}, {y}) orientation {orient * 90}°")