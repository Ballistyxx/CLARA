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
import time
from typing import Dict, Tuple, Optional, List, Any
from enum import Enum
from curriculum_config import CurriculumManager, get_metrics_for_stage


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
    
    def __init__(self, grid_size: int = 64, max_components: int = 10, enable_action_masking: bool = True):
        super().__init__()
        
        self.grid_size = grid_size
        self.max_components = max_components
        self.max_steps = max_components * 1000  # Limit episode length
        self.enable_action_masking = enable_action_masking
        
        # Action space: [target_component_id, spatial_relation_type, orientation]
        self.action_space = spaces.MultiDiscrete([
            max_components,  # target component (reference for placement)
            len(SpatialRelation),  # spatial relation type
            4  # orientation (0°, 90°, 180°, 270°)
        ])
        
        # Observation space
        obs_dict = {
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
                low=0, high=10.0,
                shape=(max_components, 4),  # [component_type, width, height, matched_component]
                dtype=np.float32
            ),
            "placement_mask": spaces.Box(
                low=0, high=1,
                shape=(max_components,),
                dtype=np.int8
            )
        }
        
        # Add action mask to observation space if enabled
        if self.enable_action_masking:
            obs_dict["action_mask"] = spaces.Box(
                low=0, high=1,
                shape=(max_components,),  # Mask for valid target components
                dtype=np.int8
            )
            
        self.observation_space = spaces.Dict(obs_dict)
        
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
        self.component_rectangles = {}  # component_id -> (x, y, width, height) for collision detection
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
        width = component_attrs.get('width', 1.0)
        height = component_attrs.get('height', 1.0)
        
        # Adjust dimensions based on orientation
        if orientation in [1, 3]:  # 90° or 270°
            width, height = height, width
        
        # Store component rectangle for collision detection
        self.component_rectangles[component_id] = (float(x), float(y), float(width), float(height))
        
        self.component_positions[component_id] = (x, y, orientation)
        self.placed_mask[component_id] = True
    
    def _calculate_relational_position(self, target_pos: Tuple[int, int, int], 
                                     relation: int, component_id: int) -> Optional[Tuple[int, int]]:
        """Calculate position based on spatial relation to target component."""
        target_x, target_y, target_orient = target_pos
        component_attrs = self.circuit.nodes[component_id]
        comp_width = component_attrs.get('width', 1.0)
        comp_height = component_attrs.get('height', 1.0)
        
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
        
        # Check for overlap with existing components using rectangle intersection
        new_rect = (float(new_x), float(new_y), float(comp_width), float(comp_height))
        for existing_rect in self.component_rectangles.values():
            if self._rectangles_overlap(new_rect, existing_rect):
                return None
        
        return (new_x, new_y)
    
    def _rectangles_overlap(self, rect1: Tuple[float, float, float, float], 
                          rect2: Tuple[float, float, float, float]) -> bool:
        """Check if two rectangles overlap.
        
        Args:
            rect1: (x, y, width, height)
            rect2: (x, y, width, height)
        
        Returns:
            True if rectangles overlap, False otherwise
        """
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        # Check for non-overlap conditions (rectangles are separate)
        if (x1 >= x2 + w2 or  # rect1 is to the right of rect2
            x2 >= x1 + w1 or  # rect2 is to the right of rect1
            y1 >= y2 + h2 or  # rect1 is below rect2
            y2 >= y1 + h1):   # rect2 is below rect1
            return False
        
        return True  # Rectangles overlap
    
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
                    attrs.get('width', 1.0),
                    attrs.get('height', 1.0),
                    attrs.get('matched_component', -1)
                ]
        
        obs = {
            "component_graph": adj_matrix.astype(np.float32),
            "placed_components_list": placed_components_list.astype(np.float32),
            "netlist_features": netlist_features.astype(np.float32),
            "placement_mask": self.placed_mask.astype(np.int8)
        }
        
        # Add action mask if enabled
        if self.enable_action_masking:
            obs["action_mask"] = self._get_action_mask()
            
        return obs
    
    def _get_action_mask(self) -> np.ndarray:
        """Generate action mask for valid target components."""
        action_mask = np.zeros(self.max_components, dtype=np.int8)
        
        # Only placed components can be used as targets
        for i in range(min(self.num_components, self.max_components)):
            if self.placed_mask[i]:
                action_mask[i] = 1
        
        return action_mask
    
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


class EnhancedAnalogLayoutEnv(AnalogLayoutEnv):
    """
    Enhanced analog layout environment with curriculum learning and advanced metrics integration.
    
    Implements a 4-stage curriculum that gradually introduces sophisticated metrics into training
    rewards, improving layout quality while maintaining training stability.
    """
    
    def __init__(self, 
                 grid_size: int = 15,
                 max_components: int = 6,
                 enable_action_masking: bool = True,
                 curriculum_enabled: bool = True,
                 manual_stage: int = None,
                 metric_calculation_interval: int = 3,
                 curriculum_config: Dict[str, Any] = None):
        """Initialize enhanced environment with curriculum learning.
        
        Args:
            grid_size: Size of the layout grid
            max_components: Maximum number of components  
            enable_action_masking: Enable action masking
            curriculum_enabled: Enable curriculum learning
            manual_stage: Force specific curriculum stage (1-4)
            metric_calculation_interval: Calculate expensive metrics every N steps
            curriculum_config: Optional curriculum configuration overrides
        """
        # Initialize curriculum attributes BEFORE calling parent constructor
        # (because parent calls reset() which needs these attributes)
        self.curriculum_enabled = curriculum_enabled
        self.curriculum_manager = None
        if curriculum_enabled:
            self.curriculum_manager = CurriculumManager(
                manual_stage=manual_stage,
                episode_window=curriculum_config.get("episode_window", 1000) if curriculum_config else 1000
            )
        
        # Initialize other attributes
        self.metric_calculation_interval = metric_calculation_interval
        self.step_count = 0
        self.cached_metrics = {}
        self.last_metric_calculation = -1
        self.episode_count = 0
        self.current_episode_success = False
        self._metrics_calculator = None
        self._layout_grid = None
        
        # Now call parent constructor
        super().__init__(grid_size, max_components, enable_action_masking)
        
        print(f"Enhanced environment initialized:")
        print(f"  Curriculum enabled: {curriculum_enabled}")
        if curriculum_enabled and manual_stage:
            print(f"  Manual stage override: {manual_stage}")
        print(f"  Metric calculation interval: {metric_calculation_interval}")
    
    def _get_metrics_calculator(self):
        """Lazy initialization of metrics calculator."""
        if self._metrics_calculator is None:
            try:
                from metrics.metrics import MetricsCalculator
                self._metrics_calculator = MetricsCalculator()
            except ImportError as e:
                print(f"Warning: Could not import metrics system: {e}")
                print("Falling back to simple reward calculation")
                return None
        return self._metrics_calculator
    
    def _get_layout_grid(self):
        """Convert current state to LayoutGrid for metrics calculation."""
        if self._layout_grid is None:
            try:
                from env.layout_grid import LayoutGrid, ComponentType as GridComponentType
                from env.layout_grid import ComponentPlacement
                
                # Create layout grid
                grid = LayoutGrid(self.grid_size, self.grid_size)
                
                # Add placed components
                for comp_id, (x, y, orientation) in self.component_positions.items():
                    if comp_id in self.circuit.nodes:
                        attrs = self.circuit.nodes[comp_id]
                        width = attrs.get('width', 1.0)
                        height = attrs.get('height', 1.0)
                        
                        # Adjust for orientation
                        if orientation in [1, 3]:  # 90° or 270°
                            width, height = height, width
                        
                        # Map component type
                        comp_type_val = attrs.get('component_type', 0)
                        if comp_type_val == 0:
                            comp_type = GridComponentType.NMOS
                        elif comp_type_val == 1:
                            comp_type = GridComponentType.PMOS
                        elif comp_type_val == 2:
                            comp_type = GridComponentType.RESISTOR
                        elif comp_type_val == 3:
                            comp_type = GridComponentType.CAPACITOR
                        else:
                            comp_type = GridComponentType.OTHER
                        
                        placement = ComponentPlacement(
                            component_id=comp_id,
                            x=int(x),
                            y=int(y),
                            width=int(max(1, width)),
                            height=int(max(1, height)),
                            orientation=orientation,
                            component_type=comp_type
                        )
                        
                        grid.place_component(placement)
                
                self._layout_grid = grid
                return grid
                
            except ImportError as e:
                print(f"Warning: Could not import layout grid: {e}")
                return None
        
        return self._layout_grid
    
    def reset(self, *, seed: Optional[int] = None, circuit_graph: Optional[nx.Graph] = None, **kwargs):
        """Reset environment and update curriculum state."""
        # Reset parent environment
        result = super().reset(seed=seed, circuit_graph=circuit_graph, **kwargs)
        
        # Reset curriculum tracking
        self.step_count = 0
        self.cached_metrics = {}
        self.last_metric_calculation = -1
        self.current_episode_success = False
        self._layout_grid = None  # Clear layout grid cache
        
        # Update episode count for curriculum
        if self.curriculum_enabled and self.curriculum_manager:
            self.episode_count += 1
            
        return result
    
    def step(self, action):
        """Step with enhanced reward calculation using curriculum metrics."""
        self.step_count += 1
        
        # Call parent step method (handle both gym/gymnasium formats)
        parent_result = super().step(action)
        
        if len(parent_result) == 5:  # Gymnasium format (obs, reward, terminated, truncated, info)
            obs, base_reward, terminated, truncated, info = parent_result
            done = terminated or truncated
        else:  # Gym format (obs, reward, done, info)
            obs, base_reward, done, info = parent_result
        
        # Calculate enhanced reward if curriculum is enabled
        if self.curriculum_enabled and self.curriculum_manager:
            enhanced_reward = self._calculate_enhanced_reward(base_reward, done, info)
            
            # Update curriculum manager on episode completion
            if done:
                self.current_episode_success = info.get('success', False) or self._check_episode_success()
                self.curriculum_manager.update_episode(
                    episode=self.episode_count,
                    success=self.current_episode_success,
                    total_reward=enhanced_reward
                )
                
                # Add curriculum info to info dict
                info['curriculum_status'] = self.curriculum_manager.get_curriculum_status()
            
            # Return in same format as parent
            if len(parent_result) == 5:
                return obs, enhanced_reward, terminated, truncated, info
            else:
                return obs, enhanced_reward, done, info
        else:
            return parent_result
    
    def _check_episode_success(self) -> bool:
        """Check if current episode should be considered successful."""
        if self.circuit is None:
            return False
        
        # Success = all components placed
        return len(self.component_positions) >= self.num_components
    
    def _calculate_enhanced_reward(self, base_reward: Dict[str, float], 
                                   done: bool, info: Dict[str, Any]) -> float:
        """Calculate enhanced reward using curriculum-appropriate metrics.
        
        Args:
            base_reward: Original simple reward components
            done: Whether episode is done
            info: Additional information
            
        Returns:
            Enhanced reward value combining simple and advanced metrics
        """
        if not self.curriculum_manager:
            # Fallback to simple reward
            return sum(base_reward.values())
        
        current_stage = self.curriculum_manager.get_current_stage(self.episode_count)
        simple_weight, advanced_weight = self.curriculum_manager.get_reward_weights(current_stage)
        
        # Calculate simple reward component
        if isinstance(base_reward, dict):
            simple_reward = sum(base_reward.values()) * simple_weight
        else:
            simple_reward = base_reward * simple_weight
        
        # Calculate advanced reward component (if weight > 0)
        advanced_reward = 0.0
        if advanced_weight > 0.0:
            advanced_reward = self._calculate_stage_metrics(current_stage) * advanced_weight
        
        total_reward = simple_reward + advanced_reward
        
        # Add curriculum info to info for logging (not base_reward since it might be a float)
        info.update({
            'curriculum_stage': float(current_stage),
            'simple_component': simple_reward,
            'advanced_component': advanced_reward,
            'curriculum_total': total_reward
        })
        
        return total_reward
    
    def _calculate_stage_metrics(self, stage: int) -> float:
        """Calculate metrics appropriate for the current curriculum stage.
        
        Args:
            stage: Current curriculum stage (1-4)
            
        Returns:
            Weighted metric score for the stage
        """
        # Performance optimization: only calculate expensive metrics periodically
        if (self.step_count - self.last_metric_calculation) < self.metric_calculation_interval:
            return self.cached_metrics.get('stage_score', 0.0)
        
        try:
            # Use lightweight metrics for better performance
            from metrics.lightweight_metrics import get_lightweight_calculator
            
            lightweight_calc = get_lightweight_calculator()
            metrics = lightweight_calc.calculate_stage_metrics(
                stage=stage,
                component_positions=self.component_positions,
                circuit=self.circuit,
                grid_size=self.grid_size,
                step_count=self.step_count
            )
            
            # Cache results
            self.cached_metrics = metrics
            self.last_metric_calculation = self.step_count
            
            return metrics.get('stage_score', 0.0)
            
        except Exception as e:
            print(f"Warning: Error calculating lightweight metrics: {e}")
            return self._calculate_simple_metrics(stage)
    
    def _extract_stage_score(self, stage: int, metrics) -> float:
        """Extract appropriate score for curriculum stage from full metrics.
        
        Args:
            stage: Curriculum stage (1-4)
            metrics: LayoutMetrics object with all calculated metrics
            
        Returns:
            Weighted score for the stage
        """
        if stage == 1:
            # Stage 1: Basic placement
            return (metrics.completion * 2.0 + 
                   metrics.row_consistency * 1.0) / 3.0
                   
        elif stage == 2:
            # Stage 2: Quality introduction
            return (metrics.completion * 1.5 +
                   metrics.row_consistency * 1.0 +
                   metrics.symmetry_score * 1.5 +
                   metrics.abutment_alignment * 1.0) / 5.0
                   
        elif stage == 3:
            # Stage 3: Routing awareness
            crossings_penalty = max(0, 1.0 - metrics.crossings / 10.0)  # Penalty for crossings
            return (metrics.completion * 1.0 +
                   metrics.row_consistency * 0.5 +
                   metrics.symmetry_score * 1.0 +
                   metrics.abutment_alignment * 1.0 +
                   metrics.rail_alignment * 1.5 +
                   (1.0 - min(1.0, metrics.avg_connection_distance / 10.0)) * 1.0 +
                   crossings_penalty * 1.0) / 7.0
                   
        else:  # Stage 4: Full optimization
            # Use the full analog score
            return metrics.analog_score
    
    def _calculate_simple_metrics(self, stage: int) -> float:
        """Fallback simple metrics calculation when advanced metrics unavailable.
        
        Args:
            stage: Curriculum stage
            
        Returns:
            Simple metric score
        """
        if self.circuit is None:
            return 0.0
        
        # Basic completion metric
        completion = len(self.component_positions) / max(1, self.num_components)
        
        if stage == 1:
            return completion
        elif stage <= 3:
            # Add simple compactness
            if len(self.component_positions) >= 2:
                positions = list(self.component_positions.values())
                xs = [pos[0] for pos in positions]
                ys = [pos[1] for pos in positions]
                
                width = max(xs) - min(xs) + 1
                height = max(ys) - min(ys) + 1
                compactness = 1.0 / (1.0 + width * height / len(positions))
                
                return (completion + compactness) / 2.0
            else:
                return completion
        else:  # Stage 4
            return completion * 0.8  # More demanding for final stage
    
    def get_curriculum_info(self) -> Dict[str, Any]:
        """Get current curriculum learning information for monitoring."""
        if not self.curriculum_enabled or not self.curriculum_manager:
            return {"curriculum_enabled": False}
        
        return self.curriculum_manager.get_curriculum_status()
    
    def force_curriculum_stage(self, stage: int):
        """Force curriculum to specific stage (for debugging/testing)."""
        if self.curriculum_manager:
            self.curriculum_manager.manual_stage = stage
            print(f"Forced curriculum to stage {stage}")
    
    def disable_curriculum(self):
        """Disable curriculum learning (fallback to simple rewards)."""
        self.curriculum_enabled = False
        print("Curriculum learning disabled")