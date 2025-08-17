"""
Lightweight metrics for curriculum learning integration.

Provides fast, incremental metric calculations optimized for RL training 
without full layout grid construction.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Set
import time
from collections import defaultdict


class LightweightMetricsCalculator:
    """
    Fast metric calculations for curriculum learning.
    
    Optimized for speed during RL training with caching and incremental updates.
    """
    
    def __init__(self, cache_size: int = 1000):
        """Initialize lightweight calculator.
        
        Args:
            cache_size: Maximum number of cached metric results
        """
        self.cache_size = cache_size
        self.metric_cache = {}
        self.last_positions = {}
        self.last_circuit_hash = None
        
    def calculate_stage_metrics(self, 
                               stage: int,
                               component_positions: Dict[int, Tuple[int, int, int]],
                               circuit: nx.Graph,
                               grid_size: int,
                               step_count: int) -> Dict[str, float]:
        """Calculate metrics appropriate for curriculum stage.
        
        Args:
            stage: Curriculum stage (1-4)
            component_positions: Dict of component_id -> (x, y, orientation)
            circuit: NetworkX circuit graph
            grid_size: Layout grid size
            step_count: Current step count for caching
            
        Returns:
            Dictionary of metric name -> value
        """
        # Check cache
        cache_key = (stage, step_count, len(component_positions))
        if cache_key in self.metric_cache:
            # Verify positions haven't changed significantly
            if self._positions_similar(component_positions, self.last_positions):
                return self.metric_cache[cache_key]
        
        start_time = time.time()
        
        # Calculate stage-specific metrics
        if stage == 1:
            metrics = self._calculate_basic_metrics(component_positions, circuit)
        elif stage == 2:
            metrics = self._calculate_quality_metrics(component_positions, circuit, grid_size)
        elif stage == 3:
            metrics = self._calculate_routing_metrics(component_positions, circuit, grid_size)
        else:  # Stage 4
            metrics = self._calculate_full_metrics(component_positions, circuit, grid_size)
        
        # Add timing info
        metrics['calculation_time'] = time.time() - start_time
        
        # Cache results
        self.metric_cache[cache_key] = metrics
        self.last_positions = component_positions.copy()
        
        # Limit cache size
        if len(self.metric_cache) > self.cache_size:
            # Remove oldest entries
            keys = list(self.metric_cache.keys())
            for key in keys[:len(keys) - self.cache_size]:
                del self.metric_cache[key]
        
        return metrics
    
    def _positions_similar(self, pos1: Dict, pos2: Dict, threshold: float = 0.1) -> bool:
        """Check if two position dictionaries are similar enough to use cache."""
        if len(pos1) != len(pos2):
            return False
        
        if not pos1 or not pos2:
            return len(pos1) == len(pos2)
        
        # Check if all components are in similar positions
        for comp_id in pos1:
            if comp_id not in pos2:
                return False
            
            x1, y1, o1 = pos1[comp_id]
            x2, y2, o2 = pos2[comp_id]
            
            if abs(x1 - x2) > threshold or abs(y1 - y2) > threshold or o1 != o2:
                return False
        
        return True
    
    def _calculate_basic_metrics(self, positions: Dict, circuit: nx.Graph) -> Dict[str, float]:
        """Stage 1: Basic placement metrics."""
        num_components = len(circuit.nodes) if circuit else 1
        completion = len(positions) / max(1, num_components)
        
        # Basic row consistency (simplified)
        row_consistency = 1.0  # Assume good for basic stage
        if positions:
            # Check if PMOS components are in upper half, NMOS in lower half
            pmos_positions = []
            nmos_positions = []
            
            for comp_id, (x, y, orient) in positions.items():
                if comp_id in circuit.nodes:
                    comp_type = circuit.nodes[comp_id].get('component_type', 0)
                    if comp_type == 1:  # PMOS
                        pmos_positions.append(y)
                    elif comp_type == 0:  # NMOS
                        nmos_positions.append(y)
            
            if pmos_positions and nmos_positions:
                avg_pmos_y = np.mean(pmos_positions)
                avg_nmos_y = np.mean(nmos_positions)
                # Good if PMOS is above NMOS on average
                row_consistency = max(0.0, min(1.0, (avg_nmos_y - avg_pmos_y + 5) / 10))
        
        return {
            'completion': completion,
            'row_consistency': row_consistency,
            'basic_compactness': self._calculate_basic_compactness(positions),
            'stage_score': (completion * 2.0 + row_consistency * 1.0) / 3.0
        }
    
    def _calculate_quality_metrics(self, positions: Dict, circuit: nx.Graph, grid_size: int) -> Dict[str, float]:
        """Stage 2: Quality introduction metrics."""
        basic_metrics = self._calculate_basic_metrics(positions, circuit)
        
        # Add symmetry and alignment
        symmetry_score = self._calculate_fast_symmetry(positions, circuit)
        abutment_alignment = self._calculate_fast_alignment(positions)
        
        stage_score = (basic_metrics['completion'] * 1.5 +
                      basic_metrics['row_consistency'] * 1.0 +
                      symmetry_score * 1.5 +
                      abutment_alignment * 1.0) / 5.0
        
        return {
            **basic_metrics,
            'symmetry_score': symmetry_score,
            'abutment_alignment': abutment_alignment,
            'stage_score': stage_score
        }
    
    def _calculate_routing_metrics(self, positions: Dict, circuit: nx.Graph, grid_size: int) -> Dict[str, float]:
        """Stage 3: Routing awareness metrics."""
        quality_metrics = self._calculate_quality_metrics(positions, circuit, grid_size)
        
        # Add routing metrics
        avg_connection_distance = self._calculate_fast_connection_distance(positions, circuit)
        crossings = self._estimate_crossings(positions, circuit)
        rail_alignment = self._calculate_fast_rail_alignment(positions, grid_size)
        
        # Calculate stage score
        crossings_penalty = max(0, 1.0 - crossings / 10.0)
        stage_score = (quality_metrics['completion'] * 1.0 +
                      quality_metrics['row_consistency'] * 0.5 +
                      quality_metrics['symmetry_score'] * 1.0 +
                      quality_metrics['abutment_alignment'] * 1.0 +
                      rail_alignment * 1.5 +
                      (1.0 - min(1.0, avg_connection_distance / 10.0)) * 1.0 +
                      crossings_penalty * 1.0) / 7.0
        
        return {
            **quality_metrics,
            'avg_connection_distance': avg_connection_distance,
            'crossings': crossings,
            'rail_alignment': rail_alignment,
            'stage_score': stage_score
        }
    
    def _calculate_full_metrics(self, positions: Dict, circuit: nx.Graph, grid_size: int) -> Dict[str, float]:
        """Stage 4: Full optimization metrics."""
        routing_metrics = self._calculate_routing_metrics(positions, circuit, grid_size)
        
        # Add advanced metrics
        matching_accuracy = self._calculate_fast_matching(positions, circuit)
        spacing_violations = self._count_spacing_violations(positions)
        overlap_penalties = self._count_overlaps(positions, circuit)
        
        # Calculate analog score (simplified)
        analog_score = (routing_metrics['completion'] * 0.3 +
                       routing_metrics['symmetry_score'] * 0.25 +
                       routing_metrics['abutment_alignment'] * 0.2 +
                       matching_accuracy * 0.15 +
                       max(0, 1.0 - spacing_violations / 10.0) * 0.1)
        
        return {
            **routing_metrics,
            'matching_accuracy': matching_accuracy,
            'spacing_violations': spacing_violations,
            'overlap_penalties': overlap_penalties,
            'analog_score': analog_score,
            'stage_score': analog_score
        }
    
    def _calculate_basic_compactness(self, positions: Dict) -> float:
        """Fast compactness calculation."""
        if len(positions) < 2:
            return 1.0
        
        xs = [pos[0] for pos in positions.values()]
        ys = [pos[1] for pos in positions.values()]
        
        width = max(xs) - min(xs) + 1
        height = max(ys) - min(ys) + 1
        area = width * height
        
        # Normalize by number of components
        return min(1.0, len(positions) / max(1, area * 0.5))
    
    def _calculate_fast_symmetry(self, positions: Dict, circuit: nx.Graph) -> float:
        """Fast symmetry score calculation."""
        if not positions or not circuit:
            return 1.0
        
        # Find matched component pairs
        matched_pairs = []
        for comp_id in positions:
            if comp_id in circuit.nodes:
                matched_comp = circuit.nodes[comp_id].get('matched_component', -1)
                if matched_comp != -1 and matched_comp in positions:
                    pair = tuple(sorted([comp_id, matched_comp]))
                    if pair not in matched_pairs:
                        matched_pairs.append(pair)
        
        if not matched_pairs:
            return 1.0
        
        # Calculate symmetry for each pair
        symmetry_scores = []
        for comp1, comp2 in matched_pairs:
            x1, y1, o1 = positions[comp1]
            x2, y2, o2 = positions[comp2]
            
            # Check for horizontal or vertical symmetry
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            
            # Good symmetry = similar distance from center
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            symmetry = 1.0 / (1.0 + 0.5 * (dx + dy))
            symmetry_scores.append(symmetry)
        
        return np.mean(symmetry_scores)
    
    def _calculate_fast_alignment(self, positions: Dict) -> float:
        """Fast abutment alignment calculation."""
        if len(positions) < 2:
            return 1.0
        
        # Check for components aligned in rows/columns
        y_positions = defaultdict(list)
        x_positions = defaultdict(list)
        
        for comp_id, (x, y, orient) in positions.items():
            y_positions[y].append(x)
            x_positions[x].append(y)
        
        # Score based on how many components are aligned
        row_alignment = sum(len(comps) - 1 for comps in y_positions.values() if len(comps) > 1)
        col_alignment = sum(len(comps) - 1 for comps in x_positions.values() if len(comps) > 1)
        
        total_alignment = row_alignment + col_alignment
        max_alignment = len(positions) * 2  # Maximum possible alignment
        
        return total_alignment / max(1, max_alignment)
    
    def _calculate_fast_connection_distance(self, positions: Dict, circuit: nx.Graph) -> float:
        """Fast average connection distance calculation."""
        if not circuit or len(positions) < 2:
            return 0.0
        
        distances = []
        for edge in circuit.edges():
            comp1, comp2 = edge
            if comp1 in positions and comp2 in positions:
                x1, y1, _ = positions[comp1]
                x2, y2, _ = positions[comp2]
                distance = abs(x1 - x2) + abs(y1 - y2)  # Manhattan distance
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def _estimate_crossings(self, positions: Dict, circuit: nx.Graph) -> int:
        """Fast estimate of net crossings."""
        if not circuit or len(positions) < 3:
            return 0
        
        # Simple heuristic: count edge intersections
        edges = []
        for edge in circuit.edges():
            comp1, comp2 = edge
            if comp1 in positions and comp2 in positions:
                x1, y1, _ = positions[comp1]
                x2, y2, _ = positions[comp2]
                edges.append(((x1, y1), (x2, y2)))
        
        crossings = 0
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                if self._segments_intersect(edges[i], edges[j]):
                    crossings += 1
        
        return crossings
    
    def _segments_intersect(self, seg1: Tuple[Tuple[int, int], Tuple[int, int]], 
                           seg2: Tuple[Tuple[int, int], Tuple[int, int]]) -> bool:
        """Check if two line segments intersect."""
        (x1, y1), (x2, y2) = seg1
        (x3, y3), (x4, y4) = seg2
        
        # Check if segments share endpoints
        if (x1, y1) in [(x3, y3), (x4, y4)] or (x2, y2) in [(x3, y3), (x4, y4)]:
            return False
        
        # Use cross product to check intersection
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return (ccw((x1, y1), (x3, y3), (x4, y4)) != ccw((x2, y2), (x3, y3), (x4, y4)) and
                ccw((x1, y1), (x2, y2), (x3, y3)) != ccw((x1, y1), (x2, y2), (x4, y4)))
    
    def _calculate_fast_rail_alignment(self, positions: Dict, grid_size: int) -> float:
        """Fast rail alignment calculation."""
        if not positions:
            return 1.0
        
        # Assume power rails are at top and bottom
        top_rail_y = 0
        bottom_rail_y = grid_size - 1
        
        # Score components by distance to appropriate rail
        rail_scores = []
        for comp_id, (x, y, orient) in positions.items():
            # Distance to nearest rail
            dist_to_top = abs(y - top_rail_y)
            dist_to_bottom = abs(y - bottom_rail_y)
            min_dist = min(dist_to_top, dist_to_bottom)
            
            # Good score for components close to rails
            score = max(0.0, 1.0 - min_dist / (grid_size / 2))
            rail_scores.append(score)
        
        return np.mean(rail_scores) if rail_scores else 0.0
    
    def _calculate_fast_matching(self, positions: Dict, circuit: nx.Graph) -> float:
        """Fast matching accuracy calculation."""
        return self._calculate_fast_symmetry(positions, circuit)  # Reuse symmetry calculation
    
    def _count_spacing_violations(self, positions: Dict) -> int:
        """Count spacing violations between components."""
        violations = 0
        components = list(positions.items())
        
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                comp1_id, (x1, y1, o1) = components[i]
                comp2_id, (x2, y2, o2) = components[j]
                
                # Simple spacing check (assume components are 1x1)
                dx = abs(x1 - x2)
                dy = abs(y1 - y2)
                
                # Violation if components are too close (adjacent is okay)
                if dx == 0 and dy == 0:  # Overlapping
                    violations += 2
                elif dx <= 1 and dy <= 1 and (dx + dy) < 2:  # Too close
                    violations += 1
        
        return violations
    
    def _count_overlaps(self, positions: Dict, circuit: nx.Graph) -> int:
        """Count overlapping components."""
        overlaps = 0
        components = list(positions.items())
        
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                comp1_id, (x1, y1, o1) = components[i]
                comp2_id, (x2, y2, o2) = components[j]
                
                # Get component sizes from circuit
                width1 = height1 = 1
                width2 = height2 = 1
                
                if circuit and comp1_id in circuit.nodes:
                    width1 = circuit.nodes[comp1_id].get('width', 1.0)
                    height1 = circuit.nodes[comp1_id].get('height', 1.0)
                
                if circuit and comp2_id in circuit.nodes:
                    width2 = circuit.nodes[comp2_id].get('width', 1.0)
                    height2 = circuit.nodes[comp2_id].get('height', 1.0)
                
                # Adjust for orientation
                if o1 in [1, 3]:
                    width1, height1 = height1, width1
                if o2 in [1, 3]:
                    width2, height2 = height2, width2
                
                # Check overlap
                if (x1 < x2 + width2 and x1 + width1 > x2 and
                    y1 < y2 + height2 and y1 + height1 > y2):
                    overlaps += 1
        
        return overlaps
    
    def clear_cache(self):
        """Clear metric cache."""
        self.metric_cache.clear()
        self.last_positions.clear()


# Global lightweight calculator instance
_lightweight_calculator = None

def get_lightweight_calculator() -> LightweightMetricsCalculator:
    """Get global lightweight metrics calculator instance."""
    global _lightweight_calculator
    if _lightweight_calculator is None:
        _lightweight_calculator = LightweightMetricsCalculator()
    return _lightweight_calculator