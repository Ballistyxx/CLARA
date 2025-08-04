#!/usr/bin/env python3
"""
Comprehensive metrics for analog IC layout evaluation.
Includes symmetry, row consistency, crossings, congestion, perimeter/area, and more.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import math

from env.layout_grid import LayoutGrid, ComponentType, RowType


@dataclass
class LayoutMetrics:
    """Complete set of layout metrics."""
    # Core metrics
    completion: float           # Fraction of components placed
    row_consistency: float      # Fraction placed in correct rows
    symmetry_score: float       # Symmetry quality for matched pairs
    pattern_validity: float     # Locked pattern preservation
    
    # Routing metrics (proxies)
    crossings: int             # Number of net crossings
    congestion_variance: float # Routing congestion uniformity
    avg_connection_distance: float  # Average Manhattan distance
    
    # Layout quality
    perimeter_area_ratio: float     # Compactness measure
    abutment_alignment: float       # Gate/drain alignment quality
    aspect_ratio: float            # Width/height ratio
    utilization: float             # Area utilization
    
    # Advanced analog metrics
    rail_alignment: float          # Source/drain to power rail distance
    matching_accuracy: float       # Precision of matched device placement
    device_spacing: float          # Average spacing between devices
    hierarchy_preservation: float  # Hierarchical block organization
    
    # Constraint violations
    overlap_count: int            # Number of overlapping cells
    row_violations: int           # Components in wrong rows
    spacing_violations: int       # Min spacing violations
    boundary_violations: int      # Out-of-bounds placements
    
    # Summary scores
    analog_score: float           # Overall analog-friendliness score
    total_score: float           # Weighted total score


class MetricsCalculator:
    """
    Comprehensive metrics calculator for analog layout evaluation.
    """
    
    def __init__(self, 
                 weight_completion: float = 4.0,
                 weight_symmetry: float = 3.0,
                 weight_row_consistency: float = 2.0,
                 weight_pattern_validity: float = 2.0,
                 weight_abutment: float = 1.5,
                 weight_crossings: float = -1.5,
                 weight_congestion: float = -1.0,
                 weight_perimeter: float = -1.0):
        """
        Initialize metrics calculator with weights.
        
        Args:
            weight_*: Importance weights for different metrics
        """
        self.weights = {
            'completion': weight_completion,
            'symmetry': weight_symmetry,
            'row_consistency': weight_row_consistency,
            'pattern_validity': weight_pattern_validity,
            'abutment_alignment': weight_abutment,
            'crossings': weight_crossings,
            'congestion_variance': weight_congestion,
            'perimeter_area_ratio': weight_perimeter
        }
    
    def calculate_all_metrics(self, 
                             layout_grid: LayoutGrid,
                             circuit: nx.Graph,
                             target_components: Optional[int] = None) -> LayoutMetrics:
        """
        Calculate all layout metrics.
        
        Args:
            layout_grid: Current layout state
            circuit: Circuit graph with connectivity
            target_components: Expected number of components (for completion)
            
        Returns:
            Complete LayoutMetrics object
        """
        if target_components is None:
            target_components = len(circuit.nodes) if circuit else len(layout_grid.placements)
        
        # Core metrics
        completion = self._calculate_completion(layout_grid, target_components)
        row_consistency = layout_grid.get_row_consistency_score()
        symmetry_score = self._calculate_symmetry_score(layout_grid, circuit)
        pattern_validity = self._calculate_pattern_validity(layout_grid)
        
        # Routing metrics
        connections = list(circuit.edges()) if circuit else []
        crossings = layout_grid.count_crossings(connections)
        congestion_variance = layout_grid.get_congestion_variance(connections)
        avg_connection_distance = self._calculate_avg_connection_distance(layout_grid, connections)
        
        # Layout quality metrics
        perimeter_area_ratio = layout_grid.get_perimeter_area_ratio()
        abutment_alignment = self._calculate_abutment_alignment(layout_grid)
        aspect_ratio = self._calculate_aspect_ratio(layout_grid)
        utilization = self._calculate_utilization(layout_grid)
        
        # Advanced analog metrics
        rail_alignment = self._calculate_rail_alignment(layout_grid)
        matching_accuracy = self._calculate_matching_accuracy(layout_grid, circuit)
        device_spacing = self._calculate_device_spacing(layout_grid)
        hierarchy_preservation = self._calculate_hierarchy_preservation(layout_grid, circuit)
        
        # Violation counts
        overlap_count = layout_grid.get_overlap_count()
        row_violations = self._count_row_violations(layout_grid)
        spacing_violations = self._count_spacing_violations(layout_grid)
        boundary_violations = self._count_boundary_violations(layout_grid)
        
        # Summary scores
        analog_score = self._calculate_analog_score(
            row_consistency, symmetry_score, rail_alignment, matching_accuracy
        )
        total_score = self._calculate_total_score(
            completion, row_consistency, symmetry_score, pattern_validity,
            abutment_alignment, crossings, congestion_variance, perimeter_area_ratio
        )
        
        return LayoutMetrics(
            completion=completion,
            row_consistency=row_consistency,
            symmetry_score=symmetry_score,
            pattern_validity=pattern_validity,
            crossings=crossings,
            congestion_variance=congestion_variance,
            avg_connection_distance=avg_connection_distance,
            perimeter_area_ratio=perimeter_area_ratio,
            abutment_alignment=abutment_alignment,
            aspect_ratio=aspect_ratio,
            utilization=utilization,
            rail_alignment=rail_alignment,
            matching_accuracy=matching_accuracy,
            device_spacing=device_spacing,
            hierarchy_preservation=hierarchy_preservation,
            overlap_count=overlap_count,
            row_violations=row_violations,
            spacing_violations=spacing_violations,
            boundary_violations=boundary_violations,
            analog_score=analog_score,
            total_score=total_score
        )
    
    def _calculate_completion(self, layout_grid: LayoutGrid, target_components: int) -> float:
        """Calculate completion fraction."""
        if target_components == 0:
            return 1.0
        return len(layout_grid.placements) / target_components
    
    def _calculate_symmetry_score(self, layout_grid: LayoutGrid, circuit: nx.Graph) -> float:
        """Calculate symmetry score for matched components."""
        if not circuit:
            return 1.0
        
        total_pairs = 0
        symmetric_pairs = 0
        
        for node_id in circuit.nodes():
            if node_id not in layout_grid.placements:
                continue
                
            matched_comp = circuit.nodes[node_id].get('matched_component', -1)
            if (matched_comp != -1 and matched_comp in layout_grid.placements and 
                node_id < matched_comp):  # Count each pair once
                
                total_pairs += 1
                
                if self._check_pair_symmetry(layout_grid, node_id, matched_comp):
                    symmetric_pairs += 1
        
        return symmetric_pairs / max(total_pairs, 1)
    
    def _check_pair_symmetry(self, layout_grid: LayoutGrid, comp1_id: int, comp2_id: int) -> bool:
        """Check if two components are placed symmetrically."""
        p1 = layout_grid.placements[comp1_id]
        p2 = layout_grid.placements[comp2_id]
        
        # Check for various symmetry types
        
        # Horizontal symmetry (mirrored across vertical axis)
        grid_center_x = layout_grid.grid_width // 2
        expected_x2 = 2 * grid_center_x - p1.x - p1.width
        if abs(p2.x - expected_x2) <= 1 and abs(p1.y - p2.y) <= 1:
            return True
        
        # Vertical symmetry (mirrored across horizontal axis)  
        grid_center_y = layout_grid.grid_height // 2
        expected_y2 = 2 * grid_center_y - p1.y - p1.height
        if abs(p2.y - expected_y2) <= 1 and abs(p1.x - p2.x) <= 1:
            return True
        
        # Row-aligned symmetry (same row, different columns)
        if abs(p1.y - p2.y) <= 1 and abs(p1.x - p2.x) >= 3:
            return True
        
        return False
    
    def _calculate_pattern_validity(self, layout_grid: LayoutGrid) -> float:
        """Calculate pattern validity score for locked groups."""
        if not layout_grid.locked_groups:
            return 1.0
        
        valid_patterns = 0
        total_patterns = len(layout_grid.locked_groups)
        
        for group_id, group_info in layout_grid.locked_groups.items():
            if self._validate_pattern(layout_grid, group_info):
                valid_patterns += 1
        
        return valid_patterns / total_patterns
    
    def _validate_pattern(self, layout_grid: LayoutGrid, group_info: Dict[str, Any]) -> bool:
        """Validate a specific pattern."""
        pattern_type = group_info['pattern_type']
        component_ids = group_info['component_ids']
        
        # Check all components are placed
        for comp_id in component_ids:
            if comp_id not in layout_grid.placements:
                return False
        
        if pattern_type == 'mirror':
            return self._validate_mirror_pattern(layout_grid, component_ids)
        elif pattern_type == 'common_centroid':
            return self._validate_common_centroid_pattern(layout_grid, component_ids)
        elif pattern_type == 'interdigitated':
            return self._validate_interdigitated_pattern(layout_grid, component_ids)
        
        return True
    
    def _validate_mirror_pattern(self, layout_grid: LayoutGrid, component_ids: List[int]) -> bool:
        """Validate mirror pattern."""
        if len(component_ids) != 2:
            return False
        
        return self._check_pair_symmetry(layout_grid, component_ids[0], component_ids[1])
    
    def _validate_common_centroid_pattern(self, layout_grid: LayoutGrid, component_ids: List[int]) -> bool:
        """Validate common-centroid pattern."""
        if len(component_ids) < 2:
            return False
        
        # Calculate centroid
        total_x, total_y = 0, 0
        for comp_id in component_ids:
            p = layout_grid.placements[comp_id]
            total_x += p.x + p.width // 2
            total_y += p.y + p.height // 2
        
        centroid_x = total_x / len(component_ids)
        centroid_y = total_y / len(component_ids)
        
        # Check if components are arranged around centroid
        # For basic validation, check if distances from centroid are balanced
        distances = []
        for comp_id in component_ids:
            p = layout_grid.placements[comp_id]
            comp_center_x = p.x + p.width // 2
            comp_center_y = p.y + p.height // 2
            dist = ((comp_center_x - centroid_x)**2 + (comp_center_y - centroid_y)**2)**0.5
            distances.append(dist)
        
        # Check if distances are roughly equal (within tolerance)
        avg_dist = sum(distances) / len(distances)
        tolerance = 2.0
        return all(abs(d - avg_dist) <= tolerance for d in distances)
    
    def _validate_interdigitated_pattern(self, layout_grid: LayoutGrid, component_ids: List[int]) -> bool:
        """Validate interdigitated pattern."""
        if len(component_ids) < 2:
            return False
        
        # For basic validation, check if components are in alternating arrangement
        # Sort by position
        positions = [(layout_grid.placements[comp_id].x, comp_id) for comp_id in component_ids]
        positions.sort()
        
        # Check alternating pattern (simplified)
        for i in range(1, len(positions)):
            prev_comp = positions[i-1][1]
            curr_comp = positions[i][1]
            # In true interdigitation, adjacent components should be different types
            # For now, just check they're not the same component
            if prev_comp == curr_comp:
                return False
        
        return True
    
    def _calculate_avg_connection_distance(self, layout_grid: LayoutGrid, connections: List[Tuple[int, int]]) -> float:
        """Calculate average Manhattan distance for connections."""
        if not connections:
            return 0.0
        
        total_distance = 0.0
        valid_connections = 0
        
        for comp1_id, comp2_id in connections:
            if comp1_id in layout_grid.placements and comp2_id in layout_grid.placements:
                p1 = layout_grid.placements[comp1_id]
                p2 = layout_grid.placements[comp2_id]
                
                # Calculate Manhattan distance between centers
                center1_x = p1.x + p1.width // 2
                center1_y = p1.y + p1.height // 2
                center2_x = p2.x + p2.width // 2
                center2_y = p2.y + p2.height // 2
                
                distance = abs(center1_x - center2_x) + abs(center1_y - center2_y)
                total_distance += distance
                valid_connections += 1
        
        return total_distance / max(valid_connections, 1)
    
    def _calculate_abutment_alignment(self, layout_grid: LayoutGrid) -> float:
        """Calculate abutment and alignment quality."""
        if len(layout_grid.placements) < 2:
            return 1.0
        
        abutment_score = 0.0
        alignment_score = 0.0
        total_pairs = 0
        
        placements = list(layout_grid.placements.values())
        
        for i, p1 in enumerate(placements):
            for p2 in placements[i+1:]:
                total_pairs += 1
                
                # Check for abutment (adjacent with shared edge)
                if self._check_abutment(p1, p2):
                    abutment_score += 1.0
                
                # Check for alignment (same row/column)
                if self._check_alignment(p1, p2):
                    alignment_score += 1.0
        
        if total_pairs == 0:
            return 1.0
        
        return (abutment_score + alignment_score) / (2 * total_pairs)
    
    def _check_abutment(self, p1, p2) -> bool:
        """Check if two placements are abutted (sharing an edge)."""
        # Horizontal abutment
        if (p1.y == p2.y and p1.height == p2.height and
            (p1.x + p1.width == p2.x or p2.x + p2.width == p1.x)):
            return True
        
        # Vertical abutment
        if (p1.x == p2.x and p1.width == p2.width and
            (p1.y + p1.height == p2.y or p2.y + p2.height == p1.y)):
            return True
        
        return False
    
    def _check_alignment(self, p1, p2) -> bool:
        """Check if two placements are aligned."""
        # Row alignment (same Y coordinate)
        if abs(p1.y - p2.y) <= 1:
            return True
        
        # Column alignment (same X coordinate)
        if abs(p1.x - p2.x) <= 1:
            return True
        
        return False
    
    def _calculate_aspect_ratio(self, layout_grid: LayoutGrid) -> float:
        """Calculate aspect ratio of bounding box."""
        if not layout_grid.placements:
            return 1.0
        
        min_x, min_y, max_x, max_y = layout_grid.get_bounding_box()
        width = max_x - min_x
        height = max_y - min_y
        
        if height == 0:
            return float('inf')
        
        return width / height
    
    def _calculate_utilization(self, layout_grid: LayoutGrid) -> float:
        """Calculate area utilization."""
        if not layout_grid.placements:
            return 0.0
        
        # Total component area
        total_component_area = sum(p.width * p.height for p in layout_grid.placements.values())
        
        # Bounding box area
        min_x, min_y, max_x, max_y = layout_grid.get_bounding_box()
        bounding_area = (max_x - min_x) * (max_y - min_y)
        
        if bounding_area == 0:
            return 0.0
        
        return total_component_area / bounding_area
    
    def _calculate_rail_alignment(self, layout_grid: LayoutGrid) -> float:
        """Calculate power rail alignment quality."""
        if not layout_grid.placements:
            return 1.0
        
        total_score = 0.0
        scored_components = 0
        
        for comp_id, placement in layout_grid.placements.items():
            if placement.component_type in [ComponentType.NMOS, ComponentType.PMOS]:
                # Find the row this component is in
                row = layout_grid.get_row_for_position(placement.y)
                if row:
                    # Score based on how close component is to ideal row position
                    if placement.component_type == ComponentType.PMOS and row.row_type == RowType.PMOS_ROW:
                        # PMOS in PMOS row - check distance to VDD rail (top of row)
                        distance_to_rail = abs(placement.y - row.y_start)
                        max_distance = row.y_end - row.y_start
                        score = 1.0 - (distance_to_rail / max(max_distance, 1))
                        total_score += score
                        scored_components += 1
                    elif placement.component_type == ComponentType.NMOS and row.row_type == RowType.NMOS_ROW:
                        # NMOS in NMOS row - check distance to VSS rail (bottom of row)
                        distance_to_rail = abs((placement.y + placement.height) - row.y_end)
                        max_distance = row.y_end - row.y_start
                        score = 1.0 - (distance_to_rail / max(max_distance, 1))
                        total_score += score
                        scored_components += 1
        
        return total_score / max(scored_components, 1)
    
    def _calculate_matching_accuracy(self, layout_grid: LayoutGrid, circuit: nx.Graph) -> float:
        """Calculate precision of matched device placement."""
        if not circuit:
            return 1.0
        
        total_accuracy = 0.0
        matched_pairs = 0
        
        for node_id in circuit.nodes():
            if node_id not in layout_grid.placements:
                continue
            
            matched_comp = circuit.nodes[node_id].get('matched_component', -1)
            if (matched_comp != -1 and matched_comp in layout_grid.placements and 
                node_id < matched_comp):  # Count each pair once
                
                matched_pairs += 1
                p1 = layout_grid.placements[node_id]
                p2 = layout_grid.placements[matched_comp]
                
                # Calculate matching accuracy based on position precision
                # Perfect match would have exact symmetry
                grid_center_x = layout_grid.grid_width // 2
                grid_center_y = layout_grid.grid_height // 2
                
                # Check horizontal symmetry accuracy
                expected_x2 = 2 * grid_center_x - p1.x - p1.width
                x_error = abs(p2.x - expected_x2)
                y_error = abs(p2.y - p1.y)
                
                # Convert errors to accuracy (0-1 scale)
                max_error = max(layout_grid.grid_width, layout_grid.grid_height)
                accuracy = 1.0 - min(1.0, (x_error + y_error) / max_error)
                total_accuracy += accuracy
        
        return total_accuracy / max(matched_pairs, 1)
    
    def _calculate_device_spacing(self, layout_grid: LayoutGrid) -> float:
        """Calculate average spacing between devices."""
        if len(layout_grid.placements) < 2:
            return 0.0
        
        total_spacing = 0.0
        spacing_count = 0
        
        placements = list(layout_grid.placements.values())
        
        for i, p1 in enumerate(placements):
            for p2 in placements[i+1:]:
                # Calculate minimum distance between rectangles
                min_distance = self._calculate_min_distance(p1, p2)
                total_spacing += min_distance
                spacing_count += 1
        
        return total_spacing / max(spacing_count, 1)
    
    def _calculate_min_distance(self, p1, p2) -> float:
        """Calculate minimum distance between two rectangular placements."""
        # Calculate closest points between rectangles
        left1, right1 = p1.x, p1.x + p1.width
        top1, bottom1 = p1.y, p1.y + p1.height
        left2, right2 = p2.x, p2.x + p2.width
        top2, bottom2 = p2.y, p2.y + p2.height
        
        # Check if rectangles overlap
        if (left1 < right2 and right1 > left2 and top1 < bottom2 and bottom1 > top2):
            return 0.0  # Overlapping
        
        # Calculate minimum distance
        dx = max(0, max(left1 - right2, left2 - right1))
        dy = max(0, max(top1 - bottom2, top2 - bottom1))
        
        return (dx * dx + dy * dy) ** 0.5
    
    def _calculate_hierarchy_preservation(self, layout_grid: LayoutGrid, circuit: nx.Graph) -> float:
        """Calculate how well the layout preserves hierarchical organization."""
        # For now, return a basic score based on clustering
        # TODO: Implement proper hierarchy analysis
        return 1.0
    
    def _count_row_violations(self, layout_grid: LayoutGrid) -> int:
        """Count components placed in wrong rows."""
        violations = 0
        
        for comp_id, placement in layout_grid.placements.items():
            for y in range(placement.y, placement.y + placement.height):
                row = layout_grid.get_row_for_position(y)
                if row and placement.component_type not in row.allowed_components:
                    violations += 1
                    break  # Count each component violation once
        
        return violations
    
    def _count_spacing_violations(self, layout_grid: LayoutGrid, min_spacing: int = 1) -> int:
        """Count minimum spacing violations."""
        violations = 0
        placements = list(layout_grid.placements.values())
        
        for i, p1 in enumerate(placements):
            for p2 in placements[i+1:]:
                min_distance = self._calculate_min_distance(p1, p2)
                if min_distance < min_spacing and min_distance > 0:
                    violations += 1
        
        return violations
    
    def _count_boundary_violations(self, layout_grid: LayoutGrid) -> int:
        """Count out-of-bounds violations."""
        violations = 0
        
        for placement in layout_grid.placements.values():
            if (placement.x < 0 or placement.y < 0 or
                placement.x + placement.width > layout_grid.grid_width or
                placement.y + placement.height > layout_grid.grid_height):
                violations += 1
        
        return violations
    
    def _calculate_analog_score(self, row_consistency: float, symmetry_score: float,
                               rail_alignment: float, matching_accuracy: float) -> float:
        """Calculate overall analog-friendliness score."""
        return (row_consistency + symmetry_score + rail_alignment + matching_accuracy) / 4.0
    
    def _calculate_total_score(self, completion: float, row_consistency: float,
                              symmetry_score: float, pattern_validity: float,
                              abutment_alignment: float, crossings: int,
                              congestion_variance: float, perimeter_area_ratio: float) -> float:
        """Calculate weighted total score."""
        score = 0.0
        
        score += self.weights['completion'] * completion
        score += self.weights['row_consistency'] * row_consistency
        score += self.weights['symmetry'] * symmetry_score
        score += self.weights['pattern_validity'] * pattern_validity
        score += self.weights['abutment_alignment'] * abutment_alignment
        
        # Penalty terms (negative weights)
        normalized_crossings = min(crossings / 10.0, 1.0)  # Normalize crossings
        score += self.weights['crossings'] * normalized_crossings
        
        normalized_congestion = min(congestion_variance / 10.0, 1.0)
        score += self.weights['congestion_variance'] * normalized_congestion
        
        normalized_perimeter = min(perimeter_area_ratio / 2.0, 1.0)
        score += self.weights['perimeter_area_ratio'] * normalized_perimeter
        
        return score


# Convenience functions
def calculate_layout_metrics(layout_grid: LayoutGrid, 
                           circuit: nx.Graph,
                           target_components: Optional[int] = None,
                           custom_weights: Optional[Dict[str, float]] = None) -> LayoutMetrics:
    """
    Convenience function to calculate all layout metrics.
    
    Args:
        layout_grid: Current layout state
        circuit: Circuit connectivity graph
        target_components: Expected number of components
        custom_weights: Custom metric weights
        
    Returns:
        Complete LayoutMetrics object
    """
    if custom_weights:
        calculator = MetricsCalculator(**custom_weights)
    else:
        calculator = MetricsCalculator()
    
    return calculator.calculate_all_metrics(layout_grid, circuit, target_components)


def get_metrics_summary(metrics: LayoutMetrics) -> Dict[str, float]:
    """Get a summary of key metrics for logging."""
    return {
        'completion': metrics.completion,
        'row_consistency': metrics.row_consistency,
        'symmetry_score': metrics.symmetry_score,
        'crossings': float(metrics.crossings),
        'analog_score': metrics.analog_score,
        'total_score': metrics.total_score,
        'violations': float(metrics.overlap_count + metrics.row_violations + 
                          metrics.spacing_violations + metrics.boundary_violations)
    }


def meets_acceptance_criteria(metrics: LayoutMetrics, 
                            min_completion: float = 1.0,
                            min_row_consistency: float = 0.95,
                            min_symmetry: float = 0.90,
                            min_pattern_validity: float = 0.90,
                            max_violations: int = 0) -> bool:
    """
    Check if layout meets acceptance criteria.
    
    Args:
        metrics: Calculated layout metrics
        min_*: Minimum thresholds for key metrics
        max_violations: Maximum allowed violations
        
    Returns:
        True if all criteria are met
    """
    total_violations = (metrics.overlap_count + metrics.row_violations + 
                       metrics.spacing_violations + metrics.boundary_violations)
    
    return (metrics.completion >= min_completion and
            metrics.row_consistency >= min_row_consistency and
            metrics.symmetry_score >= min_symmetry and
            metrics.pattern_validity >= min_pattern_validity and
            total_violations <= max_violations)