#!/usr/bin/env python3
"""
Row snap and slide legalizer for analog layout with locked pattern preservation.
Ensures components are properly aligned to rows and resolves overlaps while maintaining 
symmetry, common-centroid, and interdigitation patterns.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum
import copy

from env.layout_grid import LayoutGrid, ComponentType, RowType, ComponentPlacement


class LegalizationMode(Enum):
    """Legalization modes with different priorities."""
    PRESERVE_PATTERNS = 0    # Prioritize pattern preservation
    MINIMIZE_MOVEMENT = 1    # Minimize component movement
    OPTIMIZE_DENSITY = 2     # Optimize overall density


@dataclass
class LegalizationResult:
    """Result of legalization process."""
    success: bool
    movements: Dict[int, Tuple[int, int]]  # comp_id -> (old_pos, new_pos) 
    violations_fixed: int
    patterns_preserved: int
    total_displacement: float
    error_message: Optional[str] = None


class RowSnapLegalizer:
    """
    Legalizer that snaps components to appropriate rows and resolves overlaps
    while preserving locked symmetry/common-centroid patterns.
    """
    
    def __init__(self, 
                 min_spacing: int = 1,
                 max_iterations: int = 100,
                 preservation_weight: float = 10.0):
        """
        Initialize legalizer.
        
        Args:
            min_spacing: Minimum spacing between components
            max_iterations: Maximum legalization iterations
            preservation_weight: Weight for pattern preservation vs. movement
        """
        self.min_spacing = min_spacing
        self.max_iterations = max_iterations
        self.preservation_weight = preservation_weight
    
    def legalize(self, 
                layout_grid: LayoutGrid,
                mode: LegalizationMode = LegalizationMode.PRESERVE_PATTERNS) -> LegalizationResult:
        """
        Legalize the current layout.
        
        Args:
            layout_grid: Layout grid to legalize
            mode: Legalization mode
            
        Returns:
            LegalizationResult with success status and movements
        """
        movements = {}
        violations_fixed = 0
        initial_violations = self._count_violations(layout_grid)
        
        if initial_violations == 0:
            return LegalizationResult(
                success=True,
                movements=movements,
                violations_fixed=0,
                patterns_preserved=len(layout_grid.locked_groups),
                total_displacement=0.0
            )
        
        # Create working copy
        original_placements = copy.deepcopy(layout_grid.placements)
        
        try:
            # Phase 1: Row snapping
            row_movements = self._snap_to_rows(layout_grid)
            movements.update(row_movements)
            
            # Phase 2: Overlap resolution with pattern preservation
            overlap_movements = self._resolve_overlaps(layout_grid, mode)
            movements.update(overlap_movements)
            
            # Phase 3: Pattern validation and repair
            if mode == LegalizationMode.PRESERVE_PATTERNS:
                pattern_movements = self._repair_patterns(layout_grid)
                movements.update(pattern_movements)
            
            # Calculate final metrics
            final_violations = self._count_violations(layout_grid)
            violations_fixed = initial_violations - final_violations
            patterns_preserved = self._count_preserved_patterns(layout_grid)
            total_displacement = self._calculate_total_displacement(movements)
            
            success = final_violations == 0
            
            return LegalizationResult(
                success=success,
                movements=movements,
                violations_fixed=violations_fixed,
                patterns_preserved=patterns_preserved,
                total_displacement=total_displacement
            )
            
        except Exception as e:
            # Restore original placements on failure
            layout_grid.placements = original_placements
            self._restore_grid_from_placements(layout_grid)
            
            return LegalizationResult(
                success=False,
                movements={},
                violations_fixed=0,
                patterns_preserved=0,
                total_displacement=0.0,
                error_message=str(e)
            )
    
    def _snap_to_rows(self, layout_grid: LayoutGrid) -> Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Snap components to appropriate row boundaries.
        
        Returns:
            Dictionary of movements: comp_id -> ((old_x, old_y), (new_x, new_y))
        """
        movements = {}
        
        for comp_id, placement in layout_grid.placements.items():
            # Find appropriate row for this component type
            target_row = self._find_target_row(layout_grid, placement.component_type)
            
            if target_row is None:
                continue
                
            # Calculate snap position
            old_pos = (placement.x, placement.y)
            new_y = self._snap_y_to_row(placement, target_row)
            new_pos = (placement.x, new_y)
            
            if old_pos != new_pos:
                # Update placement
                layout_grid.remove_component(comp_id)
                success = layout_grid.place_component(
                    comp_id, new_pos[0], new_pos[1],
                    placement.width, placement.height,
                    placement.orientation, placement.component_type
                )
                
                if success:
                    movements[comp_id] = (old_pos, new_pos)
                else:
                    # Restore if snap failed
                    layout_grid.place_component(
                        comp_id, old_pos[0], old_pos[1],
                        placement.width, placement.height,
                        placement.orientation, placement.component_type
                    )
        
        return movements
    
    def _resolve_overlaps(self, 
                         layout_grid: LayoutGrid, 
                         mode: LegalizationMode) -> Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Resolve overlapping components by sliding them apart.
        
        Returns:
            Dictionary of movements
        """
        movements = {}
        max_iterations = self.max_iterations
        
        for iteration in range(max_iterations):
            overlaps = self._find_overlaps(layout_grid)
            
            if not overlaps:
                break  # No more overlaps
            
            # Process overlaps in order of severity
            overlaps.sort(key=lambda x: x[2], reverse=True)  # Sort by overlap area
            
            for comp1_id, comp2_id, overlap_area in overlaps:
                if self._has_overlap(layout_grid, comp1_id, comp2_id):
                    movement = self._resolve_single_overlap(
                        layout_grid, comp1_id, comp2_id, mode
                    )
                    if movement:
                        movements.update(movement)
        
        return movements
    
    def _repair_patterns(self, layout_grid: LayoutGrid) -> Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Repair broken symmetry/common-centroid patterns.
        
        Returns:
            Dictionary of movements
        """
        movements = {}
        
        for group_id, group_info in layout_grid.locked_groups.items():
            pattern_type = group_info['pattern_type']
            component_ids = group_info['component_ids']
            
            if pattern_type == 'mirror':
                repair_movements = self._repair_mirror_pattern(layout_grid, component_ids, group_info)
            elif pattern_type == 'common_centroid':
                repair_movements = self._repair_common_centroid_pattern(layout_grid, component_ids, group_info)
            elif pattern_type == 'interdigitated':
                repair_movements = self._repair_interdigitated_pattern(layout_grid, component_ids, group_info)
            else:
                repair_movements = {}
            
            movements.update(repair_movements)
        
        return movements
    
    def _find_target_row(self, layout_grid: LayoutGrid, comp_type: ComponentType) -> Optional[Any]:
        """Find the appropriate row for a component type."""
        for row in layout_grid.rows:
            if comp_type in row.allowed_components:
                return row
        return None
    
    def _snap_y_to_row(self, placement: ComponentPlacement, target_row) -> int:
        """Calculate Y coordinate to snap component to row."""
        # Snap to row center or boundary
        row_center = (target_row.y_start + target_row.y_end) // 2
        
        # Prefer to center the component in the row
        preferred_y = row_center - placement.height // 2
        
        # Ensure component fits within row bounds
        min_y = target_row.y_start
        max_y = target_row.y_end - placement.height
        
        return max(min_y, min(preferred_y, max_y))
    
    def _find_overlaps(self, layout_grid: LayoutGrid) -> List[Tuple[int, int, int]]:
        """
        Find all overlapping component pairs.
        
        Returns:
            List of (comp1_id, comp2_id, overlap_area) tuples
        """
        overlaps = []
        placed_components = list(layout_grid.placements.keys())
        
        for i, comp1_id in enumerate(placed_components):
            for comp2_id in placed_components[i+1:]:
                overlap_area = self._calculate_overlap_area(layout_grid, comp1_id, comp2_id)
                if overlap_area > 0:
                    overlaps.append((comp1_id, comp2_id, overlap_area))
        
        return overlaps
    
    def _calculate_overlap_area(self, layout_grid: LayoutGrid, comp1_id: int, comp2_id: int) -> int:
        """Calculate overlap area between two components."""
        if comp1_id not in layout_grid.placements or comp2_id not in layout_grid.placements:
            return 0
        
        p1 = layout_grid.placements[comp1_id]
        p2 = layout_grid.placements[comp2_id]
        
        # Calculate intersection rectangle
        left = max(p1.x, p2.x)
        right = min(p1.x + p1.width, p2.x + p2.width)
        top = max(p1.y, p2.y)
        bottom = min(p1.y + p1.height, p2.y + p2.height)
        
        if left < right and top < bottom:
            return (right - left) * (bottom - top)
        
        return 0
    
    def _has_overlap(self, layout_grid: LayoutGrid, comp1_id: int, comp2_id: int) -> bool:
        """Check if two components overlap."""
        return self._calculate_overlap_area(layout_grid, comp1_id, comp2_id) > 0
    
    def _resolve_single_overlap(self, 
                               layout_grid: LayoutGrid,
                               comp1_id: int, 
                               comp2_id: int,
                               mode: LegalizationMode) -> Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Resolve overlap between two specific components.
        
        Returns:
            Movement dictionary
        """
        p1 = layout_grid.placements[comp1_id]
        p2 = layout_grid.placements[comp2_id]
        
        # Determine which component to move based on mode and constraints
        if mode == LegalizationMode.PRESERVE_PATTERNS:
            # Prefer moving non-locked components
            if p1.locked and not p2.locked:
                moving_id = comp2_id
            elif p2.locked and not p1.locked:
                moving_id = comp1_id
            else:
                # Both locked or both unlocked - move the one that's easier
                moving_id = comp2_id if comp2_id > comp1_id else comp1_id
        else:
            # Default: move the second component
            moving_id = comp2_id
        
        # Calculate movement direction
        moving_placement = layout_grid.placements[moving_id]
        other_id = comp1_id if moving_id == comp2_id else comp2_id
        other_placement = layout_grid.placements[other_id]
        
        # Try horizontal movement first (preferred for row alignment)
        new_x = other_placement.x + other_placement.width + self.min_spacing
        if new_x + moving_placement.width <= layout_grid.grid_width:
            new_pos = (new_x, moving_placement.y)
        else:
            # Try left movement
            new_x = other_placement.x - moving_placement.width - self.min_spacing
            if new_x >= 0:
                new_pos = (new_x, moving_placement.y)
            else:
                # Try vertical movement
                new_y = other_placement.y + other_placement.height + self.min_spacing
                if new_y + moving_placement.height <= layout_grid.grid_height:
                    new_pos = (moving_placement.x, new_y)
                else:
                    new_y = other_placement.y - moving_placement.height - self.min_spacing
                    if new_y >= 0:
                        new_pos = (moving_placement.x, new_y)
                    else:
                        return {}  # Can't resolve overlap
        
        # Attempt the move
        old_pos = (moving_placement.x, moving_placement.y)
        layout_grid.remove_component(moving_id)
        
        success = layout_grid.place_component(
            moving_id, new_pos[0], new_pos[1],
            moving_placement.width, moving_placement.height,
            moving_placement.orientation, moving_placement.component_type
        )
        
        if success:
            return {moving_id: (old_pos, new_pos)}
        else:
            # Restore original position
            layout_grid.place_component(
                moving_id, old_pos[0], old_pos[1],
                moving_placement.width, moving_placement.height,
                moving_placement.orientation, moving_placement.component_type
            )
            return {}
    
    def _repair_mirror_pattern(self, 
                              layout_grid: LayoutGrid,
                              component_ids: List[int],
                              group_info: Dict[str, Any]) -> Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Repair a broken mirror pattern."""
        movements = {}
        
        if len(component_ids) != 2:
            return movements
        
        comp1_id, comp2_id = component_ids
        if comp1_id not in layout_grid.placements or comp2_id not in layout_grid.placements:
            return movements
        
        p1 = layout_grid.placements[comp1_id]
        p2 = layout_grid.placements[comp2_id]
        
        # Calculate ideal mirror position for comp2 based on comp1
        grid_center_x = layout_grid.grid_width // 2
        ideal_x = 2 * grid_center_x - p1.x - p1.width
        ideal_y = p1.y
        
        # Check if move is needed and valid
        if (p2.x, p2.y) != (ideal_x, ideal_y):
            old_pos = (p2.x, p2.y)
            
            if (0 <= ideal_x < layout_grid.grid_width - p2.width and
                0 <= ideal_y < layout_grid.grid_height - p2.height):
                
                layout_grid.remove_component(comp2_id)
                success = layout_grid.place_component(
                    comp2_id, ideal_x, ideal_y,
                    p2.width, p2.height, p2.orientation, p2.component_type
                )
                
                if success:
                    movements[comp2_id] = (old_pos, (ideal_x, ideal_y))
                else:
                    # Restore if failed
                    layout_grid.place_component(
                        comp2_id, old_pos[0], old_pos[1],
                        p2.width, p2.height, p2.orientation, p2.component_type
                    )
        
        return movements
    
    def _repair_common_centroid_pattern(self,
                                       layout_grid: LayoutGrid,
                                       component_ids: List[int],
                                       group_info: Dict[str, Any]) -> Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Repair a broken common-centroid pattern."""
        movements = {}
        
        # For now, implement basic centroid alignment
        # TODO: Implement full common-centroid patterns (ABBA, etc.)
        
        if len(component_ids) < 2:
            return movements
        
        # Calculate centroid of current positions
        total_x, total_y = 0, 0
        valid_components = []
        
        for comp_id in component_ids:
            if comp_id in layout_grid.placements:
                p = layout_grid.placements[comp_id]
                total_x += p.x + p.width // 2
                total_y += p.y + p.height // 2
                valid_components.append(comp_id)
        
        if len(valid_components) < 2:
            return movements
        
        centroid_x = total_x // len(valid_components)
        centroid_y = total_y // len(valid_components)
        
        # For basic implementation, ensure components are arranged symmetrically around centroid
        # This is a simplified version - full CC patterns would be more complex
        
        return movements
    
    def _repair_interdigitated_pattern(self,
                                     layout_grid: LayoutGrid,
                                     component_ids: List[int],
                                     group_info: Dict[str, Any]) -> Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Repair a broken interdigitated pattern."""
        movements = {}
        
        # For now, implement basic alternating arrangement
        # TODO: Implement full interdigitation patterns
        
        return movements
    
    def _count_violations(self, layout_grid: LayoutGrid) -> int:
        """Count total layout violations."""
        violations = 0
        
        # Count overlaps
        violations += layout_grid.get_overlap_count()
        
        # Count row violations
        for comp_id, placement in layout_grid.placements.items():
            for y in range(placement.y, placement.y + placement.height):
                row = layout_grid.get_row_for_position(y)
                if row and placement.component_type not in row.allowed_components:
                    violations += 1
        
        return violations
    
    def _count_preserved_patterns(self, layout_grid: LayoutGrid) -> int:
        """Count number of preserved patterns."""
        preserved = 0
        
        for group_id, group_info in layout_grid.locked_groups.items():
            if self._validate_pattern_preservation(layout_grid, group_info):
                preserved += 1
        
        return preserved
    
    def _validate_pattern_preservation(self, layout_grid: LayoutGrid, group_info: Dict[str, Any]) -> bool:
        """Validate that a pattern is still preserved after legalization."""
        # TODO: Implement specific pattern validation
        return True
    
    def _calculate_total_displacement(self, movements: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]]) -> float:
        """Calculate total displacement from movements."""
        total_displacement = 0.0
        
        for comp_id, (old_pos, new_pos) in movements.items():
            dx = new_pos[0] - old_pos[0]
            dy = new_pos[1] - old_pos[1]
            displacement = (dx * dx + dy * dy) ** 0.5
            total_displacement += displacement
        
        return total_displacement
    
    def _restore_grid_from_placements(self, layout_grid: LayoutGrid):
        """Restore grid state from placement dictionary."""
        layout_grid.grid.fill(0)
        
        for comp_id, placement in layout_grid.placements.items():
            for y in range(placement.y, placement.y + placement.height):
                for x in range(placement.x, placement.x + placement.width):
                    layout_grid.grid[y, x] = comp_id


# Utility functions for integration
def legalize_layout(layout_grid: LayoutGrid, 
                   mode: LegalizationMode = LegalizationMode.PRESERVE_PATTERNS,
                   **kwargs) -> LegalizationResult:
    """
    Convenience function to legalize a layout.
    
    Args:
        layout_grid: Layout grid to legalize
        mode: Legalization mode
        **kwargs: Additional arguments for legalizer
        
    Returns:
        LegalizationResult
    """
    legalizer = RowSnapLegalizer(**kwargs)
    return legalizer.legalize(layout_grid, mode)


def create_legalizer_config(min_spacing: int = 1,
                           max_iterations: int = 100,
                           preservation_weight: float = 10.0) -> Dict[str, Any]:
    """Create legalizer configuration dictionary."""
    return {
        'min_spacing': min_spacing,
        'max_iterations': max_iterations,
        'preservation_weight': preservation_weight
    }