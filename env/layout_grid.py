#!/usr/bin/env python3
"""
Layout grid with PFET/NFET row partitions and analog-friendly placement constraints.
Provides grid management, row discipline, overlap detection, and crossing analysis.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from enum import Enum
from dataclasses import dataclass
import math


class ComponentType(Enum):
    """Component types for analog layout."""
    NMOS = 0
    PMOS = 1
    RESISTOR = 2
    CAPACITOR = 3
    INDUCTOR = 4
    CURRENT_SOURCE = 5
    VOLTAGE_SOURCE = 6
    SUBCIRCUIT = 7
    OTHER = 8


class RowType(Enum):
    """Row types for analog layout discipline."""
    PMOS_ROW = 0      # Near VDD
    NMOS_ROW = 1      # Near VSS
    MIXED_ROW = 2     # For passive components
    

@dataclass
class ComponentPlacement:
    """Represents a placed component on the grid."""
    component_id: int
    x: int
    y: int
    width: int
    height: int
    orientation: int  # 0, 90, 180, 270 degrees
    component_type: ComponentType
    locked: bool = False
    group_id: Optional[int] = None  # For matched/mirrored groups


@dataclass  
class RowPartition:
    """Represents a row partition with type constraints."""
    row_id: int
    y_start: int
    y_end: int
    row_type: RowType
    allowed_components: Set[ComponentType]
    

class LayoutGrid:
    """
    Grid-based layout manager with analog-friendly row partitions.
    Supports PFET/NFET row discipline, overlap detection, and crossing analysis.
    """
    
    def __init__(self, 
                 grid_width: int = 64, 
                 grid_height: int = 64,
                 pmos_rows: int = 3,
                 nmos_rows: int = 3,
                 mixed_rows: int = 2):
        """
        Initialize layout grid with row partitions.
        
        Args:
            grid_width: Grid width in units
            grid_height: Grid height in units  
            pmos_rows: Number of PMOS-only rows (top)
            nmos_rows: Number of NMOS-only rows (bottom)
            mixed_rows: Number of mixed rows (middle) for passives
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        
        # Initialize empty grid (0 = empty, >0 = component_id)
        self.grid = np.zeros((grid_height, grid_width), dtype=int)
        
        # Component placements
        self.placements: Dict[int, ComponentPlacement] = {}
        
        # Row partitions
        self.rows = self._create_row_partitions(pmos_rows, nmos_rows, mixed_rows)
        
        # Locked groups for symmetry/common-centroid preservation
        self.locked_groups: Dict[int, Dict[str, Any]] = {}
        
        # Crossing analysis grid (coarser for performance)
        self.crossing_grid_size = 16
        self.crossing_grid = np.zeros((self.crossing_grid_size, self.crossing_grid_size), dtype=int)
        
    def _create_row_partitions(self, pmos_rows: int, nmos_rows: int, mixed_rows: int) -> List[RowPartition]:
        """Create row partitions with analog discipline."""
        rows = []
        total_rows = pmos_rows + nmos_rows + mixed_rows
        row_height = self.grid_height // total_rows
        
        current_y = 0
        
        # PMOS rows at top (near VDD)
        for i in range(pmos_rows):
            y_end = min(current_y + row_height, self.grid_height)
            rows.append(RowPartition(
                row_id=i,
                y_start=current_y,
                y_end=y_end,
                row_type=RowType.PMOS_ROW,
                allowed_components={ComponentType.PMOS, ComponentType.SUBCIRCUIT}
            ))
            current_y = y_end
            
        # Mixed rows in middle  
        for i in range(mixed_rows):
            y_end = min(current_y + row_height, self.grid_height)
            rows.append(RowPartition(
                row_id=pmos_rows + i,
                y_start=current_y,
                y_end=y_end,
                row_type=RowType.MIXED_ROW,
                allowed_components={ComponentType.RESISTOR, ComponentType.CAPACITOR, 
                                 ComponentType.INDUCTOR, ComponentType.CURRENT_SOURCE,
                                 ComponentType.VOLTAGE_SOURCE, ComponentType.OTHER}
            ))
            current_y = y_end
            
        # NMOS rows at bottom (near VSS) 
        for i in range(nmos_rows):
            y_end = min(current_y + row_height, self.grid_height)
            rows.append(RowPartition(
                row_id=pmos_rows + mixed_rows + i,
                y_start=current_y,
                y_end=y_end,
                row_type=RowType.NMOS_ROW,
                allowed_components={ComponentType.NMOS, ComponentType.SUBCIRCUIT}
            ))
            current_y = y_end
            
        return rows
    
    def get_row_for_position(self, y: int) -> Optional[RowPartition]:
        """Get the row partition containing the given y coordinate."""
        for row in self.rows:
            if row.y_start <= y < row.y_end:
                return row
        return None
    
    def is_valid_placement(self, 
                          component_id: int,
                          x: int, y: int, 
                          width: int, height: int,
                          component_type: ComponentType) -> Tuple[bool, str]:
        """
        Check if placement is valid according to analog constraints.
        
        Returns:
            (is_valid, error_message)
        """
        # Bounds check
        if x < 0 or y < 0 or x + width > self.grid_width or y + height > self.grid_height:
            return False, "Out of bounds"
            
        # Row discipline check
        for check_y in range(y, y + height):
            row = self.get_row_for_position(check_y)
            if row and component_type not in row.allowed_components:
                return False, f"Component type {component_type.name} not allowed in {row.row_type.name}"
        
        # Overlap check (excluding self)
        for check_y in range(y, y + height):
            for check_x in range(x, x + width):
                existing_id = self.grid[check_y, check_x]
                if existing_id != 0 and existing_id != component_id:
                    return False, f"Overlaps with component {existing_id}"
        
        # Locked group constraints
        if component_id in self.placements:
            current_placement = self.placements[component_id]
            if current_placement.group_id in self.locked_groups:
                if not self._check_locked_group_constraints(component_id, x, y, width, height):
                    return False, "Violates locked group constraints"
        
        return True, ""
    
    def place_component(self, 
                       component_id: int,
                       x: int, y: int,
                       width: int, height: int, 
                       orientation: int,
                       component_type: ComponentType) -> bool:
        """
        Place a component on the grid.
        
        Returns:
            True if placement successful, False otherwise
        """
        # Adjust dimensions for orientation
        actual_width, actual_height = self._get_oriented_dimensions(width, height, orientation)
        
        # Validate placement
        is_valid, error = self.is_valid_placement(component_id, x, y, actual_width, actual_height, component_type)
        if not is_valid:
            return False
            
        # Remove previous placement if exists
        if component_id in self.placements:
            self.remove_component(component_id)
            
        # Place on grid
        for place_y in range(y, y + actual_height):
            for place_x in range(x, x + actual_width):
                self.grid[place_y, place_x] = component_id
                
        # Store placement info
        self.placements[component_id] = ComponentPlacement(
            component_id=component_id,
            x=x, y=y,
            width=actual_width,
            height=actual_height,
            orientation=orientation,
            component_type=component_type
        )
        
        return True
    
    def remove_component(self, component_id: int) -> bool:
        """Remove a component from the grid."""
        if component_id not in self.placements:
            return False
            
        placement = self.placements[component_id]
        
        # Clear from grid
        for clear_y in range(placement.y, placement.y + placement.height):
            for clear_x in range(placement.x, placement.x + placement.width):
                if self.grid[clear_y, clear_x] == component_id:
                    self.grid[clear_y, clear_x] = 0
                    
        # Remove from placements
        del self.placements[component_id]
        return True
    
    def get_overlap_count(self) -> int:
        """Count overlapping cell pairs (should be 0 for valid layouts)."""
        overlap_count = 0
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if self.grid[y, x] > 0:
                    # Check for multiple components claiming same cell
                    component_id = self.grid[y, x]
                    placement = self.placements.get(component_id)
                    if placement:
                        # Count how many placed components occupy this cell
                        occupants = 0
                        for other_id, other_placement in self.placements.items():
                            if (other_placement.x <= x < other_placement.x + other_placement.width and
                                other_placement.y <= y < other_placement.y + other_placement.height):
                                occupants += 1
                        if occupants > 1:
                            overlap_count += 1
        return overlap_count
    
    def count_crossings(self, connections: List[Tuple[int, int]]) -> int:
        """
        Count line segment crossings for the given connections.
        Uses a coarse grid overlay for performance.
        """
        self.crossing_grid.fill(0)
        crossing_count = 0
        
        # Scale factor from fine grid to coarse crossing grid
        scale_x = self.grid_width / self.crossing_grid_size
        scale_y = self.grid_height / self.crossing_grid_size
        
        segments = []
        
        # Convert connections to line segments
        for comp1_id, comp2_id in connections:
            if comp1_id in self.placements and comp2_id in self.placements:
                p1 = self.placements[comp1_id]
                p2 = self.placements[comp2_id]
                
                # Component centers
                center1 = (p1.x + p1.width / 2, p1.y + p1.height / 2)
                center2 = (p2.x + p2.width / 2, p2.y + p2.height / 2)
                
                # Scale to crossing grid
                seg1 = (center1[0] / scale_x, center1[1] / scale_y)
                seg2 = (center2[0] / scale_x, center2[1] / scale_y)
                
                segments.append((seg1, seg2))
        
        # Count intersections using simple O(n²) check
        for i, (seg1_start, seg1_end) in enumerate(segments):
            for j, (seg2_start, seg2_end) in enumerate(segments[i+1:], i+1):
                if self._segments_intersect(seg1_start, seg1_end, seg2_start, seg2_end):
                    crossing_count += 1
                    
        return crossing_count
    
    def get_congestion_variance(self, connections: List[Tuple[int, int]]) -> float:
        """
        Calculate congestion variance across the crossing grid.
        Lower variance indicates more uniform routing density.
        """
        self.crossing_grid.fill(0)
        
        # Scale factor from fine grid to coarse crossing grid
        scale_x = self.grid_width / self.crossing_grid_size  
        scale_y = self.grid_height / self.crossing_grid_size
        
        # Rasterize connections onto crossing grid
        for comp1_id, comp2_id in connections:
            if comp1_id in self.placements and comp2_id in self.placements:
                p1 = self.placements[comp1_id]
                p2 = self.placements[comp2_id]
                
                center1 = (p1.x + p1.width / 2, p1.y + p1.height / 2)
                center2 = (p2.x + p2.width / 2, p2.y + p2.height / 2)
                
                # Scale to crossing grid
                x1, y1 = int(center1[0] / scale_x), int(center1[1] / scale_y)
                x2, y2 = int(center2[0] / scale_x), int(center2[1] / scale_y)
                
                # Simple line rasterization
                self._rasterize_line(x1, y1, x2, y2)
        
        # Calculate variance
        flat_grid = self.crossing_grid.flatten()
        if len(flat_grid) > 0:
            return float(np.var(flat_grid))
        return 0.0
    
    def get_row_consistency_score(self) -> float:
        """
        Calculate fraction of components placed in correct rows.
        Returns value between 0.0 and 1.0.
        """
        if not self.placements:
            return 1.0
            
        correct_placements = 0
        total_placements = len(self.placements)
        
        for placement in self.placements.values():
            # Check if component is in correct row type
            component_in_correct_row = True
            
            for check_y in range(placement.y, placement.y + placement.height):
                row = self.get_row_for_position(check_y)
                if row and placement.component_type not in row.allowed_components:
                    component_in_correct_row = False
                    break
                    
            if component_in_correct_row:
                correct_placements += 1
                
        return correct_placements / total_placements
    
    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        """Get bounding box of all placed components (x_min, y_min, x_max, y_max)."""
        if not self.placements:
            return 0, 0, 0, 0
            
        min_x = min(p.x for p in self.placements.values())
        min_y = min(p.y for p in self.placements.values())
        max_x = max(p.x + p.width for p in self.placements.values())
        max_y = max(p.y + p.height for p in self.placements.values())
        
        return min_x, min_y, max_x, max_y
    
    def get_perimeter_area_ratio(self) -> float:
        """
        Calculate perimeter/area ratio of bounding box.
        Lower values indicate more rectangular, compact layouts.
        """
        min_x, min_y, max_x, max_y = self.get_bounding_box()
        
        if max_x <= min_x or max_y <= min_y:
            return 0.0
            
        width = max_x - min_x
        height = max_y - min_y
        area = width * height
        perimeter = 2 * (width + height)
        
        return perimeter / area if area > 0 else float('inf')
    
    def create_locked_group(self, 
                           group_id: int,
                           component_ids: List[int], 
                           pattern_type: str,
                           pattern_data: Dict[str, Any]):
        """
        Create a locked group for symmetry/common-centroid preservation.
        
        Args:
            group_id: Unique group identifier
            component_ids: List of component IDs in the group
            pattern_type: 'mirror', 'common_centroid', 'interdigitated'
            pattern_data: Pattern-specific data (axis, sequence, etc.)
        """
        self.locked_groups[group_id] = {
            'component_ids': component_ids,
            'pattern_type': pattern_type,
            'pattern_data': pattern_data
        }
        
        # Mark components as part of group
        for comp_id in component_ids:
            if comp_id in self.placements:
                self.placements[comp_id].group_id = group_id
                self.placements[comp_id].locked = True
    
    def _get_oriented_dimensions(self, width: int, height: int, orientation: int) -> Tuple[int, int]:
        """Get actual width/height after applying orientation."""
        if orientation in [90, 270]:
            return height, width
        return width, height
    
    def _check_locked_group_constraints(self, 
                                       component_id: int,
                                       new_x: int, new_y: int,
                                       new_width: int, new_height: int) -> bool:
        """Check if proposed placement violates locked group constraints."""
        placement = self.placements[component_id]
        if placement.group_id is None:
            return True
            
        group = self.locked_groups.get(placement.group_id)
        if not group:
            return True
            
        # For now, implement basic constraint checking
        # TODO: Add specific pattern validation for mirror/CC/interdigitated
        return True
    
    def _segments_intersect(self, 
                           p1: Tuple[float, float], q1: Tuple[float, float],
                           p2: Tuple[float, float], q2: Tuple[float, float]) -> bool:
        """Check if two line segments intersect."""
        def orientation(p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]) -> int:
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if abs(val) < 1e-10:
                return 0  # Collinear
            return 1 if val > 0 else 2  # Clockwise or counterclockwise
        
        def on_segment(p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]) -> bool:
            return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                    q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
        
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)
        
        # General case
        if o1 != o2 and o3 != o4:
            return True
            
        # Special cases (collinear points)
        if (o1 == 0 and on_segment(p1, p2, q1)) or \
           (o2 == 0 and on_segment(p1, q2, q1)) or \
           (o3 == 0 and on_segment(p2, p1, q2)) or \
           (o4 == 0 and on_segment(p2, q1, q2)):
            return True
            
        return False
    
    def _rasterize_line(self, x1: int, y1: int, x2: int, y2: int):
        """Rasterize line onto crossing grid using Bresenham's algorithm."""
        x1, y1 = max(0, min(self.crossing_grid_size-1, x1)), max(0, min(self.crossing_grid_size-1, y1))
        x2, y2 = max(0, min(self.crossing_grid_size-1, x2)), max(0, min(self.crossing_grid_size-1, y2))
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        x, y = x1, y1
        
        x_inc = 1 if x1 < x2 else -1
        y_inc = 1 if y1 < y2 else -1
        
        error = dx - dy
        
        while True:
            self.crossing_grid[y, x] += 1
            
            if x == x2 and y == y2:
                break
                
            e2 = 2 * error
            
            if e2 > -dy:
                error -= dy
                x += x_inc
                
            if e2 < dx:
                error += dx
                y += y_inc
    
    def reset(self):
        """Reset grid to empty state."""
        self.grid.fill(0)
        self.placements.clear()
        self.locked_groups.clear()
        self.crossing_grid.fill(0)
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current grid state for debugging/logging."""
        return {
            'num_components': len(self.placements),
            'row_consistency': self.get_row_consistency_score(),
            'overlap_count': self.get_overlap_count(),
            'bounding_box': self.get_bounding_box(),
            'perimeter_area_ratio': self.get_perimeter_area_ratio(),
            'locked_groups': len(self.locked_groups)
        }