#!/usr/bin/env python3
"""
Unit tests for layout grid with row partitions and analog constraints.
"""

import pytest
import numpy as np
from env.layout_grid import LayoutGrid, ComponentType, RowType


class TestLayoutGrid:
    """Test suite for LayoutGrid functionality."""
    
    def setup_method(self):
        """Set up test grid for each test."""
        self.grid = LayoutGrid(
            grid_width=32,
            grid_height=32, 
            pmos_rows=2,
            nmos_rows=2,
            mixed_rows=2
        )
    
    def test_row_partitions_created(self):
        """Test that row partitions are created correctly."""
        assert len(self.grid.rows) == 6  # 2+2+2 rows
        
        # Check PMOS rows are at top
        pmos_rows = [r for r in self.grid.rows if r.row_type == RowType.PMOS_ROW]
        assert len(pmos_rows) == 2
        assert all(r.y_start < 16 for r in pmos_rows)  # Top half
        
        # Check NMOS rows are at bottom  
        nmos_rows = [r for r in self.grid.rows if r.row_type == RowType.NMOS_ROW]
        assert len(nmos_rows) == 2
        assert all(r.y_start >= 16 for r in nmos_rows)  # Bottom half
        
        # Check mixed rows are in middle
        mixed_rows = [r for r in self.grid.rows if r.row_type == RowType.MIXED_ROW]
        assert len(mixed_rows) == 2
    
    def test_component_type_constraints(self):
        """Test that components can only be placed in appropriate rows."""
        # PMOS should be allowed in PMOS rows
        pmos_row = next(r for r in self.grid.rows if r.row_type == RowType.PMOS_ROW)
        valid, msg = self.grid.is_valid_placement(
            component_id=1,
            x=5, y=pmos_row.y_start + 1,
            width=2, height=2,
            component_type=ComponentType.PMOS
        )
        assert valid, f"PMOS should be valid in PMOS row: {msg}"
        
        # NMOS should NOT be allowed in PMOS rows
        valid, msg = self.grid.is_valid_placement(
            component_id=2,
            x=5, y=pmos_row.y_start + 1,
            width=2, height=2,
            component_type=ComponentType.NMOS
        )
        assert not valid, "NMOS should not be valid in PMOS row"
        assert "not allowed" in msg.lower()
        
        # NMOS should be allowed in NMOS rows
        nmos_row = next(r for r in self.grid.rows if r.row_type == RowType.NMOS_ROW)
        valid, msg = self.grid.is_valid_placement(
            component_id=3,
            x=5, y=nmos_row.y_start + 1,
            width=2, height=2,
            component_type=ComponentType.NMOS
        )
        assert valid, f"NMOS should be valid in NMOS row: {msg}"
        
        # Resistor should be allowed in mixed rows
        mixed_row = next(r for r in self.grid.rows if r.row_type == RowType.MIXED_ROW)
        valid, msg = self.grid.is_valid_placement(
            component_id=4,
            x=5, y=mixed_row.y_start + 1,
            width=2, height=2,
            component_type=ComponentType.RESISTOR
        )
        assert valid, f"Resistor should be valid in mixed row: {msg}"
    
    def test_bounds_checking(self):
        """Test grid boundary validation."""
        # Out of bounds placements should be invalid
        valid, msg = self.grid.is_valid_placement(
            component_id=1,
            x=-1, y=5,  # Negative x
            width=2, height=2,
            component_type=ComponentType.PMOS
        )
        assert not valid
        assert "out of bounds" in msg.lower()
        
        valid, msg = self.grid.is_valid_placement(
            component_id=2,
            x=31, y=5,  # x + width > grid_width
            width=2, height=2,
            component_type=ComponentType.PMOS
        )
        assert not valid
        assert "out of bounds" in msg.lower()
    
    def test_overlap_detection(self):
        """Test overlap detection between components."""
        # Place first component
        pmos_row = next(r for r in self.grid.rows if r.row_type == RowType.PMOS_ROW)
        success = self.grid.place_component(
            component_id=1,
            x=5, y=pmos_row.y_start + 1,  
            width=3, height=2,
            orientation=0,
            component_type=ComponentType.PMOS
        )
        assert success
        
        # Try to place overlapping component
        valid, msg = self.grid.is_valid_placement(
            component_id=2,
            x=6, y=pmos_row.y_start + 1,  # Overlaps with component 1
            width=3, height=2,
            component_type=ComponentType.PMOS
        )
        assert not valid
        assert "overlaps" in msg.lower()
        
        # Non-overlapping should be valid
        valid, msg = self.grid.is_valid_placement(
            component_id=3,
            x=10, y=pmos_row.y_start + 1,  # No overlap
            width=3, height=2,
            component_type=ComponentType.PMOS
        )
        assert valid
    
    def test_component_placement_and_removal(self):
        """Test placing and removing components."""
        pmos_row = next(r for r in self.grid.rows if r.row_type == RowType.PMOS_ROW)
        
        # Place component
        success = self.grid.place_component(
            component_id=1,
            x=5, y=pmos_row.y_start + 1,
            width=3, height=2, 
            orientation=0,
            component_type=ComponentType.PMOS
        )
        assert success
        assert 1 in self.grid.placements
        
        # Check grid is occupied
        assert np.any(self.grid.grid == 1)
        
        # Remove component
        success = self.grid.remove_component(1)
        assert success
        assert 1 not in self.grid.placements
        
        # Check grid is clear
        assert not np.any(self.grid.grid == 1)
    
    def test_orientation_handling(self):
        """Test component orientation affects dimensions."""
        pmos_row = next(r for r in self.grid.rows if r.row_type == RowType.PMOS_ROW)
        
        # Place component with 90° rotation (width/height swapped)
        success = self.grid.place_component(
            component_id=1,
            x=5, y=pmos_row.y_start + 1,
            width=4, height=2,  # Original dimensions
            orientation=90,
            component_type=ComponentType.PMOS
        )
        assert success
        
        placement = self.grid.placements[1]
        # After 90° rotation, width=2, height=4
        assert placement.width == 2
        assert placement.height == 4
    
    def test_row_consistency_score(self):
        """Test row consistency scoring."""
        # Initially should be perfect (1.0)
        assert self.grid.get_row_consistency_score() == 1.0
        
        # Place components in correct rows
        pmos_row = next(r for r in self.grid.rows if r.row_type == RowType.PMOS_ROW)
        nmos_row = next(r for r in self.grid.rows if r.row_type == RowType.NMOS_ROW)
        
        self.grid.place_component(1, 5, pmos_row.y_start + 1, 2, 2, 0, ComponentType.PMOS)
        self.grid.place_component(2, 10, nmos_row.y_start + 1, 2, 2, 0, ComponentType.NMOS)
        
        # Should still be perfect
        assert self.grid.get_row_consistency_score() == 1.0
    
    def test_crossing_detection(self):
        """Test crossing detection for connections."""
        # Place 4 components in a crossing pattern
        pmos_row = next(r for r in self.grid.rows if r.row_type == RowType.PMOS_ROW)
        nmos_row = next(r for r in self.grid.rows if r.row_type == RowType.NMOS_ROW)
        
        self.grid.place_component(1, 5, pmos_row.y_start + 1, 2, 2, 0, ComponentType.PMOS)
        self.grid.place_component(2, 15, pmos_row.y_start + 1, 2, 2, 0, ComponentType.PMOS)
        self.grid.place_component(3, 5, nmos_row.y_start + 1, 2, 2, 0, ComponentType.NMOS)
        self.grid.place_component(4, 15, nmos_row.y_start + 1, 2, 2, 0, ComponentType.NMOS)
        
        # Create crossing connections: 1-4 and 2-3 should cross
        connections = [(1, 4), (2, 3)]
        crossings = self.grid.count_crossings(connections)
        
        assert crossings > 0, "Crossing connections should be detected"
        
        # Non-crossing connections: 1-2 and 3-4 should not cross
        non_crossing_connections = [(1, 2), (3, 4)]
        non_crossings = self.grid.count_crossings(non_crossing_connections)
        
        assert non_crossings == 0, "Non-crossing connections should not be detected as crossing"
    
    def test_congestion_variance(self):
        """Test congestion variance calculation."""
        # Place components
        pmos_row = next(r for r in self.grid.rows if r.row_type == RowType.PMOS_ROW)
        nmos_row = next(r for r in self.grid.rows if r.row_type == RowType.NMOS_ROW)
        
        self.grid.place_component(1, 5, pmos_row.y_start + 1, 2, 2, 0, ComponentType.PMOS)
        self.grid.place_component(2, 10, nmos_row.y_start + 1, 2, 2, 0, ComponentType.NMOS)
        
        # Test congestion calculation
        connections = [(1, 2)]
        variance = self.grid.get_congestion_variance(connections)
        
        assert isinstance(variance, float)
        assert variance >= 0.0
    
    def test_bounding_box_calculation(self):
        """Test bounding box calculation."""
        # Empty grid should have zero bounding box
        bbox = self.grid.get_bounding_box()
        assert bbox == (0, 0, 0, 0)
        
        # Place components
        pmos_row = next(r for r in self.grid.rows if r.row_type == RowType.PMOS_ROW)
        nmos_row = next(r for r in self.grid.rows if r.row_type == RowType.NMOS_ROW)
        
        self.grid.place_component(1, 5, pmos_row.y_start + 1, 3, 2, 0, ComponentType.PMOS)
        self.grid.place_component(2, 15, nmos_row.y_start + 1, 2, 3, 0, ComponentType.NMOS)
        
        bbox = self.grid.get_bounding_box()
        min_x, min_y, max_x, max_y = bbox
        
        assert min_x == 5
        assert max_x == 17  # 15 + 2
        assert min_y <= max_y
    
    def test_perimeter_area_ratio(self):
        """Test perimeter/area ratio calculation."""
        # Empty grid should have zero ratio
        ratio = self.grid.get_perimeter_area_ratio()
        assert ratio == 0.0
        
        # Place a compact layout (4x2 bounding box)
        pmos_row = next(r for r in self.grid.rows if r.row_type == RowType.PMOS_ROW)
        
        self.grid.place_component(1, 5, pmos_row.y_start + 1, 2, 2, 0, ComponentType.PMOS)
        self.grid.place_component(2, 7, pmos_row.y_start + 1, 2, 2, 0, ComponentType.PMOS)
        
        compact_ratio = self.grid.get_perimeter_area_ratio()
        
        # Clear and place elongated layout (12x2 bounding box)
        self.grid.reset()
        
        self.grid.place_component(1, 5, pmos_row.y_start + 1, 2, 2, 0, ComponentType.PMOS)
        self.grid.place_component(2, 15, pmos_row.y_start + 1, 2, 2, 0, ComponentType.PMOS)
        
        elongated_ratio = self.grid.get_perimeter_area_ratio()
        
        assert compact_ratio > elongated_ratio, "Compact layout should have higher perimeter/area ratio (less efficient)"
    
    def test_grid_reset(self):
        """Test grid reset functionality."""
        # Place some components
        pmos_row = next(r for r in self.grid.rows if r.row_type == RowType.PMOS_ROW)
        
        self.grid.place_component(1, 5, pmos_row.y_start + 1, 2, 2, 0, ComponentType.PMOS)
        self.grid.place_component(2, 10, pmos_row.y_start + 1, 2, 2, 0, ComponentType.PMOS)
        
        assert len(self.grid.placements) == 2
        assert np.any(self.grid.grid > 0)
        
        # Reset
        self.grid.reset()
        
        assert len(self.grid.placements) == 0
        assert not np.any(self.grid.grid > 0)
    
    def test_state_summary(self):
        """Test state summary generation."""
        summary = self.grid.get_state_summary()
        
        required_keys = ['num_components', 'row_consistency', 'overlap_count', 
                        'bounding_box', 'perimeter_area_ratio', 'locked_groups']
        
        for key in required_keys:
            assert key in summary
        
        assert summary['num_components'] == 0
        assert summary['row_consistency'] == 1.0
        assert summary['overlap_count'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])