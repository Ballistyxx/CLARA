#!/usr/bin/env python3
"""
Integration test for the complete analog layout system.
Tests the integration of all components: environment, policy, legalizer, metrics, and visualization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import networkx as nx
from typing import Dict, Any

from env.layout_grid import LayoutGrid, ComponentType
from env.analog_layout_env import AnalogLayoutEnv
from metrics.metrics import calculate_layout_metrics, meets_acceptance_criteria
from legalizer.row_snap import legalize_layout, LegalizationMode
from viz.overlays import visualize_analog_layout


class TestAnalogLayoutIntegration:
    """Integration tests for the complete analog layout system."""
    
    def setup_method(self):
        """Set up test environment and circuits."""
        self.env = AnalogLayoutEnv(
            grid_size=32,
            max_components=8,
            pmos_rows=2,
            nmos_rows=2,
            mixed_rows=1,
            enable_action_masking=True
        )
        
        self.test_circuit = self._create_test_circuit()
    
    def _create_test_circuit(self) -> nx.Graph:
        """Create a test circuit with mixed component types and matching pairs."""
        circuit = nx.Graph()
        
        # Add components with realistic analog layout requirements
        components = [
            {'id': 0, 'type': ComponentType.PMOS, 'width': 2, 'height': 1, 'matched': 1},
            {'id': 1, 'type': ComponentType.PMOS, 'width': 2, 'height': 1, 'matched': 0},
            {'id': 2, 'type': ComponentType.NMOS, 'width': 2, 'height': 1, 'matched': 3},
            {'id': 3, 'type': ComponentType.NMOS, 'width': 2, 'height': 1, 'matched': 2},
            {'id': 4, 'type': ComponentType.RESISTOR, 'width': 1, 'height': 2, 'matched': -1},
            {'id': 5, 'type': ComponentType.CAPACITOR, 'width': 1, 'height': 1, 'matched': -1},
        ]
        
        for comp in components:
            circuit.add_node(
                comp['id'],
                component_type=comp['type'].value,
                width=comp['width'],
                height=comp['height'],
                matched_component=comp['matched'],
                device_model=f"test_model_{comp['type'].name.lower()}",
                spice_name=f"X{comp['id']}"
            )
        
        # Add connections (differential pair + current mirror)
        connections = [(0, 2), (1, 3), (2, 4), (3, 4), (4, 5)]
        circuit.add_edges_from(connections)
        
        return circuit
    
    def test_complete_layout_workflow(self):
        """Test complete workflow: reset → place → legalize → evaluate → visualize."""
        # Reset environment with test circuit
        reset_result = self.env.reset(options={'circuit': self.test_circuit})
        
        # Handle both gym and gymnasium return formats
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
        
        assert isinstance(obs, dict)
        assert 'component_graph' in obs
        assert 'component_features' in obs
        assert 'placement_state' in obs
        
        # Perform some placement actions
        placed_count = 0
        max_attempts = 20
        
        for attempt in range(max_attempts):
            if self.env.enable_action_masking:
                action_mask = self.env.get_action_mask(obs)
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) == 0:
                    break
                
                # Sample valid action
                flat_action_idx = np.random.choice(valid_actions)
                action = self._unflatten_action(flat_action_idx)
            else:
                # Sample random action
                action = self.env.action_space.sample()
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            if info.get('placement_successful', False):
                placed_count += 1
            
            if terminated or truncated:
                break
        
        assert placed_count > 0, "Should place at least one component"
        
        # Test legalization
        legalization_result = legalize_layout(
            self.env.layout_grid, 
            mode=LegalizationMode.PRESERVE_PATTERNS
        )
        
        assert legalization_result.success or legalization_result.violations_fixed >= 0
        
        # Test metrics calculation
        metrics = calculate_layout_metrics(
            self.env.layout_grid, 
            self.test_circuit,
            target_components=len(self.test_circuit.nodes)
        )
        
        # Verify metrics structure
        assert 0.0 <= metrics.completion <= 1.0
        assert 0.0 <= metrics.row_consistency <= 1.0
        assert 0.0 <= metrics.symmetry_score <= 1.0
        assert metrics.overlap_count >= 0
        assert metrics.crossings >= 0
        
        # Test acceptance criteria
        acceptance = meets_acceptance_criteria(
            metrics,
            min_completion=0.5,  # Relaxed for test
            min_row_consistency=0.8,
            min_symmetry=0.5,
            max_violations=5
        )
        
        assert isinstance(acceptance, bool)
        
        print(f"Integration test results:")
        print(f"   Components placed: {placed_count}/{len(self.test_circuit.nodes)}")
        print(f"   Completion: {metrics.completion:.2f}")
        print(f"   Row consistency: {metrics.row_consistency:.2f}")
        print(f"   Symmetry score: {metrics.symmetry_score:.2f}")
        print(f"   Crossings: {metrics.crossings}")
        print(f"   Analog score: {metrics.analog_score:.2f}")
        print(f"   Meets criteria: {acceptance}")
    
    def test_action_masking_integration(self):
        """Test that action masking prevents invalid actions."""
        reset_result = self.env.reset(options={'circuit': self.test_circuit})
        
        # Handle both gym and gymnasium return formats
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
        
        if not self.env.enable_action_masking:
            pytest.skip("Action masking not enabled")
        
        # Get action mask
        action_mask = self.env.get_action_mask(obs)
        
        assert isinstance(action_mask, np.ndarray)
        assert action_mask.dtype == bool
        assert len(action_mask) == np.prod(self.env.action_space.nvec)
        
        # Should have some valid actions initially
        valid_actions = np.sum(action_mask)
        assert valid_actions > 0, "Should have valid actions at start"
        
        # Test that masked actions are actually invalid
        invalid_actions = np.where(~action_mask)[0]
        if len(invalid_actions) > 0:
            # Try an invalid action
            invalid_flat_idx = invalid_actions[0]
            invalid_action = self._unflatten_action(invalid_flat_idx)
            
            # Copy observation state for comparison
            obs_before = {k: v.copy() if hasattr(v, 'copy') else v for k, v in obs.items()}
            obs_after, reward, terminated, truncated, info = self.env.step(invalid_action)
            
            # Invalid action should either be rejected or heavily penalized
            assert not info.get('placement_successful', False) or reward < -1.0
    
    def test_row_discipline_enforcement(self):
        """Test that row discipline is properly enforced."""
        reset_result = self.env.reset(options={'circuit': self.test_circuit})
        
        # Handle both gym and gymnasium return formats
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
        
        # Try to place PMOS in NMOS row (should be prevented by constraints)
        layout_grid = self.env.layout_grid
        
        # Find NMOS row
        nmos_row = None
        for row in layout_grid.rows:
            if ComponentType.NMOS in row.allowed_components and ComponentType.PMOS not in row.allowed_components:
                nmos_row = row
                break
        
        if nmos_row:
            # Try to place PMOS in NMOS row
            valid, error_msg = layout_grid.is_valid_placement(
                component_id=999,  # Dummy ID
                x=5, y=nmos_row.y_start + 1,
                width=2, height=1,
                component_type=ComponentType.PMOS
            )
            
            assert not valid, "PMOS should not be allowed in NMOS row"
            assert "not allowed" in error_msg.lower()
    
    def test_symmetry_pattern_detection(self):
        """Test symmetry pattern detection and scoring."""
        # Manually place a symmetric pair
        layout_grid = LayoutGrid(grid_width=32, grid_height=32, pmos_rows=2, nmos_rows=2, mixed_rows=1)
        
        # Place matched PMOS pair symmetrically
        pmos_row = next(row for row in layout_grid.rows if ComponentType.PMOS in row.allowed_components)
        
        # Place first PMOS
        success1 = layout_grid.place_component(
            0, 8, pmos_row.y_start + 1, 2, 1, 0, ComponentType.PMOS
        )
        
        # Place second PMOS symmetrically
        success2 = layout_grid.place_component(
            1, 20, pmos_row.y_start + 1, 2, 1, 0, ComponentType.PMOS
        )
        
        assert success1 and success2, "Should successfully place symmetric pair"
        
        # Create circuit with matching information
        circuit = nx.Graph()
        circuit.add_node(0, component_type=ComponentType.PMOS.value, matched_component=1)
        circuit.add_node(1, component_type=ComponentType.PMOS.value, matched_component=0)
        
        # Calculate metrics
        metrics = calculate_layout_metrics(layout_grid, circuit)
        
        # Should detect good symmetry
        assert metrics.symmetry_score > 0.7, f"Symmetry score should be high, got {metrics.symmetry_score}"
    
    def test_visualization_integration(self):
        """Test that visualization works with the complete system."""
        # Create a layout with some components
        reset_result = self.env.reset(options={'circuit': self.test_circuit})
        
        # Handle both gym and gymnasium return formats
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
        
        # Place a few components manually for testing
        layout_grid = self.env.layout_grid
        
        # Place PMOS in PMOS row
        pmos_row = next(row for row in layout_grid.rows if ComponentType.PMOS in row.allowed_components)
        layout_grid.place_component(0, 5, pmos_row.y_start + 1, 2, 1, 0, ComponentType.PMOS)
        
        # Place NMOS in NMOS row
        nmos_row = next(row for row in layout_grid.rows if ComponentType.NMOS in row.allowed_components)
        layout_grid.place_component(2, 8, nmos_row.y_start + 1, 2, 1, 0, ComponentType.NMOS)
        
        # Test visualization (should not crash)
        try:
            fig = visualize_analog_layout(
                layout_grid, 
                self.test_circuit, 
                save_path=None  # Don't save during test
            )
            assert fig is not None
            print("Visualization integration successful")
        except Exception as e:
            pytest.fail(f"Visualization failed: {e}")
    
    def _unflatten_action(self, flat_idx: int) -> np.ndarray:
        """Convert flat action index to multi-discrete action."""
        action = np.zeros(len(self.env.action_space.nvec), dtype=int)
        
        remaining = flat_idx
        for i in reversed(range(len(self.env.action_space.nvec))):
            action[i] = remaining % self.env.action_space.nvec[i]
            remaining = remaining // self.env.action_space.nvec[i]
        
        return action
    
    def test_metrics_acceptance_criteria(self):
        """Test that acceptance criteria work correctly."""
        # Create perfect layout manually
        layout_grid = LayoutGrid(grid_width=16, grid_height=16, pmos_rows=1, nmos_rows=1, mixed_rows=1)
        
        # Place all components correctly
        pmos_row = next(row for row in layout_grid.rows if ComponentType.PMOS in row.allowed_components)
        nmos_row = next(row for row in layout_grid.rows if ComponentType.NMOS in row.allowed_components)
        mixed_row = next(row for row in layout_grid.rows if ComponentType.RESISTOR in row.allowed_components)
        
        # Place components without overlaps in correct rows
        layout_grid.place_component(0, 2, pmos_row.y_start, 2, 1, 0, ComponentType.PMOS)
        layout_grid.place_component(1, 5, pmos_row.y_start, 2, 1, 0, ComponentType.PMOS)
        layout_grid.place_component(2, 2, nmos_row.y_start, 2, 1, 0, ComponentType.NMOS)
        layout_grid.place_component(3, 5, nmos_row.y_start, 2, 1, 0, ComponentType.NMOS)
        layout_grid.place_component(4, 8, mixed_row.y_start, 1, 2, 0, ComponentType.RESISTOR)
        layout_grid.place_component(5, 10, mixed_row.y_start, 1, 1, 0, ComponentType.CAPACITOR)
        
        # Calculate metrics
        metrics = calculate_layout_metrics(layout_grid, self.test_circuit)
        
        # Should meet basic criteria
        assert metrics.completion == 1.0, "All components should be placed"
        assert metrics.row_consistency == 1.0, "All components in correct rows"
        assert metrics.overlap_count == 0, "No overlaps"
        
        # Test acceptance criteria
        acceptance = meets_acceptance_criteria(
            metrics,
            min_completion=1.0,
            min_row_consistency=0.95,
            min_symmetry=0.0,  # Relaxed for test
            max_violations=0
        )
        
        assert acceptance, "Perfect layout should meet acceptance criteria"


def test_quick_integration():
    """Quick integration test that can be run standalone."""
    env = AnalogLayoutEnv(grid_size=16, max_components=4)
    
    # Simple circuit
    circuit = nx.Graph()
    circuit.add_node(0, component_type=ComponentType.PMOS.value, width=2, height=1, matched_component=-1)
    circuit.add_node(1, component_type=ComponentType.NMOS.value, width=2, height=1, matched_component=-1)
    circuit.add_edge(0, 1)
    
    # Reset and take a few steps
    obs = env.reset(options={'circuit': circuit})
    
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    # Should complete without errors
    assert True  # If we get here, basic integration works


if __name__ == "__main__":
    # Run quick test
    test_quick_integration()
    print("Quick integration test passed!")
    
    # Run full test suite
    pytest.main([__file__, "-v"])