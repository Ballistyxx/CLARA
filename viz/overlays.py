#!/usr/bin/env python3
"""
Visualization overlays for analog IC layout analysis.
Provides symmetry axes, density heatmaps, crossing visualization, and pattern analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Set
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import matplotlib.gridspec as gridspec

from env.layout_grid import LayoutGrid, ComponentType, RowType
from metrics.metrics import LayoutMetrics, calculate_layout_metrics


class AnalogLayoutVisualizer:
    """
    Advanced visualizer for analog IC layouts with overlay analysis.
    """
    
    def __init__(self, 
                 grid_size: int = 64,
                 figsize: Tuple[int, int] = (16, 12),
                 dpi: int = 100):
        """
        Initialize visualizer.
        
        Args:
            grid_size: Layout grid size
            figsize: Figure size for plots
            dpi: Figure DPI
        """
        self.grid_size = grid_size
        self.figsize = figsize
        self.dpi = dpi
        
        # Color schemes
        self.component_colors = {
            ComponentType.NMOS: '#FF6B6B',      # Red
            ComponentType.PMOS: '#4ECDC4',      # Teal
            ComponentType.RESISTOR: '#45B7D1',  # Blue
            ComponentType.CAPACITOR: '#96CEB4', # Green
            ComponentType.INDUCTOR: '#FECA57',  # Yellow
            ComponentType.CURRENT_SOURCE: '#FF9FF3',  # Pink
            ComponentType.VOLTAGE_SOURCE: '#54A0FF',  # Light Blue
            ComponentType.SUBCIRCUIT: '#A569BD',      # Purple
            ComponentType.OTHER: '#95A5A6'            # Gray
        }
        
        self.row_colors = {
            RowType.PMOS_ROW: '#E8F6F3',    # Light teal
            RowType.NMOS_ROW: '#FADBD8',    # Light red
            RowType.MIXED_ROW: '#F8F9FA'    # Light gray
        }
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def create_comprehensive_layout_view(self,
                                       layout_grid: LayoutGrid,
                                       circuit: nx.Graph,
                                       metrics: Optional[LayoutMetrics] = None,
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive layout visualization with multiple overlays.
        
        Args:
            layout_grid: Current layout state
            circuit: Circuit connectivity graph  
            metrics: Pre-calculated metrics (optional)
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure object
        """
        if metrics is None:
            metrics = calculate_layout_metrics(layout_grid, circuit)
        
        # Create figure with subplots
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Main layout view
        ax_main = fig.add_subplot(gs[:2, :2])
        self._draw_main_layout(ax_main, layout_grid, circuit, metrics)
        
        # Symmetry analysis
        ax_symmetry = fig.add_subplot(gs[0, 2])
        self._draw_symmetry_overlay(ax_symmetry, layout_grid, circuit)
        
        # Density heatmap
        ax_density = fig.add_subplot(gs[1, 2])
        self._draw_density_heatmap(ax_density, layout_grid)
        
        # Crossing analysis
        ax_crossings = fig.add_subplot(gs[2, :])
        self._draw_crossing_analysis(ax_crossings, layout_grid, circuit)
        
        # Add overall title and metrics summary
        fig.suptitle('Analog IC Layout Analysis', fontsize=16, fontweight='bold')
        self._add_metrics_text(fig, metrics)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def _draw_main_layout(self, 
                         ax: plt.Axes,
                         layout_grid: LayoutGrid,
                         circuit: nx.Graph,
                         metrics: LayoutMetrics):
        """Draw main layout view with row partitions and components."""
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_aspect('equal')
        ax.set_title('Layout with Row Partitions', fontsize=12, fontweight='bold')
        
        # Draw row backgrounds
        for row in layout_grid.rows:
            color = self.row_colors.get(row.row_type, '#F8F9FA')
            rect = Rectangle((0, row.y_start), self.grid_size, row.y_end - row.y_start,
                           facecolor=color, alpha=0.3, edgecolor='none')
            ax.add_patch(rect)
            
            # Add row labels
            ax.text(-0.3, (row.y_start + row.y_end) / 2, 
                   row.row_type.name.replace('_', '\n'), 
                   rotation=90, ha='center', va='center', fontsize=8)
        
        # Draw grid
        for i in range(self.grid_size + 1):
            ax.axhline(i - 0.5, color='lightgray', linewidth=0.3, alpha=0.5)
            ax.axvline(i - 0.5, color='lightgray', linewidth=0.3, alpha=0.5)
        
        # Draw connections first
        if circuit:
            self._draw_connections(ax, layout_grid, circuit)
        
        # Draw components
        for comp_id, placement in layout_grid.placements.items():
            self._draw_component_with_details(ax, comp_id, placement, layout_grid, circuit)
        
        # Draw symmetry axes for locked groups
        self._draw_symmetry_axes(ax, layout_grid)
        
        ax.set_xlabel('Grid X')
        ax.set_ylabel('Grid Y')
        ax.invert_yaxis()
    
    def _draw_component_with_details(self,
                                   ax: plt.Axes,
                                   comp_id: int,
                                   placement,
                                   layout_grid: LayoutGrid,
                                   circuit: nx.Graph):
        """Draw component with detailed styling."""
        color = self.component_colors.get(placement.component_type, '#95A5A6')
        
        # Special styling for locked components
        if placement.locked:
            edgecolor = 'gold'
            linewidth = 3
            alpha = 0.9
        else:
            edgecolor = 'darkgray'
            linewidth = 1
            alpha = 0.8
        
        # Draw component rectangle
        rect = Rectangle((placement.x, placement.y), placement.width, placement.height,
                        facecolor=color, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha)
        ax.add_patch(rect)
        
        # Add component label
        center_x = placement.x + placement.width / 2
        center_y = placement.y + placement.height / 2
        
        # Get component info from circuit
        comp_info = circuit.nodes[comp_id] if circuit and comp_id in circuit.nodes else {}
        device_model = comp_info.get('device_model', '')
        spice_name = comp_info.get('spice_name', f'C{comp_id}')
        
        if device_model:
            label = device_model
        else:
            label = spice_name
        
        # Adaptive font size
        font_size = max(4, min(8, 60 // max(len(label), 1)))
        
        ax.text(center_x, center_y, label,
               ha='center', va='center', fontsize=font_size, fontweight='bold',
               color='white', 
               bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.7))
        
        # Add orientation indicator
        if placement.orientation != 0:
            self._add_orientation_indicator(ax, placement)
        
        # Highlight matched components
        matched_comp = comp_info.get('matched_component', -1)
        if matched_comp != -1 and matched_comp in layout_grid.placements:
            # Draw matching indicator
            circle = Circle((center_x, center_y), 0.3, 
                          facecolor='none', edgecolor='magenta', linewidth=2)
            ax.add_patch(circle)
    
    def _draw_connections(self, ax: plt.Axes, layout_grid: LayoutGrid, circuit: nx.Graph):
        """Draw connections between components."""
        for comp1_id, comp2_id in circuit.edges():
            if comp1_id in layout_grid.placements and comp2_id in layout_grid.placements:
                p1 = layout_grid.placements[comp1_id]
                p2 = layout_grid.placements[comp2_id]
                
                center1 = (p1.x + p1.width / 2, p1.y + p1.height / 2)
                center2 = (p2.x + p2.width / 2, p2.y + p2.height / 2)
                
                ax.plot([center1[0], center2[0]], [center1[1], center2[1]],
                       'k--', alpha=0.4, linewidth=1, zorder=0)
    
    def _draw_symmetry_axes(self, ax: plt.Axes, layout_grid: LayoutGrid):
        """Draw symmetry axes for locked groups."""
        for group_id, group_info in layout_grid.locked_groups.items():
            pattern_type = group_info['pattern_type']
            component_ids = group_info['component_ids']
            
            if pattern_type == 'mirror' and len(component_ids) == 2:
                self._draw_mirror_axis(ax, layout_grid, component_ids)
            elif pattern_type == 'common_centroid':
                self._draw_common_centroid_indicator(ax, layout_grid, component_ids)
    
    def _draw_mirror_axis(self, ax: plt.Axes, layout_grid: LayoutGrid, component_ids: List[int]):
        """Draw mirror axis for a pair of components."""
        if len(component_ids) != 2:
            return
        
        comp1_id, comp2_id = component_ids
        if comp1_id not in layout_grid.placements or comp2_id not in layout_grid.placements:
            return
        
        p1 = layout_grid.placements[comp1_id]
        p2 = layout_grid.placements[comp2_id]
        
        # Calculate midpoint
        center1 = (p1.x + p1.width / 2, p1.y + p1.height / 2)
        center2 = (p2.x + p2.width / 2, p2.y + p2.height / 2)
        mid_x = (center1[0] + center2[0]) / 2
        mid_y = (center1[1] + center2[1]) / 2
        
        # Draw symmetry axis
        if abs(center1[1] - center2[1]) < 1:  # Horizontal symmetry
            ax.axvline(mid_x, color='red', linestyle=':', linewidth=2, alpha=0.7,
                      ymin=0, ymax=1, label='Mirror Axis')
        elif abs(center1[0] - center2[0]) < 1:  # Vertical symmetry
            ax.axhline(mid_y, color='red', linestyle=':', linewidth=2, alpha=0.7,
                      xmin=0, xmax=1, label='Mirror Axis')
    
    def _draw_common_centroid_indicator(self, ax: plt.Axes, layout_grid: LayoutGrid, component_ids: List[int]):
        """Draw common centroid indicator."""
        if len(component_ids) < 2:
            return
        
        # Calculate centroid
        total_x, total_y = 0, 0
        valid_components = 0
        
        for comp_id in component_ids:
            if comp_id in layout_grid.placements:
                p = layout_grid.placements[comp_id]
                total_x += p.x + p.width / 2
                total_y += p.y + p.height / 2
                valid_components += 1
        
        if valid_components < 2:
            return
        
        centroid_x = total_x / valid_components
        centroid_y = total_y / valid_components
        
        # Draw centroid marker
        ax.plot(centroid_x, centroid_y, 'r*', markersize=12, markeredgecolor='black', 
               markeredgewidth=1, label='Common Centroid')
        
        # Draw lines to components
        for comp_id in component_ids:
            if comp_id in layout_grid.placements:
                p = layout_grid.placements[comp_id]
                comp_center = (p.x + p.width / 2, p.y + p.height / 2)
                ax.plot([centroid_x, comp_center[0]], [centroid_y, comp_center[1]],
                       'r:', alpha=0.5, linewidth=1)
    
    def _draw_symmetry_overlay(self, ax: plt.Axes, layout_grid: LayoutGrid, circuit: nx.Graph):
        """Draw symmetry analysis overlay."""
        ax.set_title('Symmetry Analysis', fontsize=10, fontweight='bold')
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_aspect('equal')
        
        # Draw global symmetry axes
        center_x = self.grid_size / 2
        center_y = self.grid_size / 2
        
        ax.axvline(center_x, color='blue', linestyle='-', linewidth=1, alpha=0.5, label='Global V-Axis')
        ax.axhline(center_y, color='green', linestyle='-', linewidth=1, alpha=0.5, label='Global H-Axis')
        
        # Highlight symmetric pairs
        if circuit:
            for node_id in circuit.nodes():
                if node_id in layout_grid.placements:
                    matched_comp = circuit.nodes[node_id].get('matched_component', -1)
                    if matched_comp != -1 and matched_comp in layout_grid.placements and node_id < matched_comp:
                        p1 = layout_grid.placements[node_id]
                        p2 = layout_grid.placements[matched_comp]
                        
                        # Draw symmetric pair indicators
                        center1 = (p1.x + p1.width / 2, p1.y + p1.height / 2)
                        center2 = (p2.x + p2.width / 2, p2.y + p2.height / 2)
                        
                        # Check symmetry quality
                        is_symmetric = self._check_symmetry_quality(p1, p2, center_x, center_y)
                        color = 'green' if is_symmetric else 'red'
                        
                        ax.plot([center1[0], center2[0]], [center1[1], center2[1]],
                               color=color, linewidth=2, alpha=0.7)
                        ax.scatter([center1[0], center2[0]], [center1[1], center2[1]],
                                 c=color, s=50, alpha=0.8)
        
        ax.legend(fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()
    
    def _check_symmetry_quality(self, p1, p2, center_x: float, center_y: float) -> bool:
        """Check if two placements are well-symmetrical."""
        center1 = (p1.x + p1.width / 2, p1.y + p1.height / 2)
        center2 = (p2.x + p2.width / 2, p2.y + p2.height / 2)
        
        # Check horizontal symmetry
        expected_x2 = 2 * center_x - center1[0]
        if abs(center2[0] - expected_x2) <= 1.5 and abs(center1[1] - center2[1]) <= 1.5:
            return True
        
        # Check vertical symmetry
        expected_y2 = 2 * center_y - center1[1]
        if abs(center2[1] - expected_y2) <= 1.5 and abs(center1[0] - center2[0]) <= 1.5:
            return True
        
        return False
    
    def _draw_density_heatmap(self, ax: plt.Axes, layout_grid: LayoutGrid):
        """Draw component density heatmap."""
        ax.set_title('Component Density', fontsize=10, fontweight='bold')
        
        # Create density grid (coarser than main grid)
        density_size = 16
        density_grid = np.zeros((density_size, density_size))
        
        # Map components to density grid
        scale_x = self.grid_size / density_size
        scale_y = self.grid_size / density_size
        
        for placement in layout_grid.placements.values():
            # Map component to density grid
            density_x = int(placement.x / scale_x)
            density_y = int(placement.y / scale_y)
            
            # Add component area to density
            for dy in range(max(0, int(placement.height / scale_y) + 1)):
                for dx in range(max(0, int(placement.width / scale_x) + 1)):
                    if (density_y + dy < density_size and density_x + dx < density_size):
                        density_grid[density_y + dy, density_x + dx] += 1
        
        # Create heatmap
        im = ax.imshow(density_grid, cmap='YlOrRd', interpolation='nearest', aspect='equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.6)
    
    def _draw_crossing_analysis(self, ax: plt.Axes, layout_grid: LayoutGrid, circuit: nx.Graph):
        """Draw crossing analysis and congestion heatmap."""
        ax.set_title('Routing Congestion & Crossings', fontsize=10, fontweight='bold')
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_aspect('equal')
        
        if not circuit:
            ax.text(self.grid_size/2, self.grid_size/2, 'No circuit data', 
                   ha='center', va='center', fontsize=12)
            return
        
        # Draw congestion heatmap background
        congestion_grid = np.zeros((16, 16))  # Coarse grid for congestion
        scale_x = self.grid_size / 16
        scale_y = self.grid_size / 16
        
        # Rasterize connections
        for comp1_id, comp2_id in circuit.edges():
            if comp1_id in layout_grid.placements and comp2_id in layout_grid.placements:
                p1 = layout_grid.placements[comp1_id]
                p2 = layout_grid.placements[comp2_id]
                
                center1 = (p1.x + p1.width / 2, p1.y + p1.height / 2)
                center2 = (p2.x + p2.width / 2, p2.y + p2.height / 2)
                
                # Rasterize line onto congestion grid
                self._rasterize_line_to_grid(center1, center2, congestion_grid, scale_x, scale_y)
        
        # Show congestion heatmap
        extent = [-0.5, self.grid_size - 0.5, self.grid_size - 0.5, -0.5]
        ax.imshow(congestion_grid, cmap='Blues', alpha=0.6, extent=extent, interpolation='bilinear')
        
        # Draw actual connections and highlight crossings
        connections = list(circuit.edges())
        crossing_points = self._find_crossing_points(layout_grid, connections)
        
        # Draw connections
        for comp1_id, comp2_id in connections:
            if comp1_id in layout_grid.placements and comp2_id in layout_grid.placements:
                p1 = layout_grid.placements[comp1_id]
                p2 = layout_grid.placements[comp2_id]
                
                center1 = (p1.x + p1.width / 2, p1.y + p1.height / 2)
                center2 = (p2.x + p2.width / 2, p2.y + p2.height / 2)
                
                ax.plot([center1[0], center2[0]], [center1[1], center2[1]],
                       'k-', alpha=0.5, linewidth=1)
        
        # Highlight crossing points
        for point in crossing_points:
            ax.plot(point[0], point[1], 'ro', markersize=8, alpha=0.8, 
                   markeredgecolor='black', markeredgewidth=1)
        
        # Add components as reference
        for placement in layout_grid.placements.values():
            rect = Rectangle((placement.x, placement.y), placement.width, placement.height,
                           facecolor='lightgray', edgecolor='black', alpha=0.3)
            ax.add_patch(rect)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()
        
        # Add crossing count
        ax.text(0.02, 0.98, f'Crossings: {len(crossing_points)}', 
               transform=ax.transAxes, fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
               verticalalignment='top')
    
    def _rasterize_line_to_grid(self, start: Tuple[float, float], end: Tuple[float, float],
                               grid: np.ndarray, scale_x: float, scale_y: float):
        """Rasterize line onto congestion grid."""
        x1, y1 = int(start[0] / scale_x), int(start[1] / scale_y)
        x2, y2 = int(end[0] / scale_x), int(end[1] / scale_y)
        
        # Clamp to grid bounds
        x1 = max(0, min(grid.shape[1] - 1, x1))
        y1 = max(0, min(grid.shape[0] - 1, y1))
        x2 = max(0, min(grid.shape[1] - 1, x2))
        y2 = max(0, min(grid.shape[0] - 1, y2))
        
        # Simple line rasterization
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        x, y = x1, y1
        x_inc = 1 if x1 < x2 else -1
        y_inc = 1 if y1 < y2 else -1
        
        error = dx - dy
        
        while True:
            grid[y, x] += 1
            
            if x == x2 and y == y2:
                break
                
            e2 = 2 * error
            
            if e2 > -dy:
                error -= dy
                x += x_inc
            
            if e2 < dx:
                error += dx
                y += y_inc
    
    def _find_crossing_points(self, layout_grid: LayoutGrid, connections: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """Find intersection points between connection lines."""
        crossing_points = []
        
        # Get line segments
        segments = []
        for comp1_id, comp2_id in connections:
            if comp1_id in layout_grid.placements and comp2_id in layout_grid.placements:
                p1 = layout_grid.placements[comp1_id]
                p2 = layout_grid.placements[comp2_id]
                
                center1 = (p1.x + p1.width / 2, p1.y + p1.height / 2)
                center2 = (p2.x + p2.width / 2, p2.y + p2.height / 2)
                
                segments.append((center1, center2))
        
        # Find intersections
        for i, seg1 in enumerate(segments):
            for seg2 in segments[i+1:]:
                intersection = self._line_intersection(seg1[0], seg1[1], seg2[0], seg2[1])
                if intersection:
                    crossing_points.append(intersection)
        
        return crossing_points
    
    def _line_intersection(self, p1: Tuple[float, float], q1: Tuple[float, float],
                          p2: Tuple[float, float], q2: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Find intersection point of two line segments."""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
        
        if intersect(p1, q1, p2, q2):
            # Calculate intersection point
            x1, y1 = p1
            x2, y2 = q1
            x3, y3 = p2
            x4, y4 = q2
            
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-10:
                return None
            
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            
            intersection_x = x1 + t * (x2 - x1)
            intersection_y = y1 + t * (y2 - y1)
            
            return (intersection_x, intersection_y)
        
        return None
    
    def _add_orientation_indicator(self, ax: plt.Axes, placement):
        """Add orientation indicator to component."""
        center_x = placement.x + placement.width / 2
        center_y = placement.y + placement.height / 2
        
        # Direction vectors for orientations
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # 0°, 90°, 180°, 270°
        orientation_idx = placement.orientation // 90
        dx, dy = directions[orientation_idx % 4]
        
        ax.arrow(center_x, center_y, dx * 0.3, dy * 0.3,
                head_width=0.1, head_length=0.1, fc='yellow', ec='orange',
                alpha=0.8, linewidth=2)
    
    def _add_metrics_text(self, fig: plt.Figure, metrics: LayoutMetrics):
        """Add metrics summary text to figure."""
        metrics_text = f"""
Layout Metrics Summary:
- Completion: {metrics.completion:.2f}
- Row Consistency: {metrics.row_consistency:.2f}
- Symmetry Score: {metrics.symmetry_score:.2f}
- Crossings: {metrics.crossings}
- Analog Score: {metrics.analog_score:.2f}
- Total Violations: {metrics.overlap_count + metrics.row_violations + metrics.spacing_violations}
"""
        
        fig.text(0.02, 0.02, metrics_text.strip(), fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                verticalalignment='bottom')


# Convenience functions
def visualize_analog_layout(layout_grid: LayoutGrid,
                          circuit: nx.Graph,
                          save_path: Optional[str] = None,
                          **kwargs) -> plt.Figure:
    """
    Convenience function to create comprehensive analog layout visualization.
    
    Args:
        layout_grid: Current layout state
        circuit: Circuit connectivity graph
        save_path: Path to save figure
        **kwargs: Additional arguments for visualizer
        
    Returns:
        matplotlib Figure object
    """
    visualizer = AnalogLayoutVisualizer(**kwargs)
    return visualizer.create_comprehensive_layout_view(layout_grid, circuit, save_path=save_path)


def create_symmetry_analysis(layout_grid: LayoutGrid,
                           circuit: nx.Graph,
                           save_path: Optional[str] = None) -> plt.Figure:
    """Create focused symmetry analysis visualization."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    visualizer = AnalogLayoutVisualizer()
    visualizer._draw_symmetry_overlay(ax, layout_grid, circuit)
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig


def create_crossing_heatmap(layout_grid: LayoutGrid,
                          circuit: nx.Graph,
                          save_path: Optional[str] = None) -> plt.Figure:
    """Create focused crossing and congestion analysis."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    visualizer = AnalogLayoutVisualizer()
    visualizer._draw_crossing_analysis(ax, layout_grid, circuit)
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig