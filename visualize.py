import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import networkx as nx
from typing import Dict, Tuple, Optional, List, Any
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import seaborn as sns


class AnalogLayoutVisualizer:
    """Visualization tools for analog IC layouts."""

    def __init__(self, grid_size: int = 64, figsize: Tuple[int, int] = (12, 10)):
        self.grid_size = grid_size
        self.figsize = figsize
        
        # Color scheme for different component types
        self.component_colors = {
            0: '#FF6B6B',  # MOSFET_N - Red
            1: '#4ECDC4',  # MOSFET_P - Teal  
            2: '#45B7D1',  # RESISTOR - Blue
            3: '#96CEB4',  # CAPACITOR - Green
            4: '#FECA57',  # INDUCTOR - Yellow
            5: '#FF9FF3',  # CURRENT_SOURCE - Pink
            6: '#54A0FF',  # VOLTAGE_SOURCE - Light Blue
        }
        
        self.component_names = {
            0: 'NMOS',
            1: 'PMOS', 
            2: 'R',
            3: 'C',
            4: 'L',
            5: 'I',
            6: 'V'
        }
        
        # Setup matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def visualize_layout(self, 
                        circuit: nx.Graph,
                        component_positions: Dict[int, Tuple[int, int, int]],
                        title: str = "Analog IC Layout",
                        show_connections: bool = True,
                        show_grid: bool = True,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize the current component placement layout.
        
        Args:
            circuit: NetworkX graph representing the circuit
            component_positions: Dict mapping component_id -> (x, y, orientation)
            title: Plot title
            show_connections: Whether to draw connections between components
            show_grid: Whether to show grid lines
            save_path: Path to save the figure
        """
        
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Draw grid
        if show_grid:
            for i in range(self.grid_size + 1):
                ax.axhline(i - 0.5, color='lightgray', linewidth=0.5, alpha=0.7)
                ax.axvline(i - 0.5, color='lightgray', linewidth=0.5, alpha=0.7)
        
        # Draw connections first (so they appear behind components)
        if show_connections and circuit is not None:
            self._draw_connections(ax, circuit, component_positions)
        
        # Draw components
        for comp_id, (x, y, orientation) in component_positions.items():
            if circuit is not None and comp_id in circuit.nodes:
                self._draw_component(ax, comp_id, x, y, orientation, circuit.nodes[comp_id])
        
        # Add legend
        self._add_component_legend(ax)
        
        # Add statistics
        if circuit is not None:
            self._add_layout_statistics(ax, circuit, component_positions)
        
        ax.set_xlabel('Grid X', fontsize=12)
        ax.set_ylabel('Grid Y', fontsize=12)
        
        # Invert y-axis to match typical layout convention
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _draw_component(self, ax: plt.Axes, comp_id: int, x: int, y: int, 
                       orientation: int, component_attrs: Dict[str, Any]):
        """Draw a single component on the layout."""
        
        comp_type = component_attrs.get('component_type', 0)
        width = component_attrs.get('width', 1)
        height = component_attrs.get('height', 1)
        matched_comp = component_attrs.get('matched_component', -1)
        
        # Adjust dimensions based on orientation
        if orientation in [1, 3]:  # 90° or 270°
            width, height = height, width
        
        # Base color
        color = self.component_colors.get(comp_type, '#95A5A6')
        
        # Special styling for matched components
        if matched_comp != -1:
            # Add pattern or border for matched components
            edgecolor = 'black'
            linewidth = 3
            alpha = 0.9
        else:
            edgecolor = 'darkgray'
            linewidth = 1
            alpha = 0.8
        
        # Create rectangle
        rect = patches.Rectangle(
            (x, y), width, height,
            facecolor=color, edgecolor=edgecolor,
            linewidth=linewidth, alpha=alpha
        )
        ax.add_patch(rect)
        
        # Add component label
        center_x = x + width / 2
        center_y = y + height / 2
        
        comp_name = self.component_names.get(comp_type, 'X')
        label = f"{comp_name}{comp_id}"
        
        ax.text(center_x, center_y, label, 
               ha='center', va='center', fontsize=8, fontweight='bold',
               color='white', 
               bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.7))
        
        # Add orientation indicator for transistors
        if comp_type in [0, 1] and width > 1 and height > 1:
            self._add_orientation_indicator(ax, x, y, width, height, orientation)
    
    def _add_orientation_indicator(self, ax: plt.Axes, x: int, y: int, 
                                 width: int, height: int, orientation: int):
        """Add visual indicator for component orientation."""
        # Simple arrow showing orientation
        center_x, center_y = x + width/2, y + height/2
        arrow_length = 0.3
        
        # Direction based on orientation
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # 0°, 90°, 180°, 270°
        dx, dy = directions[orientation % 4]
        
        ax.arrow(center_x, center_y, dx * arrow_length, dy * arrow_length,
                head_width=0.1, head_length=0.1, fc='yellow', ec='orange', 
                alpha=0.8, linewidth=2)
    
    def _draw_connections(self, ax: plt.Axes, circuit: nx.Graph, 
                         component_positions: Dict[int, Tuple[int, int, int]]):
        """Draw connections between components."""
        
        for edge in circuit.edges():
            comp1, comp2 = edge
            
            if comp1 in component_positions and comp2 in component_positions:
                pos1 = component_positions[comp1]
                pos2 = component_positions[comp2]
                
                # Calculate component centers
                attrs1 = circuit.nodes[comp1]
                attrs2 = circuit.nodes[comp2]
                
                width1 = attrs1.get('width', 1)
                height1 = attrs1.get('height', 1)
                width2 = attrs2.get('width', 1) 
                height2 = attrs2.get('height', 1)
                
                if pos1[2] in [1, 3]:  # 90° or 270°
                    width1, height1 = height1, width1
                if pos2[2] in [1, 3]:
                    width2, height2 = height2, width2
                
                center1 = (pos1[0] + width1/2, pos1[1] + height1/2)
                center2 = (pos2[0] + width2/2, pos2[1] + height2/2)
                
                # Draw connection line
                ax.plot([center1[0], center2[0]], [center1[1], center2[1]], 
                       'k--', alpha=0.6, linewidth=1.5, zorder=0)
                
                # Add small circles at connection points
                ax.plot(center1[0], center1[1], 'ko', markersize=3, alpha=0.7, zorder=1)
                ax.plot(center2[0], center2[1], 'ko', markersize=3, alpha=0.7, zorder=1)
    
    def _add_component_legend(self, ax: plt.Axes):
        """Add legend showing component types."""
        legend_elements = []
        
        for comp_type, color in self.component_colors.items():
            name = self.component_names.get(comp_type, f'Type {comp_type}')
            legend_elements.append(
                patches.Patch(facecolor=color, edgecolor='black', label=name)
            )
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
                 fontsize=10, title='Components', title_fontsize=12)
    
    def _add_layout_statistics(self, ax: plt.Axes, circuit: nx.Graph, 
                              component_positions: Dict[int, Tuple[int, int, int]]):
        """Add layout quality statistics to the plot."""
        if len(component_positions) == 0:
            return
        
        # Calculate statistics
        num_placed = len(component_positions)
        total_components = len(circuit.nodes) if circuit else num_placed
        
        # Bounding box
        positions = list(component_positions.values())
        xs = [pos[0] for pos in positions]
        ys = [pos[1] for pos in positions]
        bbox_area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1) if len(positions) > 1 else 1
        
        # Connected components proximity
        total_distance = 0
        connected_pairs = 0
        if circuit:
            for edge in circuit.edges():
                comp1, comp2 = edge
                if comp1 in component_positions and comp2 in component_positions:
                    pos1, pos2 = component_positions[comp1], component_positions[comp2]
                    distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                    total_distance += distance
                    connected_pairs += 1
        
        avg_distance = total_distance / connected_pairs if connected_pairs > 0 else 0
        
        # Matched pairs symmetry
        symmetric_pairs = 0
        total_matched_pairs = 0
        if circuit:
            for node_id in circuit.nodes():
                matched_comp = circuit.nodes[node_id].get('matched_component', -1)
                if (matched_comp != -1 and node_id < matched_comp and  # Count each pair once
                    node_id in component_positions and matched_comp in component_positions):
                    total_matched_pairs += 1
                    pos1, pos2 = component_positions[node_id], component_positions[matched_comp]
                    
                    # Simple symmetry check
                    if abs(pos1[1] - pos2[1]) <= 1:  # Horizontal symmetry
                        symmetric_pairs += 1
                    elif abs(pos1[0] - pos2[0]) <= 1:  # Vertical symmetry
                        symmetric_pairs += 1
        
        symmetry_ratio = symmetric_pairs / total_matched_pairs if total_matched_pairs > 0 else 1.0
        
        # Create statistics text
        stats_text = f"""Layout Statistics:
• Components: {num_placed}/{total_components}
• Bounding Box: {bbox_area} units²
• Avg Connection Distance: {avg_distance:.1f}
• Symmetry Ratio: {symmetry_ratio:.1%}"""
        
        # Add text box
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', 
               facecolor='wheat', alpha=0.8))
    
    def create_training_animation(self, 
                                episode_layouts: List[Dict],
                                save_path: str = "training_animation.gif",
                                interval: int = 500) -> FuncAnimation:
        """Create an animation showing layout evolution during training."""
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        def animate(frame):
            ax.clear()
            layout_data = episode_layouts[frame]
            
            circuit = layout_data.get('circuit')
            positions = layout_data.get('positions', {})
            episode = layout_data.get('episode', frame)
            reward = layout_data.get('reward', 0)
            
            ax.set_xlim(-0.5, self.grid_size - 0.5)
            ax.set_ylim(-0.5, self.grid_size - 0.5)
            ax.set_aspect('equal')
            ax.set_title(f"Episode {episode} - Reward: {reward:.2f}", fontsize=14)
            
            # Draw grid
            for i in range(self.grid_size + 1):
                ax.axhline(i - 0.5, color='lightgray', linewidth=0.5, alpha=0.7)
                ax.axvline(i - 0.5, color='lightgray', linewidth=0.5, alpha=0.7)
            
            # Draw layout
            if circuit and positions:
                self._draw_connections(ax, circuit, positions)
                for comp_id, (x, y, orientation) in positions.items():
                    if comp_id in circuit.nodes:
                        self._draw_component(ax, comp_id, x, y, orientation, 
                                           circuit.nodes[comp_id])
            
            ax.set_xlabel('Grid X')
            ax.set_ylabel('Grid Y')
            ax.invert_yaxis()
        
        anim = FuncAnimation(fig, animate, frames=len(episode_layouts),
                           interval=interval, blit=False, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=2)
        
        return anim
    
    def plot_training_metrics(self, 
                            metrics_data: Dict[str, List[float]], 
                            save_path: Optional[str] = None) -> plt.Figure:
        """Plot training metrics over time."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Reward components
        reward_components = ['symmetry', 'compactness', 'connectivity', 'completion']
        
        for i, component in enumerate(reward_components):
            if component in metrics_data and len(metrics_data[component]) > 0:
                axes[i].plot(metrics_data[component], linewidth=2)
                axes[i].set_title(f'{component.title()} Reward', fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Episode')
                axes[i].set_ylabel('Reward')
                axes[i].grid(True, alpha=0.3)
                
                # Add moving average
                if len(metrics_data[component]) > 20:
                    window_size = min(50, len(metrics_data[component]) // 10)
                    moving_avg = np.convolve(metrics_data[component], 
                                           np.ones(window_size)/window_size, mode='valid')
                    axes[i].plot(range(window_size-1, len(metrics_data[component])), 
                               moving_avg, 'r--', alpha=0.7, linewidth=2, label='Moving Avg')
                    axes[i].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def compare_layouts(self, 
                       layouts: List[Tuple[nx.Graph, Dict, str]], 
                       save_path: Optional[str] = None) -> plt.Figure:
        """Compare multiple layouts side by side."""
        
        n_layouts = len(layouts)
        cols = min(3, n_layouts)
        rows = (n_layouts + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
        if n_layouts == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else list(axes)
        else:
            axes = [ax for row in axes for ax in row]
        
        for i, (circuit, positions, title) in enumerate(layouts):
            ax = axes[i]
            
            ax.set_xlim(-0.5, self.grid_size - 0.5)
            ax.set_ylim(-0.5, self.grid_size - 0.5)
            ax.set_aspect('equal')
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            # Draw grid
            for j in range(self.grid_size + 1):
                ax.axhline(j - 0.5, color='lightgray', linewidth=0.5, alpha=0.7)
                ax.axvline(j - 0.5, color='lightgray', linewidth=0.5, alpha=0.7)
            
            # Draw layout
            if positions:
                self._draw_connections(ax, circuit, positions)
                for comp_id, (x, y, orientation) in positions.items():
                    if comp_id in circuit.nodes:
                        self._draw_component(ax, comp_id, x, y, orientation,
                                           circuit.nodes[comp_id])
            
            ax.set_xlabel('Grid X')
            ax.set_ylabel('Grid Y')
            ax.invert_yaxis()
        
        # Hide unused subplots
        for i in range(n_layouts, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


# Demo function
def demo_visualization():
    """Demonstrate the visualization capabilities."""
    
    from data.circuit_generator import AnalogCircuitGenerator
    
    # Create a sample circuit
    generator = AnalogCircuitGenerator()
    circuit = generator.generate_differential_pair()
    
    # Create sample positions
    positions = {
        0: (8, 10, 0),   # M1
        1: (12, 10, 0),  # M2 (matched)
        2: (10, 14, 0),  # Current source
        3: (8, 8, 0),    # R1
        4: (12, 8, 0),   # R2 (matched)
    }
    
    # Create visualizer
    visualizer = AnalogLayoutVisualizer()
    
    # Create layout visualization
    fig = visualizer.visualize_layout(
        circuit, positions, 
        title="Differential Pair Layout",
        show_connections=True,
        save_path="demo_layout.png"
    )
    
    plt.show()
    
    print("Demo visualization created! Check 'demo_layout.png'")


if __name__ == "__main__":
    demo_visualization()