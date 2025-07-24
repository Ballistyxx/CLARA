import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Any, Optional
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch


class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for processing circuit topology."""
    
    def __init__(self, 
                 node_features: int = 4,
                 hidden_dim: int = 64,
                 output_dim: int = 64,
                 num_layers: int = 3,
                 use_attention: bool = True):
        super().__init__()
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_attention = use_attention
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        
        if use_attention:
            # Use Graph Attention Network
            self.conv_layers.append(GATConv(node_features, hidden_dim, heads=4, concat=False))
            for _ in range(num_layers - 2):
                self.conv_layers.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            self.conv_layers.append(GATConv(hidden_dim, output_dim, heads=1, concat=False))
        else:
            # Use Graph Convolutional Network
            self.conv_layers.append(GCNConv(node_features, hidden_dim))
            for _ in range(num_layers - 2):
                self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.conv_layers.append(GCNConv(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(0.1)
        
        # Global pooling
        self.global_pool = nn.Linear(output_dim * 2, output_dim)  # concat mean + max pooling
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through GNN.
        
        Args:
            node_features: [N, node_features] - node feature matrix
            edge_index: [2, E] - edge connectivity
            batch: [N] - batch assignment for each node
            
        Returns:
            graph_embedding: [batch_size, output_dim] - graph-level representation
        """
        x = node_features
        
        # Apply graph convolution layers
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index)
            if i < len(self.conv_layers) - 1:  # No activation on last layer
                x = F.relu(x)
                x = self.dropout(x)
        
        # Global pooling to get graph-level representation
        if batch is None:
            # Single graph case
            mean_pool = torch.mean(x, dim=0, keepdim=True)
            max_pool = torch.max(x, dim=0, keepdim=True)[0]
        else:
            # Batch case
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
        
        # Combine pooling operations
        graph_embedding = torch.cat([mean_pool, max_pool], dim=-1)
        graph_embedding = self.global_pool(graph_embedding)
        
        return graph_embedding


class PlacementStateEncoder(nn.Module):
    """Encoder for the current placement state (placed components grid)."""
    
    def __init__(self, grid_size: int = 20, max_components: int = 10, output_dim: int = 64):
        super().__init__()
        
        self.grid_size = grid_size
        self.max_components = max_components
        
        # 3D CNN for processing placed components grid
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((4, 4, 4))  # Reduce spatial dimensions
        )
        
        # Flatten and project
        self.fc = nn.Linear(32 * 4 * 4 * 4, output_dim)
    
    def forward(self, placed_components: torch.Tensor) -> torch.Tensor:
        """
        Args:
            placed_components: [batch_size, grid_size, grid_size, max_components]
        Returns:
            placement_embedding: [batch_size, output_dim]
        """
        # Reshape for 3D conv: [batch_size, 1, grid_size, grid_size, max_components]
        x = placed_components.unsqueeze(1)
        
        # Apply 3D convolution
        x = self.conv3d(x)
        
        # Flatten and project
        x = x.view(x.size(0), -1)
        placement_embedding = self.fc(x)
        
        return placement_embedding


class RelationalActionNetwork(nn.Module):
    """Network that outputs relational placement actions."""
    
    def __init__(self, 
                 input_dim: int = 256,
                 max_components: int = 10,
                 num_relations: int = 7,
                 num_orientations: int = 4):
        super().__init__()
        
        self.max_components = max_components
        self.num_relations = num_relations
        self.num_orientations = num_orientations
        
        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Action heads
        self.target_component_head = nn.Linear(128, max_components)
        self.spatial_relation_head = nn.Linear(128, num_relations)
        self.orientation_head = nn.Linear(128, num_orientations)
        
        # Value head
        self.value_head = nn.Linear(128, 1)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [batch_size, input_dim] - combined features
            
        Returns:
            target_logits: [batch_size, max_components]
            relation_logits: [batch_size, num_relations]
            orientation_logits: [batch_size, num_orientations]
            values: [batch_size, 1]
        """
        shared = self.shared_net(features)
        
        target_logits = self.target_component_head(shared)
        relation_logits = self.spatial_relation_head(shared)
        orientation_logits = self.orientation_head(shared)
        values = self.value_head(shared)
        
        return target_logits, relation_logits, orientation_logits, values


class AnalogLayoutPolicy(ActorCriticPolicy):
    """Custom policy for analog layout placement using GNN and relational actions."""
    
    def __init__(self,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 lr_schedule: Schedule,
                 net_arch: Optional[Dict[str, Any]] = None,
                 activation_fn: type = nn.ReLU,
                 use_sde: bool = False,
                 log_std_init: float = 0.0,
                 sde_net_arch: Optional[list] = None,
                 use_expln: bool = False,
                 squash_output: bool = False,
                 features_extractor_class=None,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 normalize_images: bool = True,
                 optimizer_class: type = torch.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None):
        
        # Set default network architecture
        if net_arch is None:
            net_arch = {"gnn_hidden": 64, "gnn_layers": 3, "policy_hidden": 256}
        
        # Initialize parent class
        super().__init__(
            observation_space, action_space, lr_schedule,
            net_arch, activation_fn, use_sde, log_std_init,
            sde_net_arch, use_expln, squash_output,
            features_extractor_class, features_extractor_kwargs,
            normalize_images, optimizer_class, optimizer_kwargs
        )
        
        # Extract dimensions from observation space
        obs_dict = observation_space.spaces
        self.max_components = obs_dict["component_graph"].shape[0]
        self.grid_size = obs_dict["placed_components"].shape[0]
        self.node_features = obs_dict["netlist_features"].shape[1]
        
        # Extract action space dimensions
        self.num_relations = action_space.nvec[1]
        self.num_orientations = action_space.nvec[2]
        
        # Build network components
        self._build_networks()
    
    def _build_networks(self):
        """Build the GNN and action networks."""
        
        # Graph Neural Network for circuit topology
        self.gnn = GraphNeuralNetwork(
            node_features=self.node_features,
            hidden_dim=64,
            output_dim=64,
            num_layers=3,
            use_attention=True
        )
        
        # Placement state encoder
        self.placement_encoder = PlacementStateEncoder(
            grid_size=self.grid_size,
            max_components=self.max_components,
            output_dim=64
        )
        
        # Placement mask encoder
        self.mask_encoder = nn.Linear(self.max_components, 32)
        
        # Combined feature dimension
        combined_dim = 64 + 64 + 32  # GNN + placement + mask
        
        # Relational action network
        self.action_net = RelationalActionNetwork(
            input_dim=combined_dim,
            max_components=self.max_components,
            num_relations=self.num_relations,
            num_orientations=self.num_orientations
        )
    
    def extract_features(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from observation."""
        
        batch_size = obs["component_graph"].shape[0]
        
        # Process graph structure
        graph_embeddings = []
        for i in range(batch_size):
            # Convert adjacency matrix to edge index
            adj_matrix = obs["component_graph"][i]
            edge_index = self._adj_to_edge_index(adj_matrix)
            node_features = obs["netlist_features"][i]
            
            # Get graph embedding
            graph_emb = self.gnn(node_features, edge_index)
            graph_embeddings.append(graph_emb)
        
        graph_features = torch.stack(graph_embeddings, dim=0)
        
        # Process placement state
        placement_features = self.placement_encoder(obs["placed_components"])
        
        # Process placement mask
        mask_features = self.mask_encoder(obs["placement_mask"].float())
        
        # Combine all features
        combined_features = torch.cat([
            graph_features, placement_features, mask_features
        ], dim=-1)
        
        return combined_features
    
    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass to get actions and values.
        
        Returns:
            actions: [batch_size, 3] - target_component, relation, orientation
            values: [batch_size, 1] - state values
            log_probs: [batch_size] - log probabilities of selected actions
        """
        
        # Extract features
        features = self.extract_features(obs)
        
        # Get action logits and values
        target_logits, relation_logits, orientation_logits, values = self.action_net(features)
        
        # Apply placement mask to target component logits
        placement_mask = obs["placement_mask"].float()
        masked_target_logits = target_logits + (placement_mask - 1) * 1e9  # Large negative for unplaced
        
        # Sample actions
        if deterministic:
            target_action = torch.argmax(masked_target_logits, dim=-1)
            relation_action = torch.argmax(relation_logits, dim=-1)
            orientation_action = torch.argmax(orientation_logits, dim=-1)
        else:
            target_dist = torch.distributions.Categorical(logits=masked_target_logits)
            relation_dist = torch.distributions.Categorical(logits=relation_logits)
            orientation_dist = torch.distributions.Categorical(logits=orientation_logits)
            
            target_action = target_dist.sample()
            relation_action = relation_dist.sample()
            orientation_action = orientation_dist.sample()
        
        # Combine actions
        actions = torch.stack([target_action, relation_action, orientation_action], dim=-1)
        
        # Calculate log probabilities
        target_log_probs = F.log_softmax(masked_target_logits, dim=-1)
        relation_log_probs = F.log_softmax(relation_logits, dim=-1)
        orientation_log_probs = F.log_softmax(orientation_logits, dim=-1)
        
        selected_target_log_probs = target_log_probs.gather(1, target_action.unsqueeze(1)).squeeze(1)
        selected_relation_log_probs = relation_log_probs.gather(1, relation_action.unsqueeze(1)).squeeze(1)
        selected_orientation_log_probs = orientation_log_probs.gather(1, orientation_action.unsqueeze(1)).squeeze(1)
        
        log_probs = selected_target_log_probs + selected_relation_log_probs + selected_orientation_log_probs
        
        return actions, values.squeeze(-1), log_probs
    
    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO training.
        
        Returns:
            values: [batch_size] - state values
            log_probs: [batch_size] - log probabilities of actions
            entropy: [batch_size] - entropy of action distributions
        """
        
        # Extract features
        features = self.extract_features(obs)
        
        # Get action logits and values
        target_logits, relation_logits, orientation_logits, values = self.action_net(features)
        
        # Apply placement mask to target component logits
        placement_mask = obs["placement_mask"].float()
        masked_target_logits = target_logits + (placement_mask - 1) * 1e9
        
        # Create distributions
        target_dist = torch.distributions.Categorical(logits=masked_target_logits)
        relation_dist = torch.distributions.Categorical(logits=relation_logits)
        orientation_dist = torch.distributions.Categorical(logits=orientation_logits)
        
        # Extract individual actions
        target_actions = actions[:, 0]
        relation_actions = actions[:, 1]
        orientation_actions = actions[:, 2]
        
        # Calculate log probabilities
        target_log_probs = target_dist.log_prob(target_actions)
        relation_log_probs = relation_dist.log_prob(relation_actions)
        orientation_log_probs = orientation_dist.log_prob(orientation_actions)
        
        log_probs = target_log_probs + relation_log_probs + orientation_log_probs
        
        # Calculate entropy
        entropy = target_dist.entropy() + relation_dist.entropy() + orientation_dist.entropy()
        
        return values.squeeze(-1), log_probs, entropy
    
    def get_distribution(self, obs: Dict[str, torch.Tensor]):
        """Get action distributions for the current observation."""
        features = self.extract_features(obs)
        target_logits, relation_logits, orientation_logits, _ = self.action_net(features)
        
        # Apply placement mask
        placement_mask = obs["placement_mask"].float()
        masked_target_logits = target_logits + (placement_mask - 1) * 1e9
        
        return {
            "target": torch.distributions.Categorical(logits=masked_target_logits),
            "relation": torch.distributions.Categorical(logits=relation_logits),
            "orientation": torch.distributions.Categorical(logits=orientation_logits)
        }
    
    def predict_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict state values."""
        features = self.extract_features(obs)
        _, _, _, values = self.action_net(features)
        return values.squeeze(-1)
    
    def _adj_to_edge_index(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """Convert adjacency matrix to edge index format."""
        edge_indices = torch.nonzero(adj_matrix, as_tuple=False).t()
        return edge_indices.contiguous()


# Utility function to create the policy
def create_analog_layout_policy(env):
    """Create and return the custom policy for the analog layout environment."""
    return AnalogLayoutPolicy