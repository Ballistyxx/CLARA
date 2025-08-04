#!/usr/bin/env python3
"""
Relational policy network for analog layout with multi-head architecture.
Supports GNN-based component graph encoding and masked action selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Any, Union
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import MultiCategoricalDistribution
from gymnasium import spaces
import warnings

try:
    import torch_geometric
    from torch_geometric.nn import GATv2Conv, GCNConv, global_mean_pool
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    warnings.warn("torch_geometric not available, using fallback MLP encoder")


class GraphNeuralNetwork(nn.Module):
    """Graph neural network for component graph encoding."""
    
    def __init__(self, 
                 node_features: int = 8,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 use_gat: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_gat = use_gat
        
        if HAS_TORCH_GEOMETRIC and use_gat:
            # Use Graph Attention Network
            self.conv_layers = nn.ModuleList()
            self.conv_layers.append(GATv2Conv(node_features, hidden_dim, heads=4, concat=False))
            
            for _ in range(num_layers - 1):
                self.conv_layers.append(GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False))
                
        elif HAS_TORCH_GEOMETRIC:
            # Use Graph Convolutional Network
            self.conv_layers = nn.ModuleList()
            self.conv_layers.append(GCNConv(node_features, hidden_dim))
            
            for _ in range(num_layers - 1):
                self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        else:
            # Fallback to MLP (no graph structure)
            self.mlp = nn.Sequential(
                nn.Linear(node_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        self.dropout = nn.Dropout(0.1)
        self.norm_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GNN.
        
        Args:
            node_features: [batch_size, max_components, feature_dim]
            edge_index: [batch_size, 2, num_edges] (or None for fallback)
            
        Returns:
            Node embeddings: [batch_size, max_components, hidden_dim]
        """
        if not HAS_TORCH_GEOMETRIC:
            # Fallback MLP processing
            return self.mlp(node_features)
        
        batch_size, max_components, feature_dim = node_features.shape
        
        # Reshape for torch_geometric processing
        x = node_features.view(-1, feature_dim)  # [batch_size * max_components, feature_dim]
        
        # Create batch indices for graph processing
        batch_indices = torch.arange(batch_size).repeat_interleave(max_components).to(x.device)
        
        # Process through GNN layers
        for i, conv in enumerate(self.conv_layers):
            if edge_index is not None:
                x = conv(x, edge_index)
            else:
                # Fully connected fallback
                x = conv(x, torch.combinations(torch.arange(max_components).to(x.device), r=2).t())
            
            x = self.norm_layers[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Reshape back to batch format
        x = x.view(batch_size, max_components, self.hidden_dim)
        return x


class PlacementStateEncoder(nn.Module):
    """3D CNN encoder for placement state and row constraints."""
    
    def __init__(self, grid_size: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.grid_size = grid_size
        
        # Multi-channel input: placement_state + row_constraints_2d
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((16, 16)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((8, 8)),
            nn.Flatten()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, placement_state: torch.Tensor, row_constraints: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through placement state encoder.
        
        Args:
            placement_state: [batch_size, grid_size, grid_size]
            row_constraints: [batch_size, grid_size]
            
        Returns:
            Spatial embeddings: [batch_size, hidden_dim]
        """
        batch_size = placement_state.shape[0]
        
        # Expand row constraints to 2D
        row_constraints_2d = row_constraints.unsqueeze(-1).expand(-1, -1, self.grid_size)
        
        # Stack as multi-channel input
        spatial_input = torch.stack([placement_state, row_constraints_2d], dim=1)  # [batch, 2, grid, grid]
        
        # Process through CNN
        x = self.conv_layers(spatial_input)
        x = self.fc(x)
        
        return x


class RelationalActionHead(nn.Module):
    """Multi-head action network for relational placement."""
    
    def __init__(self, 
                 input_dim: int,
                 max_components: int = 16,
                 num_relations: int = 10,
                 num_orientations: int = 4,
                 num_regions: int = 4):
        super().__init__()
        
        self.max_components = max_components
        self.num_relations = num_relations
        self.num_orientations = num_orientations
        self.num_regions = num_regions
        
        # Separate heads for each action component
        self.target_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, max_components + 10)  # +10 for group targets
        )
        
        self.relation_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_relations)
        )
        
        self.orientation_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_orientations)
        )
        
        self.region_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_regions)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through action heads.
        
        Returns:
            Tuple of (target_logits, relation_logits, orientation_logits, region_logits)
        """
        target_logits = self.target_head(x)
        relation_logits = self.relation_head(x)
        orientation_logits = self.orientation_head(x)
        region_logits = self.region_head(x)
        
        return target_logits, relation_logits, orientation_logits, region_logits


class RelationalPolicyNetwork(nn.Module):
    """Complete relational policy network."""
    
    def __init__(self,
                 observation_space: spaces.Dict,
                 action_space: spaces.MultiDiscrete,
                 lr_schedule,
                 net_arch: Optional[List[int]] = None,
                 activation_fn: type = nn.ReLU,
                 normalize_images: bool = True):
        super().__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Network dimensions
        max_components = observation_space['component_features'].shape[0]
        feature_dim = observation_space['component_features'].shape[1]
        grid_size = observation_space['placement_state'].shape[0]
        
        hidden_dim = 128
        
        # Component graph encoder
        self.graph_encoder = GraphNeuralNetwork(
            node_features=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=3,
            use_gat=True
        )
        
        # Placement state encoder
        self.spatial_encoder = PlacementStateEncoder(
            grid_size=grid_size,
            hidden_dim=hidden_dim
        )
        
        # Locked groups encoder
        self.group_encoder = nn.Sequential(
            nn.Linear(10 * max_components, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Context fusion
        total_context_dim = hidden_dim + hidden_dim + hidden_dim // 2  # graph + spatial + groups
        self.context_fusion = nn.Sequential(
            nn.Linear(total_context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Action head
        self.action_head = RelationalActionHead(
            input_dim=hidden_dim,
            max_components=max_components,
            num_relations=len(action_space.nvec) > 1 and action_space.nvec[1] or 10,
            num_orientations=len(action_space.nvec) > 2 and action_space.nvec[2] or 4,
            num_regions=len(action_space.nvec) > 3 and action_space.nvec[3] or 4
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for actor-critic.
        
        Returns:
            (action_logits, values)
        """
        batch_size = obs['component_features'].shape[0]
        
        # Encode component graph
        node_features = obs['component_features']
        # Convert adjacency matrix to edge indices (simplified)
        adj_matrix = obs['component_graph']
        edge_index = None  # Could convert adj_matrix to edge_index for better GNN performance
        
        graph_embed = self.graph_encoder(node_features, edge_index)
        graph_context = torch.mean(graph_embed, dim=1)  # Global graph representation
        
        # Encode spatial state
        spatial_embed = self.spatial_encoder(obs['placement_state'], obs['row_constraints'])
        
        # Encode locked groups
        locked_groups_flat = obs['locked_groups'].view(batch_size, -1)
        group_embed = self.group_encoder(locked_groups_flat)
        
        # Fuse context
        combined_context = torch.cat([graph_context, spatial_embed, group_embed], dim=-1)
        fused_context = self.context_fusion(combined_context)
        
        # Generate action logits
        target_logits, relation_logits, orientation_logits, region_logits = self.action_head(fused_context)
        
        # Stack action logits for MultiCategorical distribution
        action_logits = torch.stack([target_logits, relation_logits, orientation_logits, region_logits], dim=-1)
        
        # Generate value estimates
        values = self.value_head(fused_context).squeeze(-1)
        
        return action_logits, values
    
    def get_distribution(self, obs: Dict[str, torch.Tensor]) -> MultiCategoricalDistribution:
        """Get action distribution for given observations."""
        action_logits, _ = self.forward(obs)
        
        # Split logits for each action component
        target_logits = action_logits[:, :self.action_space.nvec[0]]
        relation_logits = action_logits[:, self.action_space.nvec[0]:self.action_space.nvec[0]+self.action_space.nvec[1]]
        orientation_logits = action_logits[:, -self.action_space.nvec[3]-self.action_space.nvec[2]:-self.action_space.nvec[3]]
        region_logits = action_logits[:, -self.action_space.nvec[3]:]
        
        logits_list = [target_logits, relation_logits, orientation_logits, region_logits]
        
        return MultiCategoricalDistribution(logits_list)
    
    def predict_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict state values."""
        _, values = self.forward(obs)
        return values


class RelationalPolicy(ActorCriticPolicy):
    """
    Custom ActorCritic policy for relational analog layout placement.
    Integrates with Stable-Baselines3 PPO training.
    """
    
    def __init__(self,
                 observation_space: spaces.Dict,
                 action_space: spaces.MultiDiscrete,
                 lr_schedule,
                 net_arch: Optional[List[int]] = None,
                 activation_fn: type = nn.ReLU,
                 ortho_init: bool = True,
                 use_sde: bool = False,
                 log_std_init: float = 0.0,
                 full_std: bool = True,
                 use_expln: bool = False,
                 squash_output: bool = False,
                 features_extractor_class = None,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 share_features_extractor: bool = True,
                 normalize_images: bool = True,
                 optimizer_class: type = torch.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None):
        
        # Initialize with a dummy features extractor (we'll override)
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs
        )
        
        # Replace the default networks with our custom network
        self.policy_net = RelationalPolicyNetwork(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            normalize_images=normalize_images
        )
        
        # Override the action and value networks
        self.action_net = self.policy_net
        self.value_net = self.policy_net
        
        # Set up optimizer
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
    
    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the policy.
        
        Returns:
            actions, values, log_probs
        """
        # Get action distribution
        distribution = self.get_distribution(obs)
        
        # Sample actions
        actions = distribution.get_actions(deterministic=deterministic)
        log_probs = distribution.log_prob(actions)
        
        # Get values
        values = self.predict_values(obs)
        
        return actions, values, log_probs
    
    def get_distribution(self, obs: Dict[str, torch.Tensor]) -> MultiCategoricalDistribution:
        """Get action distribution."""
        return self.policy_net.get_distribution(obs)
    
    def predict_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict values."""
        return self.policy_net.predict_values(obs)
    
    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for policy optimization.
        
        Returns:
            values, log_probs, entropy
        """
        distribution = self.get_distribution(obs)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.predict_values(obs)
        
        return values, log_probs, entropy


# Factory function for creating the policy
def create_relational_policy(**kwargs):
    """Factory function to create RelationalPolicy with custom kwargs."""
    return RelationalPolicy