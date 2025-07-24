# CLARA: Component Layout using Analog Reinforcement Architecture

CLARA is a reinforcement learning system that learns to automatically place components (MOSFETs, resistors, capacitors) for analog ICs on a 2D grid. The system focuses on relational placement, learning how components should be placed relative to each other rather than absolute positioning. The goal is to develop generalizable placement logic (symmetry, proximity, compactness) across different circuit topologies.

## Architecture Overview

### Core Components

- **AnalogLayoutEnv**: Custom Gym environment with Dict observation space and MultiDiscrete action space for relational placement
- **GNN-based Policy**: Graph Neural Network using PyTorch Geometric to process circuit topology 
- **Reward System**: Multi-component reward function emphasizing symmetry, compactness, connectivity, and completion
- **Circuit Generator**: Generates diverse analog circuits (differential pairs, current mirrors, OTAs, etc.)
- **Visualization Tools**: Real-time layout rendering and training progress visualization

### Key Features

- **Relational Actions**: Places components relative to existing ones using spatial relations (left-of, right-of, above, below, mirrored, etc.)
- **Graph Neural Networks**: Processes circuit connectivity and component relationships
- **Adaptive Rewards**: Curriculum learning with evolving reward weights during training
- **Component Matching**: Handles symmetric component pairs for analog circuit requirements
- **Multi-format Support**: JSON-based circuit representation for flexibility

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Ballistyxx/CLARA.git
cd CLARA

# Create virtual environment
python -m venv venvCLARA
source venvCLARA/bin/activate  # On Windows: clara_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Generate Sample Circuits **In Progress**

```bash
# Generate training dataset
cd data
python3 circuit_generator.py
```

### Train the Model

```bash
# Basic training
python3 train.py
```

### Run the model:
```bash
python3 run_model.py --episodes 1 --visualize
```

## Training Progress

The system uses PPO with custom policy network and implements curriculum learning:

- **Phase 1 (Episodes 0-300)**: Focus on basic placement and compactness
- **Phase 2 (Episodes 300-700)**: Balance all reward components  
- **Phase 3 (Episodes 700+)**: Emphasize quality metrics (symmetry, connectivity)

## Training Policy:

1. GraphNeuralNetwork (GNN)
- Processes the circuit topology as a graph
- Uses either Graph Attention Networks (GAT) or Graph Convolutional Networks (GCN)
- Converts circuit connectivity into meaningful embeddings
- Handles node features (component properties) and edge relationships
2. PlacementStateEncoder
- Encodes the current placement state (which components are already placed where)
- Uses 3D CNNs to process the placed components grid
- Provides spatial awareness of the current layout
3. RelationalActionNetwork
- Outputs relational placement actions instead of absolute coordinates
- Three action heads:
    - Target Component: Which unplaced component to place next
    - Spatial Relation: How to place it relative to existing components (left-of, right-of, above, below, mirrored, etc.)
    - Orientation: Component rotation/orientation
4. AnalogLayoutPolicy (Main Policy Class)
- Inherits from ActorCriticPolicy (Stable-Baselines3)
- Actor-Critic architecture:
- Actor: Decides what actions to take (component placement)
- Critic: Evaluates how good the current state is (value function)

## Configuration

Key hyperparameters in `train.py`:

```python
config = {
    'grid_size': 20,           # Layout grid dimensions
    'max_components': 10,      # Maximum components per circuit
    'total_timesteps': 200000, # Training duration
    'learning_rate': 3e-4,     # PPO learning rate
    'n_envs': 8,              # Parallel environments
}
```

## Performance Metrics

The system tracks several key metrics:

- **Success Rate**: Percentage of episodes with all components placed
- **Symmetry Score**: Quality of matched component placement
- **Compactness**: Efficiency of layout bounding box usage  
- **Connectivity**: Average distance between connected components