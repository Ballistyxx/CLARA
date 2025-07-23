# CLARA: Component Layout using Analog Reinforcement Architecture

CLARA is a reinforcement learning system that learns to automatically place components (MOSFETs, resistors, capacitors) for analog ICs on a 2D grid. The system focuses on relational placement, learning how components should be placed relative to each other rather than absolute positioning. The goal is to develop generalizable placement logic (symmetry, proximity, compactness) across different circuit topologies.

## 🏗️ Architecture Overview

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

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd CLARA

# Create virtual environment
python -m venv clara_env
source clara_env/bin/activate  # On Windows: clara_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Generate Sample Circuits

```bash
# Generate training dataset
cd data
python circuit_generator.py
```

### Train the Model

```bash
# Basic training
python train.py

# With Weights & Biases logging
USE_WANDB=true python train.py
```

### Visualize Results

```bash
# Create layout visualizations
python visualize.py
```

## 📊 Training Progress

The system uses PPO with custom policy network and implements curriculum learning:

- **Phase 1 (Episodes 0-300)**: Focus on basic placement and compactness
- **Phase 2 (Episodes 300-700)**: Balance all reward components  
- **Phase 3 (Episodes 700+)**: Emphasize quality metrics (symmetry, connectivity)

## 🔧 Configuration

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

## 📁 Project Structure

```
CLARA/
├── analog_layout_env.py    # Gym environment
├── policy.py               # GNN-based PPO policy  
├── reward.py               # Modular reward functions
├── train.py                # Training script with callbacks
├── visualize.py            # Layout visualization tools
├── data/
│   ├── circuit_generator.py   # Circuit topology generator
│   └── circuits/              # Generated circuit files
├── tests/                  # Unit tests
└── logs/                   # Training logs and checkpoints
```

## 🎯 Deliverables Status

✅ **analog_layout_env.py** — Gym-compatible environment with Dict observation space and MultiDiscrete actions  
✅ **policy.py** — GNN-based custom PPO policy with relational action heads  
✅ **train.py** — SB3 training script with TensorBoard logging and curriculum learning  
✅ **reward.py** — Modular reward components (symmetry, compactness, connectivity)  
✅ **data/circuit_generator.py** — Analog circuit dataset generation  
✅ **visualize.py** — Layout renderer with matplotlib  
✅ **README.md** — Project overview and setup instructions

## 🧪 Testing

```bash
# Run basic environment test
python -c "
from analog_layout_env import AnalogLayoutEnv
env = AnalogLayoutEnv()
obs = env.reset()
print('Environment initialized successfully!')
print(f'Observation keys: {obs.keys()}')
"
```

## 📈 Performance Metrics

The system tracks several key metrics:

- **Success Rate**: Percentage of episodes with all components placed
- **Symmetry Score**: Quality of matched component placement
- **Compactness**: Efficiency of layout bounding box usage  
- **Connectivity**: Average distance between connected components

## 🤝 Contributing

This is a research project focused on analog IC placement using reinforcement learning. The implementation emphasizes educational clarity and experimental flexibility over production optimization.

## 📄 License

Research and educational use.# CLARA
