# CLARA: Component Layout using Analog Reinforcement Architecture

CLARA is a reinforcement learning system that learns to automatically place analog IC components with industry-standard constraints and patterns. The system features **PFET/NFET row discipline**, advanced **symmetry patterns** (mirroring, common-centroid, interdigitation), and **analog-friendly optimization** for real-world circuit layout.

## Key Features

### Analog-Friendly Layout Engine
- **Row Discipline**: Enforces PMOS/NMOS row partitioning with automatic rail alignment
- **Symmetry Patterns**: Supports mirroring, common-centroid, and interdigitated matching
- **Pattern Locking**: Preserves locked symmetry groups during placement and legalization
- **Real Circuit Integration**: Direct SPICE netlist parsing with device model support

### Advanced RL Architecture
- **Relational Actions**: Places components using spatial relations (left-of, mirror-about, common-centroid)
- **Multi-Head Policy**: Separate networks for target selection, relation type, and orientation
- **Action Masking**: Intelligent masking prevents invalid placements and constraint violations
- **Graph Neural Networks**: Processes circuit topology with GAT/GCN layers

### Comprehensive Metrics & Visualization
- **Analog Metrics**: Row consistency, symmetry accuracy, rail alignment, pattern validity
- **Routing Proxies**: Crossing count, congestion variance, connection distance
- **Interactive Visualization**: Symmetry axes, density heatmaps, crossing analysis
- **Acceptance Criteria**: Industry-inspired thresholds for layout quality

## Architecture Overview

```
CLARA Analog Layout System
├── env/
│   ├── layout_grid.py          # Grid with row partitions & constraint checking
│   └── analog_layout_env.py    # RL environment with relational actions
├── policy/
│   └── relational_policy.py    # Multi-head GNN policy network
├── legalizer/
│   └── row_snap.py            # Pattern-preserving legalization
├── metrics/
│   └── metrics.py             # Comprehensive analog metrics
├── viz/
│   └── overlays.py            # Advanced layout visualization
└── configs/
    ├── default_rewards.yaml   # Reward weights
    ├── curriculum_stages.yaml # Training curriculum  
    ├── environment_config.yaml # Environment settings
    └── training_config.yaml   # PPO hyperparameters
```

## Core Capabilities

### 1. PFET/NFET Row Discipline
- **Automatic Row Partitioning**: PMOS rows (near VDD), NMOS rows (near VSS), mixed rows (passives)
- **Hard Constraints**: Components cannot be placed in incompatible rows
- **Rail Alignment Scoring**: Rewards source/drain proximity to power rails

### 2. Advanced Symmetry Patterns
```python
# Supported spatial relations
SpatialRelation = {
    LEFT_OF, RIGHT_OF, ABOVE, BELOW,     # Basic placement
    MIRRORED_X, MIRRORED_Y,              # Mirror patterns
    MIRROR_ABOUT,                        # Custom axis mirroring
    COMMON_CENTROID,                     # CC patterns (ABBA, etc.)
    INTERDIGITATE                        # Interdigitated layouts
}
```

### 3. Comprehensive Metrics
- **Completion**: Fraction of components successfully placed
- **Row Consistency**: Components in correct row types (target: ≥95%)
- **Symmetry Score**: Accuracy of matched component placement (target: ≥90%)
- **Pattern Validity**: Preservation of locked groups (target: ≥90%)
- **Crossings**: Net intersection count (minimize)
- **Congestion Variance**: Routing density uniformity
- **Analog Score**: Combined analog-friendliness metric

## Quick Start

### Installation
```bash
# Clone repository
git clone <repository-url>
cd CLARA

# Create virtual environment  
python -m venv venvCLARA
source venvCLARA/bin/activate  # Linux/Mac
# venvCLARA\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Basic Training
```bash
# Train with default curriculum (3→6→12→32 components)
python train.py

# Train with custom settings
python train.py --config configs/training_config.yaml --total-timesteps 500000

# Enable Weights & Biases logging
USE_WANDB=true python train.py
```

### Model Inference & Evaluation
```bash
# Run trained model on test circuits
python run_model.py --episodes 10

# Test specific circuit types
python run_model.py --test-circuits --model logs/best_model.zip

# Generate comprehensive visualization
python run_model.py --episodes 1 --render --save-viz
```

### SPICE Circuit Integration
```bash
# Parse SPICE netlists
python enhanced_spice_parser.py --input data/netlists/ldo.spice

# Train on real circuits
python train_spice_real.py --spice-dir data/netlists
```

## Training Results & Acceptance Criteria

### Success Thresholds (32-component circuits)
- **Completion Rate**: ≥95% (place all components)
- **Row Consistency**: ≥95% (correct row placement)
- **Symmetry Score**: ≥90% (matched pair accuracy)  
- **Pattern Validity**: ≥90% (preserved locked groups)
- **Crossings**: ≤40% reduction vs. random baseline
- **Violation Count**: 0 (no overlaps/constraint violations)

### Training Curriculum
1. **Stage 1** (50K steps): Basic placement, 3-6 components, row discipline
2. **Stage 2** (100K steps): Symmetry patterns, 6-10 components, mirroring
3. **Stage 3** (150K steps): Advanced patterns, 10-16 components, common-centroid
4. **Stage 4** (200K steps): Large circuits, 16-32 components, optimization

## Configuration

### Environment Setup
```yaml
# configs/environment_config.yaml
environment:
  grid:
    default_size: 64
    adaptive_sizing: true
  rows:
    pmos_rows: 3
    nmos_rows: 3
    mixed_rows: 2
  constraints:
    enforce_row_discipline: true
    min_spacing: 1
```

### Reward Tuning
```yaml
# configs/default_rewards.yaml
reward_weights:
  completion: 4.0
  row_consistency: 2.0
  symmetry: 3.0
  pattern_validity: 2.0
  crossings: -1.5
  overlap_penalty: -5.0
```


## Advanced Usage

### Custom Circuit Generation
```python
from data.circuit_generator import AnalogCircuitGenerator

generator = AnalogCircuitGenerator()
circuit = generator.generate_ota(num_components=12, matching_pairs=3)
```

### Metrics Analysis
```python
from metrics.metrics import calculate_layout_metrics

metrics = calculate_layout_metrics(layout_grid, circuit)
print(f"Analog Score: {metrics.analog_score:.3f}")
print(f"Symmetry: {metrics.symmetry_score:.3f}")
print(f"Row Consistency: {metrics.row_consistency:.3f}")
```

### Attempted Legalization
```python
from legalizer.row_snap import legalize_layout, LegalizationMode

result = legalize_layout(
    layout_grid, 
    mode=LegalizationMode.PRESERVE_PATTERNS
)
print(f"Legalization success: {result.success}")
print(f"Patterns preserved: {result.patterns_preserved}")
```

## Testing & Validation

## Key Algorithms

### Row Snap Legalizer
1. **Phase 1**: Snap components to appropriate row boundaries
2. **Phase 2**: Resolve overlaps via sliding with pattern preservation
3. **Phase 3**: Repair broken symmetry/common-centroid patterns

### Curriculum Learning
- Progressive complexity increase (3→32 components)
- Adaptive reward weight evolution
- Automatic stage advancement based on success criteria

## Research Applications

### Analog IC Design
- **Operational Amplifiers**: Multi-stage OTA layout optimization
- **Data Converters**: SAR ADC, Delta-Sigma modulator placement
- **Power Management**: LDO, bandgap reference, bias circuits
- **RF Circuits**: VCO, mixer, LNA layout with matching constraints

### Layout Methodology
- **Design Rule Checking**: Constraint satisfaction with RL
- **Pattern Recognition**: Learning analog layout idioms
- **Multi-Objective Optimization**: Balancing area, performance, yield
- **Technology Scaling**: Transfer learning across process nodes



### Testing Guidelines
- Write unit tests for new components
- Maintain >90% code coverage
- Test with deterministic seeds for reproducibility
- Include integration tests for full workflows

## License & Citation

This project is licensed under the MIT License. If you use CLARA in your research, please cite:




---

For questions, bug reports, or feature requests, please open an issue on GitHub or contact the development team.
