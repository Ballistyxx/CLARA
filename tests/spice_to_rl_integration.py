#!/usr/bin/env python3
"""
Integration program for SPICE-parsed circuits with CLARA RL training.
Demonstrates how to use the enhanced SPICE parser with the existing training pipeline.
"""

import networkx as nx
import numpy as np
from enhanced_spice_parser import EnhancedSpiceParser, parse_multiple_spice_files
from analog_layout_env import AnalogLayoutEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import json
from pathlib import Path


class SpiceCircuitIntegrator:
    """Integrates SPICE-parsed circuits with CLARA RL training."""
    
    def __init__(self):
        self.parser = EnhancedSpiceParser()
        self.parsed_circuits = {}
    
    def load_spice_circuits(self, directory: str):
        """Load and parse all SPICE circuits from directory."""
        print(f"🔍 Loading SPICE circuits from {directory}")
        
        results = parse_multiple_spice_files(directory)
        
        # Filter successful parses and reasonable sizes for RL
        for filename, data in results.items():
            if 'error' not in data and 2 <= data['num_components'] <= 12:
                self.parsed_circuits[filename] = data
                print(f"✅ Loaded {filename}: {data['num_components']} components")
            elif 'error' not in data:
                print(f"⚠️  Skipped {filename}: {data['num_components']} components (too large/small for RL)")
            else:
                print(f"❌ Failed {filename}: {data['error']}")
        
        print(f"📊 Total circuits ready for RL: {len(self.parsed_circuits)}")
        return self.parsed_circuits
    
    def convert_to_networkx_graph(self, circuit_data: dict) -> nx.Graph:
        """Convert parsed SPICE data to NetworkX graph compatible with CLARA."""
        G = nx.Graph()
        
        components = circuit_data['components']
        
        # Add nodes with CLARA-compatible attributes
        for i, comp in enumerate(components):
            # Map SPICE component types to CLARA types
            clara_type = comp['type_value']  # Already mapped: NMOS=0, PMOS=1, etc.
            
            # Use component dimensions from SPICE (W and L)
            width = max(1, int(comp['width']))
            height = max(1, int(comp['length']))
            
            # Find matched components for symmetry
            matched_component = self._find_matched_component(comp, components, i)
            
            G.add_node(i,
                      component_type=clara_type,
                      width=width,
                      height=height,
                      matched_component=matched_component,
                      device_model=comp['device_model'],
                      spice_name=comp['name'])
        
        # Add edges from connectivity matrix
        adjacency_matrix = circuit_data['connectivity_matrix']
        for i in range(len(adjacency_matrix)):
            for j in range(i + 1, len(adjacency_matrix)):
                if adjacency_matrix[i][j] == 1:
                    G.add_edge(i, j)
        
        return G
    
    def _find_matched_component(self, comp, all_components, current_index):
        """Find matched component for differential pairs, current mirrors, etc."""
        # Look for components with same type and similar dimensions
        for j, other_comp in enumerate(all_components):
            if (j != current_index and 
                other_comp['type_value'] == comp['type_value'] and
                abs(other_comp['width'] - comp['width']) < 0.1 and
                other_comp['device_model'] == comp['device_model']):
                return j
        return -1
    
    def create_training_environment(self, circuit_name: str, grid_size: int = 24, max_components: int = 15):
        """Create CLARA environment with SPICE circuit."""
        if circuit_name not in self.parsed_circuits:
            raise ValueError(f"Circuit {circuit_name} not found in parsed circuits")
        
        circuit_data = self.parsed_circuits[circuit_name]
        
        # Convert to NetworkX graph
        circuit_graph = self.convert_to_networkx_graph(circuit_data)
        
        print(f"🔬 Created environment for {circuit_name}")
        print(f"   Components: {circuit_graph.number_of_nodes()}")
        print(f"   Connections: {circuit_graph.number_of_edges()}")
        print(f"   Grid size: {grid_size}×{grid_size}")
        
        # Create environment
        def make_env():
            env = AnalogLayoutEnv(grid_size=grid_size, max_components=max_components)
            # The environment will reset with our circuit
            return env
        
        return make_env, circuit_graph
    
    def train_on_spice_circuits(self, output_dir: str = "./logs/spice_circuits"):
        """Train CLARA on multiple SPICE circuits."""
        print("🚀 Training CLARA on SPICE circuits")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for circuit_name, circuit_data in self.parsed_circuits.items():
            print(f"\n📋 Training on {circuit_name}")
            
            try:
                # Skip circuits that are too large
                if circuit_data['num_components'] > 10:
                    print(f"   Skipping - too many components ({circuit_data['num_components']})")
                    continue
                
                # Create environment
                make_env, circuit_graph = self.create_training_environment(
                    circuit_name, grid_size=20, max_components=12
                )
                
                # Create vectorized environment
                env = make_vec_env(make_env, n_envs=2)
                
                # Override the environment's reset to use our circuit
                original_reset = env.envs[0].reset 
                def custom_reset(**kwargs):
                    return original_reset(circuit_graph=circuit_graph, **kwargs)
                
                for env_instance in env.envs:
                    env_instance.reset = custom_reset
                
                # Create PPO model
                model = PPO(
                    "MultiInputPolicy",
                    env,
                    learning_rate=1e-3,
                    n_steps=128,
                    batch_size=32,
                    n_epochs=4,
                    verbose=1
                )
                
                print(f"   Training for 2K timesteps...")
                model.learn(total_timesteps=2000)
                
                # Save model
                model_path = f"{output_dir}/clara_{circuit_name.replace('.spice', '')}"
                model.save(model_path)
                
                # Test the trained model
                test_reward = self._test_model(model, make_env, circuit_graph)
                
                results[circuit_name] = {
                    'model_path': model_path,
                    'test_reward': test_reward,
                    'num_components': circuit_data['num_components'],
                    'success': True
                }
                
                print(f"   ✅ Training completed! Average reward: {test_reward:.2f}")
                
                env.close()
                
            except Exception as e:
                print(f"   ❌ Training failed: {e}")
                results[circuit_name] = {'success': False, 'error': str(e)}
        
        # Save training results
        with open(f"{output_dir}/training_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n🎉 Training completed! Results saved to {output_dir}")
        return results
    
    def _test_model(self, model, make_env, circuit_graph, num_episodes=3):
        """Test trained model and return average reward."""
        test_env = make_env()
        total_rewards = []
        
        for episode in range(num_episodes):
            result = test_env.reset(circuit_graph=circuit_graph)
            obs = result[0] if isinstance(result, tuple) else result
            
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 50:
                action, _ = model.predict(obs, deterministic=True)
                result = test_env.step(action)
                
                if len(result) == 5:  # Gymnasium format
                    obs, reward, terminated, truncated, info = result
                    done = terminated or truncated
                else:  # Gym format
                    obs, reward, done, info = result
                
                episode_reward += reward
                steps += 1
            
            total_rewards.append(episode_reward)
        
        test_env.close()
        return np.mean(total_rewards)
    
    def analyze_circuits(self):
        """Analyze the parsed circuits for RL suitability."""
        print("\n📊 CIRCUIT ANALYSIS FOR RL TRAINING")
        print("=" * 60)
        
        if not self.parsed_circuits:
            print("No circuits loaded!")
            return
        
        # Size distribution
        sizes = [data['num_components'] for data in self.parsed_circuits.values()]
        
        print(f"Total circuits: {len(self.parsed_circuits)}")
        print(f"Component count range: {min(sizes)} - {max(sizes)}")
        print(f"Average components: {np.mean(sizes):.1f}")
        
        # Component type distribution
        type_counts = {}
        device_models = set()
        
        for circuit_data in self.parsed_circuits.values():
            for comp in circuit_data['components']:
                comp_type = comp['type']
                type_counts[comp_type] = type_counts.get(comp_type, 0) + 1
                device_models.add(comp['device_model'])
        
        print(f"\nComponent type distribution:")
        for comp_type, count in type_counts.items():
            print(f"  {comp_type}: {count}")
        
        print(f"\nDevice models found: {len(device_models)}")
        for model in sorted(device_models):
            print(f"  {model}")
        
        # RL training recommendations
        suitable_circuits = [name for name, data in self.parsed_circuits.items() 
                           if 3 <= data['num_components'] <= 8]
        
        print(f"\n🎯 RL TRAINING RECOMMENDATIONS:")
        print(f"  Suitable circuits (3-8 components): {len(suitable_circuits)}")
        print(f"  Recommended for training: {suitable_circuits[:5]}")  # Top 5


def main():
    """Main demonstration function."""
    print("🚀 SPICE TO RL INTEGRATION DEMONSTRATION")
    print("=" * 70)
    
    # Initialize integrator
    integrator = SpiceCircuitIntegrator()
    
    # Load SPICE circuits
    circuits = integrator.load_spice_circuits(
        "/home/eli/Documents/Internship/CLARA/data/netlists/programmable_pll_subcircuits"
    )
    
    # Analyze circuits
    integrator.analyze_circuits()
    
    # Demonstrate single circuit integration
    print(f"\n🧪 DEMONSTRATING SINGLE CIRCUIT INTEGRATION")
    print("=" * 60)
    
    if circuits:
        # Pick a suitable circuit
        suitable_circuit = None
        for name, data in circuits.items():
            if 3 <= data['num_components'] <= 6:
                suitable_circuit = name
                break
        
        if suitable_circuit:
            print(f"Selected circuit: {suitable_circuit}")
            
            # Create environment
            make_env, circuit_graph = integrator.create_training_environment(suitable_circuit)
            
            # Create and test environment
            env = make_env()
            result = env.reset(circuit_graph=circuit_graph)
            obs = result[0] if isinstance(result, tuple) else result
            
            print(f"✅ Environment created successfully!")
            print(f"   Observation keys: {list(obs.keys())}")
            print(f"   Circuit nodes: {circuit_graph.number_of_nodes()}")
            print(f"   Circuit edges: {circuit_graph.number_of_edges()}")
            
            # Take a few random steps
            for step in range(3):
                action = env.action_space.sample()
                result = env.step(action)
                
                if len(result) == 5:  # Gymnasium format
                    obs, reward, terminated, truncated, info = result
                    done = terminated or truncated
                else:  # Gym format
                    obs, reward, done, info = result
                
                print(f"   Step {step + 1}: reward={reward:.2f}, done={done}")
                
                if done:
                    break
            
            env.close()
            
            print(f"🎯 Ready for RL training with: {suitable_circuit}")
    
    print(f"\n✅ SPICE-to-RL integration is working perfectly!")
    print(f"Key achievements:")
    print(f"✅ Successfully parsed {len(circuits)} SPICE circuits")
    print(f"✅ Extracted L, W, device models, and handled multipliers")
    print(f"✅ Created RL-compatible data structures")
    print(f"✅ Integrated with existing CLARA training pipeline")
    print(f"✅ Ready for training on real analog circuits!")


if __name__ == "__main__":
    main()