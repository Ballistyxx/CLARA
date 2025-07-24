#!/usr/bin/env python3
"""
Installation and testing script for CLARA.
This script will install dependencies and run basic tests.
"""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a command and print results."""
    print(f"\n  {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        print("Success!")
        if result.stdout.strip():
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed with return code {e.returncode}")
        if e.stdout.strip():
            print(f"Output: {e.stdout.strip()}")
        if e.stderr.strip():
            print(f"Error: {e.stderr.strip()}")
        return False


def install_dependencies():
    """Install required dependencies."""
    print("Installing CLARA dependencies...")
    
    # Core dependencies first
    core_deps = [
        "numpy>=1.21.0",
        "torch>=1.12.0", 
        "gym>=0.21.0",
        "networkx>=2.8",
        "matplotlib>=3.5.0"
    ]
    
    print("Installing core dependencies...")
    for dep in core_deps:
        success = run_command(f"pip install {dep}", f"Installing {dep}")
        if not success:
            print(f"Failed to install {dep}, continuing...")
    
    # Try to install remaining dependencies
    remaining_deps = [
        "stable-baselines3>=1.7.0",
        "torch-geometric>=2.2.0",
        "seaborn>=0.11.0",
        "tqdm>=4.64.0"
    ]
    
    print("Installing additional dependencies...")
    for dep in remaining_deps:
        success = run_command(f"pip install {dep}", f"Installing {dep}")
        if not success:
            print(f"Failed to install {dep}, continuing...")


def create_simple_test():
    """Create a simple test that doesn't require all dependencies."""
    test_code = '''
import sys
import traceback

def test_basic_imports():
    """Test basic imports."""
    try:
        import numpy as np
        print("NumPy imported")
        
        import torch
        print("PyTorch imported")

        try:
            import gymnasium as gym
            print("Gymnasium imported")
        except ImportError:
            import gym
            print("OpenAI Gym imported")

        import networkx as nx
        print("NetworkX imported")
        
        import matplotlib.pyplot as plt
        print("Matplotlib imported")
        
        return True
    except Exception as e:
        print(f"Import failed: {e}")
        return False

def test_environment():
    """Test environment creation."""
    try:
        # Simple import test
        from analog_layout_env import AnalogLayoutEnv
        print("AnalogLayoutEnv imported successfully")
        
        # Create environment
        env = AnalogLayoutEnv(grid_size=5, max_components=3)
        print("Environment created")
        
        # Test reset
        obs = env.reset()
        print("Environment reset successful")
        print(f"Observation keys: {list(obs.keys())}")
        
        # Test action space
        action = env.action_space.sample()
        print(f"Sample action generated: {action}")
        
        return True
    except Exception as e:
        print(f"Environment test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("CLARA Basic Test")
    print("="*40)
    
    import_success = test_basic_imports()
    env_success = test_environment() if import_success else False
    
    print("\\n" + "="*40)
    if import_success and env_success:
        print("Basic tests passed! CLARA is ready.")
        sys.exit(0)
    else:
        print("Some tests failed. Check dependencies.")
        sys.exit(1)
'''
    
    with open("basic_test.py", "w") as f:
        f.write(test_code)
    
    return "basic_test.py"


def main():
    """Main installation and test routine."""
    print("CLARA Installation and Test Script")
    print("="*50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Python 3.8 or higher required")
        return 1

    print(f"Python {sys.version_info.major}.{sys.version_info.minor} detected")

    # Install dependencies
    install_dependencies()
    
    # Create and run basic test
    test_file = create_simple_test()
    print(f"\nRunning basic tests...")
    
    success = run_command(f"python3 {test_file}", "Running basic functionality test")
    
    if success:
        print("\nInstallation and basic tests completed successfully!")
        print("\nNext steps:")
        print("1. Run: python3 data/circuit_generator.py  (to generate training data)")
        print("2. Run: python3 train.py  (to start training)")
        print("3. Run: python3 test_environment.py  (for comprehensive tests)")
    else:
        print("\nSome issues detected. Please check error messages above.")
        print("You may need to install missing dependencies manually.")
    
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())