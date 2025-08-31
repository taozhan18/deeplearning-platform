#!/usr/bin/env python3
"""
Test script for MOOSE automation functionality
"""

import os
import sys
import json
import numpy as np
import subprocess
from pathlib import Path

# Add the scripts directory to the path
sys.path.append(os.path.dirname(__file__))

from moose_data_generator import MOOSEDataGenerator


def test_moose_executable():
    """Test if MOOSE executable is available and working"""
    moose_exec = "/home/zt/workspace/mymoose/mymoose-opt"
    
    # Check if executable exists
    if not os.path.exists(moose_exec):
        print(f"ERROR: MOOSE executable not found at {moose_exec}")
        return False
        
    # Check if executable is actually executable
    if not os.access(moose_exec, os.X_OK):
        print(f"ERROR: MOOSE executable at {moose_exec} is not executable")
        return False
    
    print(f"MOOSE executable found at {moose_exec}")
    return True


def test_simple_simulation():
    """Test running a simple MOOSE simulation"""
    moose_exec = "/home/zt/workspace/mymoose/mymoose-opt"
    input_file = "/home/zt/workspace/deeplearning-platform/data_loader/scripts/simple_test.i"
    test_params = "/home/zt/workspace/deeplearning-platform/data_loader/scripts/test_params.json"
    
    try:
        # Initialize MOOSE data generator
        generator = MOOSEDataGenerator(moose_exec)
        print("MOOSEDataGenerator initialized successfully")
        
        # Load test parameters
        with open(test_params, 'r') as f:
            param_sets = json.load(f)
        
        # Test running parametric simulations
        dataset_dir = "test_moose_dataset"
        output_dir = generator.run_parametric_sims(
            input_file, 
            param_sets, 
            output_dir=dataset_dir
        )
        print(f"Parametric simulations completed. Output directory: {output_dir}")
        
        # Check if dataset directory exists
        if not os.path.exists(dataset_dir):
            print(f"ERROR: Dataset directory {dataset_dir} not found")
            return False
        
        # Try to generate complete dataset
        print("Testing complete dataset generation...")
        combined_data = generator._extract_and_combine_data(dataset_dir, 'numpy', 'final_dataset')
        
        print("Dataset generation completed")
        print(f"Combined data keys: {combined_data.keys()}")
        
        print("MOOSE data generation test completed successfully")
        return True
        
    except Exception as e:
        print(f"ERROR in test_simple_simulation: {e}")
        import traceback
        traceback.print_exc()
        return False




def cleanup_test_files():
    """Clean up test files"""
    test_dirs = ["test_moose_dataset"]
    
    for dir_name in test_dirs:
        if os.path.exists(dir_name):
            import shutil
            shutil.rmtree(dir_name)
            print(f"Cleaned up {dir_name}")


def main():
    """Main test function"""
    print("Testing MOOSE Automation Functionality")
    print("=" * 50)
    
    # Test 1: Check MOOSE executable
    print("\n1. Testing MOOSE executable...")
    if not test_moose_executable():
        print("MOOSE executable test FAILED")
        return False
    
    # Test 2: Simple simulation
    print("\nTest 2: Testing simple MOOSE simulation")
    if not test_simple_simulation():
        print("Test 2 FAILED")
        return False
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)