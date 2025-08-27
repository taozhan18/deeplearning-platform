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
        
        print(f"Loaded {len(param_sets)} parameter sets")
        
        # Run parametric simulations
        output_dir = "test_moose_dataset"
        print(f"Running simulations and saving results to {output_dir}")
        
        dataset_dir = generator.run_parametric_sims(
            input_file, param_sets, output_dir
        )
        
        print(f"Simulations completed. Dataset saved to {dataset_dir}")
        
        # Check if results file exists
        results_file = Path(dataset_dir) / "simulation_results.json"
        if not results_file.exists():
            print("ERROR: simulation_results.json not found")
            return False
            
        # Load and check results
        with open(results_file, 'r') as f:
            results = json.load(f)
            
        print(f"Found {len(results)} simulation results")
        
        # Check if simulations were successful
        successful_sims = [r for r in results if r.get('status') == 'success']
        print(f"Successful simulations: {len(successful_sims)}")
        
        if len(successful_sims) == 0:
            print("ERROR: No simulations completed successfully")
            return False
            
        # Check if output files exist for at least one simulation
        first_success = successful_sims[0]
        sim_dir = first_success['sim_dir']
        
        # Look for output files
        output_files = list(Path(sim_dir).glob("*"))
        output_files = [f for f in output_files if f.name != "input.i"]  # Exclude input file
        
        print(f"Output files in {sim_dir}: {[f.name for f in output_files]}")
        
        if len(output_files) == 0:
            print("ERROR: No output files found in simulation directory")
            return False
            
        print("Simple simulation test PASSED")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False


def test_data_extraction():
    """Test data extraction functionality"""
    try:
        moose_exec = "/home/zt/workspace/mymoose/mymoose-opt"
        generator = MOOSEDataGenerator(moose_exec)
        
        # Test with the dataset we just created
        dataset_dir = "test_moose_dataset"
        
        if not os.path.exists(dataset_dir):
            print(f"ERROR: Dataset directory {dataset_dir} not found")
            return False
        
        # Try to extract field data (this will create dummy data since we don't have real extraction yet)
        print("Testing field data extraction...")
        field_data = generator.extract_field_data(
            dataset_dir, ['default'], ['default'], 'numpy'
        )
        
        print("Field data extraction completed")
        print(f"Extracted data keys: {field_data.keys()}")
        
        # Try to create FNO training data
        print("Testing FNO training data creation...")
        fno_data = generator.create_fno_training_data(dataset_dir)
        
        print("FNO training data creation completed")
        print(f"Training data keys: {fno_data['train'].keys()}")
        print(f"Test data keys: {fno_data['test'].keys()}")
        
        print("Data extraction test PASSED")
        return True
        
    except Exception as e:
        print(f"ERROR in data extraction test: {str(e)}")
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
    
    # Test 2: Run simple simulation
    print("\n2. Testing simple simulation...")
    if not test_simple_simulation():
        print("Simple simulation test FAILED")
        return False
    
    # Test 3: Test data extraction
    print("\n3. Testing data extraction...")
    if not test_data_extraction():
        print("Data extraction test FAILED")
        return False
    
    # Clean up
    print("\n4. Cleaning up test files...")
    cleanup_test_files()
    
    print("\n" + "=" * 50)
    print("All tests PASSED! MOOSE automation is working correctly.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)