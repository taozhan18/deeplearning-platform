#!/usr/bin/env python3
"""
Test script to check exodusii library reading functionality
"""

import sys
from pathlib import Path
import numpy as np

# Add exodusii to path
exodusii_path = '/home/zt/workspace/exodusii'
if exodusii_path not in sys.path:
    sys.path.append(exodusii_path)

print("Testing exodusii library reading functionality...")

try:
    import exodusii
    print("✓ Successfully imported exodusii library")
    
    # Create a test Exodus file
    print("Creating a test Exodus file...")
    
    # Create test data
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ])
    
    # Create a simple Exodus file
    filename = "test.exo"
    with exodusii.File(filename, mode="w") as exo:
        # Write coordinates
        exo.put_coords(coords)
        
        # Write time steps
        exo.put_time(1, 0.0)
        exo.put_time(2, 1.0)
        
        # Write nodal variables
        exo.put_node_variable_names(['temperature', 'pressure'])
        exo.put_node_variable_values('temperature', 1, np.array([20.0, 25.0, 30.0]))
        exo.put_node_variable_values('temperature', 2, np.array([22.0, 27.0, 32.0]))
        exo.put_node_variable_values('pressure', 1, np.array([1.0, 1.5, 2.0]))
        exo.put_node_variable_values('pressure', 2, np.array([1.2, 1.7, 2.2]))
        
        print("✓ Successfully created test Exodus file")
    
    # Now read the file back
    print("Reading the test Exodus file...")
    with exodusii.File(filename, mode="r") as exo:
        # Read coordinates
        read_coords = exo.get_coords()
        print(f"Coordinates shape: {read_coords.shape}")
        print(f"Coordinates:\n{read_coords}")
        
        # Read nodal variable names
        var_names = exo.get_node_variable_names()
        print(f"Node variable names: {var_names}")
        
        # Read variable values
        temp_values = exo.get_node_variable_values('temperature', 1)
        print(f"Temperature values at step 1: {temp_values}")
        
        temp_values_step2 = exo.get_node_variable_values('temperature', 2)
        print(f"Temperature values at step 2: {temp_values_step2}")
        
    print("✓ Successfully read test Exodus file")
    
    # Clean up
    import os
    if os.path.exists(filename):
        os.remove(filename)
        print("✓ Cleaned up test file")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()