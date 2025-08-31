#!/usr/bin/env python3
"""
Test script to check exodusii library reading functionality with correct usage
"""

import sys
from pathlib import Path
import numpy as np

# Add exodusii to path
exodusii_path = '/home/zt/workspace/exodusii'
if exodusii_path not in sys.path:
    sys.path.append(exodusii_path)

print("Testing exodusii library reading functionality with correct usage...")

try:
    import exodusii
    print("✓ Successfully imported exodusii library")
    
    # Test reading functionality (we'll test with an existing file if we have one)
    # First, let's see what methods are available for reading
    print("Available methods in exodusii module:")
    methods = [attr for attr in dir(exodusii) if not attr.startswith('_')]
    print(methods)
    
    # Check if we have any .e files in the system we can test with
    import glob
    e_files = glob.glob("/home/zt/workspace/**/*.e", recursive=True)
    if e_files:
        test_file = e_files[0]
        print(f"Found Exodus file to test with: {test_file}")
        
        # Try to read it
        with exodusii.File(test_file, mode="r") as exo:
            print("✓ Successfully opened Exodus file")
            
            # Try to get basic info
            if hasattr(exo, 'get_coords'):
                coords = exo.get_coords()
                print(f"✓ Successfully read coordinates. Shape: {coords.shape}")
                
            if hasattr(exo, 'get_node_variable_names'):
                var_names = exo.get_node_variable_names()
                print(f"Node variable names: {var_names}")
                
            if hasattr(exo, 'get_node_variable_values') and var_names:
                # Try to read first variable at first time step
                try:
                    values = exo.get_node_variable_values(var_names[0], 1)
                    print(f"✓ Successfully read variable '{var_names[0]}' values. Shape: {values.shape}")
                except Exception as e:
                    print(f"✗ Failed to read variable values: {e}")
    else:
        print("No existing Exodus files found for testing")
        
        # Let's try to understand the exodusii library better by checking its docstrings
        print("\nChecking exodusii_file class docstring:")
        if hasattr(exodusii, 'exodusii_file'):
            print(exodusii.exodusii_file.__doc__[:500] + "..." if exodusii.exodusii_file.__doc__ else "No docstring")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()