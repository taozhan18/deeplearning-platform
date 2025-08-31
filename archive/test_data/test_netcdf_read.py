#!/usr/bin/env python3
"""
Test script to check direct netCDF reading of Exodus files
"""

import sys
from pathlib import Path

print("Testing direct netCDF reading of Exodus files...")

try:
    import netCDF4
    print("✓ Successfully imported netCDF4 library")
    
    # Check if we have any .e files in the system we can test with
    import glob
    e_files = glob.glob("/home/zt/workspace/**/*.e", recursive=True)
    if e_files:
        test_file = e_files[0]
        print(f"Found Exodus file to test with: {test_file}")
        
        # Try to read it with netCDF4
        with netCDF4.Dataset(test_file, 'r') as dataset:
            print("✓ Successfully opened Exodus file with netCDF4")
            
            # Print basic information about the file
            print(f"File format: {dataset.file_format}")
            print(f"Dimensions: {list(dataset.dimensions.keys())}")
            print(f"Variables: {list(dataset.variables.keys())}")
            
            # Try to read coordinates if they exist
            coord_vars = ['coordx', 'coordy', 'coordz']
            coords_found = []
            for var_name in coord_vars:
                if var_name in dataset.variables:
                    coords_found.append(var_name)
                    var = dataset.variables[var_name]
                    print(f"Variable {var_name}: shape {var.shape}, dtype {var.dtype}")
                    
            if coords_found:
                print(f"✓ Found coordinate variables: {coords_found}")
                
            # Try to read nodal variables
            if 'name_nod_var' in dataset.variables:
                var_names = dataset.variables['name_nod_var']
                print(f"Nodal variable names: {var_names}")
                
            # Try to read time steps
            if 'time_whole' in dataset.variables:
                time_steps = dataset.variables['time_whole']
                print(f"Time steps: {time_steps[:]}")
                
    else:
        print("No existing Exodus files found for testing")
        
except ImportError as e:
    print(f"✗ Failed to import netCDF4 library: {e}")
    print("Try installing it with: pip install netCDF4")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()