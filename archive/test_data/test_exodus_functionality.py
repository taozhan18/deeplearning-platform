#!/usr/bin/env python3
"""
Test script to check exodusii library functionality
"""

import sys
from pathlib import Path

# Add exodusii to path
exodusii_path = '/home/zt/workspace/exodusii'
if exodusii_path not in sys.path:
    sys.path.append(exodusii_path)

print("Testing exodusii library functionality...")

try:
    import exodusii
    print("✓ Successfully imported exodusii library")
    
    # Check available methods
    print("Available methods in exodusii module:")
    print([method for method in dir(exodusii) if not method.startswith('_')])
    
    # Test File class
    if hasattr(exodusii, 'File'):
        print("✓ File class is available")
    else:
        print("✗ File class is not available")
        
    # Test other classes
    classes_to_check = ['ExodusFile', 'ExodusFileReader', 'exodusiiFile']
    for cls in classes_to_check:
        if hasattr(exodusii, cls):
            print(f"✓ {cls} class is available")
        else:
            print(f"✗ {cls} class is not available")
            
except ImportError as e:
    print(f"✗ Failed to import exodusii library: {e}")

# Test if we can create a simple Exodus file
try:
    import numpy as np
    
    # Create a simple mesh for testing
    num_nodes = 10
    num_elements = 9
    
    # Node coordinates (1D line)
    x_coords = np.linspace(0, 1, num_nodes)
    y_coords = np.zeros(num_nodes)
    z_coords = np.zeros(num_nodes)
    
    # Element connectivity (line elements)
    connectivity = []
    for i in range(num_elements):
        connectivity.extend([i+1, i+2])  # 2-node line elements
    
    print("✓ Successfully created test mesh data")
    
except Exception as e:
    print(f"✗ Failed to create test mesh data: {e}")