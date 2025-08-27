#!/usr/bin/env python3
"""
Test script to verify improved data extraction from MOOSE output files
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add the scripts directory to the path
sys.path.append(os.path.dirname(__file__))

from moose_data_generator import MOOSEDataGenerator


def create_test_csv():
    """Create a test CSV file that mimics MOOSE output"""
    # Create test data that resembles MOOSE output
    n_points = 50
    x = np.linspace(0, 1, n_points)
    u = np.sin(np.pi * x)  # Simple analytical solution
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': x,
        'u': u
    })
    
    # Save to CSV
    test_dir = Path("test_data_extraction")
    test_dir.mkdir(exist_ok=True)
    
    csv_file = test_dir / "test_output.csv"
    df.to_csv(csv_file, index=False)
    print(f"Created test CSV file: {csv_file}")
    print(f"CSV shape: {df.shape}")
    print(f"CSV columns: {list(df.columns)}")
    
    return str(test_dir)


def test_data_extraction():
    """Test the improved data extraction functionality"""
    print("Testing improved data extraction from MOOSE output files...")
    
    # Create test data
    test_dir = create_test_csv()
    
    try:
        # Initialize MOOSE data generator
        moose_exec = "/home/zt/workspace/mymoose/mymoose-opt"
        generator = MOOSEDataGenerator(moose_exec)
        
        # Create a mock simulation result for testing
        mock_result = {
            'sim_dir': test_dir,
            'parameters': {'nx': 50, 'left_bc': 0.0, 'right_bc': 1.0}
        }
        
        # Test data extraction
        input_field, output_field = generator._extract_moose_data(test_dir)
        
        print(f"Extracted input field shape: {input_field.shape}")
        print(f"Extracted output field shape: {output_field.shape}")
        print(f"Input field sample: {input_field[:5]}")
        print(f"Output field sample: {output_field[:5]}")
        
        # Verify the data looks reasonable
        assert len(input_field) == len(output_field), "Input and output fields should have same length"
        assert len(input_field) > 0, "Fields should not be empty"
        
        print("Data extraction test PASSED")
        return True
        
    except Exception as e:
        print(f"ERROR in data extraction test: {str(e)}")
        return False
    finally:
        # Clean up test files
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"Cleaned up {test_dir}")


def main():
    """Main test function"""
    print("Testing Improved MOOSE Data Extraction")
    print("=" * 40)
    
    success = test_data_extraction()
    
    print("\n" + "=" * 40)
    if success:
        print("All tests PASSED! Data extraction is working correctly.")
    else:
        print("Tests FAILED!")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)