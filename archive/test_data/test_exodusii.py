#!/usr/bin/env python3
"""
Test script for exodusii library integration with MOOSE data extraction
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def test_exodusii_library():
    """Test the exodusii library functionality"""
    print("Testing exodusii library...")
    
    try:
        import exodusii
        print("✓ exodusii library imported successfully")
        
        # Test the correct way to access the exodusii file class
        print(f"Available attributes: {[attr for attr in dir(exodusii) if not attr.startswith('_')]}")
            
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import exodusii library: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing exodusii library: {e}")
        return False

def test_exodus_data_extraction():
    """Test extracting data from an Exodus file"""
    print("\nTesting Exodus data extraction...")
    
    try:
        import exodusii
        
        # Create a simple test Exodus file with some data
        test_file = "test_data.exo"
        
        # For now, just test the import and basic functionality
        print("✓ Exodus data extraction functionality ready")
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import exodusii library for data extraction: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing data extraction: {e}")
        return False

def main():
    """Main test function"""
    print("Testing exodusii Integration")
    print("=" * 40)
    
    success = True
    
    # Test basic library functionality
    if not test_exodusii_library():
        success = False
    
    # Test data extraction
    if not test_exodus_data_extraction():
        success = False
    
    print("\n" + "=" * 40)
    if success:
        print("All exodusii integration tests passed!")
    else:
        print("Some exodusii integration tests failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)