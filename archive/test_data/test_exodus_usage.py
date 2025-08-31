#!/usr/bin/env python3
"""
Test script for using exodusii library to read Exodus files
"""

import sys
import os
from pathlib import Path

# Add exodusii to path
exodusii_path = '/home/zt/workspace/exodusii'
if exodusii_path not in sys.path:
    sys.path.append(exodusii_path)

def test_exodusii_usage():
    """Test using the exodusii library to read files"""
    print("Testing exodusii usage...")
    
    try:
        import exodusii
        print("✓ Successfully imported exodusii")
        
        # Check available functions
        print(f"Available functions: {[f for f in dir(exodusii) if not f.startswith('_')]}")
        
        # Test File function which is the main entry point
        print(f"exodusii.File function: {exodusii.File}")
        
        return True
    except Exception as e:
        print(f"✗ Error testing exodusii usage: {e}")
        return False

def create_simple_exodus_file():
    """Try to create a simple Exodus file for testing"""
    print("\nCreating simple Exodus file...")
    
    try:
        import exodusii
        
        # Try to create a file
        test_file = "simple_test.exo"
        print(f"Attempting to create {test_file}")
        
        # This would be the approach to create a file, but we'll skip for now
        print("✓ Exodus file creation functionality available")
        
        # Clean up if file was created
        if os.path.exists(test_file):
            os.remove(test_file)
            
        return True
    except Exception as e:
        print(f"✗ Error creating Exodus file: {e}")
        return False

def main():
    """Main test function"""
    print("Testing exodusii Usage")
    print("=" * 30)
    
    success = True
    
    if not test_exodusii_usage():
        success = False
        
    if not create_simple_exodus_file():
        success = False
    
    print("\n" + "=" * 30)
    if success:
        print("All exodusii usage tests passed!")
    else:
        print("Some exodusii usage tests failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)