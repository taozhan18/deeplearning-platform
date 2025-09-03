#!/usr/bin/env python3
"""
Test runner for the data loader module.

This script runs all tests for the data loader module and reports the results.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add the project root to the path so we can import modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_test_file(test_file: str) -> bool:
    """
    Run a single test file and return whether it passed.
    
    Args:
        test_file: Path to the test file to run
        
    Returns:
        True if the test passed, False otherwise
    """
    print(f"Running {test_file}...")
    
    try:
        # Run the test file directly
        result = subprocess.run(
            [sys.executable, test_file],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"✓ {test_file} passed")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"✗ {test_file} failed")
            if result.stdout:
                print("STDOUT:")
                print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ {test_file} timed out")
        return False
    except Exception as e:
        print(f"✗ {test_file} failed with exception: {e}")
        return False

def run_pytest_test(test_file: str, test_function: str = None) -> bool:
    """
    Run a test file using pytest.
    
    Args:
        test_file: Path to the test file to run
        test_function: Optional specific test function to run
        
    Returns:
        True if the test passed, False otherwise
    """
    print(f"Running {test_file} with pytest...")
    
    try:
        # Build command
        cmd = [sys.executable, "-m", "pytest", test_file, "-v"]
        if test_function:
            cmd.extend(["::" + test_function])
        
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"✓ {test_file} passed")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"✗ {test_file} failed")
            if result.stdout:
                print("STDOUT:")
                print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ {test_file} timed out")
        return False
    except Exception as e:
        print(f"✗ {test_file} failed with exception: {e}")
        return False

def main():
    """
    Main function to run all data loader tests.
    """
    print("Running Data Loader Tests")
    print("=" * 50)
    
    # Get the directory containing this script
    test_dir = Path(__file__).parent
    
    # List of test files to run
    test_files = [
        "test_data_loader.py",
        "test_custom_data_generator.py",
        "integration_test.py"
    ]
    
    # Track test results
    passed = 0
    failed = 0
    
    # Run each test file
    for test_file in test_files:
        test_path = test_dir / test_file
        if test_path.exists():
            if run_test_file(str(test_path)):
                passed += 1
            else:
                failed += 1
        else:
            print(f"⚠️  {test_file} not found, skipping")
            failed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)