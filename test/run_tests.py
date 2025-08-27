"""
Main test runner for the low-code deep learning platform
"""

import sys
import os
import subprocess


def run_test_script(script_path, description):
    """Run a test script and return the result"""
    print(f"\n{'='*50}")
    print(f"Running {description}")
    print(f"{'='*50}")
    
    try:
        # Run the test script
        result = subprocess.run([
            "python", script_path
        ], 
        cwd=os.path.join(os.path.dirname(__file__), '..'),
        capture_output=True, 
        text=True, 
        timeout=300)  # 5 minute timeout
        
        # Print output
        if result.stdout:
            print(result.stdout)
            
        if result.stderr:
            print("STDERR:", result.stderr)
            
        # Check result
        if result.returncode == 0:
            print(f"✅ {description} PASSED")
            return True
        else:
            print(f"❌ {description} FAILED")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ {description} TIMED OUT")
        return False
    except Exception as e:
        print(f"❌ {description} ERROR: {str(e)}")
        return False


def main():
    """Main test runner function"""
    print("Low-Code Deep Learning Platform - Test Suite")
    print("=" * 50)
    
    # Get the base path
    base_path = os.path.join(os.path.dirname(__file__), '..')
    
    # List of test scripts to run
    test_scripts = [
        ("test/model/test_platform.py", "Platform Integration Tests"),
        ("test/model/test_fno.py", "FNO Model Tests"),
        ("test/data_loader/test_data_loader.py", "Data Loader Tests"),
        ("test/training_engine/test_training_engine.py", "Training Engine Tests")
    ]
    
    # Run all tests
    results = []
    for script, description in test_scripts:
        script_path = os.path.join(base_path, script)
        if os.path.exists(script_path):
            result = run_test_script(script_path, description)
            results.append((description, result))
        else:
            print(f"⚠️  {description} - Script not found: {script}")
            results.append((description, False))
    
    # Print summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = 0
    failed = 0
    
    for description, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{description}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {len(results)} tests, {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())