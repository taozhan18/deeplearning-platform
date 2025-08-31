#!/usr/bin/env python3
"""
Test the complete MOOSE to ML training flow
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add the necessary paths
sys.path.append('/home/zt/workspace/deeplearning-platform/data_loader/scripts')

from moose_ml_automation import EnhancedMOOSEDataGenerator, MOOSEMLPipeline

def test_complete_flow():
    """Test the complete flow with corrected parameters"""
    
    print("🧪 Testing Complete MOOSE to ML Flow")
    print("=" * 50)
    
    # Create test configuration with corrected parameters
    test_config = {
        "moose_executable": "/home/zt/workspace/mymoose/mymoose-opt",
        "conda_env": "physics",
        "base_input_file": "/home/zt/workspace/deeplearning-platform/examples/moose_diffusion_template.i",
        "parameters": {
            "left_boundary": {
                "type": "uniform",
                "min": 0.0,
                "max": 1.0
            },
            "right_boundary": {
                "type": "uniform", 
                "min": 0.0,
                "max": 1.0
            },
            "source_term": {
                "type": "uniform",
                "min": -0.5,
                "max": 0.5
            }
        },
        "num_samples": 3,  # Small number for testing
        "simulation_output_dir": "test_moose_flow",
        "dataset_dir": "test_ml_dataset",
        "variables_to_extract": ["u", "x"],
        "input_variables": ["x"],
        "output_variables": ["u"],
        "normalize": True,
        "model_name": "fno",
        "epochs": 5,  # Small number for quick test
        "batch_size": 2,
        "learning_rate": 0.001,
        "use_gpu": False
    }
    
    # Save test config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f, indent=2)
        config_path = f.name
    
    try:
        print(f"\n1️⃣ Creating pipeline with config: {config_path}")
        pipeline = MOOSEMLPipeline(config_path)
        
        print(f"\n2️⃣ Running complete pipeline...")
        results = pipeline.run_complete_pipeline()
        
        print(f"\n3️⃣ Pipeline completed successfully!")
        print(f"   Dataset: {test_config['dataset_dir']}")
        print(f"   Config: {results['config_path']}")
        
        # Verify outputs
        dataset_path = Path(test_config['dataset_dir'])
        expected_files = [
            'parameters.npy',
            'input_data.npy', 
            'output_data.npy',
            'metadata.json'
        ]
        
        missing_files = []
        for file in expected_files:
            file_path = dataset_path / file
            if not file_path.exists():
                missing_files.append(file)
        
        if not missing_files:
            print(f"   ✅ All expected files generated")
            return True
        else:
            print(f"   ❌ Missing files: {missing_files}")
            return False
            
    except Exception as e:
        print(f"   ❌ Flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if os.path.exists(config_path):
            os.unlink(config_path)

def test_moose_input_generation():
    """Test MOOSE input file generation with corrected parameters"""
    
    print(f"\n4️⃣ Testing MOOSE input generation...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = EnhancedMOOSEDataGenerator()
        
        params = {
            "left_boundary": 0.5,
            "right_boundary": 1.0,
            "source_term": 0.2
        }
        
        base_input = "/home/zt/workspace/deeplearning-platform/examples/moose_diffusion_template.i"
        output_file = os.path.join(tmpdir, "test_input.i")
        
        generator._create_modified_input(base_input, output_file, params)
        
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                content = f.read()
                
            # Check if parameters were replaced correctly
            checks = [
                "{left_boundary}" not in content,
                "{right_boundary}" not in content,
                "{source_term}" not in content,
                "0.5" in content,
                "1.0" in content,
                "0.2" in content
            ]
            
            if all(checks):
                print(f"   ✅ Input file generation correct")
                return True
            else:
                print(f"   ❌ Input file generation failed")
                return False
        
        return False

def main():
    """Main test function"""
    
    print("🚀 Starting MOOSE Flow Test")
    print("=" * 40)
    
    tests = [
        ("MOOSE Input Generation", test_moose_input_generation),
        ("Complete Flow", test_complete_flow)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   {status} {test_name}")
        except Exception as e:
            results.append((test_name, False))
            print(f"   ❌ FAIL {test_name}: {e}")
    
    print("\n" + "=" * 40)
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! MOOSE flow is ready.")
    else:
        print("⚠️ Some tests failed. Check logs for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)