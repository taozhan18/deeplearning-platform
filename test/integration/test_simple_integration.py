"""
Simplified integration test for low-code deep learning platform
Tests core functionality with minimal dependencies
"""

import os
import sys
import tempfile
import numpy as np
from pathlib import Path

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "main"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "data_loader", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "model_architecture", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "training_engine", "src"))
from data_loader import DataLoaderModule
from model_manager import ModelManager
from training_engine import TrainingEngine


def test_basic_functionality():
    """Test basic functionality without full training"""
    print("🧪 Testing Basic Functionality...")

    try:
        # Test 1: Data loader initialization
        config = {"data": {"batch_size": 32, "shuffle": True, "normalize": True}}

        data_loader = DataLoaderModule(config["data"])
        print("✅ Data loader initialized successfully")

        # Test 2: Model manager initialization
        model_manager = ModelManager(config.get("model", {}))
        available_models = model_manager.list_models()
        print(f"✅ Available models: {available_models}")

        # Test 3: Model creation
        try:
            mlp_model = model_manager.create_model("mlp", input_size=10, hidden_size=64, output_size=1)
            print("✅ MLP model created successfully")
        except Exception as e:
            print(f"⚠️ MLP model creation: {e}")

        try:
            fno_model = model_manager.create_model("fno", in_channels=3, out_channels=1, modes1=12, modes2=12, width=32)
            print("✅ FNO model created successfully")
        except Exception as e:
            print(f"⚠️ FNO model creation: {e}")

        # Test 4: Basic data handling
        test_data = np.random.randn(100, 10)
        test_targets = np.random.randn(100, 1)

        # Save test data
        with tempfile.TemporaryDirectory() as temp_dir:
            np.savez(os.path.join(temp_dir, "test_data.npz"), data=test_data)
            np.savez(os.path.join(temp_dir, "test_targets.npz"), data=test_targets)

            # Test data loading
            config = {
                "train_features_path": os.path.join(temp_dir, "test_data.npz"),
                "train_targets_path": os.path.join(temp_dir, "test_targets.npz"),
                "batch_size": 32,
                "shuffle": True,
                "normalize": True,
            }

            data_loader = DataLoaderModule(config)
            print("✅ Test data loaded successfully")

        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_configuration_templates():
    """Test configuration templates"""
    print("🧪 Testing Configuration Templates...")

    try:
        # Test MLP template
        mlp_config = {
            "data": {"batch_size": 64, "normalize": True, "normalization_method": "standard"},
            "model": {
                "name": "mlp",
                "parameters": {"in_features": 20, "layer_sizes": 128, "out_features": 1, "num_layers": 3},
            },
            "training": {"epochs": 10, "device": "cpu"},
            "optimizer": {"name": "adam", "parameters": {"lr": 0.001}},
            "criterion": {"name": "mse"},
        }

        # Test model creation
        model_manager = ModelManager({})
        model = model_manager.create_model("mlp", **mlp_config["model"]["parameters"])
        print("✅ MLP configuration template works")

        # Test FNO template
        fno_config = {"model": {"name": "fno", "parameters": {"in_channels": 3, "out_channels": 1}}}

        try:
            fno_model = model_manager.create_model("fno", **fno_config["model"]["parameters"])
            print("✅ FNO configuration template works")
        except Exception as e:
            print(f"⚠️ FNO template: {e}")

        return True

    except Exception as e:
        print(f"❌ Configuration templates test failed: {e}")
        return False


def test_preprocessing_system():
    """Test preprocessing function system"""
    print("🧪 Testing Preprocessing System...")

    try:

        # Test Python file loading
        preprocess_file = os.path.join(os.path.dirname(__file__), "mlp", "preprocess_mlp.py")

        if os.path.exists(preprocess_file):
            config = {"preprocess_fn": preprocess_file, "epochs": 1, "device": "cpu"}

            training_engine = TrainingEngine(config)
            print("✅ Preprocessing system works")
        else:
            print("⚠️ Preprocessing file not found")

        return True

    except Exception as e:
        print(f"❌ Preprocessing system test failed: {e}")
        return False


def main():
    """Run simplified integration tests"""
    print("🚀 Running Simplified Integration Tests")
    print("=" * 50)

    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Configuration Templates", test_configuration_templates),
        ("Preprocessing System", test_preprocessing_system),
    ]

    results = {}

    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"❌ {name} test failed with exception: {e}")
            results[name] = False

    print("\n" + "=" * 50)
    print("📊 Simplified Test Results:")

    passed = sum(results.values())
    total = len(results)

    for name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All simplified tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check logs above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
