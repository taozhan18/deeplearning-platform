#!/usr/bin/env python3
"""
Test script for Transolver model
"""

import sys
import os
import torch
import numpy as np

# Add the modules to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "model_architecture", "src"))

from model_manager import ModelManager


def test_transolver_creation():
    """Test basic Transolver model creation"""
    print("Testing Transolver model creation...")

    # Initialize model manager
    model_manager = ModelManager({})

    # Test creating a basic Transolver model
    try:
        model = model_manager.create_model(
            "transolver",
            space_dim=2,
            n_layers=2,
            n_hidden=64,
            dropout=0.1,
            n_head=4,
            Time_Input=False,
            act="gelu",
            mlp_ratio=1,
            fun_dim=1,
            out_dim=1,
            slice_num=16,
            ref=4,
            unified_pos=False,
            H=32,
            W=32,
        )
        print("✓ Transolver model created successfully")
        print(f"  Model type: {type(model)}")
        print(f"  Number of parameters: {sum(p.numel() for p in model.parameters())}")
        return model
    except Exception as e:
        print(f"✗ Failed to create Transolver model: {e}")
        raise


def test_transolver_forward_pass():
    """Test Transolver model forward pass"""
    print("\nTesting Transolver model forward pass...")

    # Initialize model manager
    model_manager = ModelManager({})

    # Create a small Transolver model for testing
    model = model_manager.create_model(
        "transolver",
        space_dim=2,
        n_layers=2,
        n_hidden=64,
        dropout=0.1,
        n_head=4,
        Time_Input=False,
        act="gelu",
        mlp_ratio=1,
        fun_dim=1,
        out_dim=1,
        slice_num=16,
        ref=4,
        unified_pos=False,
        H=32,
        W=32,
    )

    # Create sample input data
    batch_size = 4
    N = 32 * 32  # H * W
    space_dim = 2

    # Input coordinates
    x = torch.randn(batch_size, N, space_dim)

    # Function values (optional)
    fx = torch.randn(batch_size, N, 1)

    try:
        # Forward pass with function values
        with torch.no_grad():
            output1 = model(x, fx)

        print("✓ Forward pass with function values completed successfully")
        print(f"  Input coordinates shape: {x.shape}")
        print(f"  Input function values shape: {fx.shape}")
        print(f"  Output shape: {output1.shape}")

        # Forward pass without function values
        with torch.no_grad():
            output2 = model(x)

        print("✓ Forward pass without function values completed successfully")
        print(f"  Output shape: {output2.shape}")

        # Check output shape
        expected_shape = (batch_size, N, 1)
        if output1.shape == expected_shape and output2.shape == expected_shape:
            print("✓ Output shapes are correct")
        else:
            print(f"✗ Output shape mismatch. Expected: {expected_shape}")
            print(f"  With fx: {output1.shape}, Without fx: {output2.shape}")
            return False

        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        raise


def test_transolver_with_different_configs():
    """Test Transolver with different configurations"""
    print("\nTesting Transolver with different configurations...")

    model_manager = ModelManager({})

    configs = [
        {
            "name": "Small Transolver",
            "params": {
                "space_dim": 2,
                "n_layers": 1,
                "n_hidden": 32,
                "dropout": 0.0,
                "n_head": 2,
                "Time_Input": False,
                "act": "relu",
                "mlp_ratio": 1,
                "fun_dim": 1,
                "out_dim": 1,
                "slice_num": 8,
                "ref": 2,
                "unified_pos": False,
                "H": 16,
                "W": 16,
            },
        },
        {
            "name": "Medium Transolver",
            "params": {
                "space_dim": 2,
                "n_layers": 3,
                "n_hidden": 128,
                "dropout": 0.1,
                "n_head": 4,
                "Time_Input": True,
                "act": "gelu",
                "mlp_ratio": 2,
                "fun_dim": 2,
                "out_dim": 2,
                "slice_num": 16,
                "ref": 4,
                "unified_pos": True,
                "H": 32,
                "W": 32,
            },
        },
    ]

    for config in configs:
        try:
            print(f"  Testing {config['name']}...")
            model = model_manager.create_model("transolver", **config["params"])

            # Test with sample data
            batch_size = 2
            N = config["params"]["H"] * config["params"]["W"]
            space_dim = config["params"]["space_dim"]
            fun_dim = config["params"]["fun_dim"]
            out_dim = config["params"]["out_dim"]

            x = torch.randn(batch_size, N, space_dim)
            fx = torch.randn(batch_size, N, fun_dim) if fun_dim > 0 else None

            with torch.no_grad():
                if fx is not None:
                    output = model(x, fx)
                else:
                    output = model(x)

            expected_shape = (batch_size, N, out_dim)
            if output.shape == expected_shape:
                print(f"  ✓ {config['name']} works correctly. Output shape: {output.shape}")
            else:
                print(f"  ✗ {config['name']} output shape mismatch. Expected: {expected_shape}, Got: {output.shape}")
                return False

        except Exception as e:
            print(f"  ✗ {config['name']} failed: {e}")
            raise

    return True


def test_transolver_parameter_retrieval():
    """Test retrieval of Transolver model parameters"""
    print("\nTesting Transolver model parameter retrieval...")

    model_manager = ModelManager({})

    try:
        # Get model parameters
        parameters = model_manager.get_model_parameters("transolver")
        print("✓ Transolver parameters retrieved successfully")
        print(f"  Number of parameters: {len(parameters)}")

        # Check for key parameters
        required_params = ["space_dim", "n_layers", "n_hidden"]
        for param in required_params:
            if param in parameters:
                print(f"  ✓ Required parameter '{param}' found")
            else:
                print(f"  ✗ Required parameter '{param}' missing")
                return False

        return True
    except Exception as e:
        print(f"✗ Failed to retrieve Transolver parameters: {e}")
        raise


def main():
    """Main test function"""
    print("Testing Transolver Model")
    print("=" * 50)

    try:
        # Test model creation
        model = test_transolver_creation()

        # Test forward pass
        test_transolver_forward_pass()

        # Test different configurations
        test_transolver_with_different_configs()

        # Test parameter retrieval
        test_transolver_parameter_retrieval()

        print("\n" + "=" * 50)
        print("All Transolver tests passed successfully!")
        return True

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
