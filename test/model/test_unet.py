"""
Test script for UNet model implementation
"""

import sys
import os
import torch

# Add the modules to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'model_architecture', 'src'))

from model_manager import ModelManager


def test_unet_basic():
    """Test basic UNet model"""
    print("Testing basic UNet model...")
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Create UNet model with default parameters
    unet = model_manager.create_model('unet',
                                    in_channels=3,
                                    out_channels=1,
                                    features=[64, 128, 256])
    
    print(f"UNet model created: {unet}")
    
    # Test with sample input
    x = torch.randn(2, 3, 128, 128)  # batch_size=2, channels=3, height=128, width=128
    output = unet(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check if output shape is correct
    assert output.shape == (2, 1, 128, 128), f"Expected (2, 1, 128, 128), got {output.shape}"
    print("Basic UNet test passed!")


def test_unet_different_channels():
    """Test UNet model with different input/output channels"""
    print("\nTesting UNet model with different input/output channels...")
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Create UNet model with different channels
    unet = model_manager.create_model('unet',
                                    in_channels=1,
                                    out_channels=3,
                                    features=[32, 64, 128, 256])
    
    print(f"UNet model with different channels created: {unet}")
    
    # Test with sample input
    x = torch.randn(1, 1, 256, 256)  # batch_size=1, channels=1, height=256, width=256
    output = unet(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check if output shape is correct
    assert output.shape == (1, 3, 256, 256), f"Expected (1, 3, 256, 256), got {output.shape}"
    print("UNet with different channels test passed!")


def test_unet_with_activation_normalization():
    """Test UNet model with different activation and normalization"""
    print("\nTesting UNet model with different activation and normalization...")
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Create UNet model with different activation and normalization
    unet = model_manager.create_model('unet',
                                    in_channels=3,
                                    out_channels=2,
                                    features=[32, 64],
                                    activation='leaky_relu',
                                    normalization='groupnorm')
    
    print(f"UNet model with different activation/normalization created: {unet}")
    
    # Test with sample input
    x = torch.randn(1, 3, 64, 64)  # batch_size=1, channels=3, height=64, width=64
    output = unet(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check if output shape is correct
    assert output.shape == (1, 2, 64, 64), f"Expected (1, 2, 64, 64), got {output.shape}"
    print("UNet with different activation/normalization test passed!")


def test_unet_hyperparameters():
    """Test UNet hyperparameter retrieval"""
    print("\nTesting UNet hyperparameter retrieval...")
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Get UNet hyperparameters
    hyperparams = model_manager.get_model_hyperparameters('unet')
    print(f"UNet hyperparameters: {list(hyperparams.keys())}")
    
    # Check if all expected hyperparameters are present
    expected_params = ['in_channels', 'out_channels', 'features', 'activation', 'normalization']
    
    for param in expected_params:
        assert param in hyperparams, f"Missing hyperparameter: {param}"
    
    print("UNet hyperparameter test passed!")


def main():
    """Main test function"""
    print("Testing UNet Models in Low-Code Deep Learning Platform")
    print("=" * 55)
    
    # Test basic UNet
    test_unet_basic()
    
    # Test UNet with different channels
    test_unet_different_channels()
    
    # Test UNet with different activation and normalization
    test_unet_with_activation_normalization()
    
    # Test hyperparameters
    test_unet_hyperparameters()
    
    print("\n" + "=" * 55)
    print("All UNet tests completed successfully!")


if __name__ == "__main__":
    main()