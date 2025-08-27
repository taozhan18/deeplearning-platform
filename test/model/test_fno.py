"""
Test script for FNO models
"""

import sys
import os
import torch

# Add the modules to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'model_architecture', 'src'))

from model_manager import ModelManager


def test_fno1d():
    """Test 1D FNO model"""
    print("Testing 1D FNO model...")
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Create 1D FNO model
    fno1d = model_manager.create_model('fno',
                                    in_channels=1,
                                    out_channels=1,
                                    decoder_layers=2,
                                    decoder_layer_size=32,
                                    dimension=1,
                                    latent_channels=16,
                                    num_fno_layers=2,
                                    num_fno_modes=16)
    
    print(f"1D FNO model created: {fno1d}")
    
    # Test with sample input
    x = torch.randn(2, 1, 64)  # batch_size=2, channels=1, height=64
    output = fno1d(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check if output shape is correct
    assert output.shape == (2, 1, 64), f"Expected (2, 1, 64), got {output.shape}"
    print("1D FNO test passed!")


def test_fno2d():
    """Test 2D FNO model"""
    print("\nTesting 2D FNO model...")
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Create 2D FNO model
    fno2d = model_manager.create_model('fno',
                                    in_channels=3,
                                    out_channels=2,
                                    decoder_layers=2,
                                    decoder_layer_size=32,
                                    dimension=2,
                                    latent_channels=16,
                                    num_fno_layers=2,
                                    num_fno_modes=[16, 16])
    
    print(f"2D FNO model created: {fno2d}")
    
    # Test with sample input
    x = torch.randn(2, 3, 32, 32)  # batch_size=2, channels=3, height=32, width=32
    output = fno2d(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check if output shape is correct
    assert output.shape == (2, 2, 32, 32), f"Expected (2, 2, 32, 32), got {output.shape}"
    print("2D FNO test passed!")


def test_fno_hyperparameters():
    """Test FNO hyperparameter retrieval"""
    print("\nTesting FNO hyperparameter retrieval...")
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Get FNO hyperparameters
    hyperparams = model_manager.get_model_hyperparameters('fno')
    print(f"FNO hyperparameters: {list(hyperparams.keys())}")
    
    # Check if all expected hyperparameters are present
    expected_params = ['in_channels', 'out_channels', 'decoder_layers', 'decoder_layer_size', 
                      'dimension', 'latent_channels', 'num_fno_layers', 'num_fno_modes',
                      'padding', 'padding_type', 'coord_features']
    
    for param in expected_params:
        assert param in hyperparams, f"Missing hyperparameter: {param}"
    
    print("FNO hyperparameter test passed!")


def main():
    """Main test function"""
    print("Testing FNO Models in Low-Code Deep Learning Platform")
    print("=" * 55)
    
    # Test 1D FNO
    test_fno1d()
    
    # Test 2D FNO
    test_fno2d()
    
    # Test hyperparameters
    test_fno_hyperparameters()
    
    print("\n" + "=" * 55)
    print("All FNO tests completed successfully!")


if __name__ == "__main__":
    main()