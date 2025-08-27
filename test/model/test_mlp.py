"""
Test script for MLP model implementation
"""

import sys
import os
import torch

# Add the modules to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'model_architecture', 'src'))

from model_manager import ModelManager


def test_mlp_basic():
    """Test basic MLP model"""
    print("Testing basic MLP model...")
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Create MLP model with default parameters
    mlp = model_manager.create_model('mlp',
                                   in_features=10,
                                   layer_sizes=32,
                                   out_features=5,
                                   num_layers=3,
                                   activation_fn='relu')
    
    print(f"MLP model created: {mlp}")
    
    # Test with sample input
    x = torch.randn(4, 10)  # batch_size=4, features=10
    output = mlp(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check if output shape is correct
    assert output.shape == (4, 5), f"Expected (4, 5), got {output.shape}"
    print("Basic MLP test passed!")


def test_mlp_variable_layers():
    """Test MLP model with variable layer sizes"""
    print("\nTesting MLP model with variable layer sizes...")
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Create MLP model with variable layer sizes
    mlp = model_manager.create_model('mlp',
                                   in_features=8,
                                   layer_sizes=[64, 32, 16],
                                   out_features=1,
                                   num_layers=3,
                                   activation_fn=['relu', 'tanh', 'sigmoid'])
    
    print(f"Variable layer MLP model created: {mlp}")
    
    # Test with sample input
    x = torch.randn(2, 8)  # batch_size=2, features=8
    output = mlp(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check if output shape is correct
    assert output.shape == (2, 1), f"Expected (2, 1), got {output.shape}"
    print("Variable layer MLP test passed!")


def test_mlp_with_dropout():
    """Test MLP model with dropout"""
    print("\nTesting MLP model with dropout...")
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Create MLP model with dropout
    mlp = model_manager.create_model('mlp',
                                   in_features=16,
                                   layer_sizes=32,
                                   out_features=8,
                                   num_layers=2,
                                   activation_fn='relu',
                                   dropout=0.3)
    
    print(f"MLP model with dropout created: {mlp}")
    
    # Test with sample input
    x = torch.randn(3, 16)  # batch_size=3, features=16
    output = mlp(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check if output shape is correct
    assert output.shape == (3, 8), f"Expected (3, 8), got {output.shape}"
    print("MLP with dropout test passed!")


def test_mlp_skip_connections():
    """Test MLP model with skip connections"""
    print("\nTesting MLP model with skip connections...")
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Create MLP model with skip connections
    mlp = model_manager.create_model('mlp',
                                   in_features=32,
                                   layer_sizes=64,
                                   out_features=16,
                                   num_layers=4,
                                   activation_fn='relu',
                                   skip_connections=True)
    
    print(f"MLP model with skip connections created: {mlp}")
    
    # Test with sample input
    x = torch.randn(2, 32)  # batch_size=2, features=32
    output = mlp(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check if output shape is correct
    assert output.shape == (2, 16), f"Expected (2, 16), got {output.shape}"
    print("MLP with skip connections test passed!")


def test_mlp_hyperparameters():
    """Test MLP hyperparameter retrieval"""
    print("\nTesting MLP hyperparameter retrieval...")
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Get MLP hyperparameters
    hyperparams = model_manager.get_model_hyperparameters('mlp')
    print(f"MLP hyperparameters: {list(hyperparams.keys())}")
    
    # Check if all expected hyperparameters are present
    expected_params = ['in_features', 'layer_sizes', 'out_features', 'num_layers', 
                      'activation_fn', 'skip_connections', 'dropout']
    
    for param in expected_params:
        assert param in hyperparams, f"Missing hyperparameter: {param}"
    
    print("MLP hyperparameter test passed!")


def main():
    """Main test function"""
    print("Testing MLP Models in Low-Code Deep Learning Platform")
    print("=" * 55)
    
    # Test basic MLP
    test_mlp_basic()
    
    # Test MLP with variable layers
    test_mlp_variable_layers()
    
    # Test MLP with dropout
    test_mlp_with_dropout()
    
    # Test MLP with skip connections
    test_mlp_skip_connections()
    
    # Test hyperparameters
    test_mlp_hyperparameters()
    
    print("\n" + "=" * 55)
    print("All MLP tests completed successfully!")


if __name__ == "__main__":
    main()