"""
Test script for CNNAttention model implementation
"""

import sys
import os
import torch
import torch.nn as nn
import pytest

# Add the modules to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'model_architecture', 'src'))

from model_manager import ModelManager


def test_cnnattention_basic_2d():
    """Test basic CNNAttention model with 2D field data"""
    print("Testing basic CNNAttention model with 2D field data...")
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Create CNNAttention model with default parameters
    model = model_manager.create_model('CNNAtention',
                                      in_channels=3,
                                      dimension=2,
                                      hidden_channels=32,
                                      num_encoder_layers=3,
                                      use_attention=True,
                                      mlp_hidden_sizes=[128, 64],
                                      dropout_rate=0.2)
    
    print(f"CNNAttention model created: {model}")
    
    # Test with sample 2D input
    x = torch.randn(4, 3, 32, 32)  # batch_size=4, channels=3, height=32, width=32
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check if output shape is correct
    assert output.shape == (4, 1), f"Expected (4, 1), got {output.shape}"
    print("Basic 2D CNNAttention test passed!")


def test_cnnattention_1d():
    """Test CNNAttention model with 1D field data"""
    print("\nTesting CNNAttention model with 1D field data...")
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Create CNNAttention model for 1D data
    model = model_manager.create_model('CNNAtention',
                                      in_channels=1,
                                      dimension=1,
                                      hidden_channels=16,
                                      num_encoder_layers=2,
                                      use_attention=False,
                                      mlp_hidden_sizes=[64, 32],
                                      dropout_rate=0.1)
    
    # Test with sample 1D input
    x = torch.randn(6, 1, 100)  # batch_size=6, channels=1, length=100
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check if output shape is correct
    assert output.shape == (6, 1), f"Expected (6, 1), got {output.shape}"
    print("1D CNNAttention test passed!")


def test_cnnattention_3d():
    """Test CNNAttention model with 3D field data"""
    print("\nTesting CNNAttention model with 3D field data...")
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Create CNNAttention model for 3D data
    model = model_manager.create_model('CNNAtention',
                                      in_channels=2,
                                      dimension=3,
                                      hidden_channels=24,
                                      num_encoder_layers=2,
                                      use_attention=True,
                                      attention_heads=4,
                                      mlp_hidden_sizes=[96, 48],
                                      dropout_rate=0.3)
    
    # Test with sample 3D input
    x = torch.randn(2, 2, 16, 16, 16)  # batch_size=2, channels=2, depth=16, height=16, width=16
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check if output shape is correct
    assert output.shape == (2, 1), f"Expected (2, 1), got {output.shape}"
    print("3D CNNAttention test passed!")


def test_cnnattention_attention_mechanism():
    """Test CNNAttention model with and without attention"""
    print("\nTesting CNNAttention model with and without attention...")
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Model with attention
    model_with_attention = model_manager.create_model('CNNAtention',
                                                     in_channels=2,
                                                     dimension=2,
                                                     hidden_channels=32,
                                                     use_attention=True,
                                                     attention_heads=4)
    
    # Model without attention
    model_without_attention = model_manager.create_model('CNNAtention',
                                                        in_channels=2,
                                                        dimension=2,
                                                        hidden_channels=32,
                                                        use_attention=False)
    
    # Test input
    x = torch.randn(4, 2, 32, 32)
    
    # Forward pass with attention
    output_with_attention = model_with_attention(x)
    print(f"With attention - Output shape: {output_with_attention.shape}")
    
    # Forward pass without attention
    output_without_attention = model_without_attention(x)
    print(f"Without attention - Output shape: {output_without_attention.shape}")
    
    # Both should have same output shape
    assert output_with_attention.shape == output_without_attention.shape, \
        f"Shapes don't match: {output_with_attention.shape} vs {output_without_attention.shape}"
    
    print("Attention mechanism test passed!")


def test_cnnattention_pooling_strategies():
    """Test CNNAttention model with different pooling strategies"""
    print("\nTesting CNNAttention model with different pooling strategies...")
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Model with adaptive pooling
    model_adaptive = model_manager.create_model('CNNAtention',
                                                in_channels=1,
                                                dimension=2,
                                                hidden_channels=16,
                                                pooling_strategy='adaptive')
    
    # Model with max pooling
    model_max = model_manager.create_model('CNNAtention',
                                          in_channels=1,
                                          dimension=2,
                                          hidden_channels=16,
                                          pooling_strategy='max')
    
    # Test input
    x = torch.randn(3, 1, 24, 24)
    
    # Forward pass with adaptive pooling
    output_adaptive = model_adaptive(x)
    print(f"Adaptive pooling - Output shape: {output_adaptive.shape}")
    
    # Forward pass with max pooling
    output_max = model_max(x)
    print(f"Max pooling - Output shape: {output_max.shape}")
    
    # Both should have same output shape
    assert output_adaptive.shape == output_max.shape, \
        f"Shapes don't match: {output_adaptive.shape} vs {output_max.shape}"
    
    print("Pooling strategies test passed!")


def test_cnnattention_activation_functions():
    """Test CNNAttention model with different activation functions"""
    print("\nTesting CNNAttention model with different activation functions...")
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    activation_functions = ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'gelu']
    
    for activation_fn in activation_functions:
        print(f"Testing with {activation_fn} activation...")
        
        model = model_manager.create_model('CNNAtention',
                                          in_channels=1,
                                          dimension=2,
                                          hidden_channels=16,
                                          num_encoder_layers=2,
                                          use_attention=False,
                                          activation_fn=activation_fn,
                                          mlp_hidden_sizes=[32, 16])
        
        # Test input
        x = torch.randn(2, 1, 16, 16)
        output = model(x)
        
        print(f"  {activation_fn} - Output shape: {output.shape}")
        assert output.shape == (2, 1), f"Expected (2, 1), got {output.shape}"
    
    print("Activation functions test passed!")


def test_cnnattention_hyperparameters():
    """Test CNNAttention model hyperparameters"""
    print("\nTesting CNNAttention model hyperparameters...")
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Get model hyperparameters
    hyperparams = model_manager.get_model_parameters('CNNAtention')
    print(f"Available hyperparameters: {list(hyperparams.keys())}")
    
    # Check if all expected hyperparameters are present
    expected_params = ['in_channels', 'dimension', 'hidden_channels', 'num_encoder_layers',
                      'encoder_kernel_size', 'use_attention', 'attention_heads',
                      'mlp_hidden_sizes', 'dropout_rate', 'activation_fn', 'pooling_strategy']
    
    for param in expected_params:
        assert param in hyperparams, f"Missing hyperparameter: {param}"
        print(f"  {param}: {hyperparams[param]['description']} (default: {hyperparams[param]['default']})")
    
    print("Hyperparameters test passed!")


def test_cnnattention_gradient_flow():
    """Test CNNAttention model gradient flow"""
    print("\nTesting CNNAttention model gradient flow...")
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Create model
    model = model_manager.create_model('CNNAtention',
                                      in_channels=2,
                                      dimension=2,
                                      hidden_channels=16,
                                      use_attention=True,
                                      mlp_hidden_sizes=[32, 16])
    
    # Test input and target
    x = torch.randn(3, 2, 16, 16, requires_grad=True)
    target = torch.randn(3, 1)
    
    # Forward pass
    output = model(x)
    
    # Compute loss
    criterion = nn.MSELoss()
    loss = criterion(output, target)
    
    # Backward pass
    loss.backward()
    
    # Check if gradients are computed
    assert x.grad is not None, "Input gradients not computed"
    assert loss.item() > 0, "Loss should be positive"
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Input gradient norm: {x.grad.norm().item():.4f}")
    
    print("Gradient flow test passed!")


def test_cnnattention_model_parameters():
    """Test CNNAttention model parameter count and initialization"""
    print("\nTesting CNNAttention model parameter count and initialization...")
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Create model
    model = model_manager.create_model('CNNAtention',
                                      in_channels=3,
                                      dimension=2,
                                      hidden_channels=32,
                                      num_encoder_layers=3,
                                      use_attention=True,
                                      mlp_hidden_sizes=[64, 32])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Check if all parameters are trainable
    assert total_params == trainable_params, "Some parameters are not trainable"
    
    # Check parameter initialization
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert not torch.isnan(param).any(), f"Parameter {name} contains NaN values"
            assert not torch.isinf(param).any(), f"Parameter {name} contains Inf values"
    
    print("Model parameters test passed!")


def test_cnnattention_batch_sizes():
    """Test CNNAttention model with different batch sizes"""
    print("\nTesting CNNAttention model with different batch sizes...")
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Create model
    model = model_manager.create_model('CNNAtention',
                                      in_channels=1,
                                      dimension=2,
                                      hidden_channels=16,
                                      use_attention=False,
                                      mlp_hidden_sizes=[32, 16])
    
    # Test with different batch sizes
    batch_sizes = [1, 2, 4, 8, 16]
    
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 1, 32, 32)
        output = model(x)
        
        print(f"Batch size {batch_size} - Output shape: {output.shape}")
        assert output.shape == (batch_size, 1), f"Expected ({batch_size}, 1), got {output.shape}"
    
    print("Batch sizes test passed!")


def test_cnnattention_edge_cases():
    """Test CNNAttention model edge cases"""
    print("\nTesting CNNAttention model edge cases...")
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Test with minimal configuration
    model_minimal = model_manager.create_model('CNNAtention',
                                              in_channels=1,
                                              dimension=2,
                                              hidden_channels=8,
                                              num_encoder_layers=1,
                                              use_attention=False,
                                              mlp_hidden_sizes=[16])
    
    # Test with small input
    x_small = torch.randn(1, 1, 8, 8)
    output_small = model_minimal(x_small)
    print(f"Minimal config - Small input: {x_small.shape} -> {output_small.shape}")
    assert output_small.shape == (1, 1)
    
    # Test with single channel input
    model_single = model_manager.create_model('CNNAtention',
                                             in_channels=1,
                                             dimension=2,
                                             hidden_channels=16,
                                             use_attention=True)
    
    x_single = torch.randn(2, 1, 16, 16)
    output_single = model_single(x_single)
    print(f"Single channel - Input: {x_single.shape} -> {output_single.shape}")
    assert output_single.shape == (2, 1)
    
    print("Edge cases test passed!")


def run_all_cnnattention_tests():
    """Run all CNNAttention tests"""
    print("=" * 60)
    print("CNNAttention Model Test Suite")
    print("=" * 60)
    
    test_functions = [
        test_cnnattention_basic_2d,
        test_cnnattention_1d,
        test_cnnattention_3d,
        test_cnnattention_attention_mechanism,
        test_cnnattention_pooling_strategies,
        test_cnnattention_activation_functions,
        test_cnnattention_hyperparameters,
        test_cnnattention_gradient_flow,
        test_cnnattention_model_parameters,
        test_cnnattention_batch_sizes,
        test_cnnattention_edge_cases
    ]
    
    passed_tests = 0
    failed_tests = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed_tests += 1
            print(f"✓ {test_func.__name__} PASSED")
        except Exception as e:
            failed_tests += 1
            print(f"✗ {test_func.__name__} FAILED: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed_tests} passed, {failed_tests} failed")
    print("=" * 60)
    
    if failed_tests == 0:
        print("🎉 All CNNAttention tests passed!")
    else:
        print(f"❌ {failed_tests} tests failed")
    
    return failed_tests == 0


if __name__ == "__main__":
    run_all_cnnattention_tests()