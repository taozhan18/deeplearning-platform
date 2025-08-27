#!/usr/bin/env python3
"""
Test script for Transformer model
"""

import sys
import os
import torch
import numpy as np

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from model_architecture.src.model_manager import ModelManager


def test_transformer_creation():
    """Test basic Transformer model creation"""
    print("Testing Transformer model creation...")
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Test creating a basic Transformer model
    try:
        model = model_manager.create_model(
            'transformer',
            input_dim=10,
            output_dim=5,
            d_model=64,
            n_layers=2,
            n_heads=4,
            pf_dim=128,
            dropout=0.1,
            max_len=50
        )
        print("✓ Transformer model created successfully")
        print(f"  Model type: {type(model)}")
        print(f"  Number of parameters: {sum(p.numel() for p in model.parameters())}")
        return model
    except Exception as e:
        print(f"✗ Failed to create Transformer model: {e}")
        raise


def test_transformer_forward_pass():
    """Test Transformer model forward pass"""
    print("\nTesting Transformer model forward pass...")
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Create a small Transformer model for testing
    model = model_manager.create_model(
        'transformer',
        input_dim=8,
        output_dim=4,
        d_model=32,
        n_layers=2,
        n_heads=2,
        pf_dim=64,
        dropout=0.1,
        max_len=20
    )
    
    # Create sample input data
    batch_size = 4
    seq_len = 10
    input_dim = 8
    
    # Input sequence
    x = torch.randn(batch_size, seq_len, input_dim)
    
    try:
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        print("✓ Forward pass completed successfully")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Check output shape
        expected_shape = (batch_size, seq_len, 4)  # output_dim=4
        if output.shape == expected_shape:
            print("✓ Output shape is correct")
        else:
            print(f"✗ Output shape mismatch. Expected: {expected_shape}, Got: {output.shape}")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        raise


def test_transformer_with_different_configs():
    """Test Transformer with different configurations"""
    print("\nTesting Transformer with different configurations...")
    
    model_manager = ModelManager({})
    
    configs = [
        {
            'name': 'Small Transformer',
            'params': {
                'input_dim': 4,
                'output_dim': 2,
                'd_model': 16,
                'n_layers': 1,
                'n_heads': 2,
                'pf_dim': 32,
                'dropout': 0.0,
                'max_len': 10
            }
        },
        {
            'name': 'Medium Transformer',
            'params': {
                'input_dim': 16,
                'output_dim': 8,
                'd_model': 64,
                'n_layers': 3,
                'n_heads': 4,
                'pf_dim': 128,
                'dropout': 0.1,
                'max_len': 30
            }
        }
    ]
    
    for config in configs:
        try:
            print(f"  Testing {config['name']}...")
            model = model_manager.create_model('transformer', **config['params'])
            
            # Test with sample data
            batch_size = 2
            seq_len = 5
            
            x = torch.randn(batch_size, seq_len, config['params']['input_dim'])
            
            with torch.no_grad():
                output = model(x)
            
            expected_shape = (batch_size, seq_len, config['params']['output_dim'])
            if output.shape == expected_shape:
                print(f"  ✓ {config['name']} works correctly. Output shape: {output.shape}")
            else:
                print(f"  ✗ {config['name']} output shape mismatch. Expected: {expected_shape}, Got: {output.shape}")
                return False
                
        except Exception as e:
            print(f"  ✗ {config['name']} failed: {e}")
            raise
    
    return True


def test_transformer_parameter_retrieval():
    """Test retrieval of Transformer model parameters"""
    print("\nTesting Transformer model parameter retrieval...")
    
    model_manager = ModelManager({})
    
    try:
        # Get model parameters
        parameters = model_manager.get_model_parameters('transformer')
        print("✓ Transformer parameters retrieved successfully")
        print(f"  Number of parameters: {len(parameters)}")
        
        # Check for key parameters
        required_params = ['input_dim', 'output_dim']
        for param in required_params:
            if param in parameters:
                print(f"  ✓ Required parameter '{param}' found")
            else:
                print(f"  ✗ Required parameter '{param}' missing")
                return False
                
        return True
    except Exception as e:
        print(f"✗ Failed to retrieve Transformer parameters: {e}")
        raise


def main():
    """Main test function"""
    print("Testing Transformer Model")
    print("=" * 50)
    
    try:
        # Test model creation
        model = test_transformer_creation()
        
        # Test forward pass
        test_transformer_forward_pass()
        
        # Test different configurations
        test_transformer_with_different_configs()
        
        # Test parameter retrieval
        test_transformer_parameter_retrieval()
        
        print("\n" + "=" * 50)
        print("All Transformer tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)