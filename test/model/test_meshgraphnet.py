#!/usr/bin/env python3
"""
Test script for MeshGraphNet model (DGL-free version)
"""

import sys
import os
import torch
import numpy as np

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from model_architecture.src.meshgraphnet.meshgraphnet import MeshGraphNet


def test_meshgraphnet_creation():
    """Test basic MeshGraphNet model creation"""
    print("Testing MeshGraphNet model creation...")
    
    try:
        # Create a basic MeshGraphNet model without DGL
        model = MeshGraphNet(
            input_dim_nodes=4,
            input_dim_edges=3,
            output_dim=2,
            processor_size=2,  # Use smaller size for testing
            hidden_dim_processor=32,
            hidden_dim_node_encoder=32,
            hidden_dim_edge_encoder=32,
            hidden_dim_node_decoder=32
        )
        print("✓ MeshGraphNet model created successfully")
        print(f"  Model type: {type(model)}")
        print(f"  Number of parameters: {sum(p.numel() for p in model.parameters())}")
        return model
    except Exception as e:
        print(f"✗ Failed to create MeshGraphNet model: {e}")
        raise


def test_meshgraphnet_forward_pass():
    """Test forward pass of MeshGraphNet model"""
    print("\nTesting MeshGraphNet forward pass...")
    
    try:
        model = MeshGraphNet(
            input_dim_nodes=4,
            input_dim_edges=3,
            output_dim=2,
            processor_size=2,
            hidden_dim_processor=32
        )
        
        # Create test data
        num_nodes = 10
        num_edges = 15
        
        node_features = torch.randn(num_nodes, 4)
        edge_features = torch.randn(num_edges, 3)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        # Forward pass
        output = model(node_features, edge_features, edge_index)
        
        print("✓ Forward pass successful")
        print(f"  Input node features shape: {node_features.shape}")
        print(f"  Input edge features shape: {edge_features.shape}")
        print(f"  Edge index shape: {edge_index.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Check output dimensions
        assert output.shape == (num_nodes, 2), f"Expected output shape ({num_nodes}, 2), got {output.shape}"
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        raise


def test_meshgraphnet_parameter_retrieval():
    """Test retrieval of MeshGraphNet model parameters"""
    print("\nTesting MeshGraphNet model parameter retrieval...")
    
    try:
        # Get model parameters directly from class
        parameters = MeshGraphNet.get_hyperparameters()
        print("✓ MeshGraphNet parameters retrieved successfully")
        print(f"  Number of parameters: {len(parameters)}")
        
        # Check for key parameters
        required_params = ['input_dim_nodes', 'input_dim_edges', 'output_dim']
        for param in required_params:
            if param in parameters:
                print(f"  ✓ Required parameter '{param}' found")
            else:
                print(f"  ✗ Required parameter '{param}' missing")
                return False
                
        return True
    except Exception as e:
        print(f"✗ Failed to retrieve MeshGraphNet parameters: {e}")
        raise


def test_meshgraphnet_different_sizes():
    """Test MeshGraphNet with different sizes"""
    print("\nTesting MeshGraphNet with different sizes...")
    
    try:
        # Test small model
        small_model = MeshGraphNet(
            input_dim_nodes=2,
            input_dim_edges=1,
            output_dim=1,
            processor_size=1,
            hidden_dim_processor=16
        )
        
        # Test large model
        large_model = MeshGraphNet(
            input_dim_nodes=10,
            input_dim_edges=5,
            output_dim=3,
            processor_size=3,
            hidden_dim_processor=64
        )
        
        print("✓ Different sizes tested successfully")
        print(f"  Small model params: {sum(p.numel() for p in small_model.parameters())}")
        print(f"  Large model params: {sum(p.numel() for p in large_model.parameters())}")
        
        return True
    except Exception as e:
        print(f"✗ Different sizes test failed: {e}")
        raise


def main():
    """Main test function"""
    print("Testing MeshGraphNet Model (DGL-free version)")
    print("=" * 60)
    
    try:
        # Test model creation
        model = test_meshgraphnet_creation()
        
        # Test forward pass
        test_meshgraphnet_forward_pass()
        
        # Test parameter retrieval
        test_meshgraphnet_parameter_retrieval()
        
        # Test different sizes
        test_meshgraphnet_different_sizes()
        
        print("\n" + "=" * 60)
        print("All MeshGraphNet tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)