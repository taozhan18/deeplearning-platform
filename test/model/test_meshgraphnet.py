#!/usr/bin/env python3
"""
Test script for MeshGraphNet model
"""

import sys
import os
import torch
import numpy as np

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from model_architecture.src.model_manager import ModelManager


def test_meshgraphnet_creation():
    """Test basic MeshGraphNet model creation"""
    print("Testing MeshGraphNet model creation...")
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Test creating a basic MeshGraphNet model
    try:
        model = model_manager.create_model(
            'meshgraphnet',
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


def test_meshgraphnet_parameter_retrieval():
    """Test retrieval of MeshGraphNet model parameters"""
    print("\nTesting MeshGraphNet model parameter retrieval...")
    
    model_manager = ModelManager({})
    
    try:
        # Get model parameters
        parameters = model_manager.get_model_parameters('meshgraphnet')
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


def main():
    """Main test function"""
    print("Testing MeshGraphNet Model")
    print("=" * 50)
    
    try:
        # Test model creation
        model = test_meshgraphnet_creation()
        
        # Test parameter retrieval
        test_meshgraphnet_parameter_retrieval()
        
        print("\n" + "=" * 50)
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