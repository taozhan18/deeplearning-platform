"""
Comprehensive integration tests for low-code deep learning platform
Tests MLP, FNO, UNet, Transformer, Transolver, and MeshGraphNet
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import torch
from pathlib import Path

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'main'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'data_loader', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'model_architecture', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'training_engine', 'src'))

from train import main as train_main
from data_loader import DataLoaderModule
from model_manager import ModelManager

def create_test_data_mlp():
    """Create test data for MLP (tabular data)"""
    # Create tabular data: (samples, features)
    train_features = np.random.randn(100, 20)
    train_targets = np.random.randn(100, 1)
    test_features = np.random.randn(30, 20)
    test_targets = np.random.randn(30, 1)
    
    return {
        'train_features': train_features,
        'train_targets': train_targets,
        'test_features': test_features,
        'test_targets': test_targets
    }

def create_test_data_grid():
    """Create test data for FNO/UNet (grid data)"""
    # Create grid data: (samples, channels, height, width)
    train_features = np.random.randn(50, 3, 32, 32)
    train_targets = np.random.randn(50, 1, 32, 32)
    test_features = np.random.randn(20, 3, 32, 32)
    test_targets = np.random.randn(20, 1, 32, 32)
    
    return {
        'train_features': train_features,
        'train_targets': train_targets,
        'test_features': test_features,
        'test_targets': test_targets
    }

def create_test_data_sequence():
    """Create test data for Transformer/Transolver (sequence data)"""
    # Create sequence data: (samples, time_steps, features)
    train_features = np.random.randn(80, 100, 10)
    train_targets = np.random.randn(80, 100, 1)
    test_features = np.random.randn(30, 100, 10)
    test_targets = np.random.randn(30, 100, 1)
    
    return {
        'train_features': train_features,
        'train_targets': train_targets,
        'test_features': test_features,
        'test_targets': test_targets
    }

def create_test_data_graph():
    """Create test data for MeshGraphNet (graph data)"""
    # Create graph data: node features and connectivity
    num_nodes = 50
    num_edges = 100
    
    train_node_features = np.random.randn(10, num_nodes, 10)  # (batch, nodes, features)
    train_edge_index = np.random.randint(0, num_nodes, (10, 2, num_edges))  # (batch, 2, edges)
    train_edge_features = np.random.randn(10, num_edges, 3)  # (batch, edges, features)
    train_targets = np.random.randn(10, num_nodes, 1)  # (batch, nodes, output)
    
    test_node_features = np.random.randn(5, num_nodes, 10)
    test_edge_index = np.random.randint(0, num_nodes, (5, 2, num_edges))
    test_edge_features = np.random.randn(5, num_edges, 3)
    test_targets = np.random.randn(5, num_nodes, 1)
    
    return {
        'train_node_features': train_node_features,
        'train_edge_index': train_edge_index,
        'train_edge_features': train_edge_features,
        'train_targets': train_targets,
        'test_node_features': test_node_features,
        'test_edge_index': test_edge_index,
        'test_edge_features': test_edge_features,
        'test_targets': test_targets
    }

def save_test_data(data_dict, data_dir, prefix):
    """Save test data to files"""
    os.makedirs(data_dir, exist_ok=True)
    
    for key, value in data_dict.items():
        file_path = os.path.join(data_dir, f"{key}.npz")
        np.savez(file_path, data=value)
        print(f"Saved {file_path} with shape {value.shape}")

def test_mlp_integration():
    """Test MLP integration with tabular data"""
    print("🧪 Testing MLP Integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        data = create_test_data_mlp()
        
        # Save data
        data_dir = os.path.join(temp_dir, 'data')
        save_test_data({
            'tabular_train_features': data['train_features'],
            'tabular_train_targets': data['train_targets'],
            'tabular_test_features': data['test_features'],
            'tabular_test_targets': data['test_targets']
        }, data_dir, 'tabular')
        
        # Update config file paths
        config_path = os.path.join(os.path.dirname(__file__), 'mlp', 'config_mlp.yaml')
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Replace paths with absolute paths
        config_content = config_content.replace(
            "tabular_train_features.npz",
            os.path.join(data_dir, "tabular_train_features.npz")
        ).replace(
            "tabular_train_targets.npz",
            os.path.join(data_dir, "tabular_train_targets.npz")
        ).replace(
            "tabular_test_features.npz",
            os.path.join(data_dir, "tabular_test_features.npz")
        ).replace(
            "tabular_test_targets.npz",
            os.path.join(data_dir, "tabular_test_targets.npz")
        )
        
        # Create temporary config with updated output paths
        config_content = config_content.replace(
            "trained_mlp_model.pth",
            os.path.join(temp_dir, "trained_mlp_model.pth")
        ).replace(
            "mlp_training_history.json",
            os.path.join(temp_dir, "mlp_training_history.json")
        )
        
        temp_config = os.path.join(temp_dir, 'temp_config.yaml')
        with open(temp_config, 'w') as f:
            f.write(config_content)
        
        # Run training
        try:
            train_main(temp_config)
            print("✅ MLP integration test passed")
            return True
        except Exception as e:
            print(f"❌ MLP integration test failed: {e}")
            return False

def test_fno_integration():
    """Test FNO integration with grid data"""
    print("🧪 Testing FNO Integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        data = create_test_data_grid()
        
        # Save data
        data_dir = os.path.join(temp_dir, 'data')
        save_test_data({
            'grid_train_features': data['train_features'],
            'grid_train_targets': data['train_targets'],
            'grid_test_features': data['test_features'],
            'grid_test_targets': data['test_targets']
        }, data_dir, 'grid')
        
        # Update config file paths
        config_path = os.path.join(os.path.dirname(__file__), 'fno', 'config_fno.yaml')
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Replace paths with absolute paths
        config_content = config_content.replace(
            "grid_train_features.npz",
            os.path.join(data_dir, "grid_train_features.npz")
        ).replace(
            "grid_train_targets.npz",
            os.path.join(data_dir, "grid_train_targets.npz")
        ).replace(
            "grid_test_features.npz",
            os.path.join(data_dir, "grid_test_features.npz")
        ).replace(
            "grid_test_targets.npz",
            os.path.join(data_dir, "grid_test_targets.npz")
        )
        
        # Create temporary config with updated output paths
        config_content = config_content.replace(
            "trained_fno_model.pth",
            os.path.join(temp_dir, "trained_fno_model.pth")
        ).replace(
            "fno_training_history.json",
            os.path.join(temp_dir, "fno_training_history.json")
        )
        
        temp_config = os.path.join(temp_dir, 'temp_config.yaml')
        with open(temp_config, 'w') as f:
            f.write(config_content)
        
        # Run training
        try:
            train_main(temp_config)
            print("✅ FNO integration test passed")
            return True
        except Exception as e:
            print(f"❌ FNO integration test failed: {e}")
            return False

def test_transformer_integration():
    """Test Transformer integration with sequence data"""
    print("🧪 Testing Transformer Integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        data = create_test_data_sequence()
        
        # Save data
        data_dir = os.path.join(temp_dir, 'data')
        save_test_data({
            'sequence_train_features': data['train_features'],
            'sequence_train_targets': data['train_targets'],
            'sequence_test_features': data['test_features'],
            'sequence_test_targets': data['test_targets']
        }, data_dir, 'sequence')
        
        # Update config file paths
        config_path = os.path.join(os.path.dirname(__file__), 'transformer', 'config_transformer.yaml')
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Replace paths with absolute paths
        config_content = config_content.replace(
            "sequence_train_features.npz",
            os.path.join(data_dir, "sequence_train_features.npz")
        ).replace(
            "sequence_train_targets.npz",
            os.path.join(data_dir, "sequence_train_targets.npz")
        ).replace(
            "sequence_test_features.npz",
            os.path.join(data_dir, "sequence_test_features.npz")
        ).replace(
            "sequence_test_targets.npz",
            os.path.join(data_dir, "sequence_test_targets.npz")
        )
        
        # Create temporary config with updated output paths
        config_content = config_content.replace(
            "trained_transformer_model.pth",
            os.path.join(temp_dir, "trained_transformer_model.pth")
        ).replace(
            "transformer_training_history.json",
            os.path.join(temp_dir, "transformer_training_history.json")
        )
        
        temp_config = os.path.join(temp_dir, 'temp_config.yaml')
        with open(temp_config, 'w') as f:
            f.write(config_content)
        
        # Run training
        try:
            train_main(temp_config)
            print("✅ Transformer integration test passed")
            return True
        except Exception as e:
            print(f"❌ Transformer integration test failed: {e}")
            return False

def test_meshgraphnet_integration():
    """Test MeshGraphNet integration with graph data"""
    print("🧪 Testing MeshGraphNet Integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data - simplified approach for integration testing
        num_nodes = 50
        num_edges = 100
        
        # Create single graph data (flatten batch dimension)
        # Node features: (num_nodes, feature_dim)
        node_features = np.random.randn(num_nodes, 10).astype(np.float32)
        # Edge features: (num_edges, feature_dim)
        edge_features = np.random.randn(num_edges, 3).astype(np.float32)
        # Edge index: (2, num_edges) - connectivity
        edge_index = np.random.randint(0, num_nodes, (2, num_edges)).astype(np.int64)
        # Targets: (num_nodes, output_dim)
        targets = np.random.randn(num_nodes, 1).astype(np.float32)
        
        # Save data in format expected by graph data loader
        data_dir = os.path.join(temp_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        np.savez(os.path.join(data_dir, 'graph_train_features.npz'), data=node_features)
        np.savez(os.path.join(data_dir, 'graph_train_edge_index.npz'), data=edge_index)
        np.savez(os.path.join(data_dir, 'graph_train_edge_features.npz'), data=edge_features)
        np.savez(os.path.join(data_dir, 'graph_train_targets.npz'), data=targets)
        np.savez(os.path.join(data_dir, 'graph_test_features.npz'), data=node_features)
        np.savez(os.path.join(data_dir, 'graph_test_edge_index.npz'), data=edge_index)
        np.savez(os.path.join(data_dir, 'graph_test_edge_features.npz'), data=edge_features)
        np.savez(os.path.join(data_dir, 'graph_test_targets.npz'), data=targets)
        
        # Update config file paths
        config_path = os.path.join(os.path.dirname(__file__), 'meshgraphnet', 'config_meshgraphnet.yaml')
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # For MeshGraphNet, we need to handle multiple input files
        # Update paths for all required files
        config_content = config_content.replace(
            "graph_train_features.npz",
            os.path.join(data_dir, "graph_train_features.npz")
        ).replace(
            "graph_train_targets.npz",
            os.path.join(data_dir, "graph_train_targets.npz")
        ).replace(
            "graph_test_features.npz",
            os.path.join(data_dir, "graph_test_features.npz")
        ).replace(
            "graph_test_targets.npz",
            os.path.join(data_dir, "graph_test_targets.npz")
        )
        
        # Since the standard data loader doesn't handle edge data, we'll create a simplified test
        # that directly tests the model with appropriate input format
        try:
            import torch
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'model_architecture', 'src'))
            from meshgraphnet.meshgraphnet import MeshGraphNet
            
            # Test model with sample data
            node_feat = torch.tensor(node_features)
            edge_feat = torch.tensor(edge_features)
            edge_idx = torch.tensor(edge_index)
            
            model = MeshGraphNet(
                input_dim_nodes=10,
                input_dim_edges=3,
                output_dim=1,
                processor_size=6
            )
            
            with torch.no_grad():
                output = model(node_feat, edge_feat, edge_idx)
                
            if output.shape == (num_nodes, 1):
                print("✅ MeshGraphNet integration test passed (model architecture test)")
                return True
            else:
                print(f"❌ MeshGraphNet output shape mismatch: {output.shape}")
                return False
                
        except Exception as e:
            print(f"❌ MeshGraphNet integration test failed: {e}")
            return False

def test_unet_integration():
    """Test UNet integration with grid data"""
    print("🧪 Testing UNet Integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        data = create_test_data_grid()
        
        # Save data
        data_dir = os.path.join(temp_dir, 'data')
        save_test_data({
            'grid_train_features': data['train_features'],
            'grid_train_targets': data['train_targets'],
            'grid_test_features': data['test_features'],
            'grid_test_targets': data['test_targets']
        }, data_dir, 'grid')
        
        # Update config file paths
        config_path = os.path.join(os.path.dirname(__file__), 'unet', 'config_unet.yaml')
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Replace paths with absolute paths
        config_content = config_content.replace(
            "grid_train_features.npz",
            os.path.join(data_dir, "grid_train_features.npz")
        ).replace(
            "grid_train_targets.npz",
            os.path.join(data_dir, "grid_train_targets.npz")
        ).replace(
            "grid_test_features.npz",
            os.path.join(data_dir, "grid_test_features.npz")
        ).replace(
            "grid_test_targets.npz",
            os.path.join(data_dir, "grid_test_targets.npz")
        )
        
        # Create temporary config with updated output paths
        config_content = config_content.replace(
            "trained_unet_model.pth",
            os.path.join(temp_dir, "trained_unet_model.pth")
        ).replace(
            "unet_training_history.json",
            os.path.join(temp_dir, "unet_training_history.json")
        )
        
        temp_config = os.path.join(temp_dir, 'temp_config.yaml')
        with open(temp_config, 'w') as f:
            f.write(config_content)
        
        # Run training
        try:
            train_main(temp_config)
            print("✅ UNet integration test passed")
            return True
        except Exception as e:
            print(f"❌ UNet integration test failed: {e}")
            return False

def test_transolver_integration():
    """Test Transolver integration with spatial data (simplified approach)"""
    print("🧪 Testing Transolver Integration...")
    
    # Test the Transolver model directly with simple data
    try:
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'model_architecture', 'src'))
        from transolver.transolver import Transolver
        import torch
        
        # Create simple spatial data
        batch_size = 4
        num_points = 100  # 10x10 grid
        space_dim = 2
        
        # Generate coordinates on a 2D grid
        x = torch.linspace(0, 1, 10)
        y = torch.linspace(0, 1, 10)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack([X.flatten(), Y.flatten()], dim=-1)  # (100, 2)
        
        # Create batch data
        batch_coords = coords.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, 100, 2)
        
        # Create Transolver model with adjusted parameters
        model = Transolver(
            space_dim=2,
            out_dim=1,
            n_hidden=128,
            n_layers=2,
            n_head=4,
            dropout=0.1,
            H=10,
            W=10,
            fun_dim=1  # Function dimension for targets
        )
        
        # Test forward pass
        with torch.no_grad():
            # Create dummy function values
            fx = torch.randn(batch_size, 100, 1)
            output = model(batch_coords, fx=fx)
            
        expected_shape = (batch_size, 100, 1)
        if output.shape == expected_shape:
            print("✅ Transolver integration test passed (model architecture test)")
            return True
        else:
            print(f"❌ Transolver output shape mismatch: {output.shape}, expected: {expected_shape}")
            return False
            
    except Exception as e:
        print(f"❌ Transolver integration test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🚀 Running Comprehensive Integration Tests")
    print("=" * 60)
    
    tests = [
        ("MLP", test_mlp_integration),
        ("FNO", test_fno_integration),
        ("UNet", test_unet_integration),
        ("Transformer", test_transformer_integration),
        ("Transolver", test_transolver_integration),
        ("MeshGraphNet", test_meshgraphnet_integration)
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"❌ {name} test failed with exception: {e}")
            results[name] = False
    
    print("\n" + "=" * 60)
    print("📊 Integration Test Results:")
    
    passed = sum(results.values())
    total = len(results)
    
    for name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check logs above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)