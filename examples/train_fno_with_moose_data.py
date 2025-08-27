#!/usr/bin/env python3
"""
Example script to train an FNO model with data generated from MOOSE simulations
"""

import sys
import os
import numpy as np
import torch

# Add the project modules to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_loader', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model_architecture', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training_engine', 'src'))

from data_loader import DataLoaderModule
from model_manager import ModelManager
from training_engine import TrainingEngine


def load_moose_data(dataset_dir):
    """
    Load MOOSE-generated data for training
    
    Args:
        dataset_dir: Directory containing the MOOSE dataset
        
    Returns:
        Tuple of (train_input, train_output, test_input, test_output)
    """
    dataset_path = os.path.join(dataset_dir)
    
    # Load training data
    train_input = np.load(os.path.join(dataset_path, "train_input.npy"))
    train_output = np.load(os.path.join(dataset_path, "train_output.npy"))
    
    # Load test data
    test_input = np.load(os.path.join(dataset_path, "test_input.npy"))
    test_output = np.load(os.path.join(dataset_path, "test_output.npy"))
    
    print(f"Training data input shape: {train_input.shape}")
    print(f"Training data output shape: {train_output.shape}")
    print(f"Test data input shape: {test_input.shape}")
    print(f"Test data output shape: {test_output.shape}")
    
    return train_input, train_output, test_input, test_output


def create_data_loaders(train_input, train_output, test_input, test_output, batch_size=32):
    """
    Create data loaders for training
    
    Args:
        train_input: Training input data
        train_output: Training output data
        test_input: Test input data
        test_output: Test output data
        batch_size: Batch size for training
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # For this example, we'll create simple data loaders using PyTorch
    from torch.utils.data import TensorDataset, DataLoader
    
    # Convert to PyTorch tensors
    train_x = torch.FloatTensor(train_input)
    train_y = torch.FloatTensor(train_output)
    test_x = torch.FloatTensor(test_input)
    test_y = torch.FloatTensor(test_output)
    
    # Create datasets
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_fno_model(train_loader, test_loader, input_channels, output_channels):
    """
    Train an FNO model with the provided data
    
    Args:
        train_loader: Training data loader
        test_loader: Test data loader
        input_channels: Number of input channels
        output_channels: Number of output channels
        
    Returns:
        Trained model and training history
    """
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Create FNO model for 1D field-to-field mapping
    model = model_manager.create_model(
        'fno',
        in_channels=input_channels,
        out_channels=output_channels,
        decoder_layers=2,
        decoder_layer_size=32,
        dimension=1,
        latent_channels=16,
        num_fno_layers=4,
        num_fno_modes=16,
        padding=8
    )
    
    print(f"Created FNO model: {model}")
    
    # Initialize training engine
    config = {
        'epochs': 20,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    training_engine = TrainingEngine(config)
    training_engine.set_model(model)
    training_engine.configure_optimizer('adam', lr=0.001)
    training_engine.configure_criterion('mse')
    training_engine.configure_scheduler('step', step_size=10, gamma=0.5)
    
    # Train the model
    history = training_engine.train(train_loader, test_loader)
    
    return model, history


def main():
    """Main training function"""
    print("Training FNO model with MOOSE-generated data")
    print("=" * 50)
    
    # For this example, we'll create dummy data since we don't have actual MOOSE data
    # In practice, you would load actual data from the MOOSE simulations
    
    # Create dummy 1D field data
    n_samples = 1000
    n_points = 128
    n_channels = 1
    
    print("Creating dummy 1D field data for demonstration...")
    
    # Generate dummy input data (e.g., source terms)
    train_input = np.random.randn(int(n_samples*0.8), n_channels, n_points)
    test_input = np.random.randn(int(n_samples*0.2), n_channels, n_points)
    
    # Generate dummy output data (e.g., solutions)
    # In practice, this would come from MOOSE simulations
    train_output = np.sin(train_input) + 0.1 * np.random.randn(*train_input.shape)
    test_output = np.sin(test_input) + 0.1 * np.random.randn(*test_input.shape)
    
    print(f"Training data input shape: {train_input.shape}")
    print(f"Training data output shape: {train_output.shape}")
    print(f"Test data input shape: {test_input.shape}")
    print(f"Test data output shape: {test_output.shape}")
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        train_input, train_output, test_input, test_output, batch_size=32
    )
    
    # Train FNO model
    print("\nTraining FNO model...")
    model, history = train_fno_model(
        train_loader, test_loader, n_channels, n_channels
    )
    
    # Print final results
    print("\nTraining completed!")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final test loss: {history['val_loss'][-1]:.6f}")
    print(f"Final train accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Final test accuracy: {history['val_acc'][-1]:.2f}%")
    
    # Save model
    torch.save(model.state_dict(), "fno_moose_model.pth")
    print("\nModel saved to fno_moose_model.pth")


if __name__ == "__main__":
    main()