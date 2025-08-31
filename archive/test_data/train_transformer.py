#!/usr/bin/env python3
"""
Example script for training a Transformer model using the low-code platform
"""

import sys
import os
import torch
import numpy as np

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_loader.src.data_loader import DataLoaderModule
from model_architecture.src.model_manager import ModelManager
from training_engine.src.training_engine import TrainingEngine


def generate_sample_data():
    """
    Generate sample sequential data for demonstration
    """
    print("Generating sample sequential data...")
    
    # Create sample sequential data
    n_samples = 1000
    seq_len = 50
    input_dim = 10
    output_dim = 10  # Make output dimension match input for sequence-to-sequence task
    
    # Generate input sequences (e.g., time series data)
    train_data = []
    train_targets = []
    
    for i in range(n_samples):
        # Generate input sequence (random walk-like pattern)
        seq = np.cumsum(np.random.randn(seq_len, input_dim), axis=0)
        
        # Generate target sequence (e.g., smoothed version of input)
        # Simple moving average as target
        target = np.zeros_like(seq)
        for j in range(seq_len):
            start_idx = max(0, j - 5)
            target[j] = np.mean(seq[start_idx:j+1], axis=0)
        
        train_data.append(seq)
        train_targets.append(target)
    
    # Convert to numpy arrays
    train_data = np.array(train_data, dtype=np.float32)
    train_targets = np.array(train_targets, dtype=np.float32)
    
    # Split into train and test
    split_idx = int(0.8 * n_samples)
    test_data = train_data[split_idx:]
    test_targets = train_targets[split_idx:]
    train_data = train_data[:split_idx]
    train_targets = train_targets[:split_idx]
    
    # Save data
    os.makedirs('data', exist_ok=True)
    np.save('data/train_data.npy', train_data)
    np.save('data/train_targets.npy', train_targets)
    np.save('data/test_data.npy', test_data)
    np.save('data/test_targets.npy', test_targets)
    
    print(f"Generated {n_samples} samples")
    print(f"Train data shape: {train_data.shape}")
    print(f"Train targets shape: {train_targets.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Test targets shape: {test_targets.shape}")


def train_transformer_model():
    """
    Train a Transformer model using the low-code platform
    """
    print("Training Transformer model...")
    
    # Initialize data loader
    data_loader = DataLoaderModule({
        'train_features_path': 'data/train_data.npy',
        'train_targets_path': 'data/train_targets.npy',
        'test_features_path': 'data/test_data.npy',
        'test_targets_path': 'data/test_targets.npy',
        'batch_size': 32,
        'shuffle': True
    })
    
    # Prepare datasets and create data loaders
    data_loader.prepare_datasets()
    data_loader.create_data_loaders()
    
    # Get data loaders
    train_loader, test_loader = data_loader.get_data_loaders()
    
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Test loader batches: {len(test_loader)}")
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Create Transformer model
    model = model_manager.create_model(
        'transformer',
        input_dim=10,
        output_dim=10,  # Match input dimension for sequence-to-sequence
        d_model=64,
        n_layers=4,
        n_heads=8,
        pf_dim=256,
        dropout=0.1,
        max_len=100
    )
    
    print(f"Created Transformer model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize training engine
    training_engine = TrainingEngine({
        'epochs': 20,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })
    
    training_engine.set_model(model)
    training_engine.configure_optimizer('adam', lr=0.001, weight_decay=0.0001)
    training_engine.configure_criterion('mse')
    # Remove scheduler configuration as it's not supported
    
    # Train the model
    print("Starting training...")
    history = training_engine.train(train_loader, test_loader)
    
    # Save the trained model
    os.makedirs('saved_models', exist_ok=True)
    model_path = 'saved_models/transformer_model.pth'
    training_engine.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    return model, history


def main():
    """
    Main function to run the example
    """
    print("Transformer Model Training Example")
    print("=" * 40)
    
    try:
        # Generate sample data
        generate_sample_data()
        
        # Train Transformer model
        model, history = train_transformer_model()
        
        # Print final results
        print("\nTraining completed successfully!")
        print(f"Final training loss: {history['train_loss'][-1]:.6f}")
        print(f"Final test loss: {history['val_loss'][-1]:.6f}")
        
        print("\nExample completed!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()