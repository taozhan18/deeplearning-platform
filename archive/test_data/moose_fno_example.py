#!/usr/bin/env python3
"""
Complete example of using MOOSE automation with FNO model training
"""

import os
import sys
import json
import numpy as np
import torch

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_loader', 'scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_loader', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model_architecture', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training_engine', 'src'))

from moose_data_generator import MOOSEDataGenerator
from data_loader import DataLoaderModule
from model_manager import ModelManager
from training_engine import TrainingEngine


def create_example_dataset():
    """Create an example dataset using MOOSE simulations"""
    print("Creating example dataset with MOOSE simulations...")
    
    # MOOSE executable path
    moose_exec = "/home/zt/workspace/mymoose/mymoose-opt"
    
    # Create parameter sets for parametric study
    param_sets = [
        {"nx": 64, "left_bc": 0.0, "right_bc": 1.0},
        {"nx": 64, "left_bc": 0.0, "right_bc": 2.0},
        {"nx": 128, "left_bc": 1.0, "right_bc": 2.0},
        {"nx": 128, "left_bc": 0.5, "right_bc": 1.5},
        {"nx": 256, "left_bc": 0.0, "right_bc": 1.0},
        {"nx": 256, "left_bc": 1.0, "right_bc": 3.0},
    ]
    
    # Initialize MOOSE data generator
    generator = MOOSEDataGenerator(moose_exec)
    
    # Path to input template
    input_template = os.path.join(
        os.path.dirname(__file__), 
        '..', 'data_loader', 'scripts', 'simple_test.i'
    )
    
    # Run simulations
    dataset_dir = generator.run_parametric_sims(
        input_template, param_sets, "moose_fno_dataset"
    )
    
    # Extract field data
    print("Extracting field data from simulations...")
    field_data = generator.extract_field_data(
        dataset_dir, ['default'], ['default'], 'numpy'
    )
    
    # Create FNO training data
    print("Creating FNO training data...")
    fno_data = generator.create_fno_training_data(dataset_dir)
    
    print(f"Dataset created successfully in {dataset_dir}")
    return dataset_dir


def train_fno_model(dataset_dir):
    """Train an FNO model with the MOOSE-generated dataset"""
    print("Training FNO model with MOOSE data...")
    
    # Load training data
    train_input = np.load(os.path.join(dataset_dir, "train_input.npy"))
    train_output = np.load(os.path.join(dataset_dir, "train_output.npy"))
    test_input = np.load(os.path.join(dataset_dir, "test_input.npy"))
    test_output = np.load(os.path.join(dataset_dir, "test_output.npy"))
    
    print(f"Training data shape: {train_input.shape} -> {train_output.shape}")
    print(f"Test data shape: {test_input.shape} -> {test_output.shape}")
    
    # Create PyTorch data loaders
    from torch.utils.data import TensorDataset, DataLoader
    
    train_dataset = TensorDataset(
        torch.FloatTensor(train_input), 
        torch.FloatTensor(train_output)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(test_input), 
        torch.FloatTensor(test_output)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Create 1D FNO model
    model = model_manager.create_model(
        'fno',
        in_channels=1,
        out_channels=1,
        dimension=1,
        latent_channels=16,
        num_fno_layers=4,
        num_fno_modes=16,
        decoder_layers=2,
        decoder_layer_size=32
    )
    
    print(f"Created FNO model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize training engine
    config = {
        'epochs': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    training_engine = TrainingEngine(config)
    training_engine.set_model(model)
    training_engine.configure_optimizer('adam', lr=0.001)
    training_engine.configure_criterion('mse')
    
    # Train the model
    print("Starting training...")
    history = training_engine.train(train_loader, test_loader)
    
    # Save the trained model
    model_path = os.path.join(dataset_dir, "trained_fno_model.pth")
    training_engine.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    return model, history


def main():
    """Main function demonstrating the complete workflow"""
    print("MOOSE-FNO Integration Example")
    print("=" * 40)
    
    try:
        # Step 1: Generate dataset using MOOSE simulations
        dataset_dir = create_example_dataset()
        
        # Step 2: Train FNO model with the dataset
        model, history = train_fno_model(dataset_dir)
        
        # Print final results
        print("\nTraining completed successfully!")
        print(f"Final training loss: {history['train_loss'][-1]:.6f}")
        print(f"Final test loss: {history['val_loss'][-1]:.6f}")
        
        print("\nExample workflow completed!")
        print(f"Dataset is located at: {os.path.abspath(dataset_dir)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()