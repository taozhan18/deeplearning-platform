#!/usr/bin/env python3
"""
Simplified workflow: Generate synthetic data -> Train ML model
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import required modules
from model_architecture.src.model_manager import ModelManager
from training_engine.src.training_engine import TrainingEngine


def create_synthetic_dataset():
    """Create synthetic dataset for demonstration purposes"""
    print("Creating synthetic dataset for demonstration...")
    
    # Create directories
    dataset_dir = Path("moose_ml_dataset")
    dataset_dir.mkdir(exist_ok=True)
    
    # Generate synthetic data
    num_samples = 20
    num_points = 50
    
    # Generate input fields (coordinates)
    input_fields = []
    output_fields = []
    parameters = []
    
    for i in range(num_samples):
        # Coordinates
        x_coords = np.linspace(0, 1, num_points)
        
        # Random parameters
        left_bc = np.random.uniform(0, 1)
        right_bc = np.random.uniform(0, 1)
        params = {'left_bc': left_bc, 'right_bc': right_bc}
        
        # Linear solution (simplified diffusion)
        solution = left_bc + (right_bc - left_bc) * x_coords
        
        # Add some noise
        solution += np.random.normal(0, 0.05, num_points)
        
        input_fields.append(x_coords)
        output_fields.append(solution)
        parameters.append(params)
    
    # Convert to arrays
    input_fields = np.array(input_fields)
    output_fields = np.array(output_fields)
    parameters = np.array(parameters)
    
    # Save data
    np.save(dataset_dir / "input_default.npy", input_fields)
    np.save(dataset_dir / "output_default.npy", output_fields)
    np.save(dataset_dir / "parameters.npy", parameters)
    
    print(f"Created synthetic dataset with {num_samples} samples")
    return {
        'input_fields': {'default': input_fields},
        'output_fields': {'default': output_fields},
        'parameters': parameters
    }


def load_and_prepare_data():
    """Load and prepare data for training"""
    print("\nStep 1: Loading and preparing data...")
    
    # Check if dataset exists
    dataset_dir = Path("moose_ml_dataset")
    if not dataset_dir.exists():
        print("Dataset not found. Creating synthetic dataset...")
        dataset = create_synthetic_dataset()
        return dataset
    
    # Load data
    try:
        input_data = np.load(dataset_dir / "input_default.npy", allow_pickle=True)
        output_data = np.load(dataset_dir / "output_default.npy", allow_pickle=True)
        parameters = np.load(dataset_dir / "parameters.npy", allow_pickle=True)
        
        dataset = {
            'input_fields': {'default': input_data},
            'output_fields': {'default': output_data},
            'parameters': parameters
        }
        
        print(f"Loaded dataset with {len(input_data)} samples")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating synthetic dataset as fallback...")
        return create_synthetic_dataset()


def create_and_train_model(dataset):
    """Create and train ML model"""
    print("\nStep 2: Creating and training ML model...")
    
    try:
        # Create model manager
        model_manager = ModelManager({})
        
        # Create FNO model for 1D problems
        import torch.nn as nn
        model = model_manager.create_model(
            'fno',
            in_channels=1,
            out_channels=1,
            dimension=1,  # 1D problem
            num_fno_modes=8,
            latent_channels=20,
            num_fno_layers=4,
            activation_fn=nn.GELU()  # 传递实际的激活函数实例而不是字符串
        )
        
        print(f"Created FNO model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Prepare data for training
        input_data = dataset['input_fields']['default']
        output_data = dataset['output_fields']['default']
        
        # Reshape data for FNO (batch, channels, x)
        x_train = input_data.reshape(input_data.shape[0], 1, -1)
        y_train = output_data.reshape(output_data.shape[0], 1, -1)
        
        # Convert to PyTorch tensors
        import torch
        x_train = torch.FloatTensor(x_train)
        y_train = torch.FloatTensor(y_train)
        
        # Create data loader
        from torch.utils.data import TensorDataset, DataLoader
        dataset_obj = TensorDataset(x_train, y_train)
        dataloader = DataLoader(dataset_obj, batch_size=4, shuffle=True)
        
        # Setup training
        training_engine = TrainingEngine({
            'device': 'cpu',
            'epochs': 5
        })
        training_engine.set_model(model)
        training_engine.configure_optimizer('adam', lr=0.001)
        training_engine.configure_criterion('mse')
        
        print("Starting training...")
        
        # Train for a few epochs，使用train_epoch方法而不是train_batch
        for epoch in range(5):
            avg_loss, accuracy = training_engine.train_epoch(dataloader)
            print(f"Epoch {epoch+1}/5, Average Loss: {avg_loss:.6f}")
        
        print("Training completed!")
        
        # Save model
        torch.save(model.state_dict(), "trained_moose_model.pth")
        print("Model saved as trained_moose_model.pth")
        
        return model
        
    except Exception as e:
        print(f"Error in model creation/training: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main workflow function"""
    print("MOOSE to ML Simplified Workflow")
    print("=" * 40)
    
    # Step 1: Load and prepare data
    dataset = load_and_prepare_data()
    
    # Step 2: Create and train model
    model = create_and_train_model(dataset)
    
    if model is not None:
        print("\nWorkflow completed successfully!")
        print("✓ Data loaded and prepared")
        print("✓ Model trained and saved")
    else:
        print("\nWorkflow completed with some issues.")
        print("✓ Data loaded and prepared")
        print("✗ Model training failed")


if __name__ == "__main__":
    main()