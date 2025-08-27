"""
Test script for FNO1D model with generated dataset
"""

import sys
import os
import yaml
import json
import torch

# Add the modules to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'data_loader', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'model_architecture', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'training_engine', 'src'))

from data_loader import DataLoaderModule
from model_manager import ModelManager
from training_engine import TrainingEngine


def load_config(config_path: str) -> dict:
    """
    Load configuration from a YAML file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def test_fno1d_model():
    """Test FNO1D model with generated dataset"""
    print("Testing FNO1D model with generated dataset...")
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'test_fno1d_config.yaml')
    config = load_config(config_path)
    
    # Initialize data loader
    print("1. Initializing data loader...")
    data_loader = DataLoaderModule(config.get('data', {}))
    data_loader.prepare_datasets()
    data_loader.create_data_loaders()
    train_loader, test_loader = data_loader.get_data_loaders()
    
    # Print dataset info
    dataset_info = data_loader.get_dataset_info()
    print(f"   Dataset info: {dataset_info}")
    
    # Check data shapes
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"   Train batch {batch_idx}: Data shape: {data.shape}, Target shape: {target.shape}")
        break  # Only check the first batch
    
    for batch_idx, (data, target) in enumerate(test_loader):
        print(f"   Test batch {batch_idx}: Data shape: {data.shape}, Target shape: {target.shape}")
        break  # Only check the first batch
    
    # Initialize model manager
    print("2. Initializing model manager...")
    model_manager = ModelManager(config.get('model', {}))
    
    # Create FNO1D model
    model_config = config['model']
    model_name = model_config['name']
    model_params = model_config['parameters']
    model = model_manager.create_model(model_name, **model_params)
    print(f"   FNO1D model created successfully")
    print(f"   Model: {model}")
    
    # Initialize training engine
    print("3. Initializing training engine...")
    training_engine = TrainingEngine(config['training'])
    training_engine.set_model(model)
    
    # Configure optimizer
    optimizer_config = config['optimizer']
    optimizer_name = optimizer_config['name']
    optimizer_params = optimizer_config['parameters']
    training_engine.configure_optimizer(optimizer_name, **optimizer_params)
    
    # Configure criterion
    criterion_config = config['criterion']
    criterion_name = criterion_config['name']
    criterion_params = criterion_config['parameters']
    training_engine.configure_criterion(criterion_name, **criterion_params)
    
    # Configure scheduler if specified
    scheduler_config = config.get('scheduler', {})
    if scheduler_config:
        scheduler_name = scheduler_config['name']
        scheduler_params = scheduler_config['parameters']
        training_engine.configure_scheduler(scheduler_name, **scheduler_params)
    
    # Start training
    print("4. Starting training...")
    history = training_engine.train(train_loader, test_loader)
    
    # Save model
    output_config = config['output']
    model_path = output_config['model_path']
    training_engine.save_model(model_path)
    print(f"   Model saved to {model_path}")
    
    # Save training history
    history_path = output_config['history_path']
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f"   Training history saved to {history_path}")
    
    print("\nFNO1D model test completed successfully!")
    print(f"Final training loss: {history['train_loss'][-1]:.6f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.6f}")


def main():
    """Main test function"""
    print("Testing FNO1D Model with Generated Dataset")
    print("=" * 50)
    
    try:
        test_fno1d_model()
        print("\n" + "=" * 50)
        print("All FNO1D tests completed successfully!")
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        raise


if __name__ == "__main__":
    main()