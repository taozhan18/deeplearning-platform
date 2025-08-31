"""
Test script for the low-code deep learning platform
"""

import sys
import os

# Add the modules to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_loader', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'model_architecture', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'training_engine', 'src'))

from data_loader import DataLoaderModule
from model_manager import ModelManager, ModelTemplate
from training_engine import TrainingEngine
import numpy as np


def test_data_loader():
    """Test the data loader module"""
    print("Testing Data Loader Module...")
    
    config = {
        'train_features_path': 'data/train_features.csv',
        'train_targets_path': 'data/train_targets.csv',
        'test_features_path': 'data/test_features.csv',
        'test_targets_path': 'data/test_targets.csv',
        'batch_size': 32,
        'shuffle': True
    }
    
    data_loader = DataLoaderModule(config)
    data_loader.prepare_datasets()
    data_loader.create_data_loaders()
    train_loader, test_loader = data_loader.get_data_loaders()
    
    # Print dataset info
    dataset_info = data_loader.get_dataset_info()
    print(f"Dataset info: {dataset_info}")
    
    # Check a batch of data
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Data shape: {data.shape}, Target shape: {target.shape}")
        print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")
        print(f"Target range: [{target.min()}, {target.max()}]")
        break  # Only check the first batch
    
    return train_loader, test_loader


def test_model_manager():
    """Test the model manager module"""
    print("\nTesting Model Manager Module...")
    
    model_manager = ModelManager({})
    
    # List available models
    available_models = model_manager.get_available_models()
    print(f"Available models: {available_models}")
    
    # Create a model using the template
    model = model_manager.create_model('ModelTemplate', 
                                      input_size=5, 
                                      hidden_size=64, 
                                      num_classes=10,
                                      dropout_rate=0.2)
    
    print(f"Model created: {model}")
    
    # Get model hyperparameters
    hyperparams = model_manager.get_model_hyperparameters('ModelTemplate')
    print(f"Model hyperparameters: {list(hyperparams.keys())}")
    
    return model


def test_training_engine(train_loader, test_loader, model):
    """Test the training engine module"""
    print("\nTesting Training Engine Module...")
    
    config = {
        'epochs': 3,
        'device': 'cpu'
    }
    
    training_engine = TrainingEngine(config)
    training_engine.set_model(model)
    training_engine.configure_optimizer('adam', lr=0.001)
    training_engine.configure_criterion('cross_entropy')
    
    # Train for a few epochs
    history = training_engine.train(train_loader, test_loader)
    
    print("Training completed!")
    print(f"Final train accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    
    # Save model
    training_engine.save_model('data/test_model.pth')
    print("Model saved to data/test_model.pth")
    
    return training_engine


def main():
    """Main test function"""
    print("Testing Low-Code Deep Learning Platform")
    print("=" * 50)
    
    # Test data loader
    train_loader, test_loader = test_data_loader()
    
    # Test model manager
    model = test_model_manager()
    
    # Test training engine
    training_engine = test_training_engine(train_loader, test_loader, model)
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")


if __name__ == "__main__":
    main()