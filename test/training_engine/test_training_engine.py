"""
Test script for training engine module
"""

import sys
import os
import torch
import torch.nn as nn

# Add the modules to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'training_engine', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'model_architecture', 'src'))

from training_engine import TrainingEngine
from model_manager import ModelTemplate


def test_training_engine_functionality():
    """Test the basic functionality of the training engine"""
    print("Testing Training Engine Functionality...")
    
    # Configuration
    config = {
        'epochs': 2,
        'device': 'cpu'
    }
    
    # Initialize training engine
    training_engine = TrainingEngine(config)
    print("Training engine initialized successfully")
    
    # Create a simple model for testing
    model = ModelTemplate(input_size=5, hidden_size=32, num_classes=10, dropout_rate=0.1)
    training_engine.set_model(model)
    print("Model set successfully")
    
    # Configure optimizer
    training_engine.configure_optimizer('adam', lr=0.001)
    print("Optimizer configured successfully")
    
    # Configure criterion
    training_engine.configure_criterion('cross_entropy')
    print("Criterion configured successfully")
    
    # Configure scheduler
    training_engine.configure_scheduler('step', step_size=1, gamma=0.5)
    print("Scheduler configured successfully")
    
    print("Training engine functionality test passed!")


def test_training_process():
    """Test the training process"""
    print("\nTesting Training Process...")
    
    # Configuration
    config = {
        'epochs': 2,
        'device': 'cpu'
    }
    
    # Initialize training engine
    training_engine = TrainingEngine(config)
    
    # Create a simple model for testing
    model = ModelTemplate(input_size=5, hidden_size=32, num_classes=10, dropout_rate=0.1)
    training_engine.set_model(model)
    
    # Configure optimizer and criterion
    training_engine.configure_optimizer('adam', lr=0.001)
    training_engine.configure_criterion('cross_entropy')
    
    # Create dummy data loaders
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=100):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            # Return random data and target
            return torch.randn(5), torch.randint(0, 10, (1,)).item()
    
    train_dataset = DummyDataset(100)
    test_dataset = DummyDataset(20)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Run training
    history = training_engine.train(train_loader, test_loader)
    
    # Check history
    assert 'train_loss' in history
    assert 'train_acc' in history
    assert 'val_loss' in history
    assert 'val_acc' in history
    assert 'epochs' in history
    
    print(f"Training history keys: {list(history.keys())}")
    print(f"Number of epochs recorded: {len(history['epochs'])}")
    
    print("Training process test passed!")


def main():
    """Main test function for training engine"""
    print("Testing Training Engine Module")
    print("=" * 35)
    
    # Test basic functionality
    test_training_engine_functionality()
    
    # Test training process
    test_training_process()
    
    print("\n" + "=" * 35)
    print("All training engine tests completed successfully!")


if __name__ == "__main__":
    main()