"""
Comprehensive test script for training engine module
Tests multi-source data handling, custom preprocessing, and all training features
"""

import sys
import os
import torch
import torch.nn as nn
import tempfile
import shutil

# Add the modules to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'training_engine', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'model_architecture', 'src'))

from training_engine import TrainingEngine


def test_basic_training_engine():
    """Test basic training engine initialization and configuration"""
    print("Testing basic training engine functionality...")
    
    # Configuration
    config = {
        'epochs': 2,
        'device': 'cpu'
    }
    
    # Initialize training engine
    training_engine = TrainingEngine(config)
    assert training_engine.epochs == 2
    assert str(training_engine.device) == 'cpu'
    assert training_engine.preprocess_fn is None
    
    print("✓ Basic initialization test passed")


def test_model_configuration():
    """Test model configuration and optimizer setup"""
    print("Testing model configuration...")
    
    config = {'epochs': 1, 'device': 'cpu'}
    training_engine = TrainingEngine(config)
    
    # Create a simple test model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 5)
    )
    
    training_engine.set_model(model)
    assert training_engine.model is not None
    
    # Test optimizer configuration
    training_engine.configure_optimizer('adam', lr=0.001)
    assert isinstance(training_engine.optimizer, torch.optim.Adam)
    
    # Test criterion configuration
    training_engine.configure_criterion('mse')
    assert isinstance(training_engine.criterion, nn.MSELoss)
    
    # Test scheduler configuration
    training_engine.configure_scheduler('step', step_size=1, gamma=0.5)
    assert isinstance(training_engine.scheduler, torch.optim.lr_scheduler.StepLR)
    
    print("✓ Model configuration test passed")


def test_single_source_training():
    """Test training with single source data (backward compatibility)"""
    print("Testing single source training...")
    
    config = {'epochs': 2, 'device': 'cpu'}
    training_engine = TrainingEngine(config)
    
    # Create simple model
    model = nn.Linear(5, 1)
    training_engine.set_model(model)
    training_engine.configure_optimizer('adam', lr=0.01)
    training_engine.configure_criterion('mse')
    
    # Create dummy single source data
    class SingleSourceDataset(torch.utils.data.Dataset):
        def __init__(self, size=50):
            self.size = size
            self.data = torch.randn(size, 5)
            self.targets = torch.randn(size, 1)
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]
    
    train_dataset = SingleSourceDataset(50)
    test_dataset = SingleSourceDataset(20)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)
    
    # Run training
    history = training_engine.train(train_loader, test_loader)
    
    assert len(history['train_loss']) == 2
    assert len(history['val_loss']) == 2
    print("✓ Single source training test passed")


def test_multi_source_training():
    """Test training with multi-source data and custom preprocessing"""
    print("Testing multi-source training...")
    
    # Test with concatenation preprocessing
    config = {
        'epochs': 2,
        'device': 'cpu',
        'preprocess_fn': TrainingEngine.create_concat_preprocess_fn()
    }
    
    training_engine = TrainingEngine(config)
    
    # Create model that expects concatenated features
    model = nn.Linear(8, 1)  # 3 + 5 = 8 features after concatenation
    training_engine.set_model(model)
    training_engine.configure_optimizer('adam', lr=0.01)
    training_engine.configure_criterion('mse')
    
    # Create dummy multi-source data
    class MultiSourceDataset(torch.utils.data.Dataset):
        def __init__(self, size=50):
            self.size = size
            self.x1 = torch.randn(size, 3)  # Source 1: 3 features
            self.x2 = torch.randn(size, 5)  # Source 2: 5 features
            self.targets = torch.randn(size, 1)
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            return {'x1': self.x1[idx], 'x2': self.x2[idx]}, self.targets[idx]
    
    train_dataset = MultiSourceDataset(50)
    test_dataset = MultiSourceDataset(20)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)
    
    # Run training
    history = training_engine.train(train_loader, test_loader)
    
    assert len(history['train_loss']) == 2
    assert len(history['val_loss']) == 2
    print("✓ Multi-source training test passed")


def test_custom_preprocessing():
    """Test custom preprocessing functions"""
    print("Testing custom preprocessing...")
    
    # Create custom preprocessing function
    def custom_preprocess(data):
        if isinstance(data, dict):
            # Weighted combination of sources
            return 0.7 * data['x1'] + 0.3 * data['x2']
        return data
    
    config = {
        'epochs': 1,
        'device': 'cpu',
        'preprocess_fn': custom_preprocess
    }
    
    training_engine = TrainingEngine(config)
    
    # Create model for processed features
    model = nn.Linear(3, 1)  # After processing, we have 3 features
    training_engine.set_model(model)
    training_engine.configure_optimizer('adam', lr=0.01)
    training_engine.configure_criterion('mse')
    
    # Test with same multi-source data
    class MultiSourceDataset(torch.utils.data.Dataset):
        def __init__(self, size=20):
            self.size = size
            self.x1 = torch.randn(size, 3)
            self.x2 = torch.randn(size, 3)
            self.targets = torch.randn(size, 1)
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            return {'x1': self.x1[idx], 'x2': self.x2[idx]}, self.targets[idx]
    
    dataset = MultiSourceDataset(20)
    loader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True)
    
    # Run training
    history = training_engine.train(loader)
    
    assert len(history['train_loss']) == 1
    print("✓ Custom preprocessing test passed")


def test_preprocessing_helper_functions():
    """Test the built-in preprocessing helper functions"""
    print("Testing preprocessing helper functions...")
    
    # Test concatenation function
    concat_fn = TrainingEngine.create_concat_preprocess_fn()
    
    # Test with dict data
    data_dict = {'x1': torch.randn(4, 3), 'x2': torch.randn(4, 5)}
    concatenated = concat_fn(data_dict)
    assert concatenated.shape == (4, 8)  # 3 + 5 = 8 features
    
    # Test with single tensor
    single_tensor = torch.randn(4, 10)
    result = concat_fn(single_tensor)
    assert torch.equal(result, single_tensor)
    
    # Test selection function
    select_fn = TrainingEngine.create_select_preprocess_fn('x1')
    selected = select_fn(data_dict)
    assert torch.equal(selected, data_dict['x1'])
    
    print("✓ Preprocessing helper functions test passed")


def test_classification_training():
    """Test classification training with CrossEntropyLoss"""
    print("Testing classification training...")
    
    config = {'epochs': 1, 'device': 'cpu'}
    training_engine = TrainingEngine(config)
    
    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 3)  # 3 classes
    )
    training_engine.set_model(model)
    training_engine.configure_optimizer('adam', lr=0.01)
    training_engine.configure_criterion('cross_entropy')
    
    class ClassificationDataset(torch.utils.data.Dataset):
        def __init__(self, size=30):
            self.size = size
            self.data = torch.randn(size, 4)
            self.targets = torch.randint(0, 3, (size,))
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]
    
    dataset = ClassificationDataset(30)
    loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    
    history = training_engine.train(loader)
    
    assert len(history['train_loss']) == 1
    assert len(history['train_acc']) == 1
    print("✓ Classification training test passed")


def test_model_save_load():
    """Test model saving and loading functionality"""
    print("Testing model save/load...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, 'test_model.pth')
        
        # Create and train a simple model
        config = {'epochs': 1, 'device': 'cpu'}
        training_engine = TrainingEngine(config)
        
        model = nn.Linear(3, 1)
        training_engine.set_model(model)
        training_engine.configure_optimizer('adam', lr=0.01)
        training_engine.configure_criterion('mse')
        
        # Create dummy data
        data = torch.randn(10, 3)
        targets = torch.randn(10, 1)
        dataset = torch.utils.data.TensorDataset(data, targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=5)
        
        # Train and save
        training_engine.train(loader)
        training_engine.save_model(model_path)
        
        assert os.path.exists(model_path)
        
        # Load model into new engine
        new_engine = TrainingEngine(config)
        new_model = nn.Linear(3, 1)
        new_engine.set_model(new_model)
        new_engine.load_model(model_path)
        
        print("✓ Model save/load test passed")


def test_scheduler_functionality():
    """Test learning rate scheduler"""
    print("Testing scheduler functionality...")
    
    config = {'epochs': 2, 'device': 'cpu'}
    training_engine = TrainingEngine(config)
    
    model = nn.Linear(2, 1)
    training_engine.set_model(model)
    training_engine.configure_optimizer('adam', lr=0.1)
    training_engine.configure_criterion('mse')
    training_engine.configure_scheduler('step', step_size=1, gamma=0.5)
    
    # Get initial learning rate
    initial_lr = training_engine.optimizer.param_groups[0]['lr']
    
    # Create dummy data
    data = torch.randn(20, 2)
    targets = torch.randn(20, 1)
    dataset = torch.utils.data.TensorDataset(data, targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=10)
    
    training_engine.train(loader)
    
    # Check learning rate was reduced
    final_lr = training_engine.optimizer.param_groups[0]['lr']
    expected_lr = initial_lr * 0.25  # Applied twice (0.1 -> 0.05 -> 0.025)
    assert abs(final_lr - expected_lr) < 1e-6, f"Expected LR: {expected_lr}, Actual LR: {final_lr}"
    
    print("✓ Scheduler functionality test passed")


def test_error_handling():
    """Test error handling and validation"""
    print("Testing error handling...")
    
    training_engine = TrainingEngine({'epochs': 1})
    
    # Test missing model
    try:
        training_engine.configure_optimizer('adam')
        assert False, "Should raise AttributeError for missing model"
    except (ValueError, AttributeError):
        pass
    
    # Test missing optimizer
    model = nn.Linear(2, 1)
    training_engine.set_model(model)
    try:
        training_engine.train_epoch([])
        assert False, "Should raise ValueError for missing optimizer"
    except ValueError:
        pass
    
    print("✓ Error handling test passed")


def main():
    """Main test function for training engine"""
    print("🚀 Testing Training Engine Module")
    print("=" * 50)
    
    try:
        # Basic functionality tests
        test_basic_training_engine()
        test_model_configuration()
        
        # Training tests
        test_single_source_training()
        test_multi_source_training()
        test_custom_preprocessing()
        
        # Helper functions
        test_preprocessing_helper_functions()
        test_classification_training()
        
        # Advanced features
        test_model_save_load()
        test_scheduler_functionality()
        test_error_handling()
        
        print("\n" + "=" * 50)
        print("🎉 All training engine tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)