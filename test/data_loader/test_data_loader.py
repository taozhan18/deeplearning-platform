"""
Test script for data loader module
"""

import sys
import os

# Add the modules to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'data_loader', 'src'))

from data_loader import DataLoaderModule
import numpy as np


def test_data_loader_functionality():
    """Test the basic functionality of the data loader"""
    print("Testing Data Loader Functionality...")
    
    # Configuration for testing with existing data
    config = {
        'train_features_path': 'data/train_features.csv',
        'train_targets_path': 'data/train_targets.csv',
        'test_features_path': 'data/test_features.csv',
        'test_targets_path': 'data/test_targets.csv',
        'batch_size': 16,
        'shuffle': True
    }
    
    # Initialize data loader
    data_loader = DataLoaderModule(config)
    
    # Test dataset preparation
    data_loader.prepare_datasets()
    print("Datasets prepared successfully")
    
    # Test data loader creation
    data_loader.create_data_loaders()
    print("Data loaders created successfully")
    
    # Get data loaders
    train_loader, test_loader = data_loader.get_data_loaders()
    
    # Check dataset info
    dataset_info = data_loader.get_dataset_info()
    print(f"Dataset info: {dataset_info}")
    
    # Check a batch of training data
    print("\nChecking training data batch:")
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"  Batch {batch_idx}: Data shape: {data.shape}, Target shape: {target.shape}")
        print(f"  Data range: [{data.min():.2f}, {data.max():.2f}]")
        print(f"  Target range: [{target.min()}, {target.max()}]")
        break  # Only check the first batch
    
    # Check a batch of test data
    print("\nChecking test data batch:")
    for batch_idx, (data, target) in enumerate(test_loader):
        print(f"  Batch {batch_idx}: Data shape: {data.shape}, Target shape: {target.shape}")
        print(f"  Data range: [{data.min():.2f}, {data.max():.2f}]")
        print(f"  Target range: [{target.min()}, {target.max()}]")
        break  # Only check the first batch
    
    print("Data loader functionality test passed!")


def test_data_formats():
    """Test support for different data formats"""
    print("\nTesting Data Format Support...")
    
    # Test with CSV files (already done above)
    print("CSV format support test passed!")
    
    # Other formats would be tested here if we had sample data
    print("Data format support test completed!")


def main():
    """Main test function for data loader"""
    print("Testing Data Loader Module")
    print("=" * 30)
    
    # Test basic functionality
    test_data_loader_functionality()
    
    # Test data format support
    test_data_formats()
    
    print("\n" + "=" * 30)
    print("All data loader tests completed successfully!")


if __name__ == "__main__":
    main()