"""
Data Loader Module for Low-Code Deep Learning Platform
"""

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Tuple, Optional
import torch


class BaseDataset(Dataset):
    """
    Base dataset class that handles data loading for common formats
    """
    
    def __init__(self, data, targets=None):
        """
        Initialize the dataset
        
        Args:
            data: Input data (numpy array, pandas dataframe, etc.)
            targets: Target values (optional)
        """
        self.data = data
        self.targets = targets
    
    def __len__(self):
        """Return the size of the dataset"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single data sample
        
        Args:
            idx: Index of the sample
            
        Returns:
            A tuple of (data, target) if targets are provided, otherwise just data
        """
        sample = self.data[idx]
        if self.targets is not None:
            target = self.targets[idx]
            # For classification tasks, targets should be Long type
            # For regression tasks, targets should be Float type
            # We'll use Float by default and let the training engine handle conversion if needed
            return torch.tensor(sample, dtype=torch.float32), torch.tensor(target, dtype=torch.long if target.dtype in [np.int32, np.int64] else torch.float32)
        return torch.tensor(sample, dtype=torch.float32)


class FNO1DDataset(Dataset):
    """
    Dataset class specifically for FNO1D data with dimensions (batch, channel, height)
    """
    
    def __init__(self, data, targets=None):
        """
        Initialize the FNO1D dataset
        
        Args:
            data: Input data with shape (batch, channel, height)
            targets: Target values with shape (batch, channel, height) (optional)
        """
        self.data = data
        self.targets = targets
    
    def __len__(self):
        """Return the size of the dataset"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single data sample
        
        Args:
            idx: Index of the sample
            
        Returns:
            A tuple of (data, target) if targets are provided, otherwise just data
        """
        sample = self.data[idx]
        if self.targets is not None:
            target = self.targets[idx]
            # For FNO, targets are typically regression values, so use Float
            return torch.tensor(sample, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
        return torch.tensor(sample, dtype=torch.float32)


class DataLoaderModule:
    """
    Data Loader Module for handling various data formats and preprocessing
    """
    
    SUPPORTED_FORMATS = ['csv', 'json', 'npy', 'npz']
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data loader module
        
        Args:
            config: Configuration dictionary containing data paths and parameters
        """
        self.config = config
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
    
    def load_data(self, file_path: str, data_type: str = 'features') -> np.ndarray:
        """
        Load data from various formats
        
        Args:
            file_path: Path to the data file
            data_type: Type of data ('features' or 'targets')
            
        Returns:
            Loaded data as numpy array
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower().lstrip('.')
        
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats: {self.SUPPORTED_FORMATS}")
        
        if ext == 'csv':
            data = pd.read_csv(file_path)
            if data_type == 'targets':
                # For targets, we usually want a 1D array
                return data.values.flatten() if data.shape[1] == 1 else data.values
            return data.values
        
        elif ext == 'json':
            data = pd.read_json(file_path)
            if data_type == 'targets':
                return data.values.flatten() if data.shape[1] == 1 else data.values
            return data.values
        
        elif ext == 'npy':
            return np.load(file_path)
        
        elif ext == 'npz':
            loaded = np.load(file_path)
            # Assume the first array in the npz file
            return loaded[loaded.files[0]]
    
    def prepare_datasets(self):
        """
        Prepare training and testing datasets
        """
        # Check if we're dealing with FNO1D data
        fno1d_data = self.config.get('fno1d_data', False)
        
        if fno1d_data:
            self.prepare_fno1d_datasets()
        else:
            self.prepare_standard_datasets()
    
    def prepare_standard_datasets(self):
        """
        Prepare standard datasets (the original method)
        """
        # Load training data
        train_features_path = self.config.get('train_features_path')
        train_targets_path = self.config.get('train_targets_path')
        
        if train_features_path:
            train_features = self.load_data(train_features_path, 'features')
            train_targets = None
            if train_targets_path:
                train_targets = self.load_data(train_targets_path, 'targets')
            
            self.train_dataset = BaseDataset(train_features, train_targets)
        
        # Load testing data
        test_features_path = self.config.get('test_features_path')
        test_targets_path = self.config.get('test_targets_path')
        
        if test_features_path:
            test_features = self.load_data(test_features_path, 'features')
            test_targets = None
            if test_targets_path:
                test_targets = self.load_data(test_targets_path, 'targets')
            
            self.test_dataset = BaseDataset(test_features, test_targets)
    
    def prepare_fno1d_datasets(self):
        """
        Prepare FNO1D datasets with dimensions (batch, channel, height)
        """
        # Load training data
        train_features_path = self.config.get('train_features_path')
        train_targets_path = self.config.get('train_targets_path')
        
        if train_features_path and train_targets_path:
            train_features = self.load_data(train_features_path, 'features')
            train_targets = self.load_data(train_targets_path, 'targets')
            
            # Ensure the data has the correct dimensions
            if len(train_features.shape) != 3:
                raise ValueError(f"Train features must have 3 dimensions (batch, channel, height), got {train_features.shape}")
            
            if len(train_targets.shape) != 3:
                raise ValueError(f"Train targets must have 3 dimensions (batch, channel, height), got {train_targets.shape}")
            
            self.train_dataset = FNO1DDataset(train_features, train_targets)
        
        # Load testing data
        test_features_path = self.config.get('test_features_path')
        test_targets_path = self.config.get('test_targets_path')
        
        if test_features_path and test_targets_path:
            test_features = self.load_data(test_features_path, 'features')
            test_targets = self.load_data(test_targets_path, 'targets')
            
            # Ensure the data has the correct dimensions
            if len(test_features.shape) != 3:
                raise ValueError(f"Test features must have 3 dimensions (batch, channel, height), got {test_features.shape}")
            
            if len(test_targets.shape) != 3:
                raise ValueError(f"Test targets must have 3 dimensions (batch, channel, height), got {test_targets.shape}")
            
            self.test_dataset = FNO1DDataset(test_features, test_targets)
    
    def create_data_loaders(self):
        """
        Create PyTorch data loaders for training and testing
        """
        batch_size = self.config.get('batch_size', 32)
        shuffle = self.config.get('shuffle', True)
        
        if self.train_dataset:
            self.train_loader = DataLoader(
                self.train_dataset, 
                batch_size=batch_size, 
                shuffle=shuffle
            )
        
        if self.test_dataset:
            self.test_loader = DataLoader(
                self.test_dataset, 
                batch_size=batch_size, 
                shuffle=False
            )
    
    def get_data_loaders(self) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        """
        Get the prepared data loaders
        
        Returns:
            Tuple of (train_loader, test_loader)
        """
        return self.train_loader, self.test_loader
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded datasets
        
        Returns:
            Dictionary containing dataset information
        """
        info = {}
        if self.train_dataset:
            info['train_samples'] = len(self.train_dataset)
            info['train_data_shape'] = self.train_dataset.data.shape
            if self.train_dataset.targets is not None:
                info['train_targets_shape'] = self.train_dataset.targets.shape
        if self.test_dataset:
            info['test_samples'] = len(self.test_dataset)
            info['test_data_shape'] = self.test_dataset.data.shape
            if self.test_dataset.targets is not None:
                info['test_targets_shape'] = self.test_dataset.targets.shape
        return info