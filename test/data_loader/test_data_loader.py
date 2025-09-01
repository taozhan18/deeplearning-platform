"""
Comprehensive test suite for data loader module
"""

import os
import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
import torch
from unittest.mock import patch, MagicMock

# Add the src path to sys.path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../data_loader/src'))

from data_loader import (
    BaseDataset, 
    MultiSourceDataset, 
    DataLoaderModule, 
    DataNormalizer
)


class TestDataNormalizer:
    """Test cases for DataNormalizer class"""
    
    def test_init_standard_scaler(self):
        """Test initialization with standard scaler"""
        normalizer = DataNormalizer(method='standard')
        assert normalizer.method == 'standard'
        
    def test_init_minmax_scaler(self):
        """Test initialization with minmax scaler"""
        normalizer = DataNormalizer(method='minmax')
        assert normalizer.method == 'minmax'
        
    def test_init_robust_scaler(self):
        """Test initialization with robust scaler"""
        normalizer = DataNormalizer(method='robust')
        assert normalizer.method == 'robust'
        
    def test_init_invalid_method(self):
        """Test initialization with invalid method raises error"""
        with pytest.raises(ValueError):
            DataNormalizer(method='invalid')
    
    def test_fit_transform_single_source(self):
        """Test fit_transform with single data source"""
        normalizer = DataNormalizer(method='standard')
        data = {'features': np.array([[1, 2], [3, 4], [5, 6]])}
        
        normalized = normalizer.fit_transform(data)
        
        assert 'features' in normalized
        assert normalized['features'].shape == (3, 2)
        assert np.allclose(normalized['features'].mean(axis=0), 0, atol=1e-10)
        
    def test_fit_transform_multi_source(self):
        """Test fit_transform with multiple data sources"""
        normalizer = DataNormalizer(method='standard')
        data = {
            'x1': np.array([[1, 2], [3, 4]]),
            'x2': np.array([[5], [6]])
        }
        
        normalized = normalizer.fit_transform(data)
        
        assert len(normalized) == 2
        assert 'x1' in normalized and 'x2' in normalized
        assert normalized['x1'].shape == (2, 2)
        assert normalized['x2'].shape == (2, 1)
        
    def test_fit_transform_1d_data(self):
        """Test fit_transform with 1D data"""
        normalizer = DataNormalizer(method='standard')
        data = {'features': np.array([1, 2, 3, 4, 5])}
        
        normalized = normalizer.fit_transform(data)
        
        assert normalized['features'].shape == (5, 1)
        assert np.allclose(normalized['features'].mean(), 0, atol=1e-10)
        
    def test_transform_with_fitted_scaler(self):
        """Test transform using fitted scaler"""
        normalizer = DataNormalizer(method='standard')
        
        train_data = {'features': np.array([[1, 2], [3, 4]])}
        normalizer.fit_transform(train_data)
        
        test_data = {'features': np.array([[5, 6], [7, 8]])}
        normalized = normalizer.transform(test_data)
        
        assert 'features' in normalized
        assert normalized['features'].shape == (2, 2)
        
    def test_transform_without_fitted_scaler(self):
        """Test transform without fitted scaler raises error"""
        normalizer = DataNormalizer(method='standard')
        test_data = {'features': np.array([[1, 2], [3, 4]])}
        
        with pytest.raises(ValueError):
            normalizer.transform(test_data)


class TestBaseDataset:
    """Test cases for BaseDataset class"""
    
    def test_init_without_targets(self):
        """Test initialization without targets"""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        dataset = BaseDataset(data)
        
        assert len(dataset) == 3
        assert np.array_equal(dataset.data, data)
        assert dataset.targets is None
        
    def test_init_with_targets(self):
        """Test initialization with targets"""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        targets = np.array([0, 1, 0])
        dataset = BaseDataset(data, targets)
        
        assert len(dataset) == 3
        assert np.array_equal(dataset.data, data)
        assert np.array_equal(dataset.targets, targets)


class TestMultiSourceDataset:
    """Test cases for MultiSourceDataset class"""
    
    def test_init_single_source(self):
        """Test initialization with single data source"""
        data_dict = {'x1': np.array([[1, 2], [3, 4], [5, 6]])}
        dataset = MultiSourceDataset(data_dict)
        
        assert len(dataset) == 3
        assert 'x1' in dataset.data_dict
        
    def test_init_multi_source(self):
        """Test initialization with multiple data sources"""
        data_dict = {
            'x1': np.array([[1, 2], [3, 4]]),
            'x2': np.array([[5], [6]]),
            'x3': np.array([[7, 8, 9], [10, 11, 12]])
        }
        dataset = MultiSourceDataset(data_dict)
        
        assert len(dataset) == 2
        assert len(dataset.data_dict) == 3
        
    def test_init_with_targets(self):
        """Test initialization with targets"""
        data_dict = {'x1': np.array([[1, 2], [3, 4]])}
        targets = np.array([0, 1])
        dataset = MultiSourceDataset(data_dict, targets)
        
        assert len(dataset) == 2
        assert dataset.targets is not None
        assert np.array_equal(dataset.targets, targets)
        
    def test_init_empty_data(self):
        """Test initialization with empty data raises error"""
        with pytest.raises(ValueError):
            MultiSourceDataset({})
            
    def test_init_inconsistent_shapes(self):
        """Test initialization with inconsistent shapes raises error"""
        data_dict = {
            'x1': np.array([[1, 2], [3, 4]]),
            'x2': np.array([[5]])  # Different number of samples
        }
        with pytest.raises(ValueError):
            MultiSourceDataset(data_dict)
            
    def test_init_inconsistent_targets(self):
        """Test initialization with inconsistent targets raises error"""
        data_dict = {'x1': np.array([[1, 2], [3, 4]])}
        targets = np.array([0, 1, 2])  # Different number of samples
        with pytest.raises(ValueError):
            MultiSourceDataset(data_dict, targets)


class TestDataLoaderModule:
    """Test cases for DataLoaderModule class"""
    
    def setup_method(self):
        """Setup temporary directory for test files"""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Cleanup temporary directory"""
        shutil.rmtree(self.temp_dir)
        
    def create_test_csv(self, filename, data, columns=None):
        """Helper to create test CSV file"""
        filepath = os.path.join(self.temp_dir, filename)
        if columns is None:
            columns = [f'col_{i}' for i in range(data.shape[1])]
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(filepath, index=False)
        return filepath
        
    def create_test_npy(self, filename, data):
        """Helper to create test NPY file"""
        filepath = os.path.join(self.temp_dir, filename)
        np.save(filepath, data)
        return filepath
        
    def test_init(self):
        """Test initialization"""
        config = {}
        loader = DataLoaderModule(config)
        assert loader.config == config
        assert loader.train_dataset is None
        assert loader.test_dataset is None
        
    def test_load_data_csv(self):
        """Test loading CSV data"""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        filepath = self.create_test_csv('test.csv', data)
        
        loader = DataLoaderModule({})
        loaded = loader.load_data(filepath, "features")
        
        assert loaded.shape == (3, 2)
        np.testing.assert_array_equal(loaded, data)
        
    def test_load_data_npy(self):
        """Test loading NPY data"""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        filepath = self.create_test_npy('test.npy', data)
        
        loader = DataLoaderModule({})
        loaded = loader.load_data(filepath, "features")
        
        assert loaded.shape == (3, 2)
        np.testing.assert_array_equal(loaded, data)
        
    def test_load_data_targets(self):
        """Test loading targets"""
        data = np.array([[1], [2], [3]])
        filepath = self.create_test_csv('targets.csv', data, ['target'])
        
        loader = DataLoaderModule({})
        loaded = loader.load_data(filepath, "targets")
        
        assert loaded.shape == (3,)
        np.testing.assert_array_equal(loaded, [1, 2, 3])
        
    def test_prepare_standard_datasets(self):
        """Test preparing standard datasets"""
        train_features = np.array([[1, 2], [3, 4], [5, 6]])
        train_targets = np.array([0, 1, 0])
        
        train_features_path = self.create_test_csv('train_features.csv', train_features)
        train_targets_path = self.create_test_csv('train_targets.csv', train_targets.reshape(-1, 1), ['target'])
        
        config = {
            'train_features_path': train_features_path,
            'train_targets_path': train_targets_path
        }
        
        loader = DataLoaderModule(config)
        loader.prepare_standard_datasets()
        
        assert loader.train_dataset is not None
        assert len(loader.train_dataset) == 3
        
    def test_prepare_multi_source_datasets(self):
        """Test preparing multi-source datasets"""
        train_x1 = np.array([[1, 2], [3, 4]])
        train_x2 = np.array([[5], [6]])
        train_y = np.array([0, 1])
        
        train_x1_path = self.create_test_csv('train_x1.csv', train_x1)
        train_x2_path = self.create_test_csv('train_x2.csv', train_x2)
        train_y_path = self.create_test_csv('train_y.csv', train_y.reshape(-1, 1), ['target'])
        
        config = {
            'train_features_paths': {
                'x1': train_x1_path,
                'x2': train_x2_path
            },
            'train_targets_path': train_y_path
        }
        
        loader = DataLoaderModule(config)
        loader.prepare_multi_source_datasets()
        
        assert loader.train_dataset is not None
        assert len(loader.train_dataset) == 2
        assert hasattr(loader.train_dataset, 'data_dict')
        
    def test_prepare_datasets_auto_detection(self):
        """Test automatic detection of dataset type"""
        train_x1 = np.array([[1, 2], [3, 4]])
        train_x2 = np.array([[5], [6]])
        train_y = np.array([0, 1])
        
        train_x1_path = self.create_test_csv('train_x1.csv', train_x1)
        train_x2_path = self.create_test_csv('train_x2.csv', train_x2)
        train_y_path = self.create_test_csv('train_y.csv', train_y.reshape(-1, 1), ['target'])
        
        config = {
            'train_features_paths': {
                'x1': train_x1_path,
                'x2': train_x2_path
            },
            'train_targets_path': train_y_path
        }
        
        loader = DataLoaderModule(config)
        loader.prepare_datasets()
        
        assert hasattr(loader.train_dataset, 'data_dict')
        assert 'x1' in loader.train_dataset.data_dict
        assert 'x2' in loader.train_dataset.data_dict
        
    def test_normalize_data(self):
        """Test data normalization"""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        filepath = self.create_test_csv('test.csv', data)
        
        config = {
            'train_features_path': filepath,
            'normalize': True,
            'normalization_method': 'standard'
        }
        
        loader = DataLoaderModule(config)
        loader.prepare_standard_datasets()
        
        assert loader.train_dataset is not None
        assert loader.normalizer is not None
        
    def test_get_dataset_info(self):
        """Test getting dataset information"""
        train_features = np.array([[1, 2], [3, 4], [5, 6]])
        train_targets = np.array([0, 1, 0])
        
        train_features_path = self.create_test_csv('train_features.csv', train_features)
        train_targets_path = self.create_test_csv('train_targets.csv', train_targets.reshape(-1, 1), ['target'])
        
        config = {
            'train_features_path': train_features_path,
            'train_targets_path': train_targets_path,
            'normalize': True
        }
        
        loader = DataLoaderModule(config)
        loader.prepare_datasets()
        
        info = loader.get_dataset_info()
        
        assert 'train_samples' in info
        assert 'train_data_shape' in info
        assert 'train_targets_shape' in info
        assert info['train_samples'] == 3
        assert info['normalized'] is True
        
    def test_get_dataset_info_multi_source(self):
        """Test getting dataset info for multi-source"""
        train_x1 = np.array([[1, 2], [3, 4]])
        train_x2 = np.array([[5], [6]])
        train_y = np.array([0, 1])
        
        train_x1_path = self.create_test_csv('train_x1.csv', train_x1)
        train_x2_path = self.create_test_csv('train_x2.csv', train_x2)
        train_y_path = self.create_test_csv('train_y.csv', train_y.reshape(-1, 1), ['target'])
        
        config = {
            'train_features_paths': {
                'x1': train_x1_path,
                'x2': train_x2_path
            },
            'train_targets_path': train_y_path
        }
        
        loader = DataLoaderModule(config)
        loader.prepare_datasets()
        
        info = loader.get_dataset_info()
        
        assert 'train_data_sources' in info
        assert 'train_data_shapes' in info
        assert 'x1' in info['train_data_shapes']
        assert 'x2' in info['train_data_shapes']
        assert info['multi_source'] is True


# Test runner
if __name__ == '__main__':
    pytest.main([__file__, '-v'])