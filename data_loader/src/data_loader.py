"""
Data Loader Module for Low-Code Deep Learning Platform
"""

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Tuple, Optional, List, Union
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class DataNormalizer:
    """
    Data normalization utilities supporting multiple normalization methods
    """
    
    SCALERS = {
        'standard': StandardScaler,
        'minmax': MinMaxScaler,
        'robust': RobustScaler
    }
    
    def __init__(self, method: str = 'standard'):
        """
        Initialize normalizer
        
        Args:
            method: Normalization method ('standard', 'minmax', 'robust')
        """
        if method not in self.SCALERS:
            raise ValueError(f"Unsupported normalization method: {method}")
        self.method = method
        self.scalers = {}
        
    def fit_transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Fit and transform multiple data sources
        
        Args:
            data: Dictionary mapping data source names to numpy arrays
            
        Returns:
            Dictionary with normalized data
        """
        normalized_data = {}
        
        for name, array in data.items():
            was_1d = array.ndim == 1
            
            if was_1d:
                array = array.reshape(-1, 1)
            
            scaler = self.SCALERS[self.method]()
            normalized_array = scaler.fit_transform(array)
            
            # Keep 2D shape for consistency
            normalized_data[name] = normalized_array
            self.scalers[name] = scaler
            
        return normalized_data
    
    def transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Transform data using fitted scalers
        
        Args:
            data: Dictionary mapping data source names to numpy arrays
            
        Returns:
            Dictionary with normalized data
        """
        normalized_data = {}
        
        for name, array in data.items():
            if name not in self.scalers:
                raise ValueError(f"No fitted scaler found for {name}")
                
            was_1d = array.ndim == 1
            
            if was_1d:
                array = array.reshape(-1, 1)
            
            normalized_array = self.scalers[name].transform(array)
            
            # Keep 2D shape for consistency
            normalized_data[name] = normalized_array
            
        return normalized_data


class MultiSourceDataset(Dataset):
    """
    Dataset class supporting multiple input data sources and optional targets
    """

    def __init__(self, data_dict: Dict[str, np.ndarray], targets: Optional[np.ndarray] = None):
        """
        Initialize the dataset with multiple data sources

        Args:
            data_dict: Dictionary mapping feature names to numpy arrays
            targets: Target values (optional)
        """
        self.data_dict = data_dict
        self.targets = targets
        
        # Validate data consistency
        self._validate_data_consistency()
        
    def _validate_data_consistency(self):
        """Validate that all data sources have consistent shapes"""
        if not self.data_dict:
            raise ValueError("No data provided")
            
        # Get sample count from first data source
        sample_count = None
        for name, array in self.data_dict.items():
            if sample_count is None:
                sample_count = len(array)
            elif len(array) != sample_count:
                raise ValueError(
                    f"Inconsistent sample count: {name} has {len(array)} samples, "
                    f"expected {sample_count}"
                )
        
        # Validate targets if provided
        if self.targets is not None and len(self.targets) != sample_count:
            raise ValueError(
                f"Targets have {len(self.targets)} samples, expected {sample_count}"
            )

    def __len__(self):
        """Return the size of the dataset"""
        return len(next(iter(self.data_dict.values())))

    def __getitem__(self, idx):
        """
        Get a single data sample

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (data_dict, target) if targets are provided, otherwise just data_dict
        """
        sample_dict = {name: torch.tensor(array[idx], dtype=torch.float32) 
                      for name, array in self.data_dict.items()}
        
        if self.targets is not None:
            target = self.targets[idx]
            target_tensor = torch.tensor(
                target, 
                dtype=torch.long if target.dtype in [np.int32, np.int64] else torch.float32
            )
            return sample_dict, target_tensor
        
        return sample_dict


class GraphDataset(Dataset):
    """
    Dataset class specifically for graph data (MeshGraphNet)
    """

    def __init__(self, node_features: np.ndarray, edge_index: np.ndarray, 
                 edge_features: np.ndarray, targets: Optional[np.ndarray] = None):
        """
        Initialize the graph dataset

        Args:
            node_features: Node features array of shape (num_graphs, num_nodes, feature_dim)
            edge_index: Edge connectivity array of shape (num_graphs, 2, num_edges)
            edge_features: Edge features array of shape (num_graphs, num_edges, edge_dim)
            targets: Target values array of shape (num_graphs, num_nodes, output_dim)
        """
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_features = edge_features
        self.targets = targets
        
        # Validate data consistency
        self._validate_data_consistency()
        
    def _validate_data_consistency(self):
        """Validate that all data sources have consistent shapes"""
        num_graphs = len(self.node_features)
        
        if len(self.edge_index) != num_graphs:
            raise ValueError(
                f"Edge index has {len(self.edge_index)} samples, expected {num_graphs}"
            )
        if len(self.edge_features) != num_graphs:
            raise ValueError(
                f"Edge features have {len(self.edge_features)} samples, expected {num_graphs}"
            )
        if self.targets is not None and len(self.targets) != num_graphs:
            raise ValueError(
                f"Targets have {len(self.targets)} samples, expected {num_graphs}"
            )

    def __len__(self):
        """Return the size of the dataset"""
        return len(self.node_features)

    def __getitem__(self, idx):
        """
        Get a single graph sample

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (graph_data, target) if targets are provided, otherwise just graph_data
        """
        graph_data = {
            "node_features": torch.tensor(self.node_features[idx], dtype=torch.float32),
            "edge_index": torch.tensor(self.edge_index[idx], dtype=torch.long),
            "edge_features": torch.tensor(self.edge_features[idx], dtype=torch.float32)
        }
        
        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return graph_data, target
        
        return graph_data


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
            return torch.tensor(sample, dtype=torch.float32), torch.tensor(
                target, dtype=torch.long if target.dtype in [np.int32, np.int64] else torch.float32
            )
        return torch.tensor(sample, dtype=torch.float32)


class DataLoaderModule:
    """
    Data Loader Module for handling various data formats and preprocessing
    
    Configuration Parameters:
    -------------------------
    The config dictionary should contain the following parameters:
    
    Required Parameters:
    - train_features_path (str): Path to training features data file
    - train_targets_path (str): Path to training targets data file
    - test_features_path (str): Path to testing features data file
    - test_targets_path (str): Path to testing targets data file
    
    Graph Data Configuration (for MeshGraphNet):
    - train_edge_index_path (str): Path to training edge indices file
    - train_edge_features_path (str): Path to training edge features file
    - test_edge_index_path (str): Path to testing edge indices file
    - test_edge_features_path (str): Path to testing edge features file
    
    Optional Parameters with Defaults:
    - batch_size (int): Batch size for data loading, default: 32
    - shuffle (bool): Whether to shuffle training data, default: True
    - normalize (bool): Whether to normalize the data, default: False
    - normalization_method (str): Method for normalization ('standard', 'minmax', 'robust'), default: 'standard'
    - normalize_targets (bool): Whether to normalize target values, default: False
    - memory_map (bool): Whether to use memory mapping for large files, default: False
    - train_features_paths (dict): Dictionary of paths for multi-source training features
    - test_features_paths (dict): Dictionary of paths for multi-source testing features
    - data_type (str): Type of data ('standard', 'graph'), default: 'standard'
    
    Example Configuration:
    ----------------------
    Single source data:
    config = {
        "train_features_path": "data/train_features.npy",
        "train_targets_path": "data/train_targets.npy", 
        "test_features_path": "data/test_features.npy",
        "test_targets_path": "data/test_targets.npy",
        "batch_size": 64,
        "shuffle": True,
        "normalize": True,
        "normalization_method": "standard"
    }
    
    Multi-source data:
    config = {
        "train_features_paths": {
            "pressure": "data/pressure_train.npy",
            "velocity": "data/velocity_train.npy"
        },
        "train_targets_path": "data/train_targets.npy",
        "test_features_paths": {
            "pressure": "data/pressure_test.npy", 
            "velocity": "data/velocity_test.npy"
        },
        "test_targets_path": "data/test_targets.npy",
        "batch_size": 32,
        "shuffle": True
    }
    
    Graph data (MeshGraphNet):
    config = {
        "data_type": "graph",
        "train_features_path": "data/node_features_train.npy",
        "train_targets_path": "data/node_targets_train.npy",
        "train_edge_index_path": "data/edge_index_train.npy",
        "train_edge_features_path": "data/edge_features_train.npy",
        "test_features_path": "data/node_features_test.npy",
        "test_targets_path": "data/node_targets_test.npy",
        "test_edge_index_path": "data/edge_index_test.npy",
        "test_edge_features_path": "data/edge_features_test.npy",
        "batch_size": 16,
        "shuffle": True
    }
    """

    SUPPORTED_FORMATS = ["csv", "json", "npy", "npz"]

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data loader module

        Args:
            config: Configuration dictionary containing data paths and parameters.
                   See class docstring for detailed configuration parameters.
        """
        self.config = config
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
        self.normalizer = None
        self.feature_names = []

    def load_data(self, file_path: str, data_type: str = "features") -> np.ndarray:
        """
        Load data from various formats

        Args:
            file_path: Path to the data file
            data_type: Type of data ('features' or 'targets')

        Returns:
            Loaded data as numpy array
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower().lstrip(".")

        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats: {self.SUPPORTED_FORMATS}")

        if ext == "csv":
            data = pd.read_csv(file_path)
            if data_type == "targets":
                # For targets, we usually want a 1D array
                return data.values.flatten() if data.shape[1] == 1 else data.values
            return data.values

        elif ext == "json":
            data = pd.read_json(file_path)
            if data_type == "targets":
                return data.values.flatten() if data.shape[1] == 1 else data.values
            return data.values

        elif ext == "npy":
            return np.load(file_path)

        elif ext == "npz":
            loaded = np.load(file_path)
            # Handle both 'data' key and first array cases
            if 'data' in loaded:
                return loaded['data']
            else:
                # Assume the first array in the npz file
                return loaded[loaded.files[0]]

    def prepare_datasets(self):
        """
        Prepare training and testing datasets
        """
        # Check data type
        data_type = self.config.get("data_type", "standard")
        
        if data_type == "graph":
            self.prepare_graph_datasets()
        else:
            # Check if we have multiple input data sources
            train_features_paths = self.config.get("train_features_paths")
            if train_features_paths and isinstance(train_features_paths, dict):
                self.prepare_multi_source_datasets()
            else:
                self.prepare_standard_datasets()

    def prepare_standard_datasets(self):
        """
        Prepare standard datasets with single data source
        """
        # Load training data
        train_features_path = self.config.get("train_features_path")
        train_targets_path = self.config.get("train_targets_path")

        if train_features_path:
            train_features = self.load_data(train_features_path, "features")
            train_targets = None
            if train_targets_path:
                train_targets = self.load_data(train_targets_path, "targets")

            # Apply normalization if specified
            if self.config.get("normalize", False):
                train_features = self.normalize_data(train_features, "train_features")
                if train_targets is not None and self.config.get("normalize_targets", False):
                    train_targets = self.normalize_data(train_targets, "train_targets")

            self.train_dataset = BaseDataset(train_features, train_targets)

        # Load testing data
        test_features_path = self.config.get("test_features_path")
        test_targets_path = self.config.get("test_targets_path")

        if test_features_path:
            test_features = self.load_data(test_features_path, "features")
            test_targets = None
            if test_targets_path:
                test_targets = self.load_data(test_targets_path, "targets")

            # Apply normalization if specified
            if self.config.get("normalize", False):
                test_features = self.normalize_data(test_features, "test_features", is_test=True)
                if test_targets is not None and self.config.get("normalize_targets", False):
                    test_targets = self.normalize_data(test_targets, "test_targets", is_test=True)

            self.test_dataset = BaseDataset(test_features, test_targets)

    def prepare_multi_source_datasets(self):
        """
        Prepare datasets with multiple data sources (x1, x2, ..., xn)
        """
        # Get normalization settings
        normalize = self.config.get("normalize", False)
        normalization_method = self.config.get("normalization_method", "standard")
        
        if normalize:
            self.normalizer = DataNormalizer(method=normalization_method)

        # Load training data from multiple sources
        train_data_dict = {}
        train_targets = None
        
        # Load multiple feature sources
        train_features_paths = self.config.get("train_features_paths", {})
        if isinstance(train_features_paths, dict):
            for feature_name, path in train_features_paths.items():
                if path:
                    train_data_dict[feature_name] = self.load_data(path, "features")
        else:
            # Fallback to single path
            train_features_path = self.config.get("train_features_path")
            if train_features_path:
                train_data_dict["features"] = self.load_data(train_features_path, "features")

        # Load targets
        train_targets_path = self.config.get("train_targets_path")
        if train_targets_path:
            train_targets = self.load_data(train_targets_path, "targets")

        # Apply normalization
        if normalize and train_data_dict:
            train_data_dict = self.normalizer.fit_transform(train_data_dict)
            if train_targets is not None and self.config.get("normalize_targets", False):
                # Normalize targets using a single scaler
                target_scaler = DataNormalizer(method=normalization_method)
                train_targets = target_scaler.fit_transform({"targets": train_targets})["targets"]
                self.target_normalizer = target_scaler

        # Create training dataset
        if train_data_dict:
            self.train_dataset = MultiSourceDataset(train_data_dict, train_targets)

        # Load testing data from multiple sources
        test_data_dict = {}
        test_targets = None
        
        # Load multiple feature sources
        test_features_paths = self.config.get("test_features_paths", {})
        if isinstance(test_features_paths, dict):
            for feature_name, path in test_features_paths.items():
                if path:
                    test_data_dict[feature_name] = self.load_data(path, "features")
        else:
            # Fallback to single path
            test_features_path = self.config.get("test_features_path")
            if test_features_path:
                test_data_dict["features"] = self.load_data(test_features_path, "features")

        # Load targets
        test_targets_path = self.config.get("test_targets_path")
        if test_targets_path:
            test_targets = self.load_data(test_targets_path, "targets")

        # Apply normalization using fitted scalers
        if normalize and test_data_dict and self.normalizer:
            test_data_dict = self.normalizer.transform(test_data_dict)
            if test_targets is not None and self.config.get("normalize_targets", False):
                test_targets = self.target_normalizer.transform({"targets": test_targets})["targets"]

        # Create testing dataset
        if test_data_dict:
            self.test_dataset = MultiSourceDataset(test_data_dict, test_targets)

    def create_data_loaders(self):
        """
        Create PyTorch data loaders for training and testing
        """
        batch_size = self.config.get("batch_size", 32)
        shuffle = self.config.get("shuffle", True)
        data_type = self.config.get("data_type", "standard")

        if data_type == "graph":
            # For graph data, use batch_size=1 to handle individual graphs
            # Each graph is already a batch of nodes/edges
            if self.train_dataset:
                self.train_loader = DataLoader(
                    self.train_dataset, 
                    batch_size=1,  # Process one graph at a time
                    shuffle=shuffle
                )

            if self.test_dataset:
                self.test_loader = DataLoader(
                    self.test_dataset, 
                    batch_size=1,  # Process one graph at a time
                    shuffle=False
                )
        else:
            # Standard batching for non-graph data
            if self.train_dataset:
                self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle)

            if self.test_dataset:
                self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def get_data_loaders(self) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        """
        Get the prepared data loaders

        Returns:
            Tuple of (train_loader, test_loader)
        """
        return self.train_loader, self.test_loader

    def normalize_data(self, data: np.ndarray, name: str, is_test: bool = False) -> np.ndarray:
        """
        Normalize single data array
        
        Args:
            data: Input data array
            name: Name identifier for the data
            is_test: Whether this is test data (use fitted scaler)
            
        Returns:
            Normalized data array
        """
        if self.normalizer is None:
            method = self.config.get("normalization_method", "standard")
            self.normalizer = DataNormalizer(method=method)
        
        if is_test:
            # For test data, use the corresponding training scaler
            # This ensures consistent normalization
            scaler_name = "train_features" if "features" in name else name
            if scaler_name in self.normalizer.scalers:
                return self.normalizer.transform({scaler_name: data})[scaler_name]
            else:
                # Fallback to fitting new scaler if training scaler not found
                return self.normalizer.fit_transform({name: data})[name]
        else:
            return self.normalizer.fit_transform({name: data})[name]

    def prepare_graph_datasets(self):
        """
        Prepare datasets for graph data (MeshGraphNet)
        """
        # Load training data
        node_features_path = self.config.get("train_features_path")
        targets_path = self.config.get("train_targets_path")
        edge_index_path = self.config.get("train_edge_index_path")
        edge_features_path = self.config.get("train_edge_features_path")
        
        if node_features_path and targets_path and edge_index_path and edge_features_path:
            node_features = self.load_data(node_features_path, "features")
            targets = self.load_data(targets_path, "targets")
            edge_index = self.load_data(edge_index_path, "features")
            edge_features = self.load_data(edge_features_path, "features")
            
            # Apply normalization if specified
            if self.config.get("normalize", False):
                node_features = self.normalize_data(node_features, "train_node_features")
                if self.config.get("normalize_targets", False):
                    targets = self.normalize_data(targets, "train_targets")
                edge_features = self.normalize_data(edge_features, "train_edge_features")
            
            # Create graph dataset
            self.train_dataset = GraphDataset(node_features, edge_index, edge_features, targets)

        # Load testing data
        test_node_features_path = self.config.get("test_features_path")
        test_targets_path = self.config.get("test_targets_path")
        test_edge_index_path = self.config.get("test_edge_index_path")
        test_edge_features_path = self.config.get("test_edge_features_path")
        
        if test_node_features_path and test_targets_path and test_edge_index_path and test_edge_features_path:
            test_node_features = self.load_data(test_node_features_path, "features")
            test_targets = self.load_data(test_targets_path, "targets")
            test_edge_index = self.load_data(test_edge_index_path, "features")
            test_edge_features = self.load_data(test_edge_features_path, "features")
            
            # Apply normalization using fitted scalers
            if self.config.get("normalize", False):
                test_node_features = self.normalize_data(test_node_features, "test_node_features", is_test=True)
                if self.config.get("normalize_targets", False):
                    test_targets = self.normalize_data(test_targets, "test_targets", is_test=True)
                test_edge_features = self.normalize_data(test_edge_features, "test_edge_features", is_test=True)
            
            # Create graph dataset
            self.test_dataset = GraphDataset(test_node_features, test_edge_index, test_edge_features, test_targets)

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded datasets

        Returns:
            Dictionary containing dataset information
        """
        # Check data type and dataset type
        data_type = self.config.get("data_type", "standard")
        is_multi_source = (
            hasattr(self.train_dataset, 'data_dict') or 
            hasattr(self.test_dataset, 'data_dict')
        )
        is_graph = data_type == "graph"
        
        info = {
            "multi_source": is_multi_source,
            "data_type": data_type,
            "normalized": self.config.get("normalize", False),
            "normalization_method": self.config.get("normalization_method", "standard")
        }
        
        if self.train_dataset:
            info["train_samples"] = len(self.train_dataset)
            
            if is_graph:
                # Graph dataset (MeshGraphNet)
                info["train_data_sources"] = ["node_features", "edge_index", "edge_features"]
                info["train_data_shapes"] = {
                    "node_features": self.train_dataset.node_features.shape,
                    "edge_index": self.train_dataset.edge_index.shape,
                    "edge_features": self.train_dataset.edge_features.shape
                }
            elif hasattr(self.train_dataset, 'data_dict'):
                # Multi-source dataset
                info["train_data_sources"] = list(self.train_dataset.data_dict.keys())
                info["train_data_shapes"] = {
                    name: array.shape for name, array in self.train_dataset.data_dict.items()
                }
            else:
                # Standard dataset
                info["train_data_shape"] = self.train_dataset.data.shape
            
            if self.train_dataset.targets is not None:
                info["train_targets_shape"] = self.train_dataset.targets.shape
                info["train_targets_normalized"] = self.config.get("normalize_targets", False)
                
        if self.test_dataset:
            info["test_samples"] = len(self.test_dataset)
            
            if is_graph:
                # Graph dataset (MeshGraphNet)
                info["test_data_sources"] = ["node_features", "edge_index", "edge_features"]
                info["test_data_shapes"] = {
                    "node_features": self.test_dataset.node_features.shape,
                    "edge_index": self.test_dataset.edge_index.shape,
                    "edge_features": self.test_dataset.edge_features.shape
                }
            elif hasattr(self.test_dataset, 'data_dict'):
                # Multi-source dataset
                info["test_data_sources"] = list(self.test_dataset.data_dict.keys())
                info["test_data_shapes"] = {
                    name: array.shape for name, array in self.test_dataset.data_dict.items()
                }
            else:
                # Standard dataset
                info["test_data_shape"] = self.test_dataset.data.shape
                
            if self.test_dataset.targets is not None:
                info["test_targets_shape"] = self.test_dataset.targets.shape
                info["test_targets_normalized"] = self.config.get("normalize_targets", False)
                
        return info
