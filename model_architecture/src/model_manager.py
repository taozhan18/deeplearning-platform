"""
Model Architecture Module for Low-Code Deep Learning Platform
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import importlib
import inspect


class ModelManager:
    """
    Centralized model management system for the deep learning platform.

    This class handles model creation, registration, and metadata management.
    It provides a unified interface for creating different types of models
    with their respective parameters.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ModelManager.

        Args:
            config: Configuration dictionary for the model manager. It depends on the model you want to use.
        """
        self.config = config
        self.models = {}
        self.model_metadata = {}

        # Register all available models
        self._register_models()

    def _register_models(self):
        """
        Register all available models with their metadata.
        """
        # Register FNO model
        try:
            from fno.fno import FNO

            self._register_model("fno", FNO, FNO.HYPERPARAMETERS)
        except ImportError as e:
            print(f"Warning: FNO model could not be registered: {e}")

        # Register MLP model
        try:
            from mlp.mlp import MLP

            self._register_model("mlp", MLP, MLP.HYPERPARAMETERS)
        except ImportError as e:
            print(f"Warning: MLP model could not be registered: {e}")

        # Register UNet model
        try:
            from unet.unet import UNet

            self._register_model("unet", UNet, UNet.HYPERPARAMETERS)
        except ImportError as e:
            print(f"Warning: UNet model could not be registered: {e}")

        # Register Transformer model
        try:
            from transformer.transformer import Transformer

            self._register_model("transformer", Transformer, Transformer.HYPERPARAMETERS)
        except ImportError as e:
            print(f"Warning: Transformer model could not be registered: {e}")

        # Register Transolver model
        try:
            from transolver.transolver import Transolver

            self._register_model("transolver", Transolver, Transolver.HYPERPARAMETERS)
        except ImportError as e:
            print(f"Warning: Transolver model could not be registered: {e}")

        # Register MeshGraphNet model
        try:
            from meshgraphnet.meshgraphnet import MeshGraphNet

            self._register_model("meshgraphnet", MeshGraphNet, MeshGraphNet.HYPERPARAMETERS)
        except ImportError as e:
            print(f"Warning: MeshGraphNet model could not be registered: {e}")

        # Register FieldToScalarNet model
        try:
            from CNNAtention.CNNAtention import CNNAtention

            self._register_model("CNNAtention", CNNAtention, CNNAtention.HYPERPARAMETERS)
        except ImportError as e:
            print(f"Warning: CNNAtention model could not be registered: {e}")

    def _register_model(self, name: str, model_class, hyperparameters: Dict[str, Any]):
        """
        Register a model with its class and hyperparameters.

        Args:
            name: Model name
            model_class: Model class
            hyperparameters: Model hyperparameters dictionary
        """
        self.models[name.lower()] = model_class
        self.model_metadata[name.lower()] = {"class": model_class, "hyperparameters": hyperparameters}

    def register_custom_model(self, name: str, model_class):
        """
        Register a custom model class

        Args:
            name: Name to register the model under
            model_class: The model class to register
        """
        self.models[name.lower()] = model_class

    def create_model(self, model_name: str, **kwargs) -> nn.Module:
        """
        Create a model with the specified name and parameters.

        Args:
            model_name: Name of the model to create
            **kwargs: Model-specific parameters. If not provided, will use parameters from self.config

        Returns:
            Created model instance

        Raises:
            ValueError: If the model is not found or parameters are invalid
        """
        model_name = model_name.lower()

        if model_name not in self.models:
            available_models = list(self.models.keys())
            raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")

        # If no kwargs provided, use parameters from config
        if not kwargs:
            # Get model parameters from config
            model_params = self.config.get("parameters", {})
            kwargs = model_params

        try:
            model = self.models[model_name](**kwargs)
            return model
        except Exception as e:
            raise ValueError(f"Failed to create model '{model_name}': {str(e)}")

    def get_model_metadata(self, model_name: str = None) -> Dict[str, Any]:
        """
        Get metadata for a specific model or all models.

        Args:
            model_name: Name of the model to get metadata for.
                       If None, returns metadata for all models.

        Returns:
            Model metadata dictionary
        """
        if model_name is None:
            return self.model_metadata

        model_name = model_name.lower()
        if model_name not in self.model_metadata:
            raise ValueError(f"Model '{model_name}' not found")

        return self.model_metadata[model_name]

    def list_models(self) -> List[str]:
        """
        List all available models.

        Returns:
            List of available model names
        """
        return list(self.models.keys())

    def get_model_parameters(self, model_name: str) -> Dict[str, Any]:
        """
        Get parameter information for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary of model parameters with their descriptions and types
        """
        metadata = self.get_model_metadata(model_name)
        return metadata.get("hyperparameters", {})


# Global model manager instance
_model_manager = None


def get_model_manager(config: Dict[str, Any] = None) -> ModelManager:
    """
    Get the global model manager instance.

    Args:
        config: Configuration dictionary for the model manager

    Returns:
        ModelManager instance
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(config or {})
    return _model_manager


def create_model(model_name: str, **kwargs) -> nn.Module:
    """
    Create a model using the global model manager.

    Args:
        model_name: Name of the model to create
        **kwargs: Model-specific parameters

    Returns:
        Created model instance
    """
    manager = get_model_manager()
    return manager.create_model(model_name, **kwargs)
