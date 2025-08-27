"""
Model Architecture Module for Low-Code Deep Learning Platform
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import importlib
import inspect
import sys
import os

# Add FNO, MLP and UNet modules
sys.path.append(os.path.dirname(__file__))
from fno.fno import FNO
from mlp.mlp import MLP
from unet.unet import UNet


class ModelManager:
    """
    Model Manager for handling predefined models and custom model templates
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model manager
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        self.config = config
        self.models = {}
        self.layers = {}
        self._register_predefined_models()
        self._register_template_models()
        self._register_fno_models()
        self._register_mlp_models()
        self._register_unet_models()
        self._register_predefined_layers()
    
    def _register_predefined_models(self):
        """
        Register complete neural network architectures
        """
        # These are complete models, not individual layers
        pass
    
    def _register_template_models(self):
        """
        Register template models
        """
        self.models['modeltemplate'] = ModelTemplate
    
    def _register_fno_models(self):
        """
        Register FNO models
        """
        self.models['fno'] = FNO
    
    def _register_mlp_models(self):
        """
        Register MLP models
        """
        self.models['mlp'] = MLP
    
    def _register_unet_models(self):
        """
        Register UNet models
        """
        self.models['unet'] = UNet
    
    def _register_predefined_layers(self):
        """
        Register predefined layers (not complete models)
        """
        # Register basic neural network layers as building blocks
        self.layers['linear'] = nn.Linear
        self.layers['conv2d'] = nn.Conv2d
        self.layers['lstm'] = nn.LSTM
        self.layers['gru'] = nn.GRU
        self.layers['rnn'] = nn.RNN
        
        # Register activation functions
        self.layers['relu'] = nn.ReLU
        self.layers['sigmoid'] = nn.Sigmoid
        self.layers['tanh'] = nn.Tanh
        self.layers['softmax'] = nn.Softmax
    
    def register_custom_model(self, name: str, model_class):
        """
        Register a custom model class
        
        Args:
            name: Name to register the model under
            model_class: The model class to register
        """
        self.models[name.lower()] = model_class
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available complete models (not individual layers)
        
        Returns:
            List of available model names
        """
        return list(self.models.keys())
    
    def get_available_layers(self) -> List[str]:
        """
        Get list of available layers and functions
        
        Returns:
            List of available layer names
        """
        return list(self.layers.keys())
    
    def get_model_hyperparameters(self, model_name: str) -> Dict[str, Any]:
        """
        Get hyperparameters for a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of hyperparameters with descriptions
        """
        model_name = model_name.lower()
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {self.get_available_models()}")
        
        model_class = self.models[model_name]
        
        # Check if the model has a HYPERPARAMETERS attribute (like in our template)
        if hasattr(model_class, 'HYPERPARAMETERS'):
            return model_class.HYPERPARAMETERS
        
        # For PyTorch built-in models, we can extract parameters from the constructor
        sig = inspect.signature(model_class.__init__)
        params = {}
        for name, param in sig.parameters.items():
            if name != 'self':
                params[name] = {
                    'description': f'Parameter {name}',
                    'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'unknown',
                    'default': param.default if param.default != inspect.Parameter.empty else None
                }
        return params
    
    def get_layer_parameters(self, layer_name: str) -> Dict[str, Any]:
        """
        Get parameters for a specific layer
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Dictionary of parameters with descriptions
        """
        layer_name = layer_name.lower()
        if layer_name not in self.layers:
            raise ValueError(f"Layer '{layer_name}' not found. Available layers: {self.get_available_layers()}")
        
        layer_class = self.layers[layer_name]
        
        # Extract parameters from the constructor
        sig = inspect.signature(layer_class.__init__)
        params = {}
        for name, param in sig.parameters.items():
            if name != 'self':
                params[name] = {
                    'description': f'Parameter {name}',
                    'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'unknown',
                    'default': param.default if param.default != inspect.Parameter.empty else None
                }
        return params
    
    def create_model(self, model_name: str, **kwargs) -> nn.Module:
        """
        Create an instance of a model
        
        Args:
            model_name: Name of the model to create
            **kwargs: Hyperparameters for the model
            
        Returns:
            Instantiated model
        """
        model_name = model_name.lower()
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {self.get_available_models()}")
        
        model_class = self.models[model_name]
        try:
            # Try to create model with provided parameters
            model = model_class(**kwargs)
            return model
        except Exception as e:
            raise ValueError(f"Failed to create model '{model_name}' with parameters {kwargs}: {str(e)}")
    
    def create_layer(self, layer_name: str, **kwargs) -> nn.Module:
        """
        Create an instance of a layer
        
        Args:
            layer_name: Name of the layer to create
            **kwargs: Parameters for the layer
            
        Returns:
            Instantiated layer
        """
        layer_name = layer_name.lower()
        if layer_name not in self.layers:
            raise ValueError(f"Layer '{layer_name}' not found. Available layers: {self.get_available_layers()}")
        
        layer_class = self.layers[layer_name]
        try:
            # Try to create layer with provided parameters
            layer = layer_class(**kwargs)
            return layer
        except Exception as e:
            raise ValueError(f"Failed to create layer '{layer_name}' with parameters {kwargs}: {str(e)}")
    
    def load_model_from_file(self, file_path: str, class_name: str) -> nn.Module:
        """
        Load a custom model from a Python file
        
        Args:
            file_path: Path to the Python file containing the model
            class_name: Name of the model class in the file
            
        Returns:
            Model class
        """
        spec = importlib.util.spec_from_file_location("custom_model", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if not hasattr(module, class_name):
            raise ValueError(f"Class '{class_name}' not found in '{file_path}'")
        
        model_class = getattr(module, class_name)
        self.register_custom_model(class_name.lower(), model_class)
        return model_class


class ModelTemplate(nn.Module):
    """
    A template for deep learning models.
    This is a simple example that can be extended.
    """
    
    # Model hyperparameters with descriptions
    HYPERPARAMETERS = {
        'input_size': {
            'description': 'Size of the input features',
            'type': 'int',
            'default': 784
        },
        'hidden_size': {
            'description': 'Number of units in the hidden layers',
            'type': 'int',
            'default': 256
        },
        'num_classes': {
            'description': 'Number of output classes',
            'type': 'int',
            'default': 10
        },
        'dropout_rate': {
            'description': 'Dropout rate for regularization',
            'type': 'float',
            'default': 0.2
        }
    }
    
    def __init__(self, **kwargs):
        """
        Initialize the model with hyperparameters.
        
        Args:
            **kwargs: Hyperparameters for the model
        """
        super(ModelTemplate, self).__init__()
        
        # Set hyperparameters with defaults
        self.input_size = kwargs.get('input_size', self.HYPERPARAMETERS['input_size']['default'])
        self.hidden_size = kwargs.get('hidden_size', self.HYPERPARAMETERS['hidden_size']['default'])
        self.num_classes = kwargs.get('num_classes', self.HYPERPARAMETERS['num_classes']['default'])
        self.dropout_rate = kwargs.get('dropout_rate', self.HYPERPARAMETERS['dropout_rate']['default'])
        
        # Define model layers
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, self.num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Flatten the input if it's not already flat
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.layers(x)
    
    @classmethod
    def get_hyperparameters(cls) -> Dict[str, Any]:
        """
        Get model hyperparameters with descriptions.
        
        Returns:
            Dict[str, Any]: Dictionary of hyperparameters with their descriptions
        """
        return cls.HYPERPARAMETERS