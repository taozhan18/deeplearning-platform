"""
Deep Learning Model Template

This is a template for implementing new deep learning models in the platform.
All models should follow this structure to ensure consistency and compatibility.
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class ModelTemplate(nn.Module):
    """
    A template for deep learning models.
    
    This template demonstrates the structure that all models should follow.
    Each model should have clear hyperparameter descriptions to help users
    understand their functionality.
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
        return self.layers(x)
    
    @classmethod
    def get_hyperparameters(cls) -> Dict[str, Any]:
        """
        Get model hyperparameters with descriptions.
        
        Returns:
            Dict[str, Any]: Dictionary of hyperparameters with their descriptions
        """
        return cls.HYPERPARAMETERS


# Example usage:
# model = ModelTemplate(input_size=784, hidden_size=128, num_classes=10, dropout_rate=0.3)
# print(model.get_hyperparameters())