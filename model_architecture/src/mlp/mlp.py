"""
Multi-Layer Perceptron (MLP) implementation for the low-code deep learning platform
"""

import torch
import torch.nn as nn
from typing import List, Union


class MLP(nn.Module):
    """A multi-layer perceptron model for the low-code platform.

    Parameters
    ----------
    in_features : int, optional
        Size of input features, by default 512
    layer_sizes : Union[int, List[int]], optional
        Size of hidden layers. Can be a single int for all layers or a list for each layer, by default 512
    out_features : int, optional
        Size of output features, by default 512
    num_layers : int, optional
        Number of hidden layers, by default 6
    activation_fn : Union[str, List[str]], optional
        Activation function to use, by default 'relu'
    skip_connections : bool, optional
        Add skip connections every 2 hidden layers, by default False
    dropout : Union[float, List[float]], optional
        Dropout rate. Can be a single float for all layers or a list for each layer, by default 0.0
    """

    # Model hyperparameters with descriptions
    HYPERPARAMETERS = {
        'in_features': {
            'description': 'Size of input features',
            'type': 'int',
            'default': 512
        },
        'layer_sizes': {
            'description': 'Size of hidden layers. Can be a single int for all layers or a list for each layer',
            'type': 'Union[int, List[int]]',
            'default': 512
        },
        'out_features': {
            'description': 'Size of output features',
            'type': 'int',
            'default': 512
        },
        'num_layers': {
            'description': 'Number of hidden layers',
            'type': 'int',
            'default': 6
        },
        'activation_fn': {
            'description': 'Activation function to use',
            'type': 'Union[str, List[str]]',
            'default': 'relu'
        },
        'skip_connections': {
            'description': 'Add skip connections every 2 hidden layers',
            'type': 'bool',
            'default': False
        },
        'dropout': {
            'description': 'Dropout rate. Can be a single float for all layers or a list for each layer',
            'type': 'Union[float, List[float]]',
            'default': 0.0
        }
    }

    def __init__(
        self,
        in_features: int = 512,
        layer_sizes: Union[int, List[int]] = 512,
        out_features: int = 512,
        num_layers: int = 6,
        activation_fn: Union[str, List[str]] = "relu",
        skip_connections: bool = False,
        dropout: Union[float, List[float]] = 0.0,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.skip_connections = skip_connections

        # Process layer sizes
        if isinstance(layer_sizes, int):
            layer_sizes = [layer_sizes] * num_layers
        elif len(layer_sizes) != num_layers:
            raise ValueError(f"layer_sizes list length ({len(layer_sizes)}) must match num_layers ({num_layers})")

        # Process activation functions
        if isinstance(activation_fn, str):
            activation_fn = [activation_fn] * num_layers
        elif len(activation_fn) != num_layers:
            raise ValueError(f"activation_fn list length ({len(activation_fn)}) must match num_layers ({num_layers})")

        # Process dropout rates
        if isinstance(dropout, (float, int)):
            dropout = [dropout] * num_layers
        elif len(dropout) != num_layers:
            raise ValueError(f"dropout list length ({len(dropout)}) must match num_layers ({num_layers})")

        # Create activation function mapping
        activation_map = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'leaky_relu': nn.LeakyReLU,
            'elu': nn.ELU,
            'selu': nn.SELU,
            'gelu': nn.GELU,
            'silu': nn.SiLU,
            'none': None
        }

        # Build the layers
        self.layers = nn.ModuleList()
        
        # Input to first hidden layer
        layer_in_features = in_features
        for i in range(num_layers):
            # Linear layer
            self.layers.append(nn.Linear(layer_in_features, layer_sizes[i]))
            
            # Activation function
            if activation_fn[i] != 'none' and activation_fn[i] in activation_map:
                act_fn = activation_map[activation_fn[i]]()
                self.layers.append(act_fn)
            
            # Dropout
            if dropout[i] > 0:
                self.layers.append(nn.Dropout(dropout[i]))
            
            # Update for next layer
            layer_in_features = layer_sizes[i]

        # Final layer
        self.final_layer = nn.Linear(layer_in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = None
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Handle skip connections
            if self.skip_connections and isinstance(layer, nn.Linear):
                # Check if this is every 2nd linear layer (0-indexed)
                linear_layer_count = sum(1 for l in self.layers[:i] if isinstance(l, nn.Linear))
                if linear_layer_count > 0 and linear_layer_count % 2 == 0:
                    if x_skip is not None:
                        x = x + x_skip
                    x_skip = x

        x = self.final_layer(x)
        return x

    @classmethod
    def get_hyperparameters(cls) -> dict:
        """Get model hyperparameters with descriptions.

        Returns:
            Dict[str, Any]: Dictionary of hyperparameters with their descriptions
        """
        return cls.HYPERPARAMETERS