# Multi-Layer Perceptron (MLP) Models

This directory contains the implementation of Multi-Layer Perceptron (MLP) models for the low-code deep learning platform.

## Overview

The Multi-Layer Perceptron (MLP) is a class of feedforward artificial neural network that consists of at least three layers of nodes: an input layer, a hidden layer, and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function.

## Implemented Models

1. **MLP**: A flexible multi-layer perceptron implementation

## Key Features

- Configurable number of hidden layers
- Support for different activation functions per layer
- Skip connections for deeper networks
- Configurable dropout rates
- Flexible layer sizing

## Hyperparameters

The MLP model supports the following hyperparameters:

- `in_features`: Size of input features
- `layer_sizes`: Size of hidden layers. Can be a single int for all layers or a list for each layer
- `out_features`: Size of output features
- `num_layers`: Number of hidden layers
- `activation_fn`: Activation function to use (supports 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'selu', 'gelu', 'silu', 'none')
- `skip_connections`: Add skip connections every 2 hidden layers
- `dropout`: Dropout rate. Can be a single float for all layers or a list for each layer

## Usage

To use the MLP model in the low-code platform:

```python
from model_manager import ModelManager

# Initialize model manager
model_manager = ModelManager({})

# Create MLP model
mlp_model = model_manager.create_model('mlp', 
                                     in_features=784,
                                     layer_sizes=256,
                                     out_features=10,
                                     num_layers=4,
                                     activation_fn='relu',
                                     dropout=0.2)
```

## Supported Activation Functions

- `relu`: Rectified Linear Unit
- `tanh`: Hyperbolic tangent
- `sigmoid`: Sigmoid function
- `leaky_relu`: Leaky ReLU
- `elu`: Exponential Linear Unit
- `selu`: Scaled Exponential Linear Unit
- `gelu`: Gaussian Error Linear Unit
- `silu`: Sigmoid Linear Unit (Swish)
- `none`: No activation function

## Skip Connections

Skip connections can be enabled to help with training deeper networks by allowing gradients to flow more easily through the network. When enabled, skip connections are added every 2 hidden layers.