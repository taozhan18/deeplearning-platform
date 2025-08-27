# Model Architecture Module

The Model Architecture module provides a collection of predefined deep learning models based on PyTorch.

## Features

- Predefined popular model architectures (ResNet, VGG, LSTM, etc.)
- Clear hyperparameter descriptions for each model
- Template for defining new model architectures
- Easy model selection and configuration
- Fourier Neural Operator (FNO) models for PDE-related tasks

## Usage

Users can select from predefined models or define their own architectures using our template. Each model comes with detailed hyperparameter descriptions to help users understand their functionality.

## Model Template

We provide a standard template for implementing new model architectures. All models should follow this template to ensure consistency and ease of use.

## FNO Models

The module now includes Fourier Neural Operator (FNO) models, which are particularly effective for solving partial differential equations and other problems with complex spatial patterns. See [FNO documentation](src/fno/README.md) for more details.

## Available Models

- `linear`, `conv2d`, `lstm`, `gru`, `rnn` - Basic PyTorch layers
- `relu`, `sigmoid`, `tanh`, `softmax` - Activation functions
- `modeltemplate` - Basic model template
- `fno` - Fourier Neural Operator model (1D and 2D)

To get a list of all available models programmatically:

```python
from model_manager import ModelManager
model_manager = ModelManager({})
print(model_manager.get_available_models())
```