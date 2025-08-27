# Model Guide for Low-Code Deep Learning Platform

This document provides information about all available complete neural network models in the platform and their parameters.

## Available Complete Models

The platform currently supports the following complete neural network architectures:

1. **Custom Models**:
   - `modeltemplate`: A basic feedforward neural network template
   - `fno`: Fourier Neural Operator for PDE-related tasks
   - `mlp`: Multi-Layer Perceptron with configurable layers and activation functions
   - `unet`: U-Net for image segmentation tasks

Note: The platform also includes basic PyTorch layers (like Linear, Conv2d, LSTM, etc.) and activation functions (like ReLU, Sigmoid, etc.) which can be used to build custom architectures, but these are not complete models by themselves.

## Complete Model Parameters

### 1. ModelTemplate

A basic feedforward neural network with configurable layers and dropout.

**Parameters**:
- `input_size` (int, default: 784): Size of the input features
- `hidden_size` (int, default: 256): Number of units in the hidden layers
- `num_classes` (int, default: 10): Number of output classes
- `dropout_rate` (float, default: 0.2): Dropout rate for regularization

**Example Configuration**:
```yaml
model:
  name: "ModelTemplate"
  parameters:
    input_size: 784
    hidden_size: 256
    num_classes: 10
    dropout_rate: 0.2
```

### 2. FNO (Fourier Neural Operator)

A model designed for learning mappings between function spaces, particularly effective for PDE-related tasks.

**Parameters**:
- `in_channels` (int, default: 1): Number of input channels
- `out_channels` (int, default: 1): Number of output channels
- `decoder_layers` (int, default: 1): Number of decoder layers
- `decoder_layer_size` (int, default: 32): Number of neurons in decoder layers
- `dimension` (int, default: 2): Model dimensionality (1 for 1D, 2 for 2D)
- `latent_channels` (int, default: 32): Latent features size in spectral convolutions
- `num_fno_layers` (int, default: 4): Number of spectral convolutional layers
- `num_fno_modes` (Union[int, List[int]], default: 16): Number of Fourier modes kept in spectral convolutions
- `padding` (int, default: 8): Domain padding for spectral convolutions
- `padding_type` (str, default: "constant"): Type of padding for spectral convolutions
- `coord_features` (bool, default: True): Use coordinate grid as additional feature map

**Example Configuration for FNO1D**:
```yaml
model:
  name: "fno"
  parameters:
    in_channels: 1
    out_channels: 1
    dimension: 1
    latent_channels: 16
    num_fno_layers: 3
    num_fno_modes: 16
```

**Example Configuration for FNO2D**:
```yaml
model:
  name: "fno"
  parameters:
    in_channels: 3
    out_channels: 2
    dimension: 2
    latent_channels: 32
    num_fno_layers: 4
    num_fno_modes: [16, 16]
```

### 3. MLP (Multi-Layer Perceptron)

A flexible multi-layer perceptron implementation with configurable layers, activation functions, and skip connections.

**Parameters**:
- `in_features` (int, default: 512): Size of input features
- `layer_sizes` (Union[int, List[int]], default: 512): Size of hidden layers. Can be a single int for all layers or a list for each layer
- `out_features` (int, default: 512): Size of output features
- `num_layers` (int, default: 6): Number of hidden layers
- `activation_fn` (Union[str, List[str]], default: 'relu'): Activation function to use
- `skip_connections` (bool, default: False): Add skip connections every 2 hidden layers
- `dropout` (Union[float, List[float]], default: 0.0): Dropout rate. Can be a single float for all layers or a list for each layer

**Supported Activation Functions**:
- `relu`: Rectified Linear Unit
- `tanh`: Hyperbolic tangent
- `sigmoid`: Sigmoid function
- `leaky_relu`: Leaky ReLU
- `elu`: Exponential Linear Unit
- `selu`: Scaled Exponential Linear Unit
- `gelu`: Gaussian Error Linear Unit
- `silu`: Sigmoid Linear Unit (Swish)
- `none`: No activation function

**Example Configuration**:
```yaml
model:
  name: "mlp"
  parameters:
    in_features: 784
    layer_sizes: 256
    out_features: 10
    num_layers: 4
    activation_fn: "relu"
    dropout: 0.2
```

**Example Configuration with Variable Layers**:
```yaml
model:
  name: "mlp"
  parameters:
    in_features: 784
    layer_sizes: [512, 256, 128, 64]
    out_features: 10
    num_layers: 4
    activation_fn: ["relu", "tanh", "relu", "sigmoid"]
    dropout: [0.1, 0.2, 0.2, 0.3]
    skip_connections: True
```

### 4. UNet

A standard UNet implementation for image segmentation tasks with encoder-decoder architecture and skip connections.

**Parameters**:
- `in_channels` (int, default: 3): Number of channels in the input image
- `out_channels` (int, default: 1): Number of channels in the output segmentation map
- `features` (List[int], default: [64, 128, 256, 512, 1024]): Number of feature channels at each level
- `activation` (str, default: 'relu'): Type of activation to use
- `normalization` (str, default: 'batchnorm'): Type of normalization to use

**Supported Activation Functions**:
- `relu`: Rectified Linear Unit
- `leaky_relu`: Leaky ReLU
- `elu`: Exponential Linear Unit

**Supported Normalization Types**:
- `batchnorm`: Batch Normalization
- `groupnorm`: Group Normalization

**Example Configuration**:
```yaml
model:
  name: "unet"
  parameters:
    in_channels: 3
    out_channels: 1
    features: [64, 128, 256, 512, 1024]
    activation: "relu"
    normalization: "batchnorm"
```

## How to Use Models

To use any of these models, specify the model name and parameters in your configuration file:

```yaml
model:
  name: "model_name"  # e.g., "fno", "ModelTemplate", "mlp", "unet"
  parameters:
    # Model-specific parameters
```

Then run the training with:
```bash
python main/train.py --config your_config.yaml
```

## Adding New Models

To add a new model to the platform:
1. Implement your model class
2. Register it in the ModelManager
3. Add hyperparameter documentation following the template pattern
4. Test the model with the platform

See the existing models in `model_architecture/src/` for examples.