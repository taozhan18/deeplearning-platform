# CNNAttention Model Documentation

## Overview

The CNNAttention model is a field-to-scalar neural network that extracts features from field data (1D/2D/3D) and predicts scalar values. It combines convolutional feature extraction with attention mechanisms for effective field data processing and regression tasks.

## Architecture

The model consists of three main components:

### 1. FieldEncoder
- **Purpose**: Encodes field data into latent features using convolutional layers
- **Supports**: 1D, 2D, and 3D field data
- **Features**: 
  - Convolutional layers with batch normalization and ReLU activation
  - Global pooling (adaptive or max pooling)
  - Configurable number of layers and kernel sizes

### 2. AttentionModule (Optional)
- **Purpose**: Self-attention mechanism for feature enhancement
- **Features**:
  - Multi-head attention
  - Residual connections with layer normalization
  - Dropout for regularization

### 3. MLP Head
- **Purpose**: Final scalar prediction from encoded features
- **Features**:
  - Configurable hidden layer sizes
  - Multiple activation functions (ReLU, Tanh, Sigmoid, LeakyReLU, GELU, SiLU)
  - Dropout regularization

## Model Parameters

### Required Parameters
- `in_channels`: Number of input channels in field data (int)
- `dimension`: Dimensionality of field data (1, 2, or 3)

### Optional Parameters
- `hidden_channels`: Number of hidden channels in encoder (default: 64)
- `num_encoder_layers`: Number of encoder layers (default: 4)
- `encoder_kernel_size`: Kernel size for encoder convolutions (default: 3)
- `use_attention`: Use attention mechanism (default: True)
- `attention_heads`: Number of attention heads (default: 8)
- `mlp_hidden_sizes`: Hidden layer sizes for MLP head (default: [256, 128, 64])
- `dropout_rate`: Dropout rate for MLP layers (default: 0.2)
- `activation_fn`: Activation function for MLP (default: 'relu')
- `pooling_strategy`: Pooling strategy ('adaptive', 'max') (default: 'adaptive')

## Usage Examples

### Basic Usage
```python
from model_manager import ModelManager

# Initialize model manager
model_manager = ModelManager({})

# Create CNNAttention model
model = model_manager.create_model('CNNAtention',
                                 in_channels=3,        # RGB image
                                 dimension=2,          # 2D field data
                                 hidden_channels=64,
                                 use_attention=True,
                                 mlp_hidden_sizes=[128, 64])

# Forward pass
field_data = torch.randn(16, 3, 64, 64)  # batch=16, channels=3, 64x64
predictions = model(field_data)  # Output: (16, 1)
```

### 1D Field Data (Time Series)
```python
model = model_manager.create_model('CNNAtention',
                                 in_channels=1,        # Single channel
                                 dimension=1,          # 1D data
                                 hidden_channels=32,
                                 use_attention=False)

# Input: (batch, channels, length)
time_series = torch.randn(8, 1, 1000)
predictions = model(time_series)
```

### 3D Field Data (Volumetric)
```python
model = model_manager.create_model('CNNAtention',
                                 in_channels=2,        # Multi-channel 3D
                                 dimension=3,          # 3D data
                                 hidden_channels=24,
                                 attention_heads=4)

# Input: (batch, channels, depth, height, width)
volume_data = torch.randn(4, 2, 32, 32, 32)
predictions = model(volume_data)
```

## Input/Output Specifications

### Input Shapes
- **1D**: `(batch_size, channels, length)`
- **2D**: `(batch_size, channels, height, width)`
- **3D**: `(batch_size, channels, depth, height, width)`

### Output Shape
- Always: `(batch_size, 1)` - scalar predictions for each batch item

## Supported Activation Functions
- `relu`: Rectified Linear Unit
- `tanh`: Hyperbolic Tangent
- `sigmoid`: Sigmoid
- `leaky_relu`: Leaky ReLU
- `elu`: Exponential Linear Unit
- `gelu`: Gaussian Error Linear Unit
- `silu`: Sigmoid Linear Unit

## Pooling Strategies
- `adaptive`: Adaptive average pooling (default)
- `max`: Global max pooling

## Applications

This model is suitable for:
- **Physics simulations**: Predicting scalar quantities from field data (temperature, pressure, velocity fields)
- **Medical imaging**: Extracting diagnostic values from medical scans
- **Geospatial analysis**: Predicting environmental metrics from spatial data
- **Signal processing**: Extracting features from time series or sensor data
- **Quality control**: Predicting quality metrics from inspection images

## Performance Characteristics

### Computational Complexity
- **Encoder**: O(N × C × H × W × D) where N is batch size, C is channels, H/W/D are spatial dimensions
- **Attention**: O(N × F²) where F is feature dimension
- **MLP**: O(N × H) where H is total hidden units

### Memory Usage
- Scales with input dimensions and hidden channel size
- Attention mechanism adds moderate overhead
- Batch processing improves efficiency

## Testing

The model includes comprehensive tests covering:
- Basic functionality for all dimensions (1D, 2D, 3D)
- Attention mechanism variants
- Pooling strategies
- Activation functions
- Gradient flow and parameter initialization
- Batch size variations
- Edge cases and error handling

### Running Tests
```bash
# Run standalone tests
python test_cnnattention.py

# Run pytest version
python -m pytest test_cnnattention_pytest.py -v

# Run all CNNAttention tests
python run_cnnattention_tests.py
```

## Integration with Model Manager

The model is fully integrated with the platform's model management system:

```python
# Check available models
model_manager.list_models()  # Includes 'CNNAtention'

# Get model parameters
params = model_manager.get_model_parameters('CNNAtention')

# Create model with configuration
config = {
    "parameters": {
        "in_channels": 1,
        "dimension": 2,
        "hidden_channels": 32,
        "use_attention": True
    }
}
model = model_manager.create_model('CNNAtention')
```

## Hyperparameter Tuning

Key hyperparameters to tune:
1. **hidden_channels**: Controls model capacity (16-256 recommended)
2. **num_encoder_layers**: Depth of feature extraction (2-6 recommended)
3. **mlp_hidden_sizes**: Width of prediction head
4. **use_attention**: Enable/disable attention mechanism
5. **dropout_rate**: Regularization strength (0.1-0.5 recommended)

## Limitations and Considerations

- Input dimensions should be consistent within batches
- Feature dimension must be divisible by number of attention heads
- Very small input sizes (<8 in any dimension) may reduce performance
- Memory usage increases significantly with 3D data and large batch sizes

## Future Enhancements

Potential improvements:
- Support for variable-sized inputs within batches
- Additional attention mechanisms (cross-attention, self-attention variants)
- Multi-scale feature extraction
- Regularization techniques beyond dropout
- Mixed precision training support