# Fourier Neural Operator (FNO) Models

This directory contains the implementation of Fourier Neural Operator (FNO) models for the low-code deep learning platform.

## Overview

The Fourier Neural Operator (FNO) is a deep learning architecture designed for learning mappings between function spaces. It uses Fourier transforms to efficiently capture global dependencies in the data, making it particularly effective for solving partial differential equations (PDEs) and other problems with complex spatial patterns.

## Implemented Models

1. **FNO1DEncoder**: 1D Spectral encoder for FNO
2. **FNO2DEncoder**: 2D Spectral encoder for FNO
3. **FNO**: Full FNO model supporting both 1D and 2D cases

## Key Features

- Supports 1D and 2D domains
- Spectral convolutions using FFT for efficient global operations
- Coordinate feature augmentation
- Configurable number of layers, modes, and channels
- Compatible with the low-code platform's model interface

## Hyperparameters

The FNO model supports the following hyperparameters:

- `in_channels`: Number of input channels
- `out_channels`: Number of output channels
- `decoder_layers`: Number of decoder layers
- `decoder_layer_size`: Number of neurons in decoder layers
- `dimension`: Model dimensionality (1 or 2)
- `latent_channels`: Latent features size in spectral convolutions
- `num_fno_layers`: Number of spectral convolutional layers
- `num_fno_modes`: Number of Fourier modes kept in spectral convolutions
- `padding`: Domain padding for spectral convolutions
- `padding_type`: Type of padding for spectral convolutions
- `coord_features`: Use coordinate grid as additional feature map

## Usage

To use the FNO model in the low-code platform:

```python
from model_manager import ModelManager

# Initialize model manager
model_manager = ModelManager({})

# Create FNO model
fno_model = model_manager.create_model('fno', 
                                      in_channels=1,
                                      out_channels=1,
                                      dimension=2,
                                      latent_channels=32,
                                      num_fno_layers=4,
                                      num_fno_modes=16)
```

## References

- Li, Zongyi, et al. "Fourier neural operator for parametric partial differential equations." arXiv preprint arXiv:2010.08895 (2020).