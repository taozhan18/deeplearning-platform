# UNet Models

This directory contains the implementation of UNet models for the low-code deep learning platform.

## Overview

UNet is a convolutional neural network architecture originally designed for biomedical image segmentation. The architecture consists of an encoder (contracting path) and a decoder (expansive path), with skip connections between corresponding levels of the encoder and decoder to help preserve spatial information.

## Implemented Models

1. **UNet**: A standard UNet implementation

## Key Features

- Standard encoder-decoder architecture with skip connections
- Configurable number of feature channels at each level
- Support for different activation functions
- Support for different normalization techniques
- Flexible input and output channel configurations

## Architecture

The UNet architecture consists of:

1. **Encoder Path**: A series of convolutional blocks with max pooling for downsampling
2. **Bottleneck**: The deepest part of the network connecting encoder and decoder
3. **Decoder Path**: A series of transposed convolutional blocks with skip connections for upsampling
4. **Skip Connections**: Connections from encoder to corresponding decoder levels to preserve spatial information

Each encoder and decoder block contains two convolutional layers with optional normalization and activation.

## Hyperparameters

The UNet model supports the following hyperparameters:

- `in_channels`: Number of channels in the input image
- `out_channels`: Number of channels in the output segmentation map
- `features`: Number of feature channels at each level
- `activation`: Type of activation to use ('relu', 'leaky_relu', 'elu')
- `normalization`: Type of normalization to use ('batchnorm', 'groupnorm')

## Usage

To use the UNet model in the low-code platform:

```python
from model_manager import ModelManager

# Initialize model manager
model_manager = ModelManager({})

# Create UNet model
unet_model = model_manager.create_model('unet', 
                                     in_channels=3,
                                     out_channels=1,
                                     features=[64, 128, 256, 512, 1024])
```

## Skip Connections

Skip connections in UNet help preserve spatial information that might be lost during downsampling in the encoder path. These connections concatenate feature maps from the encoder with corresponding feature maps in the decoder, allowing the network to learn more precise localization.

## Applications

UNet is particularly well-suited for:

- Image segmentation tasks
- Medical image analysis
- Satellite image processing
- Any task requiring pixel-level predictions