# Transolver Model

## Overview

Transolver is a transformer-based solver specifically designed for solving Partial Differential Equations (PDEs). It was modified from the original implementation at https://github.com/thuml/Transolver.

The key features of Transolver include:
- Physics-informed attention mechanisms
- Structured mesh processing capabilities
- Flexible architecture for various PDE problems

## Architecture

Transolver consists of several key components:

1. **Physics Attention**: Specialized attention mechanisms designed for physics problems
2. **Transformer Blocks**: Standard transformer encoder blocks with physics-informed modifications
3. **MLP Layers**: Multi-layer perceptrons for feature processing
4. **Embedding Layers**: Positional and timestep embeddings

## Key Features

- Designed for structured 2D mesh data
- Physics-informed attention mechanisms
- Support for time-dependent PDEs with timestep embeddings
- Configurable number of layers, heads, and hidden dimensions
- Multiple activation functions supported

## Parameters

| Parameter | Description | Type | Default |
|----------|-------------|------|---------|
| `space_dim` | The spatial dimension of the input data | int | 1 |
| `n_layers` | The number of transformer layers | int | 5 |
| `n_hidden` | The hidden dimension of the transformer | int | 256 |
| `dropout` | The dropout rate | float | 0.0 |
| `n_head` | The number of attention heads | int | 8 |
| `Time_Input` | Whether to include time embeddings | bool | False |
| `act` | The activation function | str | 'gelu' |
| `mlp_ratio` | The ratio of hidden dimension in the MLP | int | 1 |
| `fun_dim` | The dimension of the function | int | 1 |
| `out_dim` | The output dimension | int | 1 |
| `slice_num` | The number of slices in the structured attention | int | 32 |
| `ref` | The reference dimension | int | 8 |
| `unified_pos` | Whether to use unified positional embeddings | bool | False |
| `H` | The height of the mesh | int | 85 |
| `W` | The width of the mesh | int | 85 |

## Usage

To use the Transolver model in your configuration:

```yaml
model:
  name: "transolver"
  parameters:
    space_dim: 2
    n_layers: 6
    n_hidden: 256
    dropout: 0.1
    n_head: 8
    Time_Input: False
    act: "gelu"
    mlp_ratio: 1
    fun_dim: 1
    out_dim: 1
    slice_num: 32
    ref: 8
    unified_pos: False
    H: 85
    W: 85
```

## References

1. Transolver repository: https://github.com/thuml/Transolver
2. MIT License Copyright (c) 2024 THUML @ Tsinghua University