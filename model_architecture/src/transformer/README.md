# Transformer Model

## Overview

The Transformer model is a powerful architecture for sequence-to-sequence tasks, originally proposed in the paper "Attention Is All You Need" by Vaswani et al. This implementation provides a flexible Transformer architecture suitable for various sequence modeling tasks.

## Architecture

The Transformer consists of an encoder and a decoder:

1. **Encoder**: Comprises a stack of identical layers, each containing:
   - Multi-head self-attention mechanism
   - Position-wise feedforward network
   - Residual connections and layer normalization

2. **Decoder**: Also comprises a stack of identical layers, each containing:
   - Masked multi-head self-attention
   - Multi-head attention over encoder output
   - Position-wise feedforward network
   - Residual connections and layer normalization

3. **Positional Encoding**: Since the Transformer contains no recurrence or convolution, positional information is injected using sinusoidal functions.

## Key Features

- Configurable number of layers, attention heads, and hidden dimensions
- Support for variable sequence lengths
- Dropout for regularization
- Layer normalization for training stability

## Parameters

| Parameter | Description | Type | Default |
|----------|-------------|------|---------|
| `input_dim` | Dimension of input features | int | Required |
| `output_dim` | Dimension of output features | int | Required |
| `d_model` | The dimension of the model embeddings | int | 512 |
| `n_layers` | Number of encoder/decoder layers | int | 6 |
| `n_heads` | Number of attention heads | int | 8 |
| `pf_dim` | Position-wise feedforward hidden dimension | int | 2048 |
| `dropout` | Dropout rate | float | 0.1 |
| `max_len` | Maximum sequence length | int | 100 |

## Usage

To use the Transformer model in your configuration:

```yaml
model:
  name: "transformer"
  parameters:
    input_dim: 10
    output_dim: 5
    d_model: 256
    n_layers: 4
    n_heads: 8
    pf_dim: 1024
    dropout: 0.1
    max_len: 50
```

## References

1. Vaswani, A., et al. "Attention is all you need." Advances in Neural Information Processing Systems 30 (2017).