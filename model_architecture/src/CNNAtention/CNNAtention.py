"""
Field-to-Scalar Neural Network implementation for the low-code deep learning platform

This model extracts features from field data (1D/2D/3D) and predicts a scalar value.
It combines convolutional feature extraction with attention mechanisms for
effective field data processing and regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Optional


class FieldEncoder(nn.Module):
    """Encodes field data into latent features using convolutional layers

    Parameters
    ----------
    in_channels : int
        Number of input channels
    hidden_channels : int
        Number of hidden channels in convolutional layers
    num_layers : int
        Number of convolutional layers
    kernel_size : int
        Kernel size for convolutional layers
    pooling : str
        Pooling strategy ('max', 'avg', 'adaptive')
    dimension : int
        Dimensionality of field data (1, 2, or 3)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        kernel_size: int = 3,
        pooling: str = "adaptive",
        dimension: int = 2,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.dimension = dimension

        # Choose appropriate convolution based on dimension
        if dimension == 1:
            self.conv_layer = nn.Conv1d
            self.bn_layer = nn.BatchNorm1d
            self.pool_layer = nn.AdaptiveAvgPool1d if pooling == "adaptive" else nn.MaxPool1d
        elif dimension == 2:
            self.conv_layer = nn.Conv2d
            self.bn_layer = nn.BatchNorm2d
            self.pool_layer = nn.AdaptiveAvgPool2d if pooling == "adaptive" else nn.MaxPool2d
        elif dimension == 3:
            self.conv_layer = nn.Conv3d
            self.bn_layer = nn.BatchNorm3d
            self.pool_layer = nn.AdaptiveAvgPool3d if pooling == "adaptive" else nn.MaxPool3d
        else:
            raise ValueError(f"Unsupported dimension: {dimension}")

        # Build encoder layers
        self.encoder_layers = nn.ModuleList()

        # Input layer
        self.encoder_layers.append(self._make_conv_block(in_channels, hidden_channels))

        # Hidden layers
        for i in range(num_layers - 1):
            self.encoder_layers.append(self._make_conv_block(hidden_channels, hidden_channels))

    def _make_conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """Create a convolutional block with batch norm and activation"""
        if self.dimension == 1:
            padding = self.kernel_size // 2
            conv = self.conv_layer(in_ch, out_ch, self.kernel_size, padding=padding)
        elif self.dimension == 2:
            padding = self.kernel_size // 2
            conv = self.conv_layer(in_ch, out_ch, self.kernel_size, padding=padding)
        elif self.dimension == 3:
            padding = self.kernel_size // 2
            conv = self.conv_layer(in_ch, out_ch, self.kernel_size, padding=padding)

        return nn.Sequential(conv, self.bn_layer(out_ch), nn.ReLU(), nn.Dropout(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder"""
        for layer in self.encoder_layers:
            x = layer(x)

        # Global pooling
        if self.pooling == "adaptive":
            if self.dimension == 1:
                x = F.adaptive_avg_pool1d(x, 1)
            elif self.dimension == 2:
                x = F.adaptive_avg_pool2d(x, 1)
            elif self.dimension == 3:
                x = F.adaptive_avg_pool3d(x, 1)
        else:
            # Global max pooling
            if self.dimension == 1:
                x = F.max_pool1d(x, kernel_size=x.shape[-1])
            elif self.dimension == 2:
                x = F.max_pool2d(x, kernel_size=(x.shape[-2], x.shape[-1]))
            elif self.dimension == 3:
                x = F.max_pool3d(x, kernel_size=(x.shape[-3], x.shape[-2], x.shape[-1]))

        # Flatten
        return x.view(x.size(0), -1)


class AttentionModule(nn.Module):
    """Self-attention module for feature enhancement"""

    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        assert self.head_dim * num_heads == feature_dim, "feature_dim must be divisible by num_heads"

        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.out = nn.Linear(feature_dim, feature_dim)

        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with self-attention"""
        batch_size = x.size(0)

        # Multi-head attention
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.feature_dim)

        # Output projection
        output = self.out(attended)
        output = self.dropout(output)

        # Residual connection and layer norm
        return self.layer_norm(output + x)


class CNNAtention(nn.Module):
    """Field-to-Scalar Neural Network for regression from field data

    This network processes field data (1D/2D/3D) and predicts a scalar value.
    It uses convolutional feature extraction followed by attention mechanisms
    and fully connected layers for final prediction.

    Parameters
    ----------
    in_channels : int
        Number of input channels in field data
    dimension : int
        Dimensionality of field data (1, 2, or 3)
    hidden_channels : int, optional
        Number of hidden channels in encoder, by default 64
    num_encoder_layers : int, optional
        Number of encoder layers, by default 4
    encoder_kernel_size : int, optional
        Kernel size for encoder convolutions, by default 3
    use_attention : bool, optional
        Use attention mechanism, by default True
    attention_heads : int, optional
        Number of attention heads, by default 8
    mlp_hidden_sizes : List[int], optional
        Hidden layer sizes for MLP head, by default [256, 128, 64]
    dropout_rate : float, optional
        Dropout rate for MLP layers, by default 0.2
    activation_fn : str, optional
        Activation function for MLP, by default 'relu'
    pooling_strategy : str, optional
        Pooling strategy ('adaptive', 'max'), by default 'adaptive'
    """

    # Model hyperparameters with descriptions
    HYPERPARAMETERS = {
        "in_channels": {"description": "Number of input channels in field data", "type": "int", "default": 1},
        "dimension": {"description": "Dimensionality of field data (1, 2, or 3)", "type": "int", "default": 2},
        "hidden_channels": {"description": "Number of hidden channels in encoder", "type": "int", "default": 64},
        "num_encoder_layers": {"description": "Number of encoder layers", "type": "int", "default": 4},
        "encoder_kernel_size": {"description": "Kernel size for encoder convolutions", "type": "int", "default": 3},
        "use_attention": {"description": "Use attention mechanism", "type": "bool", "default": True},
        "attention_heads": {"description": "Number of attention heads", "type": "int", "default": 8},
        "mlp_hidden_sizes": {
            "description": "Hidden layer sizes for MLP head",
            "type": "List[int]",
            "default": [256, 128, 64],
        },
        "dropout_rate": {"description": "Dropout rate for MLP layers", "type": "float", "default": 0.2},
        "activation_fn": {"description": "Activation function for MLP", "type": "str", "default": "relu"},
        "pooling_strategy": {"description": "Pooling strategy (adaptive, max)", "type": "str", "default": "adaptive"},
    }

    def __init__(
        self,
        in_channels: int,
        dimension: int = 2,
        hidden_channels: int = 64,
        num_encoder_layers: int = 4,
        encoder_kernel_size: int = 3,
        use_attention: bool = True,
        attention_heads: int = 8,
        mlp_hidden_sizes: List[int] = [256, 128, 64],
        dropout_rate: float = 0.2,
        activation_fn: str = "relu",
        pooling_strategy: str = "adaptive",
    ):

        super().__init__()

        self.in_channels = in_channels
        self.dimension = dimension
        self.hidden_channels = hidden_channels
        self.use_attention = use_attention

        # Field encoder
        self.encoder = FieldEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_encoder_layers,
            kernel_size=encoder_kernel_size,
            pooling=pooling_strategy,
            dimension=dimension,
        )

        # Calculate feature dimension after encoding
        self.feature_dim = hidden_channels

        # Attention module
        if use_attention:
            self.attention = AttentionModule(self.feature_dim, attention_heads)

        # MLP head for scalar prediction
        self.mlp_head = self._build_mlp_head(mlp_hidden_sizes, dropout_rate, activation_fn)

    def _build_mlp_head(self, hidden_sizes: List[int], dropout_rate: float, activation_fn: str) -> nn.Module:
        """Build MLP head for scalar prediction"""
        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU,
            "elu": nn.ELU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }

        layers = []
        input_size = self.feature_dim

        for hidden_size in hidden_sizes:
            layers.extend(
                [nn.Linear(input_size, hidden_size), activation_map[activation_fn](), nn.Dropout(dropout_rate)]
            )
            input_size = hidden_size

        # Final output layer
        layers.append(nn.Linear(input_size, 1))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x: Input field data tensor
               - 1D: (batch, channels, length)
               - 2D: (batch, channels, height, width)
               - 3D: (batch, channels, depth, height, width)

        Returns:
            torch.Tensor: Scalar predictions (batch, 1)
        """
        # Encode field data
        features = self.encoder(x)

        # Apply attention if enabled
        if self.use_attention:
            # Add sequence dimension for attention
            features = features.unsqueeze(1)  # (batch, 1, feature_dim)
            features = self.attention(features)
            features = features.squeeze(1)  # (batch, feature_dim)

        # Predict scalar
        output = self.mlp_head(features)

        return output

    @classmethod
    def get_hyperparameters(cls) -> dict:
        """Get model hyperparameters with descriptions

        Returns:
            Dict[str, Any]: Dictionary of hyperparameters with their descriptions
        """
        return cls.HYPERPARAMETERS
