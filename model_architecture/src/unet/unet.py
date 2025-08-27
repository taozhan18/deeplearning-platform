"""
UNet implementation for the low-code deep learning platform
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union


class ConvBlock(nn.Module):
    """
    A convolutional block with optional normalization and activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple] = 3,
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 1,
        activation: str = "relu",
        normalization: str = "batchnorm",
    ):
        super().__init__()
        
        # Convolution layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False  # Bias is handled by normalization
        )
        
        # Normalization layer
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm2d(out_channels)
        elif normalization == "groupnorm":
            self.norm = nn.GroupNorm(min(32, out_channels), out_channels)
        else:
            self.norm = nn.Identity()
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class DoubleConv(nn.Module):
    """
    Double convolution block: conv -> norm -> activation -> conv -> norm -> activation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple] = 3,
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 1,
        activation: str = "relu",
        normalization: str = "batchnorm",
    ):
        super().__init__()
        self.double_conv = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                activation=activation,
                normalization=normalization,
            ),
            ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                activation=activation,
                normalization=normalization,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple] = 3,
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 1,
        activation: str = "relu",
        normalization: str = "batchnorm",
    ):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                activation=activation,
                normalization=normalization,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling then double conv
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple] = 3,
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 1,
        activation: str = "relu",
        normalization: str = "batchnorm",
        bilinear: bool = True,
    ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                activation=activation,
                normalization=normalization,
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                activation=activation,
                normalization=normalization,
            )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Output convolution layer
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net model with encoder-decoder architecture and skip connections.
    
    Parameters
    ----------
    in_channels : int, optional
        Number of channels in the input image, by default 3
    out_channels : int, optional
        Number of channels in the output segmentation map, by default 1
    features : List[int], optional
        Number of feature channels at each level, by default [64, 128, 256, 512, 1024]
    activation : str, optional
        Type of activation to use, by default 'relu'
    normalization : str, optional
        Type of normalization to use, by default 'batchnorm'
    bilinear : bool, optional
        Whether to use bilinear upsampling or transposed convolutions, by default True
    """

    # Model hyperparameters with descriptions
    HYPERPARAMETERS = {
        'in_channels': {
            'description': 'Number of channels in the input image',
            'type': 'int',
            'default': 3
        },
        'out_channels': {
            'description': 'Number of channels in the output segmentation map',
            'type': 'int',
            'default': 1
        },
        'features': {
            'description': 'Number of feature channels at each level',
            'type': 'List[int]',
            'default': [64, 128, 256, 512]
        },
        'activation': {
            'description': 'Type of activation to use',
            'type': 'str',
            'default': 'relu'
        },
        'normalization': {
            'description': 'Type of normalization to use',
            'type': 'str',
            'default': 'batchnorm'
        },
        'bilinear': {
            'description': 'Whether to use bilinear upsampling or transposed convolutions',
            'type': 'bool',
            'default': True
        }
    }

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: List[int] = [64, 128, 256, 512],
        activation: str = "relu",
        normalization: str = "batchnorm",
        bilinear: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.bilinear = bilinear

        # Initial convolution
        self.inc = DoubleConv(
            in_channels=in_channels,
            out_channels=features[0],
            activation=activation,
            normalization=normalization,
        )
        
        # Encoder path
        self.down_layers = nn.ModuleList()
        for i in range(len(features) - 1):
            self.down_layers.append(
                Down(
                    in_channels=features[i],
                    out_channels=features[i + 1],
                    activation=activation,
                    normalization=normalization,
                )
            )
        
        # Decoder path
        self.up_layers = nn.ModuleList()
        rev_features = list(reversed(features))
        for i in range(len(features) - 1):
            self.up_layers.append(
                Up(
                    in_channels=rev_features[i] + rev_features[i + 1],  # Concatenated channels
                    out_channels=rev_features[i + 1],
                    activation=activation,
                    normalization=normalization,
                    bilinear=bilinear,
                )
            )
        
        # Output layer
        self.outc = OutConv(features[0], out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial convolution
        x1 = self.inc(x)
        
        # Encoder
        xs = [x1]
        for down_layer in self.down_layers:
            x_new = down_layer(xs[-1])
            xs.append(x_new)
        
        # Decoder with skip connections
        x = xs[-1]  # Start with the deepest features
        xs = list(reversed(xs[:-1]))  # Skip connections (excluding the deepest)
        
        for i, up_layer in enumerate(self.up_layers):
            x = up_layer(x, xs[i])
        
        # Output
        logits = self.outc(x)
        return logits

    @classmethod
    def get_hyperparameters(cls) -> dict:
        """Get model hyperparameters with descriptions.

        Returns:
            Dict[str, Any]: Dictionary of hyperparameters with their descriptions
        """
        return cls.HYPERPARAMETERS