"""
Fourier Neural Operator (FNO) implementation for the low-code deep learning platform
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union
from .spectral_layers import SpectralConv1d, SpectralConv2d


class FNO1DEncoder(nn.Module):
    """1D Spectral encoder for FNO

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels, by default 1
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    fno_layer_size : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding :  Union[int, List[int]], optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : nn.Module, optional
        Activation function, by default nn.GELU
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_fno_layers: int = 4,
        fno_layer_size: int = 32,
        num_fno_modes: Union[int, List[int]] = 16,
        padding: Union[int, List[int]] = 8,
        padding_type: str = "constant",
        activation_fn: nn.Module = nn.GELU(),
        coord_features: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        self.activation_fn = activation_fn

        # Add relative coordinate feature
        self.coord_features = coord_features
        if self.coord_features:
            self.in_channels = self.in_channels + 1

        # Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding]
        self.pad = padding[:1]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = padding_type

        if isinstance(num_fno_modes, int):
            num_fno_modes = [num_fno_modes]

        # Build lift network
        self.build_lift_network()
        self.build_fno(num_fno_modes)

    def build_lift_network(self) -> None:
        """Construct network for lifting variables to latent space."""
        self.lift_network = torch.nn.Sequential()
        self.lift_network.append(
            nn.Conv1d(self.in_channels, int(self.fno_width / 2), 1)
        )
        self.lift_network.append(self.activation_fn)
        self.lift_network.append(
            nn.Conv1d(int(self.fno_width / 2), self.fno_width, 1)
        )

    def build_fno(self, num_fno_modes: List[int]) -> None:
        """Construct FNO block.

        Parameters
        ----------
        num_fno_modes : List[int]
            Number of Fourier modes kept in spectral convolutions
        """
        # Build Neural Fourier Operators
        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_fno_layers):
            self.spconv_layers.append(
                SpectralConv1d(self.fno_width, self.fno_width, num_fno_modes[0])
            )
            self.conv_layers.append(nn.Conv1d(self.fno_width, self.fno_width, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_network(x)
        # (left, right)
        x = F.pad(x, (0, self.pad[0]), mode=self.padding_type)
        # Spectral layers
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(conv(x) + w(x))
            else:
                x = conv(x) + w(x)

        x = x[..., : self.ipad[0]]
        return x

    def meshgrid(self, shape: List[int], device: torch.device) -> torch.Tensor:
        """Creates 1D meshgrid feature

        Parameters
        ----------
        shape : List[int]
            Tensor shape
        device : torch.device
            Device model is on

        Returns
        -------
        torch.Tensor
            Meshgrid tensor
        """
        bsize, size_x = shape[0], shape[2]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1)
        return grid_x


class FNO2DEncoder(nn.Module):
    """2D Spectral encoder for FNO

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels, by default 1
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    fno_layer_size : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding :  Union[int, List[int]], optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : nn.Module, optional
        Activation function, by default nn.GELU
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_fno_layers: int = 4,
        fno_layer_size: int = 32,
        num_fno_modes: Union[int, List[int]] = 16,
        padding: Union[int, List[int]] = 8,
        padding_type: str = "constant",
        activation_fn: nn.Module = nn.GELU(),
        coord_features: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        self.coord_features = coord_features
        self.activation_fn = activation_fn

        # Add relative coordinate feature
        if self.coord_features:
            self.in_channels = self.in_channels + 2

        # Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding, padding]
        padding = padding + [0, 0]  # Pad with zeros for smaller lists
        self.pad = padding[:2]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = padding_type

        if isinstance(num_fno_modes, int):
            num_fno_modes = [num_fno_modes, num_fno_modes]

        # Build lift network
        self.build_lift_network()
        self.build_fno(num_fno_modes)

    def build_lift_network(self) -> None:
        """Construct network for lifting variables to latent space."""
        self.lift_network = torch.nn.Sequential()
        self.lift_network.append(
            nn.Conv2d(self.in_channels, int(self.fno_width / 2), 1)
        )
        self.lift_network.append(self.activation_fn)
        self.lift_network.append(
            nn.Conv2d(int(self.fno_width / 2), self.fno_width, 1)
        )

    def build_fno(self, num_fno_modes: List[int]) -> None:
        """Construct FNO block.

        Parameters
        ----------
        num_fno_modes : List[int]
            Number of Fourier modes kept in spectral convolutions
        """
        # Build Neural Fourier Operators
        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_fno_layers):
            self.spconv_layers.append(
                SpectralConv2d(
                    self.fno_width, self.fno_width, num_fno_modes[0], num_fno_modes[1]
                )
            )
            self.conv_layers.append(nn.Conv2d(self.fno_width, self.fno_width, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                "Only 4D tensors [batch, in_channels, grid_x, grid_y] accepted for 2D FNO"
            )

        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_network(x)
        # (left, right, top, bottom)
        x = F.pad(x, (0, self.pad[1], 0, self.pad[0]), mode=self.padding_type)
        # Spectral layers
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(conv(x) + w(x))
            else:
                x = conv(x) + w(x)

        # remove padding
        x = x[..., : self.ipad[0], : self.ipad[1]]

        return x

    def meshgrid(self, shape: List[int], device: torch.device) -> torch.Tensor:
        """Creates 2D meshgrid feature

        Parameters
        ----------
        shape : List[int]
            Tensor shape
        device : torch.device
            Device model is on

        Returns
        -------
        torch.Tensor
            Meshgrid tensor
        """
        bsize, size_x, size_y = shape[0], shape[2], shape[3]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing="ij")
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1)
        return torch.cat((grid_x, grid_y), dim=1)


class FNO(nn.Module):
    """Fourier Neural Operator (FNO) model for the low-code platform.

    The FNO architecture supports options for 1D and 2D fields which can
    be controlled using the `dimension` parameter.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    decoder_layers : int, optional
        Number of decoder layers, by default 1
    decoder_layer_size : int, optional
        Number of neurons in decoder layers, by default 32
    decoder_activation_fn : nn.Module, optional
        Activation function for decoder, by default nn.SiLU
    dimension : int
        Model dimensionality (supports 1, 2).
    latent_channels : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding : int, optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : nn.Module, optional
        Activation function, by default nn.GELU
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    """

    # Model hyperparameters with descriptions
    HYPERPARAMETERS = {
        'in_channels': {
            'description': 'Number of input channels',
            'type': 'int',
            'default': 1
        },
        'out_channels': {
            'description': 'Number of output channels',
            'type': 'int',
            'default': 1
        },
        'decoder_layers': {
            'description': 'Number of decoder layers',
            'type': 'int',
            'default': 1
        },
        'decoder_layer_size': {
            'description': 'Number of neurons in decoder layers',
            'type': 'int',
            'default': 32
        },
        'dimension': {
            'description': 'Model dimensionality (supports 1, 2)',
            'type': 'int',
            'default': 2
        },
        'latent_channels': {
            'description': 'Latent features size in spectral convolutions',
            'type': 'int',
            'default': 32
        },
        'num_fno_layers': {
            'description': 'Number of spectral convolutional layers',
            'type': 'int',
            'default': 4
        },
        'num_fno_modes': {
            'description': 'Number of Fourier modes kept in spectral convolutions',
            'type': 'Union[int, List[int]]',
            'default': 16
        },
        'padding': {
            'description': 'Domain padding for spectral convolutions',
            'type': 'int',
            'default': 8
        },
        'padding_type': {
            'description': 'Type of padding for spectral convolutions',
            'type': 'str',
            'default': 'constant'
        },
        'coord_features': {
            'description': 'Use coordinate grid as additional feature map',
            'type': 'bool',
            'default': True
        }
    }

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        decoder_layers: int = 1,
        decoder_layer_size: int = 32,
        decoder_activation_fn: nn.Module = nn.SiLU(),
        dimension: int = 2,
        latent_channels: int = 32,
        num_fno_layers: int = 4,
        num_fno_modes: Union[int, List[int]] = 16,
        padding: int = 8,
        padding_type: str = "constant",
        activation_fn: nn.Module = nn.GELU(),
        coord_features: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.decoder_layers = decoder_layers
        self.decoder_layer_size = decoder_layer_size
        self.dimension = dimension
        self.latent_channels = latent_channels
        self.num_fno_layers = num_fno_layers
        self.num_fno_modes = num_fno_modes
        self.padding = padding
        self.padding_type = padding_type
        self.coord_features = coord_features

        # Decoder network
        self.decoder_net = nn.Sequential()
        self.decoder_net.append(nn.Linear(latent_channels, decoder_layer_size))
        self.decoder_net.append(decoder_activation_fn)
        
        for _ in range(decoder_layers - 1):
            self.decoder_net.append(nn.Linear(decoder_layer_size, decoder_layer_size))
            self.decoder_net.append(decoder_activation_fn)
            
        self.decoder_net.append(nn.Linear(decoder_layer_size, out_channels))

        # Select FNO encoder based on dimension
        FNOModel = self.get_fno_encoder()

        self.spec_encoder = FNOModel(
            in_channels,
            num_fno_layers=self.num_fno_layers,
            fno_layer_size=latent_channels,
            num_fno_modes=self.num_fno_modes,
            padding=self.padding,
            padding_type=self.padding_type,
            activation_fn=activation_fn,
            coord_features=self.coord_features,
        )

    def get_fno_encoder(self):
        """Get FNO encoder based on dimension"""
        if self.dimension == 1:
            return FNO1DEncoder
        elif self.dimension == 2:
            return FNO2DEncoder
        else:
            raise NotImplementedError(
                "Invalid dimensionality. Only 1D and 2D FNO implemented"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fourier encoder
        y_latent = self.spec_encoder(x)

        # Reshape to pointwise inputs
        y_shape = y_latent.shape
        if self.dimension == 1:
            # For 1D: (batch, channels, x) -> (batch*x, channels)
            y_latent = y_latent.permute(0, 2, 1).reshape(-1, y_latent.shape[1])
        elif self.dimension == 2:
            # For 2D: (batch, channels, x, y) -> (batch*x*y, channels)
            y_latent = y_latent.permute(0, 2, 3, 1).reshape(-1, y_latent.shape[1])

        # Decoder
        y = self.decoder_net(y_latent)

        # Convert back into grid
        if self.dimension == 1:
            # For 1D: (batch*x, out_channels) -> (batch, out_channels, x)
            y = y.reshape(y_shape[0], y_shape[2], y.shape[1]).permute(0, 2, 1)
        elif self.dimension == 2:
            # For 2D: (batch*x*y, out_channels) -> (batch, out_channels, x, y)
            y = y.reshape(y_shape[0], y_shape[2], y_shape[3], y.shape[1]).permute(0, 3, 1, 2)

        return y

    @classmethod
    def get_hyperparameters(cls) -> dict:
        """Get model hyperparameters with descriptions.

        Returns:
            Dict[str, Any]: Dictionary of hyperparameters with their descriptions
        """
        return cls.HYPERPARAMETERS