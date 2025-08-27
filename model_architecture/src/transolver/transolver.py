"""
Transolver Model for Low-Code Deep Learning Platform

This implementation provides a physics-informed transformer architecture suitable for 
solving Partial Differential Equations (PDEs).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
from timm.layers import trunc_normal_

from .Embedding import timestep_embedding
from .Physics_Attention import Physics_Attention_Structured_Mesh_2D


class Transolver(nn.Module):
    """
    Transolver model for PDE solving.
    
    This implementation follows the structure defined in the model template,
    with clear hyperparameter descriptions to help users understand their functionality.
    """
    
    # Model hyperparameters with descriptions
    HYPERPARAMETERS = {
        'space_dim': {
            'description': 'The spatial dimension of the input data',
            'type': 'int',
            'default': 1
        },
        'n_layers': {
            'description': 'The number of transformer layers',
            'type': 'int',
            'default': 5
        },
        'n_hidden': {
            'description': 'The hidden dimension of the transformer',
            'type': 'int',
            'default': 256
        },
        'dropout': {
            'description': 'The dropout rate',
            'type': 'float',
            'default': 0.0
        },
        'n_head': {
            'description': 'The number of attention heads',
            'type': 'int',
            'default': 8
        },
        'Time_Input': {
            'description': 'Whether to include time embeddings',
            'type': 'bool',
            'default': False
        },
        'act': {
            'description': 'The activation function',
            'type': 'str',
            'default': 'gelu'
        },
        'mlp_ratio': {
            'description': 'The ratio of hidden dimension in the MLP',
            'type': 'int',
            'default': 1
        },
        'fun_dim': {
            'description': 'The dimension of the function',
            'type': 'int',
            'default': 1
        },
        'out_dim': {
            'description': 'The output dimension',
            'type': 'int',
            'default': 1
        },
        'slice_num': {
            'description': 'The number of slices in the structured attention',
            'type': 'int',
            'default': 32
        },
        'ref': {
            'description': 'The reference dimension',
            'type': 'int',
            'default': 8
        },
        'unified_pos': {
            'description': 'Whether to use unified positional embeddings',
            'type': 'bool',
            'default': False
        },
        'H': {
            'description': 'The height of the mesh',
            'type': 'int',
            'default': 85
        },
        'W': {
            'description': 'The width of the mesh',
            'type': 'int',
            'default': 85
        }
    }
    
    def __init__(self, **kwargs):
        """
        Initialize the Transolver model.
        
        Args:
            **kwargs: Hyperparameters for the model
        """
        super(Transolver, self).__init__()
        
        # Set hyperparameters with defaults
        self.space_dim = kwargs.get('space_dim', self.HYPERPARAMETERS['space_dim']['default'])
        self.n_layers = kwargs.get('n_layers', self.HYPERPARAMETERS['n_layers']['default'])
        self.n_hidden = kwargs.get('n_hidden', self.HYPERPARAMETERS['n_hidden']['default'])
        self.dropout = kwargs.get('dropout', self.HYPERPARAMETERS['dropout']['default'])
        self.n_head = kwargs.get('n_head', self.HYPERPARAMETERS['n_head']['default'])
        self.Time_Input = kwargs.get('Time_Input', self.HYPERPARAMETERS['Time_Input']['default'])
        self.act = kwargs.get('act', self.HYPERPARAMETERS['act']['default'])
        self.mlp_ratio = kwargs.get('mlp_ratio', self.HYPERPARAMETERS['mlp_ratio']['default'])
        self.fun_dim = kwargs.get('fun_dim', self.HYPERPARAMETERS['fun_dim']['default'])
        self.out_dim = kwargs.get('out_dim', self.HYPERPARAMETERS['out_dim']['default'])
        self.slice_num = kwargs.get('slice_num', self.HYPERPARAMETERS['slice_num']['default'])
        self.ref = kwargs.get('ref', self.HYPERPARAMETERS['ref']['default'])
        self.unified_pos = kwargs.get('unified_pos', self.HYPERPARAMETERS['unified_pos']['default'])
        self.H = kwargs.get('H', self.HYPERPARAMETERS['H']['default'])
        self.W = kwargs.get('W', self.HYPERPARAMETERS['W']['default'])
        
        # Initialize model components
        self.__name__ = "Transolver_2D"
        self._initialize_model_components()
    
    def _initialize_model_components(self):
        """Initialize the model components based on hyperparameters."""
        if self.unified_pos:
            self.pos = self.get_grid()
            # Adjust preprocess layer for unified position case
            self.preprocess = MLP(
                self.fun_dim + self.ref * self.ref,
                self.n_hidden * 2,
                self.n_hidden,
                n_layers=0,
                res=False,
                act=self.act,
            )
        else:
            # Fix the input dimension for the preprocess layer
            self.preprocess = MLP(
                self.space_dim + self.fun_dim,  # Space dimension + function dimension
                self.n_hidden * 2,
                self.n_hidden,
                n_layers=0,
                res=False,
                act=self.act,
            )
            
            # Add a separate preprocess layer for when fx is not provided
            self.preprocess_no_fx = MLP(
                self.space_dim,  # Only space dimension
                self.n_hidden * 2,
                self.n_hidden,
                n_layers=0,
                res=False,
                act=self.act,
            )

        self.n_hidden = self.n_hidden
        self.space_dim = self.space_dim
        if self.Time_Input:
            self.time_fc = nn.Sequential(
                nn.Linear(self.n_hidden, self.n_hidden), nn.SiLU(), nn.Linear(self.n_hidden, self.n_hidden)
            )

        self.blocks = nn.ModuleList(
            [
                Transolver_block(
                    num_heads=self.n_head,
                    hidden_dim=self.n_hidden,
                    dropout=self.dropout,
                    act=self.act,
                    mlp_ratio=self.mlp_ratio,
                    out_dim=self.out_dim,
                    slice_num=self.slice_num,
                    H=self.H,
                    W=self.W,
                    last_layer=(_ == self.n_layers - 1),
                )
                for _ in range(self.n_layers)
            ]
        )
        self.initialize_weights()
        self.placeholder = nn.Parameter(
            (1 / (self.n_hidden)) * torch.rand(self.n_hidden, dtype=torch.float)
        )
    
    def initialize_weights(self):
        """Initialize model weights."""
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights for a module."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, batchsize=1):
        """Generate grid for positional encoding."""
        size_x, size_y = self.H, self.W
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1)  # B H W 2

        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1)  # B H W 8 8 2

        pos = (
            torch.sqrt(
                torch.sum(
                    (grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :])
                    ** 2,
                    dim=-1,
                )
            )
            .reshape(batchsize, size_x, size_y, self.ref * self.ref)
            .contiguous()
        )
        return pos

    def forward(self, x, fx=None, T=None):
        """
        Forward pass for the transolver model.
        
        Args:
            x: Input tensor of shape (batch_size, N, space_dim)
            fx: Function values tensor of shape (batch_size, N, fun_dim) (optional)
            T: Time tensor (optional)
            
        Returns:
            Output tensor of shape (batch_size, N, out_dim)
        """
        if self.unified_pos:
            x = (
                self.pos.repeat(x.shape[0], 1, 1, 1)
                .reshape(x.shape[0], self.H * self.W, self.ref * self.ref)
                .to(x.device)
            )
        
        # Process input with function values if provided
        if fx is not None:
            # Concatenate spatial coordinates with function values
            input_features = torch.cat((x, fx), dim=-1)
            fx = self.preprocess(input_features)
        else:
            # Use the separate preprocess layer for when fx is not provided
            fx = self.preprocess_no_fx(x)
            fx = fx + self.placeholder[None, None, :]

        if T is not None:
            Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        for block in self.blocks:
            fx = block(fx)

        return fx
    
    @classmethod
    def get_hyperparameters(cls) -> Dict[str, Any]:
        """
        Get model hyperparameters with descriptions.
        
        Returns:
            Dict[str, Any]: Dictionary of hyperparameters with their descriptions
        """
        return cls.HYPERPARAMETERS


class MLP(nn.Module):
    """Multi-Layer Perceptron module."""
    
    ACTIVATION = {
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU(0.1),
        "softplus": nn.Softplus,
        "ELU": nn.ELU,
        "silu": nn.SiLU,
    }

    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True):
        super(MLP, self).__init__()

        if act in self.ACTIVATION.keys():
            act = self.ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(n_hidden, n_hidden), act())
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_block(nn.Module):
    """Transformer encoder block for Transolver."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=32,
        H=85,
        W=85,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Structured_Mesh_2D(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
            H=H,
            W=W,
        )

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(
            hidden_dim,
            hidden_dim * mlp_ratio,
            hidden_dim,
            n_layers=0,
            res=False,
            act=act,
        )
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx