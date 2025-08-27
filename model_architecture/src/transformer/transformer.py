#!/usr/bin/env python3
"""
Transformer Model for Low-Code Deep Learning Platform

This implementation provides a flexible Transformer architecture suitable for 
various sequence-to-sequence tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any


class Transformer(nn.Module):
    """
    Full Transformer model for sequence-to-sequence tasks.
    
    This implementation follows the structure defined in the model template,
    with clear hyperparameter descriptions to help users understand their functionality.
    """
    
    # Model hyperparameters with descriptions
    HYPERPARAMETERS = {
        'input_dim': {
            'description': 'Dimension of input features',
            'type': 'int',
            'default': 512
        },
        'output_dim': {
            'description': 'Dimension of output features',
            'type': 'int',
            'default': 512
        },
        'd_model': {
            'description': 'The dimension of the model embeddings',
            'type': 'int',
            'default': 512
        },
        'n_layers': {
            'description': 'Number of encoder/decoder layers',
            'type': 'int',
            'default': 6
        },
        'n_heads': {
            'description': 'Number of attention heads',
            'type': 'int',
            'default': 8
        },
        'pf_dim': {
            'description': 'Position-wise feedforward hidden dimension',
            'type': 'int',
            'default': 2048
        },
        'dropout': {
            'description': 'Dropout rate',
            'type': 'float',
            'default': 0.1
        },
        'max_len': {
            'description': 'Maximum sequence length',
            'type': 'int',
            'default': 100
        }
    }
    
    def __init__(self, **kwargs):
        """
        Initialize the Transformer model.
        
        Args:
            **kwargs: Hyperparameters for the model
        """
        super(Transformer, self).__init__()
        
        # Set hyperparameters with defaults
        self.input_dim = kwargs.get('input_dim', self.HYPERPARAMETERS['input_dim']['default'])
        self.output_dim = kwargs.get('output_dim', self.HYPERPARAMETERS['output_dim']['default'])
        self.d_model = kwargs.get('d_model', self.HYPERPARAMETERS['d_model']['default'])
        self.n_layers = kwargs.get('n_layers', self.HYPERPARAMETERS['n_layers']['default'])
        self.n_heads = kwargs.get('n_heads', self.HYPERPARAMETERS['n_heads']['default'])
        self.pf_dim = kwargs.get('pf_dim', self.HYPERPARAMETERS['pf_dim']['default'])
        self.dropout = kwargs.get('dropout', self.HYPERPARAMETERS['dropout']['default'])
        self.max_len = kwargs.get('max_len', self.HYPERPARAMETERS['max_len']['default'])
        
        # For compatibility with standard training procedures, we'll create a simplified version
        # that works with single input tensors rather than separate source/target tensors
        self.embedding = nn.Linear(self.input_dim, self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model, self.max_len, self.dropout)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(self.d_model, self.n_heads, self.pf_dim, self.dropout)
            for _ in range(self.n_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(self.d_model, self.n_heads, self.pf_dim, self.dropout)
            for _ in range(self.n_layers)
        ])
        self.fc_out = nn.Linear(self.d_model, self.output_dim)
        self.dropout_layer = nn.Dropout(self.dropout)
    
    def forward(self, x):
        """
        Forward pass for the transformer model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        # Embedding and positional encoding
        x = self.embedding(x)
        x = self.pos_encoding(x.permute(1, 0, 2)).permute(1, 0, 2)
        
        # Apply encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Apply decoder layers
        for layer in self.decoder_layers:
            x = layer(x)
        
        # Final linear layer
        x = self.fc_out(x)
        return x
    
    @classmethod
    def get_hyperparameters(cls) -> Dict[str, Any]:
        """
        Get model hyperparameters with descriptions.
        
        Returns:
            Dict[str, Any]: Dictionary of hyperparameters with their descriptions
        """
        return cls.HYPERPARAMETERS


class PositionalEncoding(nn.Module):
    """
    Positional Encoding module for Transformer models.
    
    This module adds positional information to the input embeddings.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize the PositionalEncoding module.
        
        Args:
            d_model: The dimension of the model embeddings
            max_len: The maximum sequence length
            dropout: Dropout rate
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register as buffer (not a parameter, but part of the model state)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for positional encoding.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            Output tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module for Transformer models.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize the MultiHeadAttention module.
        
        Args:
            d_model: The dimension of the model
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for multi-head attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key: Key tensor of shape (batch_size, seq_len, d_model)
            value: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor
            
        Returns:
            Output tensor
        """
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Split into heads
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        
        # Scaled dot-product attention
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(Q.device)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = F.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        x = torch.matmul(attention, V)
        
        # Concatenate heads
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.d_model)
        
        # Output linear transformation
        x = self.W_o(x)
        
        return x


class PositionwiseFeedforward(nn.Module):
    """
    Position-wise Feedforward module for Transformer models.
    """
    
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        """
        Initialize the PositionwiseFeedforward module.
        
        Args:
            d_model: The dimension of the model
            hidden_dim: Hidden dimension of the feedforward network
            dropout: Dropout rate
        """
        super(PositionwiseFeedforward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for position-wise feedforward.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Single layer of the Transformer Encoder.
    """
    
    def __init__(self, d_model: int, n_heads: int, pf_dim: int, dropout: float = 0.1):
        """
        Initialize the TransformerEncoderLayer.
        
        Args:
            d_model: The dimension of the model
            n_heads: Number of attention heads
            pf_dim: Position-wise feedforward hidden dimension
            dropout: Dropout rate
        """
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedforward(d_model, pf_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for transformer encoder layer.
        
        Args:
            src: Source sequence tensor
            
        Returns:
            Output tensor
        """
        # Self-attention
        attn_output = self.self_attn(src, src, src)
        src = self.norm1(src + self.dropout(attn_output))
        
        # Position-wise feedforward
        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout(ff_output))
        
        return src


class TransformerDecoderLayer(nn.Module):
    """
    Single layer of the Transformer Decoder.
    """
    
    def __init__(self, d_model: int, n_heads: int, pf_dim: int, dropout: float = 0.1):
        """
        Initialize the TransformerDecoderLayer.
        
        Args:
            d_model: The dimension of the model
            n_heads: Number of attention heads
            pf_dim: Position-wise feedforward hidden dimension
            dropout: Dropout rate
        """
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.encoder_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedforward(d_model, pf_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for transformer decoder layer.
        
        Args:
            tgt: Target sequence tensor
            
        Returns:
            Output tensor
        """
        # Self-attention
        attn_output = self.self_attn(tgt, tgt, tgt)
        tgt = self.norm1(tgt + self.dropout(attn_output))
        
        # Encoder-decoder attention - simplified for our use case
        attn_output = self.encoder_attn(tgt, tgt, tgt)
        tgt = self.norm2(tgt + self.dropout(attn_output))
        
        # Position-wise feedforward
        ff_output = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout(ff_output))
        
        return tgt
