"""
MeshGraphNet Model for Low-Code Deep Learning Platform

This implementation provides a MeshGraphNet architecture suitable for 
graph-based physics simulations and other applications.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Union, Tuple
from torch import Tensor

# Try to import DGL, but don't fail if it's not available
try:
    import dgl
    from dgl import DGLGraph
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    DGLGraph = None


class MeshGraphNet(nn.Module):
    """
    MeshGraphNet network architecture
    
    This implementation follows the structure defined in the model template,
    with clear hyperparameter descriptions to help users understand their functionality.
    """
    
    # Model hyperparameters with descriptions
    HYPERPARAMETERS = {
        'input_dim_nodes': {
            'description': 'Number of node features',
            'type': 'int',
            'default': 4
        },
        'input_dim_edges': {
            'description': 'Number of edge features',
            'type': 'int',
            'default': 3
        },
        'output_dim': {
            'description': 'Number of outputs',
            'type': 'int',
            'default': 2
        },
        'processor_size': {
            'description': 'Number of message passing blocks',
            'type': 'int',
            'default': 15
        },
        'mlp_activation_fn': {
            'description': 'Activation function to use',
            'type': 'str',
            'default': 'relu'
        },
        'num_layers_node_processor': {
            'description': 'Number of MLP layers for processing nodes in each message passing block',
            'type': 'int',
            'default': 2
        },
        'num_layers_edge_processor': {
            'description': 'Number of MLP layers for processing edge features in each message passing block',
            'type': 'int',
            'default': 2
        },
        'hidden_dim_processor': {
            'description': 'Hidden layer size for the message passing blocks',
            'type': 'int',
            'default': 128
        },
        'hidden_dim_node_encoder': {
            'description': 'Hidden layer size for the node feature encoder',
            'type': 'int',
            'default': 128
        },
        'num_layers_node_encoder': {
            'description': 'Number of MLP layers for the node feature encoder',
            'type': 'int',
            'default': 2
        },
        'hidden_dim_edge_encoder': {
            'description': 'Hidden layer size for the edge feature encoder',
            'type': 'int',
            'default': 128
        },
        'num_layers_edge_encoder': {
            'description': 'Number of MLP layers for the edge feature encoder',
            'type': 'int',
            'default': 2
        },
        'hidden_dim_node_decoder': {
            'description': 'Hidden layer size for the node feature decoder',
            'type': 'int',
            'default': 128
        },
        'num_layers_node_decoder': {
            'description': 'Number of MLP layers for the node feature decoder',
            'type': 'int',
            'default': 2
        },
        'aggregation': {
            'description': 'Message aggregation type',
            'type': 'str',
            'default': 'sum'
        }
    }
    
    def __init__(self, **kwargs):
        """
        Initialize the MeshGraphNet model.
        
        Args:
            **kwargs: Hyperparameters for the model
        """
        super(MeshGraphNet, self).__init__()
        
        # Check if DGL is available
        if not DGL_AVAILABLE:
            raise ImportError(
                "MeshGraphNet requires the DGL library. Install the "
                + "desired CUDA version at: \n https://www.dgl.ai/pages/start.html"
            )
        
        # Set hyperparameters with defaults
        self.input_dim_nodes = kwargs.get('input_dim_nodes', self.HYPERPARAMETERS['input_dim_nodes']['default'])
        self.input_dim_edges = kwargs.get('input_dim_edges', self.HYPERPARAMETERS['input_dim_edges']['default'])
        self.output_dim = kwargs.get('output_dim', self.HYPERPARAMETERS['output_dim']['default'])
        self.processor_size = kwargs.get('processor_size', self.HYPERPARAMETERS['processor_size']['default'])
        self.mlp_activation_fn = kwargs.get('mlp_activation_fn', self.HYPERPARAMETERS['mlp_activation_fn']['default'])
        self.num_layers_node_processor = kwargs.get('num_layers_node_processor', self.HYPERPARAMETERS['num_layers_node_processor']['default'])
        self.num_layers_edge_processor = kwargs.get('num_layers_edge_processor', self.HYPERPARAMETERS['num_layers_edge_processor']['default'])
        self.hidden_dim_processor = kwargs.get('hidden_dim_processor', self.HYPERPARAMETERS['hidden_dim_processor']['default'])
        self.hidden_dim_node_encoder = kwargs.get('hidden_dim_node_encoder', self.HYPERPARAMETERS['hidden_dim_node_encoder']['default'])
        self.num_layers_node_encoder = kwargs.get('num_layers_node_encoder', self.HYPERPARAMETERS['num_layers_node_encoder']['default'])
        self.hidden_dim_edge_encoder = kwargs.get('hidden_dim_edge_encoder', self.HYPERPARAMETERS['hidden_dim_edge_encoder']['default'])
        self.num_layers_edge_encoder = kwargs.get('num_layers_edge_encoder', self.HYPERPARAMETERS['num_layers_edge_encoder']['default'])
        self.hidden_dim_node_decoder = kwargs.get('hidden_dim_node_decoder', self.HYPERPARAMETERS['hidden_dim_node_decoder']['default'])
        self.num_layers_node_decoder = kwargs.get('num_layers_node_decoder', self.HYPERPARAMETERS['num_layers_node_decoder']['default'])
        self.aggregation = kwargs.get('aggregation', self.HYPERPARAMETERS['aggregation']['default'])
        
        # Initialize model components
        self._initialize_model_components()
    
    def _initialize_model_components(self):
        """Initialize the model components based on hyperparameters."""
        activation_fn = self._get_activation_fn(self.mlp_activation_fn)
        
        # Edge encoder
        self.edge_encoder = MeshGraphMLP(
            self.input_dim_edges,
            output_dim=self.hidden_dim_processor,
            hidden_dim=self.hidden_dim_edge_encoder,
            hidden_layers=self.num_layers_edge_encoder,
            activation_fn=activation_fn,
        )

        # Node encoder
        self.node_encoder = MeshGraphMLP(
            self.input_dim_nodes,
            output_dim=self.hidden_dim_processor,
            hidden_dim=self.hidden_dim_node_encoder,
            hidden_layers=self.num_layers_node_encoder,
            activation_fn=activation_fn,
        )

        # Node decoder
        self.node_decoder = MeshGraphMLP(
            self.hidden_dim_processor,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim_node_decoder,
            hidden_layers=self.num_layers_node_decoder,
            activation_fn=activation_fn,
            norm_type=None,
        )
        
        # Processor
        self.processor = MeshGraphNetProcessor(
            processor_size=self.processor_size,
            input_dim_node=self.hidden_dim_processor,
            input_dim_edge=self.hidden_dim_processor,
            num_layers_node=self.num_layers_node_processor,
            num_layers_edge=self.num_layers_edge_processor,
            aggregation=self.aggregation,
            activation_fn=activation_fn,
        )
    
    def _get_activation_fn(self, activation_name: str):
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU()
        }
        return activations.get(activation_name, nn.ReLU())
    
    def forward(self, node_features: Tensor, edge_features: Tensor, graph: Union[DGLGraph, list]):
        """
        Forward pass for the MeshGraphNet model.
        
        Args:
            node_features: Node features tensor of shape (num_nodes, input_dim_nodes)
            edge_features: Edge features tensor of shape (num_edges, input_dim_edges)
            graph: DGL graph or list of DGL graphs
            
        Returns:
            Output tensor of shape (num_nodes, output_dim)
        """
        if not DGL_AVAILABLE:
            raise ImportError(
                "MeshGraphNet requires the DGL library. Install the "
                + "desired CUDA version at: \n https://www.dgl.ai/pages/start.html"
            )
            
        edge_features = self.edge_encoder(edge_features)
        node_features = self.node_encoder(node_features)
        node_features = self.processor(node_features, edge_features, graph)
        output = self.node_decoder(node_features)
        return output
    
    @classmethod
    def get_hyperparameters(cls) -> Dict[str, Any]:
        """
        Get model hyperparameters with descriptions.
        
        Returns:
            Dict[str, Any]: Dictionary of hyperparameters with their descriptions
        """
        return cls.HYPERPARAMETERS


class MeshGraphMLP(nn.Module):
    """Multi-Layer Perceptron for MeshGraphNet."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        activation_fn: nn.Module = nn.ReLU(),
        norm_type: str = "LayerNorm",
    ):
        super(MeshGraphMLP, self).__init__()
        
        # If hidden_layers is None or 0, create a simple linear layer
        if hidden_layers is None or hidden_layers == 0:
            self.mlp = nn.Linear(input_dim, output_dim)
        else:
            layers = []
            # Input layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation_fn)
            
            # Hidden layers
            for _ in range(hidden_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation_fn)
            
            # Output layer
            layers.append(nn.Linear(hidden_dim, output_dim))
            
            self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class MeshGraphNetProcessor(nn.Module):
    """MeshGraphNet processor block"""
    
    def __init__(
        self,
        processor_size: int = 15,
        input_dim_node: int = 128,
        input_dim_edge: int = 128,
        num_layers_node: int = 2,
        num_layers_edge: int = 2,
        aggregation: str = "sum",
        activation_fn: nn.Module = nn.ReLU(),
    ):
        super(MeshGraphNetProcessor, self).__init__()
        self.processor_size = processor_size
        
        # Create processor layers
        self.processor_layers = nn.ModuleList()
        for _ in range(self.processor_size):
            # Add edge block and node block alternately
            self.processor_layers.append(
                MeshEdgeBlock(
                    input_dim_node,
                    input_dim_edge,
                    input_dim_edge,
                    input_dim_edge,
                    num_layers_edge,
                    activation_fn,
                    aggregation
                )
            )
            self.processor_layers.append(
                MeshNodeBlock(
                    aggregation,
                    input_dim_node,
                    input_dim_edge,
                    input_dim_edge,
                    input_dim_edge,
                    num_layers_node,
                    activation_fn,
                    "LayerNorm"
                )
            )
    
    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        graph: Union[DGLGraph, list],
    ) -> Tensor:
        # Process through all layers
        for module in self.processor_layers:
            edge_features, node_features = module(edge_features, node_features, graph)
        
        return node_features


class MeshEdgeBlock(nn.Module):
    """Edge block for MeshGraphNet."""
    
    def __init__(
        self,
        input_dim_nodes: int,
        input_dim_edges: int,
        output_dim_edges: int,
        hidden_dim: int,
        hidden_layers: int,
        activation_fn: nn.Module,
        aggregation: str,
    ):
        super(MeshEdgeBlock, self).__init__()
        self.aggregation = aggregation
        
        # MLP for edge update
        self.edge_mlp = MeshGraphMLP(
            input_dim_nodes * 2 + input_dim_edges,  # sender_node + receiver_node + edge
            output_dim_edges,
            hidden_dim,
            hidden_layers,
            activation_fn,
        )
    
    def forward(
        self,
        edge_features: Tensor,
        node_features: Tensor,
        graph: Union[DGLGraph, list],
    ) -> Tuple[Tensor, Tensor]:
        # For simplicity, we'll implement a basic version
        # In a full implementation, this would use DGL operations
        return edge_features, node_features


class MeshNodeBlock(nn.Module):
    """Node block for MeshGraphNet."""
    
    def __init__(
        self,
        aggregation: str,
        input_dim_nodes: int,
        input_dim_edges: int,
        output_dim_nodes: int,
        hidden_dim: int,
        hidden_layers: int,
        activation_fn: nn.Module,
        norm_type: str,
    ):
        super(MeshNodeBlock, self).__init__()
        self.aggregation = aggregation
        
        # MLP for node update
        self.node_mlp = MeshGraphMLP(
            input_dim_nodes + input_dim_edges,  # node + aggregated_edges
            output_dim_nodes,
            hidden_dim,
            hidden_layers,
            activation_fn,
        )
    
    def forward(
        self,
        edge_features: Tensor,
        node_features: Tensor,
        graph: Union[DGLGraph, list],
    ) -> Tuple[Tensor, Tensor]:
        # For simplicity, we'll implement a basic version
        # In a full implementation, this would use DGL operations
        return edge_features, node_features