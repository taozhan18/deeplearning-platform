import torch
import numpy as np

def preprocess_fn(data):
    """
    Preprocessing for MeshGraphNet graph structure data
    
    Args:
        data: Dict containing node features, edge indices, and edge features
    Returns:
        Processed data dict for MeshGraphNet
    """
    if isinstance(data, dict):
        result = {}
        
        # Handle node features
        if 'node_features' in data:
            node_features = data['node_features']
            # Ensure proper shape: (num_nodes, feature_dim)
            if len(node_features.shape) == 1:
                node_features = node_features.unsqueeze(-1)
            elif len(node_features.shape) == 3:  # (b, num_nodes, feature_dim)
                # Flatten batch dimension for message passing
                batch_size, num_nodes, feature_dim = node_features.shape
                node_features = node_features.view(-1, feature_dim)
            result['node_features'] = node_features
        
        # Handle edge connectivity
        if 'edge_index' in data:
            edge_index = data['edge_index']
            # Ensure proper shape: (2, num_edges)
            if len(edge_index.shape) == 2 and edge_index.shape[0] == 2:
                result['edge_index'] = edge_index.long()
            elif len(edge_index.shape) == 3 and edge_index.shape[0] == 2:
                # Handle batched edge indices
                result['edge_index'] = edge_index.long()
        
        # Handle edge features
        if 'edge_features' in data:
            edge_features = data['edge_features']
            # Ensure proper shape: (num_edges, edge_dim)
            if len(edge_features.shape) == 1:
                edge_features = edge_features.unsqueeze(-1)
            elif len(edge_features.shape) == 2:
                pass  # Already correct
            elif len(edge_features.shape) == 3:  # (b, num_edges, edge_dim)
                # Flatten batch dimension
                batch_size, num_edges, edge_dim = edge_features.shape
                edge_features = edge_features.view(-1, edge_dim)
            result['edge_features'] = edge_features
        
        # Handle batch information
        if 'batch' in data:
            result['batch'] = data['batch'].long()
        
        # Handle targets
        if 'targets' in data:
            targets = data['targets']
            if len(targets.shape) == 1:
                # Node-level targets
                result['targets'] = targets
            elif len(targets.shape) == 2:
                # Node-level targets with feature dimension
                result['targets'] = targets
            elif len(targets.shape) == 3:
                # Flatten batch dimension for node-level targets
                batch_size, num_nodes, output_dim = targets.shape
                result['targets'] = targets.view(-1, output_dim)
        
        return result
    else:
        # Single tensor input - treat as node features
        if isinstance(data, torch.Tensor):
            if len(data.shape) == 2:  # (num_nodes, feature_dim)
                return {'node_features': data}
            elif len(data.shape) == 1:  # (num_nodes,)
                return {'node_features': data.unsqueeze(-1)}
            else:
                raise ValueError(f"Unexpected tensor shape for graph data: {data.shape}")
        else:
            return data