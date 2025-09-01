import torch

def preprocess_fn(data):
    """
    Preprocessing for tabular data (b,h) format
    
    Args:
        data: Dict of tensors or single tensor
    Returns:
        Processed tensor for MLP input
    """
    if isinstance(data, dict):
        # Concatenate all features
        features = []
        for key in sorted(data.keys()):
            tensor = data[key]
            if len(tensor.shape) == 1:
                tensor = tensor.unsqueeze(-1)
            features.append(tensor)
        return torch.cat(features, dim=-1)
    else:
        # Single tensor input
        if len(data.shape) == 1:
            return data.unsqueeze(-1)
        return data