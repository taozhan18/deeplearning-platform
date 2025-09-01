import torch

def preprocess_fn(data):
    """
    Preprocessing for Transolver sequence data (b, nt, n) format
    
    Args:
        data: Dict of tensors or single tensor
    Returns:
        Processed tensor for Transolver input: (b, nt, n)
    """
    if isinstance(data, dict):
        # Handle multi-source sequence data
        sequences = []
        for key in sorted(data.keys()):
            tensor = data[key]
            # Ensure proper shape for Transolver: (b, nt, n)
            if len(tensor.shape) == 2:  # (b, nt) - single feature
                tensor = tensor.unsqueeze(-1)
            elif len(tensor.shape) == 1:  # (nt,) - single sequence
                tensor = tensor.unsqueeze(0).unsqueeze(-1)
            elif len(tensor.shape) == 3:  # (b, nt, n) - already correct
                pass
            else:
                raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
            sequences.append(tensor)
        
        if len(sequences) == 1:
            return sequences[0]
        else:
            # Concatenate along feature dimension
            return torch.cat(sequences, dim=-1)
    else:
        # Single tensor input
        if len(data.shape) == 2:  # (b, nt)
            return data.unsqueeze(-1)
        elif len(data.shape) == 1:  # (nt,)
            return data.unsqueeze(0).unsqueeze(-1)
        elif len(data.shape) == 3:  # (b, nt, n)
            return data
        else:
            raise ValueError(f"Unexpected tensor shape: {data.shape}")