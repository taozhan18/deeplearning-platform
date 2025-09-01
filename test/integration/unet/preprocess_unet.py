import torch

def preprocess_fn(data):
    """
    Preprocessing for UNet grid data (b,c,h,w) format
    
    Args:
        data: Dict of tensors or single tensor
    Returns:
        Processed tensor for UNet input: (b, c, h, w)
    """
    if isinstance(data, dict):
        # Handle multi-channel grid data
        channels = []
        for key in sorted(data.keys()):
            tensor = data[key]
            # Ensure proper shape for UNet: (b, c, h, w)
            if len(tensor.shape) == 3:  # (b, h, w)
                tensor = tensor.unsqueeze(1)
            elif len(tensor.shape) == 2:  # (h, w) - assume single sample
                tensor = tensor.unsqueeze(0).unsqueeze(0)
            elif len(tensor.shape) == 4:  # (b, c, h, w)
                pass  # Already correct
            else:
                raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
            channels.append(tensor)
        
        if len(channels) == 1:
            return channels[0]
        else:
            return torch.cat(channels, dim=1)
    else:
        # Single tensor input
        if len(data.shape) == 3:  # (b, h, w)
            return data.unsqueeze(1)
        elif len(data.shape) == 2:  # (h, w)
            return data.unsqueeze(0).unsqueeze(0)
        elif len(data.shape) == 4:  # (b, c, h, w)
            return data
        else:
            raise ValueError(f"Unexpected tensor shape: {data.shape}")