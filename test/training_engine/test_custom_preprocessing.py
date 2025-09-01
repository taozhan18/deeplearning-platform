"""
Custom preprocessing functions for testing Python file path loading
"""

import torch


def preprocess_fn(data):
    """
    Simple concatenation preprocessing function for testing
    """
    if isinstance(data, dict):
        return torch.cat([data[key] for key in sorted(data.keys())], dim=1)
    else:
        return data


def weighted_sum(data):
    """
    Weighted sum preprocessing function for testing
    """
    if isinstance(data, dict):
        # Simple weighted sum with equal weights
        result = None
        weight = 1.0 / len(data)
        
        for tensor in data.values():
            if result is None:
                result = weight * tensor
            else:
                result += weight * tensor
        
        return result
    else:
        return data


def select_first(data):
    """
    Select the first source for testing
    """
    if isinstance(data, dict):
        return data[list(data.keys())[0]]
    else:
        return data


def custom_normalize(data):
    """
    Custom normalization function for testing
    """
    if isinstance(data, dict):
        normalized = {}
        for key, tensor in data.items():
            normalized[key] = (tensor - tensor.mean()) / (tensor.std() + 1e-8)
        return normalized
    else:
        return (data - data.mean()) / (data.std() + 1e-8)