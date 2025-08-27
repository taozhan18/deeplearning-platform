"""
Script to list all available models and their parameters in the low-code deep learning platform
"""

import sys
import os

# Add the modules to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'model_architecture', 'src'))

from model_manager import ModelManager


def list_available_models():
    """List all available complete models in the platform"""
    print("Complete Neural Network Models in the Low-Code Deep Learning Platform")
    print("=" * 70)
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Get available models (complete neural network architectures)
    available_models = model_manager.get_available_models()
    
    print(f"Total complete models available: {len(available_models)}")
    print("\nAvailable complete models:")
    for i, model_name in enumerate(available_models, 1):
        print(f"  {i}. {model_name}")
    
    return model_manager, available_models


def list_available_layers():
    """List all available layers and functions in the platform"""
    print("\n\nAvailable Layers and Functions")
    print("=" * 40)
    
    # Initialize model manager
    model_manager = ModelManager({})
    
    # Get available layers
    available_layers = model_manager.get_available_layers()
    
    print(f"Total layers and functions available: {len(available_layers)}")
    print("\nAvailable layers and functions:")
    for i, layer_name in enumerate(available_layers, 1):
        print(f"  {i}. {layer_name}")
    
    return model_manager, available_layers


def show_model_parameters(model_manager, model_name):
    """Show parameters for a specific complete model"""
    print(f"\n\nParameters for '{model_name}' model:")
    print("-" * 40)
    
    try:
        # Get model hyperparameters
        hyperparams = model_manager.get_model_hyperparameters(model_name)
        
        if hyperparams:
            for param_name, param_info in hyperparams.items():
                description = param_info.get('description', 'No description available')
                param_type = param_info.get('type', 'Unknown')
                default_value = param_info.get('default', 'No default')
                
                print(f"  {param_name}:")
                print(f"    Description: {description}")
                print(f"    Type: {param_type}")
                print(f"    Default: {default_value}")
                print()
        else:
            print("  No parameters information available for this model.")
            
    except Exception as e:
        print(f"  Error retrieving parameters: {str(e)}")


def show_layer_parameters(model_manager, layer_name):
    """Show parameters for a specific layer"""
    print(f"\n\nParameters for '{layer_name}' layer:")
    print("-" * 40)
    
    try:
        # Get layer parameters
        params = model_manager.get_layer_parameters(layer_name)
        
        if params:
            for param_name, param_info in params.items():
                description = param_info.get('description', 'No description available')
                param_type = param_info.get('type', 'Unknown')
                default_value = param_info.get('default', 'No default')
                
                print(f"  {param_name}:")
                print(f"    Description: {description}")
                print(f"    Type: {param_type}")
                print(f"    Default: {default_value}")
                print()
        else:
            print("  No parameters information available for this layer.")
            
    except Exception as e:
        print(f"  Error retrieving parameters: {str(e)}")


def main():
    """Main function to list models and their parameters"""
    print("Low-Code Deep Learning Platform - Model Information")
    print("=" * 60)
    
    # List available complete models
    model_manager, available_models = list_available_models()
    
    # Show parameters for each complete model
    print("\n\nDetailed Parameter Information for Complete Models")
    print("=" * 60)
    
    for model_name in available_models:
        show_model_parameters(model_manager, model_name)
    
    # List available layers and functions
    layer_manager, available_layers = list_available_layers()
    
    # Show parameters for selected layers (not all, to keep output manageable)
    print("\n\nDetailed Parameter Information for Selected Layers")
    print("=" * 60)
    
    # Show parameters for some key layers
    key_layers = ['linear', 'conv2d', 'lstm']
    for layer_name in key_layers:
        if layer_name in available_layers:
            show_layer_parameters(layer_manager, layer_name)


if __name__ == "__main__":
    main()