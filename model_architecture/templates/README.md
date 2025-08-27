# Model Definition Template

This directory contains a template for implementing new deep learning models in the platform.

## Template Structure

The [model_template.py](model_template.py) file demonstrates the structure that all models should follow:

1. **Hyperparameter Documentation**: Each model includes a `HYPERPARAMETERS` dictionary that describes all available hyperparameters, including their descriptions, types, and default values.

2. **Standard Constructor**: The `__init__` method accepts keyword arguments for hyperparameters and sets defaults when not provided.

3. **Forward Method**: All models implement a `forward` method that defines the forward pass of the network.

4. **Hyperparameter Retrieval**: A class method `get_hyperparameters()` allows users to retrieve information about all available hyperparameters.

## Usage

To create a new model:

1. Copy the template file
2. Rename it appropriately
3. Modify the hyperparameters dictionary to reflect your model's actual hyperparameters
4. Update the model architecture in the `__init__` and `forward` methods
5. Update the documentation to describe your specific model

This ensures all models in the platform follow a consistent interface and provide clear documentation for users.