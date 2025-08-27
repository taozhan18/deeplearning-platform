# Using the Main Training Function for Different Tasks

This document explains how to use the main training function for different deep learning tasks in the low-code platform.

## Overview

The main training function in [main/train.py](file:///home/zt/workspace/deeplearning-platform/main/train.py) is designed to be generic and can handle various deep learning tasks by simply changing the configuration file. This approach avoids code duplication and ensures consistency across different tasks.

## How to Use

1. Create a configuration file for your specific task
2. Run the main training function with your configuration file:

```bash
python main/train.py --config your_config.yaml
```

## Configuration File Structure

All configuration files should follow this structure:

```yaml
# Data configuration
data:
  # Data paths and parameters
  train_features_path: "path/to/train_features"
  train_targets_path: "path/to/train_targets"
  test_features_path: "path/to/test_features"
  test_targets_path: "path/to/test_targets"
  batch_size: 32
  shuffle: true

# Model configuration
model:
  name: "model_name"
  parameters:
    # Model-specific parameters

# Training configuration
training:
  epochs: 10
  device: "cpu"  # or "cuda"

# Optimizer configuration
optimizer:
  name: "adam"
  parameters:
    lr: 0.001

# Criterion configuration
criterion:
  name: "cross_entropy"  # or "mse" for regression tasks

# Scheduler configuration (optional)
scheduler:
  name: "step"
  parameters:
    step_size: 5
    gamma: 0.1

# Output configuration
output:
  model_path: "path/to/save/model.pth"
  history_path: "path/to/save/history.json"
```

## Examples

### 1. Standard Classification Task

For standard classification tasks with the template model:

```yaml
model:
  name: "ModelTemplate"
  parameters:
    input_size: 784
    hidden_size: 256
    num_classes: 10
    dropout_rate: 0.2
```

### 2. FNO1D Task

For FNO1D tasks with 1D data:

```yaml
data:
  # FNO1D-specific data configuration
  train_features_path: "data/fno1d/train_features.npy"
  train_targets_path: "data/fno1d/train_targets.npy"
  test_features_path: "data/fno1d/test_features.npy"
  test_targets_path: "data/fno1d/test_targets.npy"
  batch_size: 16
  shuffle: true
  fno1d_data: true

model:
  name: "fno"
  parameters:
    in_channels: 1
    out_channels: 1
    dimension: 1
    # Other FNO-specific parameters
```

## Benefits

1. **Code Reusability**: No need to write separate training scripts for each task
2. **Consistency**: All tasks follow the same training procedure
3. **Maintainability**: Changes to the training process only need to be made in one place
4. **Flexibility**: Easy to add new models and tasks by just creating new configuration files

## Adding New Tasks

To add a new task:

1. Create a new configuration file following the structure above
2. Ensure your data is in a supported format (CSV, JSON, NPY, NPZ)
3. If needed, extend the data loader to support new data formats
4. If needed, register new models in the model manager
5. Run the training with your new configuration file