# MOOSE-FNO Integration Examples

This directory contains examples showing how to use MOOSE simulation data with the deep learning platform, particularly for training FNO models.

## Overview

The workflow for using MOOSE simulations with the deep learning platform consists of:

1. Generating simulation data using the MOOSE data generator
2. Preprocessing the data for training
3. Training FNO models with the data
4. Evaluating and using the trained models

## Directory Structure

```
examples/
├── train_fno_with_moose_data.py  # Example training script
├── moose_fno_workflow.md         # Detailed workflow documentation
└── README.md                     # This file
```

## Workflow

### 1. Generate Dataset with MOOSE

First, use the MOOSE data generator to run simulations and create a dataset:

```bash
cd ../data_loader/scripts
python moose_data_generator.py \
  --moose-exec /home/zt/workspace/mymoose/mymoose-opt \
  --create-example \
  --output-dir ../../data/moose_fno_dataset
```

### 2. Train FNO Model

Then, use the example training script to train an FNO model with the generated data:

```bash
cd ../..
python examples/train_fno_with_moose_data.py
```

## Customization

### Using Your Own MOOSE Input Files

1. Create a MOOSE input template with parameter placeholders:

```text
[Mesh]
  type = GeneratedMesh
  dim = 1
  nx = {{nx}}
[]

[Kernels]
  [source]
    type = CoefficientKernel
    variable = u
    coefficient = {{source_strength}}
  []
[]
```

2. Create a parameter study file:

```json
[
  {"nx": 64, "source_strength": 1.0},
  {"nx": 128, "source_strength": 1.5},
  {"nx": 256, "source_strength": 2.0}
]
```

3. Generate the dataset:

```bash
python moose_data_generator.py \
  --moose-exec /path/to/your/moose-app \
  --input-template your_template.i \
  --param-file your_params.json \
  --output-dir your_dataset
```

### Modifying the FNO Model

Adjust the FNO model parameters in the training script:

```python
model = model_manager.create_model(
    'fno',
    in_channels=1,           # Number of input fields
    out_channels=1,          # Number of output fields
    dimension=1,             # 1D, 2D, or 3D
    latent_channels=32,      # Hidden representation size
    num_fno_layers=4,        # Number of FNO layers
    num_fno_modes=16,        # Number of Fourier modes
    # ... other parameters
)
```

## Best Practices

1. **Data Quality**: Ensure your MOOSE simulations have converged and are sufficiently resolved
2. **Parameter Space Coverage**: Design your parameter studies to cover the range of interest
3. **Data Size**: Generate enough training data for your model complexity
4. **Validation**: Always validate your trained models on test data not used during training
5. **Physical Consistency**: Check that your ML models respect physical constraints

## Advanced Usage

### Multi-Field Models

For models that take multiple input fields and produce multiple output fields:

```python
# Multiple input channels
train_input = np.stack([field1, field2, field3], axis=1)  # (n_samples, 3, n_points)

# Multiple output channels
train_output = np.stack([solution1, solution2], axis=1)   # (n_samples, 2, n_points)

# Create model with appropriate channels
model = model_manager.create_model(
    'fno',
    in_channels=3,     # 3 input fields
    out_channels=2,    # 2 output fields
    # ... other parameters
)
```

### Time-Dependent Problems

For time-dependent simulations, you can treat time as an additional input dimension:

```python
# Input: (space, time)
# Reshape to: (n_samples, 1, n_space, n_time)
train_input = field_data[:, np.newaxis, :, :]  

# Create 2D FNO model
model = model_manager.create_model(
    'fno',
    in_channels=1,
    out_channels=1,
    dimension=2,         # 2D (space and time)
    # ... other parameters
)
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size or model complexity
2. **Convergence Problems**: Adjust learning rate or training schedule
3. **Poor Accuracy**: Check data quality and increase training data size
4. **Overfitting**: Add regularization or reduce model complexity

### Performance Optimization

1. Use GPU acceleration when available
2. Optimize data loading with appropriate batch sizes
3. Use mixed precision training for faster computation
4. Profile your code to identify bottlenecks