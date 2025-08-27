# MOOSE-FNO Integration Workflow

This document describes the complete workflow for using MOOSE simulations to generate data and train FNO models.

## Overview

The workflow consists of two main steps:
1. Generate training data using MOOSE simulations
2. Train an FNO model with the generated data

## Step 1: Generate Training Data

### Using the MOOSE Data Generator

The MOOSE data generator automates the process of running parametric studies and extracting field data.

```bash
# Run the example script
python examples/moose_fno_example.py
```

This will:
1. Run 6 MOOSE simulations with different parameters
2. Extract field data from the simulation results
3. Prepare the data in a format suitable for FNO training

### Customizing the Workflow

To use your own MOOSE input files and parameters:

1. Create a MOOSE input template with placeholders:
```text
[Mesh]
  type = GeneratedMesh
  dim = 1
  nx = {{nx}}
[]

[Variables]
  [u]
  []
[]

[Kernels]
  [diff]
    type = Diffusion
    variable = u
  []
[]

[BCs]
  [left]
    type = DirichletBC
    variable = u
    boundary = left
    value = {{left_bc}}
  []
  [right]
    type = DirichletBC
    variable = u
    boundary = right
    value = {{right_bc}}
  []
[]

[Executioner]
  type = Steady
  solve_type = 'PJFNK'
[]

[Outputs]
  csv = true
  exodus = true
[]
```

2. Create a parameter file (JSON):
```json
[
  {"nx": 64, "left_bc": 0.0, "right_bc": 1.0},
  {"nx": 128, "left_bc": 0.0, "right_bc": 2.0}
]
```

3. Use the MOOSE data generator directly:
```python
from moose_data_generator import MOOSEDataGenerator

generator = MOOSEDataGenerator("/path/to/moose/executable")
generator.run_parametric_sims("input_template.i", param_sets, "output_dataset")
```

## Step 2: Train FNO Model

Once the dataset is generated, you can train an FNO model:

```python
# Load the data
train_input = np.load("output_dataset/train_input.npy")
train_output = np.load("output_dataset/train_output.npy")

# Create data loaders
from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(torch.FloatTensor(train_input), torch.FloatTensor(train_output))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create and train FNO model
from model_manager import ModelManager
from training_engine import TrainingEngine

model_manager = ModelManager({})
model = model_manager.create_model('fno', in_channels=1, out_channels=1, dimension=1, ...)

training_engine = TrainingEngine({'epochs': 50, 'device': 'cuda' if torch.cuda.is_available() else 'cpu'})
training_engine.set_model(model)
training_engine.configure_optimizer('adam', lr=0.001)
training_engine.configure_criterion('mse')
training_engine.train(train_loader, test_loader)
```

## Understanding the Output

The generated dataset contains:
- `train_input.npy`: Training input fields with shape (n_samples, channels, spatial_dims)
- `train_output.npy`: Training output fields with shape (n_samples, channels, spatial_dims)
- `test_input.npy`: Test input fields
- `test_output.npy`: Test output fields
- `simulation_results.json`: Metadata about the simulations

## Best Practices

1. **Parameter Space Coverage**: Design your parameter studies to adequately cover the space of interest
2. **Data Quality**: Ensure MOOSE simulations are well-converged and sufficiently resolved
3. **Data Size**: Generate enough training data for your model complexity
4. **Validation**: Always validate your trained models on test data not used during training

## Troubleshooting

Common issues and solutions:

1. **MOOSE Simulation Failures**:
   - Check that all kernel and boundary condition types are supported by your MOOSE application
   - Verify that parameter values are physically reasonable

2. **Memory Issues**:
   - Reduce batch size
   - Use smaller spatial discretizations
   - Reduce model complexity

3. **Convergence Problems**:
   - Adjust learning rate
   - Try different optimizers
   - Increase training epochs

4. **Poor Accuracy**:
   - Check data quality
   - Increase training data size
   - Adjust model hyperparameters