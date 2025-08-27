# MOOSE Data Generator

This tool automates the process of running MOOSE simulations and generating datasets for training field-to-field deep learning models, particularly FNO models.

## Features

- Run parametric MOOSE simulations
- Extract field data from simulation outputs
- Prepare data in formats suitable for deep learning models
- Create training datasets for FNO models

## Installation

Make sure you have the required dependencies installed:

```bash
# Activate your physics environment
source activate physics

# The script uses standard Python libraries that should already be available
```

## Usage

### Creating an Example Dataset

To create an example 1D field dataset:

```bash
python moose_data_generator.py --moose-exec /home/zt/workspace/mymoose/mymoose-opt --create-example --output-dir example_dataset
```

### Using Custom Input Template and Parameters

1. Create a MOOSE input file template with placeholders:

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
  [source]
    type = CoefficientKernel
    variable = u
    coefficient = {{input_amplitude}}
  []
[]

[BCs]
  [left]
    type = DirichletBC
    variable = u
    boundary = left
    value = 0
  []
  [right]
    type = DirichletBC
    variable = u
    boundary = right
    value = 0
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

2. Create a parameter file (JSON format):

```json
[
  {
    "nx": 64,
    "input_amplitude": 1.0,
    "output_amplitude": 0.5
  },
  {
    "nx": 128,
    "input_amplitude": 1.5,
    "output_amplitude": 0.7
  }
]
```

3. Run the data generator:

```bash
python moose_data_generator.py \
  --moose-exec /home/zt/workspace/mymoose/mymoose-opt \
  --input-template your_input.i \
  --param-file your_params.json \
  --output-dir my_dataset
```

## Output Structure

The tool generates the following output structure:

```
my_dataset/
├── sim_0000/
│   ├── input.i
│   ├── output.csv
│   └── output.e
├── sim_0001/
│   ├── input.i
│   ├── output.csv
│   └── output.e
├── simulation_results.json
├── parameters.npy
├── input_default.npy
├── output_default.npy
├── train_input.npy
├── train_output.npy
├── test_input.npy
└── test_output.npy
```

## Customization

The script is designed to be generic and extensible:

1. **Different MOOSE Applications**: Simply change the MOOSE executable path
2. **Different Field Types**: Modify the `extract_field_data` method to handle different output formats
3. **Different ML Models**: Adjust the data preparation methods for other model types
4. **Advanced Parameter Studies**: Extend the parameter substitution mechanism for complex scenarios

## Integration with Deep Learning Platform

The generated datasets can be directly used with the deep learning platform:

1. Training data is saved in NumPy format compatible with the data loader
2. FNO training data is pre-formatted for the FNO model
3. Metadata is saved for reproducibility and analysis