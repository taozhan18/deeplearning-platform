# Low-Code Deep Learning Training Platform

A low-code platform for deep learning training that allows users to focus on their models and data rather than implementation details.

## Features

- **Modular Design**: Separates data loading, model architecture, and training engine into independent modules
- **Predefined Models**: Built-in popular model architectures from PyTorch
- **Easy Configuration**: Specify dataset paths and select model architecture to start training
- **Flexible Usage**: Support both programmatic and UI-based operations
- **Extensible**: Easy to add new models and features

## Modules

1. [Data Loader](./data_loader) - Handles dataset loading and preprocessing
2. [Model Architecture](./model_architecture) - Provides predefined model architectures and templates
3. [Training Engine](./training_engine) - Controls the training process

## Installation

```bash
pip install -r requirements.txt
```

## Available Models

For a complete list of available models and their parameters, see:
- [Model Guide](MODEL_GUIDE.md) - Detailed information about all available models
- [模型指南](MODEL_GUIDE_CN.md) - 所有可用模型的详细信息（中文版）

## Usage

Users simply need to:
1. Specify the paths to training and testing datasets
2. Select a model architecture
3. Configure training hyperparameters (optional)
4. Start training

The platform will automatically handle the entire training workflow.

### Configuration-driven approach

Create a configuration file (YAML or JSON format) that specifies:

```yaml
# Example config.yaml
data:
  train_features_path: "data/train_features.csv"
  train_targets_path: "data/train_targets.csv"
  test_features_path: "data/test_features.csv"
  test_targets_path: "data/test_targets.csv"
  batch_size: 32

model:
  name: "ModelTemplate"
  parameters:
    input_size: 784
    hidden_size: 256
    num_classes: 10
    dropout_rate: 0.2

training:
  epochs: 10
  device: "cpu"

optimizer:
  name: "adam"
  parameters:
    lr: 0.001

criterion:
  name: "cross_entropy"
```

Then run the training:

```bash
python main/train.py --config config.yaml
```

### Programmatic approach

```python
from data_loader.src.data_loader import DataLoaderModule
from model_architecture.src.model_manager import ModelManager
from training_engine.src.training_engine import TrainingEngine

# Configure data loading
data_config = {
    'train_features_path': 'data/train_features.csv',
    'train_targets_path': 'data/train_targets.csv',
    'batch_size': 32
}

# Load data
data_loader = DataLoaderModule(data_config)
data_loader.prepare_datasets()
data_loader.create_data_loaders()
train_loader, test_loader = data_loader.get_data_loaders()

# Configure model
model_manager = ModelManager({})
model = model_manager.create_model('ModelTemplate', 
                                   input_size=784, 
                                   hidden_size=256, 
                                   num_classes=10)

# Configure training
training_config = {'epochs': 10, 'device': 'cpu'}
training_engine = TrainingEngine(training_config)
training_engine.set_model(model)
training_engine.configure_optimizer('adam', lr=0.001)
training_engine.configure_criterion('cross_entropy')

# Start training
history = training_engine.train(train_loader, test_loader)
```

## Project Structure

```
deeplearning-platform/
├── data_loader/
│   ├── src/
│   │   └── data_loader.py
│   ├── README.md
│   └── README_CN.md
├── model_architecture/
│   ├── src/
│   │   └── model_manager.py
│   ├── templates/
│   │   ├── model_template.py
│   │   ├── README.md
│   │   └── README_CN.md
│   ├── README.md
│   └── README_CN.md
├── training_engine/
│   ├── src/
│   │   └── training_engine.py
│   ├── README.md
│   └── README_CN.md
├── main/
│   ├── train.py
│   └── config.yaml
├── requirements.txt
├── README.md
└── README_CN.md
```