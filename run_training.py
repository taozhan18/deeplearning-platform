"""
Example script to run training with the low-code deep learning platform
"""

import sys
import os

# Add the modules to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_loader', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'model_architecture', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'training_engine', 'src'))

from data_loader import DataLoaderModule
from model_manager import ModelManager
from training_engine import TrainingEngine


def main():
    """
    Main training function using the generated test data
    """
    # Configuration
    config = {
        'data': {
            'train_features_path': 'data/train_features.csv',
            'train_targets_path': 'data/train_targets.csv',
            'test_features_path': 'data/test_features.csv',
            'test_targets_path': 'data/test_targets.csv',
            'batch_size': 64,
            'shuffle': True
        },
        'model': {
            'name': 'ModelTemplate',
            'parameters': {
                'input_size': 5,
                'hidden_size': 128,
                'num_classes': 10,
                'dropout_rate': 0.3
            }
        },
        'training': {
            'epochs': 10,
            'device': 'cpu'
        },
        'optimizer': {
            'name': 'adam',
            'parameters': {
                'lr': 0.001,
                'weight_decay': 0.0001
            }
        },
        'criterion': {
            'name': 'cross_entropy',
            'parameters': {}
        },
        'scheduler': {
            'name': 'step',
            'parameters': {
                'step_size': 5,
                'gamma': 0.5
            }
        },
        'output': {
            'model_path': 'data/final_model.pth',
            'history_path': 'data/final_training_history.json'
        }
    }
    
    print("Starting training with the low-code deep learning platform...")
    print("=" * 60)
    
    # Initialize data loader
    print("1. Initializing data loader...")
    data_loader = DataLoaderModule(config['data'])
    data_loader.prepare_datasets()
    data_loader.create_data_loaders()
    train_loader, test_loader = data_loader.get_data_loaders()
    
    # Print dataset info
    dataset_info = data_loader.get_dataset_info()
    print(f"   Dataset info: {dataset_info}")
    
    # Initialize model manager
    print("2. Initializing model manager...")
    model_manager = ModelManager(config['model'])
    
    # Create model
    model_config = config['model']
    model_name = model_config['name']
    model_params = model_config['parameters']
    model = model_manager.create_model(model_name.lower(), **model_params)
    print(f"   Model '{model_name}' created successfully")
    
    # Initialize training engine
    print("3. Initializing training engine...")
    training_engine = TrainingEngine(config['training'])
    training_engine.set_model(model)
    
    # Configure optimizer
    optimizer_config = config['optimizer']
    optimizer_name = optimizer_config['name']
    optimizer_params = optimizer_config['parameters']
    training_engine.configure_optimizer(optimizer_name, **optimizer_params)
    
    # Configure criterion
    criterion_config = config['criterion']
    criterion_name = criterion_config['name']
    criterion_params = criterion_config['parameters']
    training_engine.configure_criterion(criterion_name, **criterion_params)
    
    # Configure scheduler if specified
    scheduler_config = config['scheduler']
    if scheduler_config:
        scheduler_name = scheduler_config['name']
        scheduler_params = scheduler_config['parameters']
        training_engine.configure_scheduler(scheduler_name, **scheduler_params)
    
    # Start training
    print("4. Starting training...")
    history = training_engine.train(train_loader, test_loader)
    
    # Save model
    output_config = config['output']
    model_path = output_config['model_path']
    training_engine.save_model(model_path)
    print(f"   Model saved to {model_path}")
    
    # Save training history
    history_path = output_config['history_path']
    import json
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f"   Training history saved to {history_path}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Final training accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Final validation accuracy: {history['val_acc'][-1]:.2f}%")


if __name__ == "__main__":
    main()