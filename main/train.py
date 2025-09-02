"""
Main training script for Low-Code Deep Learning Platform
"""

import sys
import os
import argparse
import yaml
import json

# Add the modules to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data_loader", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "model_architecture", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training_engine", "src"))

from data_loader import DataLoaderModule
from model_manager import ModelManager
from training_engine import TrainingEngine


def load_config(config_path: str) -> dict:
    """
    Load configuration from a YAML file

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            return yaml.safe_load(f)
        else:
            return json.load(f)


def main(config_path: str):
    """
    Main training function

    Args:
        config_path: Path to the configuration file
    """
    # Load configuration
    config = load_config(config_path)

    # Initialize data loader
    print("Initializing data loader...")
    data_loader = DataLoaderModule(config.get("data", {}))
    data_loader.prepare_datasets()
    data_loader.create_data_loaders()
    train_loader, test_loader = data_loader.get_data_loaders()

    # Print dataset info
    dataset_info = data_loader.get_dataset_info()
    print(f"Dataset info: {dataset_info}")

    # Initialize model manager
    print("Initializing model manager...")
    model_manager = ModelManager(config.get("model", {}))

    # Create model
    model_config = config.get("model", {})
    model_name = model_config.get("name", "ModelTemplate")

    if model_name.lower() == "modeltemplate":
        # Use the template model
        model_params = model_config.get("parameters", {})
        model = model_manager.create_model("modeltemplate", **model_params)
    else:
        # Try to use a predefined model
        model_params = model_config.get("parameters", {})
        model = model_manager.create_model(model_name.lower(), **model_params)

    print(f"Model '{model_name}' created successfully")

    # Initialize training engine
    print("Initializing training engine...")
    training_engine = TrainingEngine(model, config.get("training", {}))
    # training_engine.set_model(model)

    # Model, optimizer, criterion and scheduler are now automatically configured from config
    # No need to explicitly configure them

    # Start training
    print("Starting training...")
    history = training_engine.train(train_loader, test_loader)

    # Save training history
    history_path = config.get("output", {}).get("history_path", "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Low-Code Deep Learning Platform")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")

    args = parser.parse_args()
    main(args.config)
