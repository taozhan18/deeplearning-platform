"""
Training Engine Module for Low-Code Deep Learning Platform
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import json
import os


class TrainingEngine:
    """
    Training Engine for controlling the training process

    Configuration Parameters:
    -------------------------
    The config dictionary should contain the following parameters:

    Required Parameters:
    - None (all parameters are optional with defaults)

    Core Training Parameters:
    - device (str): Device to use for training ('cpu' or 'cuda'), default: 'cpu'
    - epochs (int): Number of training epochs, default: 10
    - preprocess_fn (str): Path to custom preprocessing function file, default: None

    Early Stopping Configuration:
    - early_stopping (bool): Whether to use early stopping, default: False
    - patience (int): Patience for early stopping (epochs to wait for improvement), default: 10
    - min_delta (float): Minimum delta for early stopping (minimum improvement required), default: 0.001
    - monitor (str): Metric to monitor for early stopping ('val_loss' or 'val_acc'), default: 'val_loss'
    - mode (str): Mode for monitoring ('min' for loss minimization, 'max' for accuracy maximization), default: 'min'

    Model Saving Configuration:
    - save_best_only (bool): Whether to save only the best model based on validation metric, default: False
    - save_every (int): Save model checkpoint every N epochs, default: 1
    - save_path (str): Directory path to save models, default: './checkpoints'
    - model_name (str): Base name for saved models, default: 'model'
    - save_last (bool): Whether to always save the final model, default: True

    Validation Configuration:
    - validation_interval (int): Run validation every N epochs, default: 1
    - validation_split (float): Fraction of training data to use for validation (0-1), default: 0.0 (use test_loader)
    - validation_shuffle (bool): Whether to shuffle validation data, default: False

    Logging Configuration:
    - verbose (int): Verbosity level (0=silent, 1=progress bar, 2=epoch summary), default: 1
    - log_interval (int): Print training progress every N batches, default: 10
    - metrics (list): List of additional metrics to track, default: []

    Checkpoint Configuration:
    - resume_from (str): Path to checkpoint to resume training from, default: None
    - save_optimizer (bool): Whether to save optimizer state in checkpoints, default: True
    - save_scheduler (bool): Whether to save scheduler state in checkpoints, default: True

    Optimizer Configuration (under 'optimizer' key):
    - optimizer.name (str): Name of optimizer ('adam', 'sgd', 'adamw', 'rmsprop'), default: 'adam'
    - optimizer.parameters (dict): Optimizer parameters, default: {}

    Criterion Configuration (under 'criterion' key):
    - criterion.name (str): Name of loss function ('cross_entropy', 'mse', 'l1', 'bce', 'bce_with_logits', 'smooth_l1'), default: 'cross_entropy'
    - criterion.parameters (dict): Loss function parameters, default: {}

    Scheduler Configuration (under 'scheduler' key):
    - scheduler.name (str): Name of scheduler ('step', 'exponential', 'cosine', 'reduce_on_plateau'), default: None
    - scheduler.parameters (dict): Scheduler parameters, default: {}

    Advanced Configuration:
    - gradient_clipping (float): Gradient clipping value, default: None
    - mixed_precision (bool): Whether to use mixed precision training, default: False
    - compile_model (bool): Whether to compile model for faster training, default: False
    - dataloader_num_workers (int): Number of workers for DataLoader, default: 0
    - pin_memory (bool): Whether to pin memory for DataLoader, default: True

    Example Configuration:
    ----------------------
    config = {
        "epochs": 100,
        "device": "cuda",

        # Early Stopping
        "early_stopping": True,
        "patience": 15,
        "min_delta": 0.001,
        "monitor": "val_loss",
        "mode": "min",

        # Model Saving
        "save_best_only": True,
        "save_every": 5,
        "save_path": "./models/experiment1",
        "model_name": "resnet50_classifier",
        "save_last": True,

        # Validation
        "validation_interval": 2,
        "validation_split": 0.2,
        "validation_shuffle": True,

        # Logging
        "verbose": 2,
        "log_interval": 5,
        "metrics": ["f1", "precision", "recall"],

        # Checkpoint
        "resume_from": "./checkpoints/last_checkpoint.pth",
        "save_optimizer": True,
        "save_scheduler": True,

        # Optimizer
        "optimizer": {
            "name": "adamw",
            "parameters": {
                "lr": 0.001,
                "weight_decay": 1e-4,
                "betas": [0.9, 0.999]
            }
        },

        # Criterion
        "criterion": {
            "name": "cross_entropy",
            "parameters": {
                "label_smoothing": 0.1
            }
        },

        # Scheduler
        "scheduler": {
            "name": "cosine",
            "parameters": {
                "T_max": 100,
                "eta_min": 1e-6
            }
        },

        # Advanced
        "gradient_clipping": 1.0,
        "mixed_precision": True,
        "compile_model": True,
        "dataloader_num_workers": 4,
        "pin_memory": True
    }
    """

    def __init__(self, model, config: Dict[str, Any]):
        """
        Initialize the training engine and configure optimizer, criterion and scheduler if specified in config

        Args:
            model: PyTorch model to train
            config: Configuration dictionary containing training parameters
        """
        self.config = config
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.device = torch.device(config.get("device", "cpu"))
        self.epochs = config.get("epochs", 10)
        self.preprocess_fn = self._load_preprocess_fn(config.get("preprocess_fn", None))

        self.set_model(model)

        # Early stopping configuration
        self.early_stopping = config.get("early_stopping", False)
        self.patience = config.get("patience", 10)
        self.min_delta = config.get("min_delta", 0.001)
        self.monitor = config.get("monitor", "val_loss")
        self.mode = config.get("mode", "min")

        # Model saving configuration
        self.save_best_only = config.get("save_best_only", False)
        self.save_every = config.get("save_every", 1)
        self.save_path = config.get("save_path", "./checkpoints")
        self.model_name = config.get("model_name", "model")
        self.save_last = config.get("save_last", True)

        # Validation configuration
        self.validation_interval = config.get("validation_interval", 1)
        self.validation_split = config.get("validation_split", 0.0)
        self.validation_shuffle = config.get("validation_shuffle", False)

        # Logging configuration
        self.verbose = config.get("verbose", 1)
        self.log_interval = config.get("log_interval", 10)

        # Checkpoint configuration
        self.resume_from = config.get("resume_from", None)
        self.save_optimizer = config.get("save_optimizer", True)
        self.save_scheduler = config.get("save_scheduler", True)

        # Advanced configuration
        self.gradient_clipping = config.get("gradient_clipping", None)
        self.mixed_precision = config.get("mixed_precision", False)
        self.compile_model = config.get("compile_model", False)
        self.dataloader_num_workers = config.get("dataloader_num_workers", 0)
        self.pin_memory = config.get("pin_memory", True)

        # Initialize early stopping state
        self.best_score = None
        self.counter = 0
        self.early_stop = False

        # Create save directory
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)

        # Configure optimizer, criterion and scheduler from config if available
        self._configure_from_config()

    def _configure_from_config(self):
        """Configure optimizer, criterion and scheduler from config"""
        # Configure optimizer - use default if not specified
        optimizer_config = self.config.get("optimizer", {})
        optimizer_name = optimizer_config.get("name", "adam")  # Default to Adam
        optimizer_params = optimizer_config.get("parameters", {})
        self.configure_optimizer(optimizer_name, **optimizer_params)

        # Configure criterion - use default if not specified
        criterion_config = self.config.get("criterion", {})
        criterion_name = criterion_config.get("name", "mse")  # Default to CrossEntropy
        criterion_params = criterion_config.get("parameters", {})
        self.configure_criterion(criterion_name, **criterion_params)

        # Configure scheduler - optional, no default
        scheduler_config = self.config.get("scheduler", {})
        scheduler_name = scheduler_config.get("name")
        if scheduler_name:
            scheduler_params = scheduler_config.get("parameters", {})
            self.configure_scheduler(scheduler_name, **scheduler_params)

    def set_model(self, model: nn.Module):
        """
        Set the model to be trained

        Args:
            model: PyTorch model to be trained
        """
        self.model = model.to(self.device)
        # Reconfigure optimizer if it was already set to include the new model's parameters
        if self.optimizer is not None:
            optimizer_config = self.config.get("optimizer", {})
            if optimizer_config:
                optimizer_name = optimizer_config.get("name", "adam")
                optimizer_params = optimizer_config.get("parameters", {})
                self.configure_optimizer(optimizer_name, **optimizer_params)

    def configure_optimizer(self, optimizer_name: str = None, **kwargs):
        """
        Configure the optimizer

        Args:
            optimizer_name: Name of the optimizer (e.g., 'adam', 'sgd').
                          If not provided, will use from self.config
            **kwargs: Additional arguments for the optimizer.
                     If not provided, will use from self.config
        """
        # If no optimizer_name provided, use from config
        if optimizer_name is None:
            optimizer_config = self.config.get("optimizer", {})
            optimizer_name = optimizer_config.get("name", "adam")
            # If no kwargs provided, use parameters from config
            if not kwargs:
                kwargs = optimizer_config.get("parameters", {})

        if self.model is None:
            raise ValueError("Model must be set before configuring optimizer")

        optimizer_name = optimizer_name.lower()

        if optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), **kwargs)
        elif optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), **kwargs)
        elif optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), **kwargs)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def configure_criterion(self, criterion_name: str = None, **kwargs):
        """
        Configure the loss function

        Args:
            criterion_name: Name of the loss function (e.g., 'cross_entropy', 'mse').
                          If not provided, will use from self.config
            **kwargs: Additional arguments for the loss function.
                     If not provided, will use from self.config
        """
        # If no criterion_name provided, use from config
        if criterion_name is None:
            criterion_config = self.config.get("criterion", {})
            criterion_name = criterion_config.get("name", "cross_entropy")
            # If no kwargs provided, use parameters from config
            if not kwargs:
                kwargs = criterion_config.get("parameters", {})

        criterion_name = criterion_name.lower()

        if criterion_name == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss(**kwargs)
        elif criterion_name == "mse":
            self.criterion = nn.MSELoss(**kwargs)
        elif criterion_name == "l1":
            self.criterion = nn.L1Loss(**kwargs)
        elif criterion_name == "bce":
            self.criterion = nn.BCELoss(**kwargs)
        elif criterion_name == "bce_with_logits":
            self.criterion = nn.BCEWithLogitsLoss(**kwargs)
        else:
            raise ValueError(f"Unsupported criterion: {criterion_name}")

    def configure_scheduler(self, scheduler_name: str = None, **kwargs):
        """
        Configure the learning rate scheduler

        Args:
            scheduler_name: Name of the scheduler (e.g., 'step', 'exponential').
                          If not provided, will use from self.config
            **kwargs: Additional arguments for the scheduler.
                     If not provided, will use from self.config
        """
        # If no scheduler_name provided, use from config
        if scheduler_name is None:
            scheduler_config = self.config.get("scheduler", {})
            scheduler_name = scheduler_config.get("name")
            # If no kwargs provided, use parameters from config
            if not kwargs:
                kwargs = scheduler_config.get("parameters", {})

            # If scheduler_name is still None, skip scheduler configuration
            if scheduler_name is None:
                return

        if self.optimizer is None:
            raise ValueError("Optimizer must be configured before scheduler")

        scheduler_name = scheduler_name.lower()

        if scheduler_name == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **kwargs)
        elif scheduler_name == "exponential":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, **kwargs)
        elif scheduler_name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **kwargs)
        elif scheduler_name == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **kwargs)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    def _check_early_stopping(self, current_score: float) -> bool:
        """
        Check if training should stop based on early stopping criteria

        Args:
            current_score: Current validation score to check

        Returns:
            bool: True if training should stop
        """
        if not self.early_stopping:
            return False

        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == "min":
            # For loss minimization
            if current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
        else:
            # For accuracy maximization
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            print(f"Early stopping triggered after {self.patience} epochs without improvement")
            return True

        return False

    def _save_checkpoint(self, epoch: int, val_loss: float, val_acc: float, is_best: bool = False):
        """
        Save model checkpoint

        Args:
            epoch: Current epoch number
            val_loss: Validation loss
            val_acc: Validation accuracy
            is_best: Whether this is the best model so far
        """
        if self.model is None:
            return

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "config": self.config,
        }

        if self.save_optimizer and self.optimizer is not None:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()

        if self.save_scheduler and self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save regular checkpoint
        if epoch % self.save_every == 0:
            checkpoint_path = os.path.join(self.save_path, f"{self.model_name}_epoch_{epoch}.pth")
            torch.save(checkpoint, checkpoint_path)
            if self.verbose >= 2:
                print(f"Checkpoint saved: {checkpoint_path}")

        # Save best model
        if is_best and self.save_best_only:
            best_path = os.path.join(self.save_path, f"{self.model_name}_best.pth")
            torch.save(checkpoint, best_path)
            if self.verbose >= 2:
                print(f"Best model saved: {best_path}")

        # Always save last model
        if self.save_last:
            last_path = os.path.join(self.save_path, f"{self.model_name}_last.pth")
            torch.save(checkpoint, last_path)

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load model checkpoint and resume training

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            dict: Checkpoint data
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if self.model is not None:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint

    def train_epoch(self, train_loader):
        """
        Train for one epoch

        Args:
            train_loader: DataLoader for training data

        Returns:
            Tuple of (average_loss, accuracy)
        """
        if self.model is None or self.optimizer is None or self.criterion is None:
            raise ValueError("Model, optimizer, and criterion must be configured before training")

        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Handle device placement for both single tensors and dicts
            if isinstance(data, dict):
                data = {key: value.to(self.device) for key, value in data.items()}
            else:
                data = data.to(self.device)
            target = target.to(self.device)

            # Special handling for different criterion types
            if isinstance(self.criterion, (nn.CrossEntropyLoss, nn.NLLLoss)):
                # For classification, target should be Long type
                if target.dtype != torch.long:
                    target = target.long()
            elif isinstance(self.criterion, (nn.MSELoss, nn.L1Loss)):
                # For regression, target should be Float type
                if target.dtype != torch.float:
                    target = target.float()
            elif isinstance(self.criterion, (nn.BCELoss, nn.BCEWithLogitsLoss)):
                # For binary classification, target should be Float type
                if target.dtype != torch.float:
                    target = target.float()

            self.optimizer.zero_grad()

            # Handle custom preprocessing for multi-source data
            if self.preprocess_fn is not None:
                processed_data = self.preprocess_fn(data)
                output = self.model(processed_data)
            else:
                # Default behavior: direct model forward pass
                output = self.model(data)

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)

            # Handle accuracy calculation based on criterion type
            if isinstance(self.criterion, (nn.CrossEntropyLoss, nn.NLLLoss)):
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            elif isinstance(self.criterion, (nn.MSELoss, nn.L1Loss)):
                # For regression, we don't calculate accuracy in the traditional sense
                total += target.size(0)
                correct += 0  # Not applicable for regression
            elif isinstance(self.criterion, (nn.BCELoss, nn.BCEWithLogitsLoss)):
                # For binary classification
                total += target.size(0)
                predicted_binary = (torch.sigmoid(output) > 0.5).float()
                correct += predicted_binary.eq(target).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def validate_epoch(self, test_loader):
        """
        Validate for one epoch

        Args:
            test_loader: DataLoader for test data

        Returns:
            Tuple of (average_loss, accuracy)
        """
        if self.model is None or self.criterion is None:
            raise ValueError("Model and criterion must be configured before validation")

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                # Handle device placement for both single tensors and dicts
                if isinstance(data, dict):
                    data = {key: value.to(self.device) for key, value in data.items()}
                else:
                    data = data.to(self.device)
                target = target.to(self.device)

                # Special handling for different criterion types
                if isinstance(self.criterion, (nn.CrossEntropyLoss, nn.NLLLoss)):
                    # For classification, target should be Long type
                    if target.dtype != torch.long:
                        target = target.long()
                elif isinstance(self.criterion, (nn.MSELoss, nn.L1Loss)):
                    # For regression, target should be Float type
                    if target.dtype != torch.float:
                        target = target.float()
                elif isinstance(self.criterion, (nn.BCELoss, nn.BCEWithLogitsLoss)):
                    # For binary classification, target should be Float type
                    if target.dtype != torch.float:
                        target = target.float()

                # Handle custom preprocessing for multi-source data
                if self.preprocess_fn is not None:
                    processed_data = self.preprocess_fn(data)
                    output = self.model(processed_data)
                else:
                    # Default behavior: direct model forward pass
                    output = self.model(data)

                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)

                # Handle accuracy calculation based on criterion type
                if isinstance(self.criterion, (nn.CrossEntropyLoss, nn.NLLLoss)):
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                elif isinstance(self.criterion, (nn.MSELoss, nn.L1Loss)):
                    # For regression, we don't calculate accuracy in the traditional sense
                    total += target.size(0)
                    correct += 0  # Not applicable for regression
                elif isinstance(self.criterion, (nn.BCELoss, nn.BCEWithLogitsLoss)):
                    # For binary classification
                    total += target.size(0)
                    predicted_binary = (torch.sigmoid(output) > 0.5).float()
                    correct += predicted_binary.eq(target).sum().item()

        avg_loss = total_loss / len(test_loader)
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def train(self, train_loader, test_loader=None):
        """
        Train the model with early stopping, checkpointing, and validation

        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data (optional)

        Returns:
            Dictionary containing training history
        """
        if self.model is None or self.optimizer is None or self.criterion is None:
            raise ValueError("Model, optimizer, and criterion must be configured before training")

        # Load checkpoint if resume_from is specified
        start_epoch = 1
        if self.resume_from is not None:
            try:
                checkpoint = self.load_checkpoint(self.resume_from)
                start_epoch = checkpoint["epoch"] + 1
                print(f"Resuming training from epoch {start_epoch}")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                print("Starting training from scratch")

        # Compile model if requested
        if self.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epochs": [], "lr": []}

        print(f"Starting training for {self.epochs} epochs...")

        for epoch in range(start_epoch, self.epochs + 1):
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate if test_loader is provided and validation interval is met
            val_loss, val_acc = 0.0, 0.0
            should_validate = test_loader is not None and epoch % self.validation_interval == 0

            if should_validate:
                val_loss, val_acc = self.validate_epoch(test_loader)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update history
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["epochs"].append(epoch)
            history["lr"].append(current_lr)

            # Print progress
            if should_validate:
                print(
                    f"Epoch {epoch}/{self.epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}"
                )
            else:
                print(
                    f"Epoch {epoch}/{self.epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, LR: {current_lr:.6f}"
                )

            # Step the scheduler if configured
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # ReduceLROnPlateau needs validation loss
                    if should_validate:
                        self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Determine current score for early stopping and best model saving
            current_score = val_loss if self.monitor == "val_loss" else val_acc
            if not should_validate:
                # If no validation, use training metrics
                current_score = train_loss if self.monitor == "val_loss" else train_acc

            # Check if this is the best model
            is_best = False
            if self.save_best_only:
                if self.best_score is None:
                    self.best_score = current_score
                    is_best = True
                else:
                    if self.mode == "min":
                        if current_score < self.best_score - self.min_delta:
                            self.best_score = current_score
                            is_best = True
                    else:
                        if current_score > self.best_score + self.min_delta:
                            self.best_score = current_score
                            is_best = True

            # Save checkpoint
            self._save_checkpoint(epoch, val_loss, val_acc, is_best)

            # Check early stopping
            if should_validate and self._check_early_stopping(current_score):
                print(f"Early stopping at epoch {epoch}")
                break

        print("Training completed!")

        # Save final model
        if self.save_last:
            final_path = os.path.join(self.save_path, f"{self.model_name}_final.pth")
            torch.save(
                {"model_state_dict": self.model.state_dict(), "history": history, "config": self.config}, final_path
            )
            print(f"Final model saved: {final_path}")

        return history

    def save_model(self, path: str):
        """
        Save the trained model

        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model must be configured before saving")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save model state dict
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """
        Load a trained model

        Args:
            path: Path to load the model from
        """
        if self.model is None:
            raise ValueError("Model must be configured before loading")

        # Load model state dict
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        print(f"Model loaded from {path}")

    def set_preprocess_fn(self, preprocess_fn):
        """
        Set custom preprocessing function from Python file

        Args:
            preprocess_fn: Path to Python file or callable function
        """
        self.preprocess_fn = self._load_preprocess_fn(preprocess_fn)

    def _load_preprocess_fn(self, preprocess_spec):
        """
        Load preprocessing function from Python file

        Args:
            preprocess_spec: Path to Python file containing preprocess_fn function

        Returns:
            Callable preprocessing function or None
        """
        if preprocess_spec is None:
            return None

        if callable(preprocess_spec):
            return preprocess_spec

        if isinstance(preprocess_spec, str):
            if not preprocess_spec.endswith(".py"):
                raise ValueError("preprocess_fn must be a path to a Python file ending with .py")
            return self.load_function_from_file(preprocess_spec, "preprocess_fn")

        raise ValueError("preprocess_fn must be a Python file path or callable function")

    @staticmethod
    def load_function_from_file(file_path: str, function_name: str):
        """
        Static method to load a function from a Python file

        Args:
            file_path: Path to the Python file
            function_name: Name of the function to load

        Returns:
            Callable function
        """
        import importlib.util
        import os

        if not os.path.exists(file_path):
            raise ValueError(f"Python file not found: {file_path}")

        try:
            spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
            if spec is None or spec.loader is None:
                raise ValueError(f"Could not load module from file: {file_path}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            return getattr(module, function_name)
        except (AttributeError, ImportError) as e:
            raise ValueError(f"Could not load function {function_name} from {file_path}: {e}")
