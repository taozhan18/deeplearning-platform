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
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the training engine
        
        Args:
            config: Configuration dictionary containing training parameters
        """
        self.config = config
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.device = torch.device(config.get('device', 'cpu'))
        self.epochs = config.get('epochs', 10)
        self.preprocess_fn = self._load_preprocess_fn(config.get('preprocess_fn', None))
    
    def set_model(self, model: nn.Module):
        """
        Set the model to be trained
        
        Args:
            model: PyTorch model to be trained
        """
        self.model = model.to(self.device)
    
    def configure_optimizer(self, optimizer_name: str, **kwargs):
        """
        Configure the optimizer
        
        Args:
            optimizer_name: Name of the optimizer (e.g., 'adam', 'sgd')
            **kwargs: Additional arguments for the optimizer
        """
        optimizer_name = optimizer_name.lower()
        
        if optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), **kwargs)
        elif optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), **kwargs)
        elif optimizer_name == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), **kwargs)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def configure_criterion(self, criterion_name: str, **kwargs):
        """
        Configure the loss function
        
        Args:
            criterion_name: Name of the loss function (e.g., 'cross_entropy', 'mse')
            **kwargs: Additional arguments for the loss function
        """
        criterion_name = criterion_name.lower()
        
        if criterion_name == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss(**kwargs)
        elif criterion_name == 'mse':
            self.criterion = nn.MSELoss(**kwargs)
        elif criterion_name == 'l1':
            self.criterion = nn.L1Loss(**kwargs)
        elif criterion_name == 'bce':
            self.criterion = nn.BCELoss(**kwargs)
        elif criterion_name == 'bce_with_logits':
            self.criterion = nn.BCEWithLogitsLoss(**kwargs)
        else:
            raise ValueError(f"Unsupported criterion: {criterion_name}")
    
    def configure_scheduler(self, scheduler_name: str, **kwargs):
        """
        Configure the learning rate scheduler
        
        Args:
            scheduler_name: Name of the scheduler (e.g., 'step', 'exponential')
            **kwargs: Additional arguments for the scheduler
        """
        scheduler_name = scheduler_name.lower()
        
        if self.optimizer is None:
            raise ValueError("Optimizer must be configured before scheduler")
        
        if scheduler_name == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **kwargs)
        elif scheduler_name == 'exponential':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, **kwargs)
        elif scheduler_name == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **kwargs)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
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
        accuracy = 100. * correct / total if total > 0 else 0.0
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
        accuracy = 100. * correct / total if total > 0 else 0.0
        return avg_loss, accuracy
    
    def train(self, train_loader, test_loader=None):
        """
        Train the model
        
        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data (optional)
            
        Returns:
            Dictionary containing training history
        """
        if self.model is None or self.optimizer is None or self.criterion is None:
            raise ValueError("Model, optimizer, and criterion must be configured before training")
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epochs': []
        }
        
        print(f"Starting training for {self.epochs} epochs...")
        
        for epoch in range(1, self.epochs + 1):
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate if test_loader is provided
            val_loss, val_acc = 0.0, 0.0
            if test_loader is not None:
                val_loss, val_acc = self.validate_epoch(test_loader)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['epochs'].append(epoch)
            
            # Print progress
            if test_loader is not None:
                print(f"Epoch {epoch}/{self.epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            else:
                print(f"Epoch {epoch}/{self.epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            
            # Step the scheduler if configured
            if self.scheduler is not None:
                self.scheduler.step()
        
        print("Training completed!")
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
            if not preprocess_spec.endswith('.py'):
                raise ValueError("preprocess_fn must be a path to a Python file ending with .py")
            return self.load_function_from_file(preprocess_spec, 'preprocess_fn')
        
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
    
    