"""
Test script for new training engine features including early stopping, checkpointing, and enhanced configuration
"""

import sys
import os
import torch
import torch.nn as nn
import tempfile
import shutil

# Add the modules to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "training_engine", "src"))

from training_engine import TrainingEngine


def test_early_stopping():
    """Test early stopping functionality"""
    print("Testing early stopping...")

    config = {
        "epochs": 100,
        "device": "cpu",
        "early_stopping": True,
        "patience": 3,
        "min_delta": 0.01,
        "monitor": "val_loss",
        "mode": "min",
        "save_path": tempfile.mkdtemp(),
    }

    try:

        # Create simple model and data
        model = nn.Sequential(nn.Linear(10, 1))
        training_engine = TrainingEngine(model, config)
        # training_engine.set_model(model)
        training_engine.configure_optimizer("adam", lr=0.01)
        training_engine.configure_criterion("mse")

        # Create dummy data
        train_data = torch.randn(20, 10)
        train_targets = torch.randn(20, 1)
        val_data = torch.randn(10, 10)
        val_targets = torch.randn(10, 1)

        train_loader = [(train_data, train_targets)]
        val_loader = [(val_data, val_targets)]

        # Run training
        history = training_engine.train(train_loader, val_loader)

        # Check that training completed (may stop early due to no improvement)
        assert len(history["train_loss"]) > 0
        assert len(history["val_loss"]) > 0

        print("✓ Early stopping test passed")

    finally:
        # Clean up
        if os.path.exists(config["save_path"]):
            shutil.rmtree(config["save_path"])


def test_checkpoint_saving():
    """Test checkpoint saving functionality"""
    print("Testing checkpoint saving...")

    temp_dir = tempfile.mkdtemp()

    config = {
        "epochs": 5,
        "device": "cpu",
        "save_every": 2,
        "save_path": temp_dir,
        "model_name": "test_model",
        "save_last": True,
        "save_optimizer": True,
        "save_scheduler": True,
    }

    try:
        model = nn.Sequential(nn.Linear(5, 1))
        training_engine = TrainingEngine(model, config)
        training_engine.set_model(model)
        training_engine.configure_optimizer("adam", lr=0.01)
        training_engine.configure_criterion("mse")
        training_engine.configure_scheduler("step", step_size=2, gamma=0.1)

        # Create dummy data
        train_data = torch.randn(10, 5)
        train_targets = torch.randn(10, 1)
        train_loader = [(train_data, train_targets)]

        # Run training
        history = training_engine.train(train_loader)

        # Check that checkpoints were saved
        expected_files = [
            "test_model_epoch_2.pth",
            "test_model_epoch_4.pth",
            "test_model_last.pth",
            "test_model_final.pth",
        ]

        saved_files = os.listdir(temp_dir)
        for expected_file in expected_files:
            assert expected_file in saved_files, f"Expected file {expected_file} not found"

        print("✓ Checkpoint saving test passed")

    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_best_model_saving():
    """Test best model saving based on validation metrics"""
    print("Testing best model saving...")

    temp_dir = tempfile.mkdtemp()

    config = {
        "epochs": 10,
        "device": "cpu",
        "save_best_only": True,
        "save_path": temp_dir,
        "model_name": "best_model",
        "monitor": "val_loss",
        "mode": "min",
    }

    try:
        # Create simple model and data
        model = nn.Sequential(nn.Linear(3, 1))
        training_engine = TrainingEngine(model, config)
        training_engine.set_model(model)
        training_engine.configure_optimizer("adam", lr=0.01)
        training_engine.configure_criterion("mse")

        # Create dummy data
        train_data = torch.randn(8, 3)
        train_targets = torch.randn(8, 1)
        val_data = torch.randn(4, 3)
        val_targets = torch.randn(4, 1)

        train_loader = [(train_data, train_targets)]
        val_loader = [(val_data, val_targets)]

        # Run training
        history = training_engine.train(train_loader, val_loader)

        # Check that best model was saved
        best_model_path = os.path.join(temp_dir, "best_model_best.pth")
        assert os.path.exists(best_model_path), "Best model file not found"

        # Load and verify checkpoint
        checkpoint = torch.load(best_model_path)
        assert "model_state_dict" in checkpoint
        assert "val_loss" in checkpoint
        assert "config" in checkpoint

        print("✓ Best model saving test passed")

    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_resume_training():
    """Test resume training from checkpoint"""
    print("Testing resume training...")

    temp_dir = tempfile.mkdtemp()

    # First training run
    config1 = {
        "epochs": 3,
        "device": "cpu",
        "save_path": temp_dir,
        "model_name": "resume_test",
        "save_last": True,
        "save_optimizer": True,
    }

    try:
        # First training
        model = nn.Sequential(nn.Linear(2, 1))
        training_engine1 = TrainingEngine(model, config1)

        training_engine1.set_model(model)
        training_engine1.configure_optimizer("adam", lr=0.01)
        training_engine1.configure_criterion("mse")

        train_data = torch.randn(6, 2)
        train_targets = torch.randn(6, 1)
        train_loader = [(train_data, train_targets)]

        history1 = training_engine1.train(train_loader)

        # Second training with resume
        config2 = {
            "epochs": 5,
            "device": "cpu",
            "resume_from": os.path.join(temp_dir, "resume_test_last.pth"),
            "save_path": temp_dir,
            "model_name": "resume_test2",
        }

        # Create model first, then load checkpoint
        model2 = nn.Sequential(nn.Linear(2, 1))
        training_engine2 = TrainingEngine(model2, config2)
        training_engine2.set_model(model2)
        training_engine2.configure_optimizer("adam", lr=0.01)
        training_engine2.configure_criterion("mse")

        history2 = training_engine2.train(train_loader)

        # Check that training resumed correctly - should train for 2 more epochs (5-3=2)
        assert len(history2["train_loss"]) == 2  # Should train for 5-3 = 2 more epochs

        print("✓ Resume training test passed")

    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_validation_interval():
    """Test validation interval configuration"""
    print("Testing validation interval...")

    temp_dir = tempfile.mkdtemp()

    config = {
        "epochs": 6,
        "device": "cpu",
        "validation_interval": 3,  # Validate every 3 epochs
        "save_path": temp_dir,
        "model_name": "interval_test",
    }

    try:

        model = nn.Sequential(nn.Linear(4, 1))
        training_engine = TrainingEngine(model, config)

        # Create simple model and data
        training_engine.configure_optimizer("adam", lr=0.01)
        training_engine.configure_criterion("mse")

        # Create dummy data
        train_data = torch.randn(8, 4)
        train_targets = torch.randn(8, 1)
        val_data = torch.randn(4, 4)
        val_targets = torch.randn(4, 1)

        train_loader = [(train_data, train_targets)]
        val_loader = [(val_data, val_targets)]

        # Run training
        history = training_engine.train(train_loader, val_loader)

        # Check validation was run at correct intervals
        val_losses = [l for l in history["val_loss"] if l != 0.0]
        # Should validate at epochs 3 and 6
        assert len(val_losses) == 2, f"Expected 2 validation runs, got {len(val_losses)}"

        print("✓ Validation interval test passed")

    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_comprehensive_configuration():
    """Test comprehensive configuration with all new features"""
    print("Testing comprehensive configuration...")

    temp_dir = tempfile.mkdtemp()

    config = {
        "epochs": 4,
        "device": "cpu",
        # Early stopping
        "early_stopping": True,
        "patience": 5,
        "min_delta": 0.01,
        "monitor": "val_loss",
        "mode": "min",
        # Model saving
        "save_best_only": True,
        "save_every": 2,
        "save_path": temp_dir,
        "model_name": "comprehensive_test",
        "save_last": True,
        "save_optimizer": True,
        "save_scheduler": True,
        # Validation
        "validation_interval": 1,
    }

    try:
        # Create training engine
        model = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1))
        training_engine = TrainingEngine(model, config)

        # Create model and set it before configuring optimizer/scheduler
        training_engine.set_model(model)
        training_engine.configure_optimizer("adamw", lr=0.001, weight_decay=1e-4)
        training_engine.configure_criterion("mse")
        training_engine.configure_scheduler("cosine", T_max=4, eta_min=1e-6)

        # Create data
        train_data = torch.randn(10, 3)
        train_targets = torch.randn(10, 1)
        val_data = torch.randn(5, 3)
        val_targets = torch.randn(5, 1)

        train_loader = [(train_data, train_targets)]
        val_loader = [(val_data, val_targets)]

        # Run training
        history = training_engine.train(train_loader, val_loader)

        # Verify configuration was applied
        assert training_engine.early_stopping == True
        assert training_engine.patience == 5
        assert training_engine.save_every == 2
        assert training_engine.validation_interval == 1

        # Check files were created
        expected_files = [
            "comprehensive_test_epoch_2.pth",
            "comprehensive_test_epoch_4.pth",
            "comprehensive_test_best.pth",
            "comprehensive_test_last.pth",
            "comprehensive_test_final.pth",
        ]

        saved_files = os.listdir(temp_dir)
        for expected_file in expected_files:
            assert expected_file in saved_files

        print("✓ Comprehensive configuration test passed")

    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("Running new features tests...")
    print("=" * 50)

    try:
        test_early_stopping()
        test_checkpoint_saving()
        test_best_model_saving()
        test_resume_training()
        test_validation_interval()
        test_comprehensive_configuration()

        print("\n" + "=" * 50)
        print("All new features tests passed! 🎉")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
