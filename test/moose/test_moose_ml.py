#!/usr/bin/env python3
"""
Test suite for MOOSE to ML Training Automation
Tests the complete pipeline from MOOSE simulations to ML training
"""

import os
import sys
import json
import tempfile
import shutil
import numpy as np
from pathlib import Path
import argparse

# Add paths for local modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "data_loader", "moose"))
from moose_ml_automation import MOOSEMLPipeline, EnhancedMOOSEDataGenerator


def main():
    """Main function for command-line usage"""

    parser = argparse.ArgumentParser(
        description="MOOSE to ML Training Pipeline",
    )
    parser.add_argument(
        "--config",
        default="/home/zt/workspace/deeplearning-platform/test/moose/moose_ml_config.json",
        help="Configuration file path (optional, will use default if not provided)",
    )

    args = parser.parse_args()

    # Determine config path
    config_path = args.config

    # Run pipeline
    pipeline = MOOSEMLPipeline(config_path)
    results = pipeline.run_complete_pipeline()

    print("\n🎉 Pipeline completed successfully!")
    print(f"Dataset saved to: {results['config_path'].replace('training_config.yaml', '')}")
    print(f"Training config: {results['config_path']}")
    print(f"To run training: python main/train.py --config {results['config_path']}")

    try:
        import subprocess

        cmd = [
            sys.executable,
            "/home/zt/workspace/deeplearning-platform/main/train.py",
            "--config",
            results["config_path"],
        ]
        print(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print("✓ Training completed!")
    except subprocess.CalledProcessError as e:
        print(f"✗ Training failed: {e}")
    except FileNotFoundError:
        print("Training script not found. You can run it manually:")
        print(f"python main/train.py --config {results['config_path']}")
    else:
        print("To run training later:")
        print(f"python main/train.py --config {results['config_path']}")


if __name__ == "__main__":
    main()
