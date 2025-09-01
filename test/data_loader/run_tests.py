#!/usr/bin/env python3
"""
Test runner for data loader module
"""

import os
import sys
import subprocess
import tempfile
import shutil
import numpy as np


def run_pytest():
    """Run pytest with verbose output"""
    test_dir = os.path.dirname(__file__)
    test_file = os.path.join(test_dir, "test_data_loader.py")

    print("Running data loader tests...")
    print("=" * 50)

    try:
        # Run pytest
        result = subprocess.run(
            ["python", "-m", "pytest", test_file, "-v", "--tb=short"], capture_output=False, text=True
        )

        if result.returncode == 0:
            print("\n" + "=" * 50)
            print("All tests passed! ✓")
        else:
            print(f"\nTests failed with return code: {result.returncode}")

    except FileNotFoundError:
        print("Pytest not found. Running basic tests...")
        run_basic_tests()


def run_basic_tests():
    """Run basic functionality tests"""
    print("Running basic functionality tests...")

    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../data_loader/src"))

    try:
        from data_loader import DataLoaderModule, DataNormalizer, BaseDataset, MultiSourceDataset

        # Test DataNormalizer
        print("Testing DataNormalizer...")
        normalizer = DataNormalizer("standard")
        data = {"test": np.array([[1, 2], [3, 4]])}
        normalized = normalizer.fit_transform(data)
        print(f"✓ DataNormalizer works: {normalized['test'].shape}")

        # Test BaseDataset
        print("Testing BaseDataset...")
        dataset = BaseDataset(np.array([[1, 2], [3, 4]]), np.array([0, 1]))
        assert len(dataset) == 2
        print(f"✓ BaseDataset works: length={len(dataset)}")

        # Test MultiSourceDataset
        print("Testing MultiSourceDataset...")
        data_dict = {"x1": np.array([[1, 2], [3, 4]]), "x2": np.array([[5], [6]])}
        ms_dataset = MultiSourceDataset(data_dict, np.array([0, 1]))
        assert len(ms_dataset) == 2
        print(f"✓ MultiSourceDataset works: length={len(ms_dataset)}")

        print("\nAll basic tests passed! ✓")

    except Exception as e:
        print(f"Error in basic tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_pytest()
