#!/usr/bin/env python3
"""
Integration test for data loader module
Tests complete end-to-end functionality
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../data_loader/src'))

from data_loader import DataLoaderModule


class IntegrationTester:
    """Integration test class for data loader"""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        
    def create_test_data(self):
        """Create temporary test data"""
        temp_dir = tempfile.mkdtemp()
        
        # Create train data
        train_x1 = np.array([[1.2, 3.4, 5.6], [2.1, 4.5, 6.7], [3.2, 5.6, 7.8], 
                           [4.1, 6.7, 8.9], [5.2, 7.8, 9.0], [6.1, 8.9, 10.1]])
        train_x2 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], 
                           [0.7, 0.8], [0.9, 1.0], [1.1, 1.2]])
        train_y = np.array([0, 1, 0, 1, 0, 1])
        
        # Create test data
        test_x1 = np.array([[1.5, 3.8, 5.9], [2.3, 4.7, 6.8], [3.4, 5.8, 7.9]])
        test_x2 = np.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]])
        test_y = np.array([1, 0, 1])
        
        # Save as CSV files
        pd.DataFrame(train_x1, columns=["f1", "f2", "f3"]).to_csv(
            Path(temp_dir) / "train_x1.csv", index=False)
        pd.DataFrame(train_x2, columns=["s1", "s2"]).to_csv(
            Path(temp_dir) / "train_x2.csv", index=False)
        pd.DataFrame(train_y, columns=["target"]).to_csv(
            Path(temp_dir) / "train_y.csv", index=False)
        pd.DataFrame(test_x1, columns=["f1", "f2", "f3"]).to_csv(
            Path(temp_dir) / "test_x1.csv", index=False)
        pd.DataFrame(test_x2, columns=["s1", "s2"]).to_csv(
            Path(temp_dir) / "test_x2.csv", index=False)
        pd.DataFrame(test_y, columns=["target"]).to_csv(
            Path(temp_dir) / "test_y.csv", index=False)
        
        return temp_dir
    
    def test_standard_pipeline(self):
        """Test standard single-source pipeline"""
        print("=" * 60)
        print("Testing Standard Pipeline...")
        
        temp_dir = self.create_test_data()
        
        try:
            config = {
                "train_features_path": str(Path(temp_dir) / "train_x1.csv"),
                "train_targets_path": str(Path(temp_dir) / "train_y.csv"),
                "test_features_path": str(Path(temp_dir) / "test_x1.csv"),
                "test_targets_path": str(Path(temp_dir) / "test_y.csv"),
                "batch_size": 2,
                "shuffle": True,
                "normalize": True,
                "normalization_method": "standard"
            }
            
            loader = DataLoaderModule(config)
            loader.prepare_datasets()
            loader.create_data_loaders()
            
            train_loader, test_loader = loader.get_data_loaders()
            info = loader.get_dataset_info()
            
            # Assertions
            assert train_loader is not None, "Train loader should not be None"
            assert test_loader is not None, "Test loader should not be None"
            assert info["multi_source"] is False, "Should be single source"
            assert info["normalized"] is True, "Should be normalized"
            
            # Check data shapes and normalization
            for batch_idx, (data, target) in enumerate(train_loader):
                print(f"Standard batch {batch_idx}: Data shape {data.shape}, Target shape {target.shape}")
                assert data.shape[0] <= config["batch_size"], "Batch size should not exceed config"
                
                # Check if normalization worked (mean should be ~0)
                if batch_idx == 0:
                    data_mean = data.numpy().mean(axis=0)
                    print(f"Data mean after normalization: {data_mean.mean():.4f}")
                break
            
            print("✓ Standard pipeline test passed!")
            return True
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_multi_source_pipeline(self):
        """Test multi-source pipeline"""
        print("=" * 60)
        print("Testing Multi-Source Pipeline...")
        
        temp_dir = self.create_test_data()
        
        try:
            config = {
                "train_features_paths": {
                    "x1": str(Path(temp_dir) / "train_x1.csv"),
                    "x2": str(Path(temp_dir) / "train_x2.csv")
                },
                "train_targets_path": str(Path(temp_dir) / "train_y.csv"),
                "test_features_paths": {
                    "x1": str(Path(temp_dir) / "test_x1.csv"),
                    "x2": str(Path(temp_dir) / "test_x2.csv")
                },
                "test_targets_path": str(Path(temp_dir) / "test_y.csv"),
                "batch_size": 2,
                "shuffle": True,
                "normalize": True,
                "normalization_method": "minmax"
            }
            
            loader = DataLoaderModule(config)
            loader.prepare_datasets()
            loader.create_data_loaders()
            
            train_loader, test_loader = loader.get_data_loaders()
            info = loader.get_dataset_info()
            
            # Assertions
            assert train_loader is not None, "Train loader should not be None"
            assert test_loader is not None, "Test loader should not be None"
            assert info["multi_source"] is True, "Should be multi-source"
            assert info["normalized"] is True, "Should be normalized"
            assert len(info["train_data_sources"]) == 2, "Should have 2 data sources"
            
            # Check data shapes with multiple sources
            for batch_idx, (data_dict, target) in enumerate(train_loader):
                print(f"Multi-source batch {batch_idx}:")
                for name, data in data_dict.items():
                    print(f"  {name}: {data.shape}")
                print(f"  Target: {target.shape}")
                
                # Verify tensor types
                assert isinstance(data_dict["x1"], np.ndarray) or hasattr(data_dict["x1"], 'numpy'), "Should be tensor-like"
                break
            
            print("✓ Multi-source pipeline test passed!")
            return True
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_data_consistency(self):
        """Test data consistency across sources"""
        print("=" * 60)
        print("Testing Data Consistency...")
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create inconsistent data
            x1_path = Path(temp_dir) / "x1.csv"
            x2_path = Path(temp_dir) / "x2.csv"
            y_path = Path(temp_dir) / "y.csv"
            
            # x1 has 3 samples, x2 has 4 samples - should fail
            pd.DataFrame([[1, 2], [3, 4], [5, 6]]).to_csv(x1_path, index=False)
            pd.DataFrame([[7], [8], [9], [10]]).to_csv(x2_path, index=False)
            pd.DataFrame([0, 1, 0, 1]).to_csv(y_path, index=False)
            
            config = {
                "train_features_paths": {
                    "x1": str(x1_path),
                    "x2": str(x2_path)
                },
                "train_targets_path": str(y_path)
            }
            
            loader = DataLoaderModule(config)
            
            try:
                loader.prepare_datasets()
                assert False, "Should have raised ValueError for inconsistent shapes"
            except ValueError as e:
                print(f"✓ Correctly caught inconsistency error: {e}")
                return True
                
        finally:
            shutil.rmtree(temp_dir)
    
    def test_edge_cases(self):
        """Test edge cases"""
        print("=" * 60)
        print("Testing Edge Cases...")
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Test 1D data normalization
            data_path = Path(temp_dir) / "data.csv"
            target_path = Path(temp_dir) / "target.csv"
            
            # Create 1D data
            data = np.array([1, 2, 3, 4, 5])
            targets = np.array([0, 1, 0, 1, 0])
            
            pd.DataFrame(data).to_csv(data_path, index=False)
            pd.DataFrame(targets).to_csv(target_path, index=False)
            
            config = {
                "train_features_path": str(data_path),
                "train_targets_path": str(target_path),
                "normalize": True,
                "normalization_method": "robust"
            }
            
            loader = DataLoaderModule(config)
            loader.prepare_datasets()
            
            info = loader.get_dataset_info()
            assert info["normalized"] is True
            assert info["train_data_shape"] == (5, 1), "1D data should be reshaped"
            print("✓ 1D data handling works correctly!")
            
        finally:
            shutil.rmtree(temp_dir)
    
    def run_all_tests(self):
        """Run all integration tests"""
        print("🚀 Starting Data Loader Integration Tests")
        print("=" * 80)
        
        tests = [
            self.test_standard_pipeline,
            self.test_multi_source_pipeline,
            self.test_data_consistency,
            self.test_edge_cases
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            try:
                test()
                passed += 1
                print()
            except Exception as e:
                print(f"❌ {test.__name__} failed: {e}")
                failed += 1
                print()
        
        print("=" * 80)
        print(f"Integration Tests Completed!")
        print(f"Passed: {passed}, Failed: {failed}")
        
        if failed == 0:
            print("🎉 All integration tests passed!")
        else:
            print(f"⚠️  {failed} test(s) failed")
        
        return failed == 0


def main():
    """Main function"""
    tester = IntegrationTester()
    success = tester.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())