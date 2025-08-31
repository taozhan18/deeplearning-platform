"""
Script to generate test datasets for the low-code deep learning platform
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_complex_function_data(n_samples=10000, noise_level=0.1):
    """
    Generate a dataset based on a complex function for testing the deep learning platform.
    
    The function is: 
    y = sin(x1) * cos(x2) + exp(-x3^2) + x4 * x5 + noise
    
    Args:
        n_samples: Number of samples to generate
        noise_level: Level of noise to add to the target values
        
    Returns:
        features: Input features (n_samples, 5)
        targets: Target values (n_samples,)
    """
    # Generate random features
    np.random.seed(42)  # For reproducibility
    x1 = np.random.uniform(-2*np.pi, 2*np.pi, n_samples)
    x2 = np.random.uniform(-2*np.pi, 2*np.pi, n_samples)
    x3 = np.random.uniform(-2, 2, n_samples)
    x4 = np.random.uniform(-5, 5, n_samples)
    x5 = np.random.uniform(-5, 5, n_samples)
    
    # Combine features
    features = np.column_stack([x1, x2, x3, x4, x5])
    
    # Compute target values using a complex function
    y = (np.sin(x1) * np.cos(x2) + 
         np.exp(-x3**2) + 
         x4 * x5)
    
    # Add noise
    noise = np.random.normal(0, noise_level, n_samples)
    targets = y + noise
    
    # Normalize targets to [0, 1] range for classification
    targets = (targets - targets.min()) / (targets.max() - targets.min())
    
    # Convert to 10-class classification problem
    targets = np.floor(targets * 10).astype(int)
    # Make sure we have exactly 10 classes (0-9)
    targets = np.clip(targets, 0, 9)
    
    return features, targets


def save_dataset(features, targets, output_dir):
    """
    Save the dataset to CSV files
    
    Args:
        features: Input features
        targets: Target values
        output_dir: Directory to save the files
    """
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )
    
    # Create dataframes
    train_features_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(1, 6)])
    train_targets_df = pd.DataFrame(y_train, columns=['target'])
    
    test_features_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(1, 6)])
    test_targets_df = pd.DataFrame(y_test, columns=['target'])
    
    # Save to CSV files
    train_features_df.to_csv(os.path.join(output_dir, 'train_features.csv'), index=False)
    train_targets_df.to_csv(os.path.join(output_dir, 'train_targets.csv'), index=False)
    test_features_df.to_csv(os.path.join(output_dir, 'test_features.csv'), index=False)
    test_targets_df.to_csv(os.path.join(output_dir, 'test_targets.csv'), index=False)
    
    print(f"Dataset saved to {output_dir}")
    print(f"Train features shape: {X_train.shape}")
    print(f"Train targets shape: {y_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Test targets shape: {y_test.shape}")
    print(f"Target classes: {np.unique(y_train)}")


def main():
    """
    Main function to generate and save the test dataset
    """
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate dataset
    print("Generating complex function dataset...")
    features, targets = generate_complex_function_data(n_samples=10000, noise_level=0.1)
    
    # Save dataset
    print("Saving dataset...")
    save_dataset(features, targets, data_dir)
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Features range: [{features.min():.2f}, {features.max():.2f}]")
    print(f"Targets range: [{targets.min()}, {targets.max()}]")
    print(f"Target distribution:")
    unique, counts = np.unique(targets, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} samples ({c/len(targets)*100:.1f}%)")


if __name__ == "__main__":
    main()