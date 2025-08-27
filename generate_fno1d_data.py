"""
Script to generate test datasets for FNO1D model
Data dimensions: (batch, channel, height)
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def generate_fno1d_data(n_samples=1000, n_channels=1, height=64, noise_level=0.1):
    """
    Generate a 1D function dataset for testing FNO1D model.
    
    The function is a combination of sine waves with different frequencies and phases.
    
    Args:
        n_samples: Number of samples to generate
        n_channels: Number of channels in the data
        height: Height (length) of the 1D data
        noise_level: Level of noise to add to the data
        
    Returns:
        features: Input features with shape (n_samples, n_channels, height)
        targets: Target values with shape (n_samples, n_channels, height)
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create spatial domain
    x = np.linspace(0, 1, height)
    
    # Initialize arrays
    features = np.zeros((n_samples, n_channels, height))
    targets = np.zeros((n_samples, n_channels, height))
    
    for i in range(n_samples):
        for c in range(n_channels):
            # Random parameters for input function
            freq1 = np.random.uniform(2, 10)
            freq2 = np.random.uniform(1, 5)
            phase1 = np.random.uniform(0, 2*np.pi)
            phase2 = np.random.uniform(0, 2*np.pi)
            amplitude1 = np.random.uniform(0.5, 2.0)
            amplitude2 = np.random.uniform(0.2, 1.0)
            
            # Input function: combination of sine waves
            features[i, c, :] = (amplitude1 * np.sin(2 * np.pi * freq1 * x + phase1) +
                                amplitude2 * np.sin(2 * np.pi * freq2 * x + phase2))
            
            # Target function: solution to a simple differential equation
            # For example, we can use a smoothed version of the derivative
            target = amplitude1 * freq1 * np.cos(2 * np.pi * freq1 * x + phase1) + \
                    amplitude2 * freq2 * np.cos(2 * np.pi * freq2 * x + phase2)
            
            # Add smoothing to make it more realistic
            target = np.convolve(target, np.ones(5)/5, mode='same')
            
            # Add noise
            noise = np.random.normal(0, noise_level, height)
            targets[i, c, :] = target + noise
    
    return features, targets


def save_fno1d_dataset(features, targets, output_dir):
    """
    Save the FNO1D dataset to numpy files
    
    Args:
        features: Input features
        targets: Target values
        output_dir: Directory to save the files
    """
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )
    
    # Save to numpy files
    np.save(os.path.join(output_dir, 'train_features.npy'), X_train)
    np.save(os.path.join(output_dir, 'train_targets.npy'), y_train)
    np.save(os.path.join(output_dir, 'test_features.npy'), X_test)
    np.save(os.path.join(output_dir, 'test_targets.npy'), y_test)
    
    print(f"FNO1D dataset saved to {output_dir}")
    print(f"Train features shape: {X_train.shape}")
    print(f"Train targets shape: {y_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Test targets shape: {y_test.shape}")
    
    # Save some statistics
    stats = {
        'train_features_min': float(X_train.min()),
        'train_features_max': float(X_train.max()),
        'train_targets_min': float(y_train.min()),
        'train_targets_max': float(y_train.max()),
        'test_features_min': float(X_test.min()),
        'test_features_max': float(X_test.max()),
        'test_targets_min': float(y_test.min()),
        'test_targets_max': float(y_test.max())
    }
    
    import json
    with open(os.path.join(output_dir, 'dataset_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("Dataset statistics saved to dataset_stats.json")


def convert_to_csv(features, targets, output_dir):
    """
    Convert a sample of the data to CSV format for inspection
    
    Args:
        features: Input features
        targets: Target values
        output_dir: Directory to save the CSV files
    """
    # Take a sample of the data for CSV conversion (first 10 samples)
    n_samples = min(10, features.shape[0])
    sample_features = features[:n_samples]
    sample_targets = targets[:n_samples]
    
    for i in range(n_samples):
        # Save each sample as a separate CSV file
        feature_df = pd.DataFrame(sample_features[i].T, columns=[f'channel_{c}' for c in range(sample_features.shape[1])])
        target_df = pd.DataFrame(sample_targets[i].T, columns=[f'channel_{c}' for c in range(sample_targets.shape[1])])
        
        feature_df.to_csv(os.path.join(output_dir, f'sample_{i}_features.csv'), index=False)
        target_df.to_csv(os.path.join(output_dir, f'sample_{i}_targets.csv'), index=False)
    
    print(f"Sample data (first {n_samples} samples) saved as CSV files")


def main():
    """
    Main function to generate and save the FNO1D test dataset
    """
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'fno1d')
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate dataset
    print("Generating FNO1D dataset...")
    features, targets = generate_fno1d_data(n_samples=1000, n_channels=1, height=64, noise_level=0.1)
    
    # Save dataset
    print("Saving FNO1D dataset...")
    save_fno1d_dataset(features, targets, data_dir)
    
    # Convert sample to CSV for inspection
    print("Converting sample data to CSV...")
    convert_to_csv(features, targets, data_dir)
    
    # Print some information
    print("\nDataset Information:")
    print(f"Data dimensions: (batch, channel, height) = {features.shape}")
    print(f"Feature range: [{features.min():.2f}, {features.max():.2f}]")
    print(f"Target range: [{targets.min():.2f}, {targets.max():.2f}]")


if __name__ == "__main__":
    main()