"""
Test script for FNO1D model using the main training function
"""

import sys
import os
import subprocess
import json


def test_fno1d_with_main_train():
    """Test FNO1D model using the main training function"""
    print("Testing FNO1D model using the main training function...")
    
    # Get the base path
    base_path = os.path.join(os.path.dirname(__file__), '..', '..')
    
    # Configuration file path
    config_path = os.path.join(base_path, 'fno1d_config.yaml')
    
    # Check if config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Run the main training function as a subprocess
    print("Running main training function...")
    try:
        result = subprocess.run([
            'python', os.path.join(base_path, 'main', 'train.py'),
            '--config', config_path
        ], 
        cwd=base_path,
        capture_output=True, 
        text=True, 
        timeout=300)  # 5 minute timeout
        
        # Print output
        if result.stdout:
            print(result.stdout)
            
        if result.stderr:
            print("STDERR:", result.stderr)
            
        # Check result
        if result.returncode == 0:
            print("✅ FNO1D training with main function completed successfully")
        else:
            print("❌ FNO1D training with main function failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ FNO1D training timed out")
        return False
    except Exception as e:
        print(f"❌ FNO1D training error: {str(e)}")
        return False
    
    # Check if output files were created
    model_path = os.path.join(base_path, 'data', 'fno1d', 'fno1d_model.pth')
    history_path = os.path.join(base_path, 'data', 'fno1d', 'fno1d_training_history.json')
    
    if os.path.exists(model_path):
        print(f"✅ Model saved to {model_path}")
    else:
        print(f"❌ Model not found at {model_path}")
        return False
        
    if os.path.exists(history_path):
        print(f"✅ Training history saved to {history_path}")
        # Read and display final metrics
        with open(history_path, 'r') as f:
            history = json.load(f)
        print(f"Final training loss: {history['train_loss'][-1]:.6f}")
        print(f"Final validation loss: {history['val_loss'][-1]:.6f}")
    else:
        print(f"❌ Training history not found at {history_path}")
        return False
    
    return True


def main():
    """Main test function"""
    print("Testing FNO1D Model with Main Training Function")
    print("=" * 50)
    
    try:
        success = test_fno1d_with_main_train()
        if success:
            print("\n" + "=" * 50)
            print("FNO1D test with main training function completed successfully!")
        else:
            print("\n" + "=" * 50)
            print("FNO1D test with main training function failed!")
            return 1
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())