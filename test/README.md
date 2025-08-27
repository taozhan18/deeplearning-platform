# Test Suite

This directory contains the test suite for the low-code deep learning platform.

## Directory Structure

```
test/
├── data_loader/           # Tests for the data loader module
├── model/                 # Tests for model implementations
├── training_engine/       # Tests for the training engine module
├── run_tests.py           # Main test runner script
└── README.md              # This file
```

## Test Organization

### Data Loader Tests
Tests for the data loading and preprocessing functionality:
- `test_data_loader.py` - Tests for basic data loader functionality

### Model Tests
Tests for model implementations:
- `test_platform.py` - General platform integration tests
- `test_fno.py` - Tests specifically for the FNO model implementation

### Training Engine Tests
Tests for the training engine functionality:
- `test_training_engine.py` - Tests for the training engine components

## Running Tests

To run all tests, execute the main test runner:

```bash
python test/run_tests.py
```

This will run all test scripts in sequence and provide a summary of the results.

## Individual Test Execution

You can also run individual test scripts directly:

```bash
# Run platform integration tests
python test/model/test_platform.py

# Run FNO model tests
python test/model/test_fno.py

# Run data loader tests
python test/data_loader/test_data_loader.py

# Run training engine tests
python test/training_engine/test_training_engine.py
```

## Test Environment

All tests should be run in the `physics` conda environment which contains all the necessary dependencies.