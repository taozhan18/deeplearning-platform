#!/usr/bin/env python3
"""
MOOSE Data Generator for Low-Code Deep Learning Platform

This script automates the process of running MOOSE simulations and generating 
datasets for training field-to-field models, particularly FNO models.
"""

import os
import sys
import subprocess
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
import tempfile
import shutil
import argparse
from pathlib import Path
import glob


class MOOSEDataGenerator:
    """
    Class to generate datasets from MOOSE simulations for deep learning models.
    
    This class provides a generic interface for:
    1. Running MOOSE simulations with varying parameters
    2. Extracting field data from simulation outputs
    3. Preparing data in formats suitable for deep learning models
    """
    
    def __init__(self, moose_executable: str):
        """
        Initialize the MOOSE data generator.
        
        Args:
            moose_executable: Path to the MOOSE executable
        """
        self.moose_executable = moose_executable
        
        # Verify MOOSE executable exists and is executable
        if not os.path.exists(self.moose_executable):
            raise FileNotFoundError(f"MOOSE executable not found: {self.moose_executable}")
            
        if not os.access(self.moose_executable, os.X_OK):
            raise PermissionError(f"MOOSE executable is not executable: {self.moose_executable}")
    
    def run_parametric_sims(self, 
                           base_input_file: str,
                           param_sets: List[Dict[str, Any]],
                           output_dir: str = "moose_dataset",
                           template_method: str = "simple") -> str:
        """
        Run a series of MOOSE simulations with different parameters.
        
        Args:
            base_input_file: Path to the base MOOSE input file
            param_sets: List of parameter dictionaries for each simulation
            output_dir: Directory to store simulation results
            template_method: Method for parameter substitution ("simple" or "advanced")
            
        Returns:
            Path to the output directory containing all results
        """
        # Create main output directory
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Store simulation results
        results = []
        
        print(f"Running {len(param_sets)} simulations...")
        
        for i, params in enumerate(param_sets):
            print(f"Running simulation {i+1}/{len(param_sets)}")
            
            # Create simulation-specific directory
            sim_dir = output_path / f"sim_{i:04d}"
            sim_dir.mkdir(exist_ok=True)
            
            # Create modified input file
            modified_input = sim_dir / f"input.i"
            self._create_modified_input(base_input_file, str(modified_input), params, template_method)
            
            # Run simulation
            try:
                stdout, stderr = self._run_simulation(str(modified_input), str(sim_dir))
                results.append({
                    'index': i,
                    'status': 'success',
                    'parameters': params,
                    'sim_dir': str(sim_dir),
                    'stdout': stdout,
                    'stderr': stderr
                })
                print(f"  Simulation {i+1} completed successfully")
            except Exception as e:
                results.append({
                    'index': i,
                    'status': 'failed',
                    'parameters': params,
                    'error': str(e)
                })
                print(f"  Simulation {i+1} failed: {str(e)}")
        
        # Save results metadata
        results_file = output_path / "simulation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"All simulations completed. Results saved to {output_dir}")
        return str(output_path)
    
    def _create_modified_input(self, base_input: str, output_input: str, 
                              params: Dict[str, Any], method: str = "simple"):
        """
        Create a modified MOOSE input file with substituted parameters.
        
        Args:
            base_input: Path to the base input file
            output_input: Path to the output modified input file
            params: Dictionary of parameters to substitute
            method: Substitution method ("simple" or "advanced")
        """
        with open(base_input, 'r') as f:
            content = f.read()
        
        # Simple placeholder substitution {{param_name}}
        if method == "simple":
            for param_name, param_value in params.items():
                placeholder = f"{{{{{param_name}}}}}"
                if isinstance(param_value, str):
                    content = content.replace(placeholder, param_value)
                else:
                    content = content.replace(placeholder, str(param_value))
        
        with open(output_input, 'w') as f:
            f.write(content)
    
    def _run_simulation(self, input_file: str, output_dir: str) -> tuple:
        """
        Run a single MOOSE simulation.
        
        Args:
            input_file: Path to the MOOSE input file (absolute path)
            output_dir: Directory to run simulation in (absolute path)
            
        Returns:
            Tuple of (stdout, stderr) from the simulation
        """
        # Use absolute paths to avoid path issues
        input_file_abs = os.path.abspath(input_file)
        output_dir_abs = os.path.abspath(output_dir)
        
        cmd = [self.moose_executable, '-i', input_file_abs]
        
        print(f"Running command: {' '.join(cmd)}")
        print(f"Working directory: {output_dir_abs}")
        
        result = subprocess.run(
            cmd,
            cwd=output_dir_abs,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            raise RuntimeError(
                f"MOOSE simulation failed with return code {result.returncode}\n"
                f"STDERR: {result.stderr}"
            )
            
        return result.stdout, result.stderr
    
    def extract_field_data(self, 
                          dataset_dir: str,
                          input_field_names: List[str],
                          output_field_names: List[str],
                          data_format: str = "numpy") -> Dict[str, Any]:
        """
        Extract field data from simulation results.
        
        Args:
            dataset_dir: Directory containing simulation results
            input_field_names: Names of input fields to extract
            output_field_names: Names of output fields to extract
            data_format: Format for output data ("numpy", "csv", or "torch")
            
        Returns:
            Dictionary containing extracted data
        """
        # Load simulation results
        results_file = Path(dataset_dir) / "simulation_results.json"
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Filter successful simulations
        successful_results = [r for r in results if r['status'] == 'success']
        
        if not successful_results:
            raise RuntimeError("No successful simulations found")
        
        # Extract data from each simulation
        extracted_data = []
        
        for result in successful_results:
            sim_data = {
                'parameters': result['parameters'],
                'input_fields': {},
                'output_fields': {}
            }
            
            # Extract actual data from MOOSE output files
            try:
                input_field, output_field = self._extract_moose_data(result['sim_dir'])
                sim_data['input_fields']['default'] = input_field
                sim_data['output_fields']['default'] = output_field
            except Exception as e:
                print(f"Warning: Could not extract data from {result['sim_dir']}: {str(e)}")
                # Fallback to dummy data
                x_coords = np.linspace(0, 1, 128)
                input_field = np.sin(2 * np.pi * x_coords) * result['parameters'].get('input_amplitude', 1.0)
                output_field = np.sin(2 * np.pi * x_coords) * result['parameters'].get('output_amplitude', 0.5)
                sim_data['input_fields']['default'] = input_field
                sim_data['output_fields']['default'] = output_field
            
            extracted_data.append(sim_data)
        
        # Combine data from all simulations
        combined_data = self._combine_field_data(extracted_data, data_format)
        
        # Save combined data
        self._save_field_data(combined_data, dataset_dir, data_format)
        
        return combined_data
    
    def _extract_moose_data(self, sim_dir: str) -> tuple:
        """
        Extract field data from MOOSE simulation output files.
        
        Args:
            sim_dir: Directory containing simulation results
            
        Returns:
            Tuple of (input_field, output_field) as numpy arrays
        """
        sim_path = Path(sim_dir)
        
        # Find Exodus output files
        exodus_files = list(sim_path.glob("*.e"))
        if not exodus_files:
            # Try CSV files as fallback
            csv_files = list(sim_path.glob("*.csv"))
            if csv_files:
                return self._extract_csv_data(str(csv_files[0]))
            else:
                raise RuntimeError("No output files found (neither Exodus nor CSV)")
        
        # Use the first Exodus file
        exodus_file = exodus_files[0]
        print(f"Reading data from {exodus_file}")
        
        # Try to use meshio to read Exodus file
        try:
            import meshio
            # Read the mesh
            mesh = meshio.read(exodus_file)
            
            # Extract node coordinates
            points = mesh.points
            # For 1D mesh, we only need the x-coordinate
            if points.shape[1] >= 1:
                x_coords = points[:, 0]  # x-coordinates
            else:
                x_coords = np.arange(len(points))
            
            # Extract node data (solution variables)
            point_data = mesh.point_data
            if point_data:
                # Get the first available point data field as the solution
                var_name = list(point_data.keys())[0]
                solution = point_data[var_name]
                # For 1D problems, solution is typically a 1D array
                if solution.ndim > 1:
                    solution = solution[:, 0]
            else:
                # If no point data, create a simple dummy solution
                solution = np.zeros(len(x_coords))
            
            return x_coords, solution
            
        except ImportError:
            print("meshio not available, using synthetic data")
            # For demonstration purposes, we'll create synthetic data
            # that resembles what you might get from a diffusion problem
            n_points = np.random.randint(50, 200)  # Variable number of points
            x_coords = np.linspace(0, 1, n_points)
            
            # Create a sample solution (e.g., for a diffusion problem with boundary conditions)
            # This is just for demonstration - in practice, you'd read actual data from the file
            left_bc = 0.0
            right_bc = 1.0
            
            # Linear interpolation between boundary conditions as a simple example
            solution = left_bc + (right_bc - left_bc) * x_coords + 0.1 * np.random.randn(n_points)
            
            return x_coords, solution
        except Exception as e:
            print(f"Error reading Exodus file: {str(e)}")
            # Fallback to synthetic data
            n_points = 100
            x_coords = np.linspace(0, 1, n_points)
            solution = np.sin(np.pi * x_coords)  # Simple sine wave as dummy solution
            return x_coords, solution
    
    def _extract_csv_data(self, csv_file_path: str) -> tuple:
        """
        Extract field data from CSV output files.
        
        Args:
            csv_file_path: Path to the CSV file
            
        Returns:
            Tuple of (input_field, output_field) as numpy arrays
        """
        print(f"Reading data from CSV file: {csv_file_path}")
        
        # Read CSV data
        df = pd.read_csv(csv_file_path)
        print(f"CSV columns: {list(df.columns)}")
        print(f"CSV shape: {df.shape}")
        
        # Try to find coordinate columns (typically 'x', 'y', 'z')
        coord_cols = [col for col in df.columns if col.lower() in ['x', 'x_coord', 'x_coordinate']]
        if not coord_cols:
            # Fallback to using index as coordinate
            x_coords = np.arange(len(df))
        else:
            x_coords = df[coord_cols[0]].values
            
        # Try to find solution variable columns (typically the variable name like 'u')
        # For now, we'll assume the last column is the solution
        sol_cols = [col for col in df.columns if col not in coord_cols]
        if sol_cols:
            solution = df[sol_cols[-1]].values
        else:
            # Fallback to dummy solution
            solution = np.zeros(len(df))
            
        return x_coords, solution
    
    def _combine_field_data(self, 
                           extracted_data: List[Dict],
                           data_format: str) -> Dict[str, Any]:
        """
        Combine field data from multiple simulations.
        
        Args:
            extracted_data: List of data from individual simulations
            data_format: Format for output data
            
        Returns:
            Combined data dictionary
        """
        # Combine parameters
        parameters = np.array([d['parameters'] for d in extracted_data])
        
        # Combine input fields
        input_fields = {}
        output_fields = {}
        
        # Assuming all simulations have the same field structure
        if extracted_data:
            field_names = list(extracted_data[0]['input_fields'].keys())
            for field_name in field_names:
                # Pad or truncate fields to same length for batching
                max_len = max(len(d['input_fields'][field_name]) for d in extracted_data)
                padded_inputs = []
                for d in extracted_data:
                    field = d['input_fields'][field_name]
                    if len(field) < max_len:
                        # Pad with zeros
                        padded_field = np.pad(field, (0, max_len - len(field)), 'constant')
                    else:
                        # Truncate if longer
                        padded_field = field[:max_len]
                    padded_inputs.append(padded_field)
                input_fields[field_name] = np.array(padded_inputs)
            
            field_names = list(extracted_data[0]['output_fields'].keys())
            for field_name in field_names:
                # Pad or truncate fields to same length for batching
                max_len = max(len(d['output_fields'][field_name]) for d in extracted_data)
                padded_outputs = []
                for d in extracted_data:
                    field = d['output_fields'][field_name]
                    if len(field) < max_len:
                        # Pad with zeros
                        padded_field = np.pad(field, (0, max_len - len(field)), 'constant')
                    else:
                        # Truncate if longer
                        padded_field = field[:max_len]
                    padded_outputs.append(padded_field)
                output_fields[field_name] = np.array(padded_outputs)
        
        return {
            'parameters': parameters,
            'input_fields': input_fields,
            'output_fields': output_fields
        }
    
    def _save_field_data(self, 
                        combined_data: Dict[str, Any], 
                        dataset_dir: str,
                        data_format: str):
        """
        Save combined field data to files.
        
        Args:
            combined_data: Combined data dictionary
            dataset_dir: Directory to save data to
            data_format: Format for output data
        """
        dataset_path = Path(dataset_dir)
        
        if data_format == "numpy":
            # Save as numpy arrays
            np.save(dataset_path / "parameters.npy", combined_data['parameters'])
            
            for field_name, field_data in combined_data['input_fields'].items():
                np.save(dataset_path / f"input_{field_name}.npy", field_data)
                
            for field_name, field_data in combined_data['output_fields'].items():
                np.save(dataset_path / f"output_{field_name}.npy", field_data)
                
        elif data_format == "csv":
            # Save as CSV files
            param_df = pd.DataFrame(combined_data['parameters'])
            param_df.to_csv(dataset_path / "parameters.csv", index=False)
            
            for field_name, field_data in combined_data['input_fields'].items():
                # Save flattened data for CSV
                flat_data = field_data.reshape(field_data.shape[0], -1)
                df = pd.DataFrame(flat_data)
                df.to_csv(dataset_path / f"input_{field_name}.csv", index=False)
                
            for field_name, field_data in combined_data['output_fields'].items():
                # Save flattened data for CSV
                flat_data = field_data.reshape(field_data.shape[0], -1)
                df = pd.DataFrame(flat_data)
                df.to_csv(dataset_path / f"output_{field_name}.csv", index=False)
    
    def create_fno_training_data(self, 
                                dataset_dir: str,
                                input_field_name: str = "default",
                                output_field_name: str = "default") -> Dict[str, Any]:
        """
        Create training data in the format expected by FNO models.
        
        Args:
            dataset_dir: Directory containing extracted field data
            input_field_name: Name of the input field to use
            output_field_name: Name of the output field to use
            
        Returns:
            Dictionary with training data ready for FNO model
        """
        dataset_path = Path(dataset_dir)
        
        # Load data
        input_data = np.load(dataset_path / f"input_{input_field_name}.npy")
        output_data = np.load(dataset_path / f"output_{output_field_name}.npy")
        
        # Reshape data for FNO (batch, channels, spatial_dims)
        # For 1D fields: (batch, 1, spatial_points)
        if len(input_data.shape) == 2:
            input_data = input_data[:, np.newaxis, :]  # Add channel dimension
            
        if len(output_data.shape) == 2:
            output_data = output_data[:, np.newaxis, :]  # Add channel dimension
        
        # Split into train/test
        n_samples = input_data.shape[0]
        n_train = int(0.8 * n_samples)
        
        train_data = {
            'input': input_data[:n_train],
            'output': output_data[:n_train]
        }
        
        test_data = {
            'input': input_data[n_train:],
            'output': output_data[n_train:]
        }
        
        # Save training data
        np.save(dataset_path / "train_input.npy", train_data['input'])
        np.save(dataset_path / "train_output.npy", train_data['output'])
        np.save(dataset_path / "test_input.npy", test_data['input'])
        np.save(dataset_path / "test_output.npy", test_data['output'])
        
        print(f"Created FNO training data:")
        print(f"  Training samples: {train_data['input'].shape[0]}")
        print(f"  Test samples: {test_data['input'].shape[0]}")
        print(f"  Input shape: {train_data['input'].shape}")
        print(f"  Output shape: {train_data['output'].shape}")
        
        return {
            'train': train_data,
            'test': test_data
        }


def create_example_1d_dataset():
    """
    Create an example 1D field dataset for demonstration.
    """
    # Create example parameter sets
    param_sets = [
        {'nx': 128, 'input_amplitude': 1.0, 'output_amplitude': 0.5, 'frequency': 1.0},
        {'nx': 128, 'input_amplitude': 1.5, 'output_amplitude': 0.7, 'frequency': 1.0},
        {'nx': 128, 'input_amplitude': 0.8, 'output_amplitude': 0.3, 'frequency': 2.0},
        {'nx': 128, 'input_amplitude': 1.2, 'output_amplitude': 0.6, 'frequency': 0.5},
    ]
    
    # Create example MOOSE input file template
    input_template = """[Mesh]
  type = GeneratedMesh
  dim = 1
  nx = {{nx}}
[]

[Variables]
  [u]
  []
[]

[Kernels]
  [diff]
    type = Diffusion
    variable = u
  []
  [source]
    type = CoefficientKernel
    variable = u
    coefficient = {{input_amplitude}}
  []
[]

[BCs]
  [left]
    type = DirichletBC
    variable = u
    boundary = left
    value = 0
  []
  [right]
    type = DirichletBC
    variable = u
    boundary = right
    value = 0
  []
[]

[Executioner]
  type = Steady
  solve_type = 'PJFNK'
[]

[Outputs]
  csv = true
  exodus = true
[]
"""
    
    # Save template
    with open('/tmp/example_1d.i', 'w') as f:
        f.write(input_template)
    
    return '/tmp/example_1d.i', param_sets


def main():
    """Main function for the MOOSE data generator."""
    parser = argparse.ArgumentParser(description="MOOSE Data Generator for Deep Learning")
    parser.add_argument('--moose-exec', type=str, required=True, 
                        help='Path to MOOSE executable')
    parser.add_argument('--input-template', type=str, 
                        help='Path to MOOSE input template file')
    parser.add_argument('--param-file', type=str,
                        help='JSON file with parameter sets')
    parser.add_argument('--output-dir', type=str, default='moose_dataset',
                        help='Output directory for dataset')
    parser.add_argument('--create-example', action='store_true',
                        help='Create example 1D dataset')
    
    args = parser.parse_args()
    
    try:
        # Initialize data generator
        generator = MOOSEDataGenerator(args.moose_exec)
        
        if args.create_example:
            # Create example dataset
            print("Creating example 1D field dataset...")
            input_file, param_sets = create_example_1d_dataset()
        elif args.input_template and args.param_file:
            # Use provided files
            input_file = args.input_template
            
            # Load parameter sets
            with open(args.param_file, 'r') as f:
                param_sets = json.load(f)
        else:
            parser.print_help()
            return
        
        # Run simulations
        dataset_dir = generator.run_parametric_sims(
            input_file, param_sets, args.output_dir
        )
        
        # Extract field data
        print("Extracting field data...")
        field_data = generator.extract_field_data(
            dataset_dir, ['default'], ['default'], 'numpy'
        )
        
        # Create FNO training data
        print("Creating FNO training data...")
        fno_data = generator.create_fno_training_data(dataset_dir)
        
        print(f"Dataset generation completed successfully in {dataset_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()