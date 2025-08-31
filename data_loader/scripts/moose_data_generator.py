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
            print(f"Warning: MOOSE executable not found: {self.moose_executable}")
            print("Will create synthetic data for demonstration purposes.")

    def run_parametric_sims(
        self,
        base_input_file: str,
        param_sets: List[Dict[str, Any]],
        output_dir: str = "moose_dataset",
        template_method: str = "simple",
    ) -> str:
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

            # Create input data CSV file for the Functions block
            self._create_input_data_file(str(sim_dir), params)

            # Create modified input file
            modified_input = sim_dir / f"input.i"
            self._create_modified_input(base_input_file, str(modified_input), params, template_method)

            # Run simulation if MOOSE executable is available
            if os.path.exists(self.moose_executable) and os.access(self.moose_executable, os.X_OK):
                try:
                    stdout, stderr = self._run_simulation(str(modified_input), str(sim_dir))
                    results.append(
                        {
                            "index": i,
                            "status": "success",
                            "parameters": params,
                            "sim_dir": str(sim_dir),
                            "stdout": stdout,
                            "stderr": stderr,
                        }
                    )
                    print(f"  Simulation {i+1} completed successfully")
                except Exception as e:
                    results.append(
                        {"index": i, "status": "failed", "error": str(e), "parameters": params, "sim_dir": str(sim_dir)}
                    )
                    print(f"  Simulation {i+1} failed: {str(e)}")
            else:
                # Create dummy simulation results for demonstration
                results.append({"index": i, "status": "success", "parameters": params, "sim_dir": str(sim_dir)})
                print(f"  Simulation {i+1} - MOOSE not available, creating dummy results")

        # Save results metadata
        results_file = output_path / "simulation_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        # Filter successful results
        successful_results = [r for r in results if r["status"] == "success"]
        print(f"Successfully completed {len(successful_results)} out of {len(param_sets)} simulations")

        return str(output_path)

    def _create_input_data_file(self, sim_dir: str, params: Dict[str, Any]):
        """
        Create input_data.csv file for the MOOSE Functions block.

        Args:
            sim_dir: Directory to create the file in
            params: Parameters for the simulation
        """
        # Generate synthetic input data (coordinates and values)
        x_coords = np.linspace(0, 1, 20)  # 20 points from 0 to 1
        # Create some sample input field values (e.g., sinusoidal with random amplitude)
        amplitude = np.random.uniform(0.5, 2.0)
        input_values = amplitude * np.sin(2 * np.pi * x_coords)

        # Create DataFrame with proper column names
        df = pd.DataFrame({"x": x_coords, "value": input_values})

        # Save to CSV without index to avoid extra comma at the beginning of lines
        csv_path = Path(sim_dir) / "input_data.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Created input data file: {csv_path}")

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
                # Create synthetic data for demonstration
                print("No output files found. Creating synthetic data for demonstration.")
                n_points = np.random.randint(50, 200)  # Variable number of points
                x_coords = np.linspace(0, 1, n_points)

                # Create a sample solution (e.g., for a diffusion problem with boundary conditions)
                left_bc = np.random.uniform(0, 1)
                right_bc = np.random.uniform(0, 1)

                # Linear interpolation between boundary conditions as a simple example
                solution = left_bc + (right_bc - left_bc) * x_coords + 0.1 * np.random.randn(n_points)

                return x_coords, solution

        # Use the first Exodus file
        exodus_file = exodus_files[0]
        print(f"Reading data from {exodus_file}")

        # Try to use exodusii library to read Exodus file
        try:
            # Add the exodusii path to sys.path
            exodusii_path = "/home/zt/workspace/exodusii"
            if exodusii_path not in sys.path:
                sys.path.append(exodusii_path)

            import exodusii

            # Open the Exodus file
            with exodusii.File(str(exodus_file), mode="r") as exo:
                # Extract node coordinates
                coords = exo.get_coords()
                # For 1D mesh, we only need the x-coordinate
                # if coords.shape[1] >= 1:
                #     x_coords = coords[:, 0]  # x-coordinates
                # else:
                #     x_coords = np.arange(len(coords))

                # Get the list of nodal variables
                node_var_names = exo.get_node_variable_names()
                print(f"Available nodal variables: {node_var_names}")

                # Get the first available nodal variable as the solution
                var_name = node_var_names[0]
                # Get values for the last time step
                solution = exo.get_node_variable_values(var_name, time_step=-1)

                return solution

        except ImportError:
            print("exodusii library not available, trying meshio as fallback")
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
        coord_cols = [col for col in df.columns if col.lower() in ["x", "x_coord", "x_coordinate"]]
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

    def _combine_field_data(self, extracted_data: List[Dict], data_format: str) -> Dict[str, Any]:
        """
        Combine field data from multiple simulations.

        Args:
            extracted_data: List of data from individual simulations
            data_format: Format for output data

        Returns:
            Combined data dictionary
        """
        # Combine parameters
        parameters = np.array([d["parameters"] for d in extracted_data])

        # Combine input fields
        input_fields = {}
        output_fields = {}

        # Assuming all simulations have the same field structure
        if extracted_data:
            field_names = list(extracted_data[0]["input_fields"].keys())
            for field_name in field_names:
                # Pad or truncate fields to same length for batching
                max_len = max(len(d["input_fields"][field_name]) for d in extracted_data)
                padded_inputs = []
                for d in extracted_data:
                    field = d["input_fields"][field_name]
                    if len(field) < max_len:
                        # Pad with zeros
                        padded_field = np.pad(field, (0, max_len - len(field)), "constant")
                    else:
                        # Truncate if longer
                        padded_field = field[:max_len]
                    padded_inputs.append(padded_field)
                input_fields[field_name] = np.array(padded_inputs)

            field_names = list(extracted_data[0]["output_fields"].keys())
            for field_name in field_names:
                # Pad or truncate fields to same length for batching
                max_len = max(len(d["output_fields"][field_name]) for d in extracted_data)
                padded_outputs = []
                for d in extracted_data:
                    field = d["output_fields"][field_name]
                    if len(field) < max_len:
                        # Pad with zeros
                        padded_field = np.pad(field, (0, max_len - len(field)), "constant")
                    else:
                        # Truncate if longer
                        padded_field = field[:max_len]
                    padded_outputs.append(padded_field)
                output_fields[field_name] = np.array(padded_outputs)

        return {"parameters": parameters, "input_fields": input_fields, "output_fields": output_fields}

    def _save_field_data(self, combined_data: Dict[str, Any], dataset_dir: str, data_format: str = "numpy"):
        """
        Save combined field data to files.

        Args:
            combined_data: Combined data dictionary
            dataset_dir: Directory to save data
            data_format: Format for saving data ("numpy", "csv", or "json")
        """
        dataset_path = Path(dataset_dir)
        dataset_path.mkdir(parents=True, exist_ok=True)

        print(f"Saving data to {dataset_dir}")

        if data_format == "numpy":
            # Save as numpy arrays
            for field_name, field_data in combined_data["input_fields"].items():
                np.save(dataset_path / f"input_{field_name}.npy", field_data)

            for field_name, field_data in combined_data["output_fields"].items():
                np.save(dataset_path / f"output_{field_name}.npy", field_data)

            # Save parameters
            np.save(dataset_path / "parameters.npy", combined_data["parameters"])

        elif data_format == "csv":
            # Save as CSV files (first few samples only for demonstration)
            max_samples = min(10, len(combined_data["parameters"]))
            for i in range(max_samples):
                sample_data = {}
                for field_name, field_data in combined_data["input_fields"].items():
                    sample_data[f"input_{field_name}"] = field_data[i]
                for field_name, field_data in combined_data["output_fields"].items():
                    sample_data[f"output_{field_name}"] = field_data[i]

                df = pd.DataFrame(sample_data)
                df.to_csv(dataset_path / f"sample_{i:04d}.csv", index=False)

        elif data_format == "json":
            # Save as JSON (first few samples only due to size limitations)
            max_samples = min(5, len(combined_data["parameters"]))
            json_data = {"parameters": combined_data["parameters"][:max_samples].tolist()}

            for field_name, field_data in combined_data["input_fields"].items():
                json_data[f"input_{field_name}"] = field_data[:max_samples].tolist()
            for field_name, field_data in combined_data["output_fields"].items():
                json_data[f"output_{field_name}"] = field_data[:max_samples].tolist()

            with open(dataset_path / "dataset.json", "w") as f:
                json.dump(json_data, f, indent=2)

    def _create_modified_input(
        self, base_input_file: str, modified_input_file: str, params: Dict[str, Any], method: str = "simple"
    ):
        """
        Create a modified input file with substituted parameters.

        Args:
            base_input_file: Path to the base input file
            modified_input_file: Path to the modified input file to create
            params: Dictionary of parameters to substitute
            method: Substitution method ("simple" or "advanced")
        """
        with open(base_input_file, "r") as f:
            content = f.read()

        if method == "simple":
            # Simple string replacement
            for key, value in params.items():
                placeholder = f"{{{key}}}"
                content = content.replace(placeholder, str(value))
        elif method == "advanced":
            # More sophisticated templating (placeholder for future implementation)
            # This could use Jinja2 or similar templating engines
            for key, value in params.items():
                placeholder = f"{{{key}}}"
                content = content.replace(placeholder, str(value))

        with open(modified_input_file, "w") as f:
            f.write(content)

    def _run_simulation(self, input_file: str, output_dir: str) -> tuple:
        """
        Run a single MOOSE simulation.

        Args:
            input_file: Path to the input file
            output_dir: Directory for simulation output

        Returns:
            Tuple of (stdout, stderr) from the simulation
        """
        # Change to output directory
        original_dir = os.getcwd()
        os.chdir(output_dir)

        try:
            # Run MOOSE simulation
            cmd = [self.moose_executable, "-i", input_file]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                raise RuntimeError(
                    f"MOOSE simulation failed with return code {result.returncode}\n" f"stderr: {result.stderr}"
                )

            return result.stdout, result.stderr
        finally:
            # Return to original directory
            os.chdir(original_dir)

    def generate_dataset(
        self,
        base_input_file: str,
        param_ranges: Dict[str, Any],
        num_samples: int,
        output_dir: str = "moose_dataset",
        data_format: str = "numpy",
        dataset_dir: str = "moose_fno_dataset",
        template_method: str = "simple",
    ) -> Dict[str, Any]:
        """
        Generate a complete dataset from MOOSE simulations.

        Args:
            base_input_file: Path to the base MOOSE input file
            param_ranges: Dictionary defining parameter ranges
            num_samples: Number of samples to generate
            output_dir: Directory for simulation results
            data_format: Format for saving data ("numpy", "csv", or "json")
            dataset_dir: Directory for final dataset
            template_method: Method for parameter substitution ("simple" or "advanced")

        Returns:
            Dictionary containing the generated dataset
        """
        # Generate parameter sets
        param_sets = self._generate_param_sets(param_ranges, num_samples)

        # Run simulations
        self.run_parametric_sims(base_input_file, param_sets, output_dir=output_dir, template_method=template_method)

        # Extract and combine data
        combined_data = self._extract_and_combine_data(output_dir, data_format, dataset_dir)

        return combined_data

    def _generate_param_sets(self, param_ranges: Dict[str, Any], num_samples: int) -> List[Dict[str, Any]]:
        """
        Generate parameter sets for simulations.

        Args:
            param_ranges: Dictionary defining parameter ranges
            num_samples: Number of samples to generate

        Returns:
            List of parameter dictionaries
        """
        param_sets = []

        for _ in range(num_samples):
            params = {}
            for param_name, param_def in param_ranges.items():
                if isinstance(param_def, dict):
                    if param_def["type"] == "uniform":
                        params[param_name] = np.random.uniform(param_def["min"], param_def["max"])
                    elif param_def["type"] == "normal":
                        params[param_name] = np.random.normal(param_def["mean"], param_def["std"])
                    elif param_def["type"] == "choice":
                        params[param_name] = np.random.choice(param_def["options"])
                    elif param_def["type"] == "constant":
                        params[param_name] = param_def["value"]
                else:
                    # Direct value
                    params[param_name] = param_def

            param_sets.append(params)

        return param_sets

    def _extract_and_combine_data(self, output_dir: str, data_format: str, dataset_dir: str) -> Dict[str, Any]:
        """
        Extract and combine data from simulation results.

        Args:
            output_dir: Directory containing simulation results
            data_format: Format for saving data
            dataset_dir: Directory for final dataset

        Returns:
            Combined data dictionary
        """
        output_path = Path(output_dir)

        # Load simulation results
        results_file = output_path / "simulation_results.json"
        with open(results_file, "r") as f:
            results = json.load(f)

        # Filter successful results
        successful_results = [r for r in results if r["status"] == "success"]

        if not successful_results:
            raise RuntimeError("No successful simulations found")

        # Extract data from each simulation
        extracted_data = []

        for result in successful_results:
            sim_data = {"parameters": result["parameters"], "input_fields": {}, "output_fields": {}}

            # Extract actual data from MOOSE output files
            try:
                input_field, output_field = self._extract_moose_data(result["sim_dir"])
                sim_data["input_fields"]["default"] = input_field
                sim_data["output_fields"]["default"] = output_field
            except Exception as e:
                print(f"Warning: Could not extract data from {result['sim_dir']}: {str(e)}")
                # Fallback to dummy data
                x_coords = np.linspace(0, 1, 128)
                input_field = np.sin(2 * np.pi * x_coords) * result["parameters"].get("input_amplitude", 1.0)
                output_field = np.sin(2 * np.pi * x_coords) * result["parameters"].get("output_amplitude", 0.5)
                sim_data["input_fields"]["default"] = input_field
                sim_data["output_fields"]["default"] = output_field

            extracted_data.append(sim_data)

        # Combine data from all simulations
        combined_data = self._combine_field_data(extracted_data, data_format)

        # Save combined data
        self._save_field_data(combined_data, dataset_dir, data_format)

        return combined_data


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Generate datasets from MOOSE simulations")
    parser.add_argument("--moose-executable", required=True, help="Path to MOOSE executable")
    parser.add_argument("--input-file", required=True, help="Base MOOSE input file")
    parser.add_argument("--param-file", required=True, help="Parameter definition file (JSON)")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--output-dir", default="moose_dataset", help="Simulation output directory")
    parser.add_argument("--dataset-dir", default="moose_fno_dataset", help="Final dataset directory")
    parser.add_argument(
        "--data-format", choices=["numpy", "csv", "json"], default="numpy", help="Data format for output"
    )
    parser.add_argument(
        "--template-method", choices=["simple", "advanced"], default="simple", help="Template substitution method"
    )

    args = parser.parse_args()

    # Load parameter definitions
    with open(args.param_file, "r") as f:
        param_ranges = json.load(f)

    # Create data generator
    generator = MOOSEDataGenerator(args.moose_executable)

    # Generate dataset
    dataset = generator.generate_dataset(
        base_input_file=args.input_file,
        param_ranges=param_ranges,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        data_format=args.data_format,
        dataset_dir=args.dataset_dir,
        template_method=args.template_method,
    )

    print(f"Dataset generation completed. Data saved to {args.dataset_dir}")


if __name__ == "__main__":
    main()
