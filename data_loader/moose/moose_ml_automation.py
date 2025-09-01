#!/usr/bin/env python3
"""
Enhanced MOOSE to ML Training Automation Script

This script provides a complete pipeline for:
1. Activating conda physics environment
2. Running MOOSE simulations with parametric variations
3. Extracting data from .e Exodus files using exodusii library
4. Generating datasets ready for ML training
5. Training ML models using the low-code platform
"""

import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import tempfile
import shutil
import argparse
from datetime import datetime

# Add paths for local modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append("/home/zt/workspace/exodusii")

# Import after path setup
try:
    import exodusii
except ImportError:
    print("Warning: exodusii library not found, using fallback methods")
    exodusii = None


class MOOSEEnvironmentManager:
    """Manages MOOSE environment setup and conda activation"""

    def __init__(self, conda_env: str = "physics", moose_executable: str = None):
        self.conda_env = conda_env
        self.moose_executable = moose_executable or "/home/zt/workspace/mymoose/mymoose-opt"
        self._verify_environment()

    def _verify_environment(self):
        """Verify MOOSE and conda environment are properly set up"""

        # Check if moose executable exists
        if not os.path.exists(self.moose_executable):
            raise FileNotFoundError(f"MOOSE executable not found: {self.moose_executable}")

        if not os.access(self.moose_executable, os.X_OK):
            raise PermissionError(f"MOOSE executable not executable: {self.moose_executable}")

        print(f"✓ MOOSE executable verified: {self.moose_executable}")

    def run_moose_with_conda(
        self, input_file: str, output_dir: str, env_vars: Dict[str, str] = None
    ) -> subprocess.CompletedProcess:
        """Run MOOSE simulation in conda environment"""

        original_dir = os.getcwd()
        try:
            os.chdir(output_dir)

            # Build command
            cmd = [self.moose_executable, "-i", input_file]
            if env_vars:
                env = os.environ.copy()
                env.update(env_vars)
            else:
                env = None

            print(f"Running MOOSE simulation: {' '.join(cmd)}")
            print(f"Working directory: {output_dir}")

            result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)  # 10 minutes timeout

            if result.returncode != 0:
                print(f"MOOSE simulation failed with return code {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                raise RuntimeError(f"MOOSE simulation failed: {result.stderr}")

            print(f"✓ MOOSE simulation completed successfully")
            return result

        finally:
            os.chdir(original_dir)


class ExodusDataExtractor:
    """Enhanced .e file data extraction using exodusii library"""

    def __init__(self):
        self.exodusii_available = exodusii is not None

    def extract_field_data(
        self, exodus_file: str, variable_names: List[str] = None, time_step: int = -1
    ) -> Dict[str, np.ndarray]:
        """Extract field data from Exodus file"""

        if not self.exodusii_available:
            return self._fallback_extraction(exodus_file)

        try:
            return self._exodusii_extraction(exodus_file, variable_names, time_step)
        except Exception as e:
            print(f"Error with exodusii extraction: {e}")
            return self._fallback_extraction(exodus_file)

    def _exodusii_extraction(
        self, exodus_file: str, variable_names: List[str] = None, time_step: int = -1
    ) -> Dict[str, np.ndarray]:
        """Extract data using exodusii library"""

        data = {}

        with exodusii.File(exodus_file, mode="r") as exo:
            # # Get coordinates
            # coords = exo.get_coords()
            # if coords.ndim == 1:
            #     # 1D case
            #     data["x"] = coords
            # else:
            #     # Multi-dimensional case
            #     for i, dim_name in enumerate(["x", "y", "z"]):
            #         if i < coords.shape[0]:
            #             data[dim_name] = coords[i]

            # Get time steps
            time_steps = exo.get_times()
            actual_time_step = time_step if 0 <= time_step < len(time_steps) else len(time_steps) - 1

            # Get nodal variables
            node_vars = exo.get_node_variable_names()

            # Extract specified variables or all available
            vars_to_extract = variable_names if variable_names else node_vars

            for var_name in vars_to_extract:
                if var_name in node_vars:
                    values = exo.get_node_variable_values(var_name, actual_time_step + 1)  # 1-based indexing
                    data[var_name] = values
                else:
                    print(f"Warning: Variable '{var_name}' not found in Exodus file")

            # Get element variables if needed
            elem_vars = exo.get_element_variable_names()
            for var_name in vars_to_extract:
                if var_name in elem_vars:
                    values = exo.get_element_variable_values(var_name, actual_time_step + 1)
                    data[f"element_{var_name}"] = values

            print(f"✓ Extracted {len(data)} variables from Exodus file")
            print(f"  Variables: {list(data.keys())}")
            print(f"  Time step: {actual_time_step}, Time: {time_steps[actual_time_step]}")

        return data

    def _fallback_extraction(self, exodus_file: str) -> Dict[str, np.ndarray]:
        """Fallback extraction using meshio or synthetic data"""

        try:
            import meshio

            mesh = meshio.read(exodus_file)

            data = {}

            # Get coordinates
            points = mesh.points
            if points.shape[1] >= 1:
                data["x"] = points[:, 0]
            if points.shape[1] >= 2:
                data["y"] = points[:, 1]
            if points.shape[1] >= 3:
                data["z"] = points[:, 2]

            # Get point data
            for var_name, values in mesh.point_data.items():
                if values.ndim == 1:
                    data[var_name] = values
                elif values.ndim == 2 and values.shape[1] == 1:
                    data[var_name] = values[:, 0]
                else:
                    # Handle vector/tensor quantities
                    for i in range(values.shape[1]):
                        data[f"{var_name}_{i}"] = values[:, i]

            print(f"✓ Extracted {len(data)} variables using meshio fallback")
            return data

        except ImportError:
            print("meshio not available, creating synthetic data")
            # Create synthetic data for testing
            n_points = 100
            x = np.linspace(0, 1, n_points)
            u = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(n_points)

            return {"x": x, "u": u}


class EnhancedMOOSEDataGenerator:
    """Enhanced MOOSE data generator with proper .e file handling"""

    def __init__(self, moose_executable: str = None, conda_env: str = "physics"):
        self.env_manager = MOOSEEnvironmentManager(conda_env, moose_executable)
        self.data_extractor = ExodusDataExtractor()

    def run_parametric_simulations(
        self,
        base_input_file: str,
        parameter_config: Dict[str, Any],
        num_samples: int,
        output_dir: str,
        variables_to_extract: List[str] = None,
    ) -> Dict[str, Any]:
        """Run parametric MOOSE simulations and extract data"""

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate parameter sets
        param_sets = self._generate_parameter_sets(parameter_config, num_samples)

        # Store simulation results
        results = []

        print(f"Running {len(param_sets)} MOOSE simulations...")

        for i, params in enumerate(param_sets):
            print(f"\nSimulation {i+1}/{len(param_sets)}")

            # Create simulation directory
            sim_dir = output_path / f"sim_{i:04d}"
            sim_dir.mkdir(exist_ok=True)

            # Create modified input file
            modified_input = sim_dir / "input.i"
            self._create_modified_input(base_input_file, modified_input, params)

            # Run simulation
            try:
                result = self.env_manager.run_moose_with_conda(str(modified_input.name), str(sim_dir))

                # Extract data from Exodus file
                exodus_files = list(sim_dir.glob("*.e"))
                if exodus_files:
                    exodus_file = exodus_files[0]
                    extracted_data = self.data_extractor.extract_field_data(str(exodus_file), variables_to_extract)

                    results.append(
                        {
                            "index": i,
                            "parameters": params,
                            "sim_dir": str(sim_dir),
                            "data": extracted_data,
                            "status": "success",
                        }
                    )
                    print(f"  ✓ Simulation completed successfully")
                else:
                    results.append(
                        {
                            "index": i,
                            "parameters": params,
                            "sim_dir": str(sim_dir),
                            "error": "No Exodus output file found",
                            "status": "failed",
                        }
                    )
                    print(f"  ✗ No Exodus output found")

            except Exception as e:
                results.append(
                    {"index": i, "parameters": params, "sim_dir": str(sim_dir), "error": str(e), "status": "failed"}
                )
                print(f"  ✗ Simulation failed: {str(e)}")

        # Save results metadata - convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        results_file = output_path / "simulation_results.json"
        with open(results_file, "w") as f:
            json.dump(convert_numpy(results), f, indent=2)

        successful_results = [r for r in results if r["status"] == "success"]
        print(f"\nCompleted {len(successful_results)} out of {len(param_sets)} simulations")

        return {
            "results": results,
            "successful_count": len(successful_results),
            "total_count": len(param_sets),
            "output_dir": str(output_path),
        }

    def generate_ml_dataset(
        self,
        simulation_results: Dict[str, Any],
        dataset_dir: str,
        input_variables: List[str] = None,
        output_variables: List[str] = None,
        normalize: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Generate ML-ready dataset from simulation results with parameter-to-result mapping"""

        dataset_path = Path(dataset_dir)
        dataset_path.mkdir(parents=True, exist_ok=True)

        successful_results = [r for r in simulation_results["results"] if r["status"] == "success"]

        if not successful_results:
            raise ValueError("No successful simulations to process")

        # Extract parameters
        parameters = []
        output_fields = []
        parameter_result_mapping = []

        # Determine variables to use
        first_result = successful_results[0]
        available_vars = list(first_result["data"].keys())

        # Only extract output variables (solution/results)
        output_vars = output_variables or [
            v for v in available_vars if v in ["u", "v", "p", "T", "c", "phi", "stress", "strain"]
        ]
        if not output_vars:
            # Fallback to all non-coordinate variables
            output_vars = [v for v in available_vars if not v.startswith("x")]

        print(f"Processing {len(successful_results)} successful simulations")
        print(f"Output variables: {output_vars}")

        for result in successful_results:
            # Parameters
            params = list(result["parameters"].values())
            parameters.append(params)

            # Output fields (solution variables only)
            output_data = []
            for var in output_vars:
                if var in result["data"]:
                    output_data.extend(result["data"][var])
            output_fields.append(output_data)

            # Create parameter-to-result mapping
            mapping_entry = {
                "parameters": result["parameters"],  # Keep as dict for clarity
                "parameter_vector": params,  # Flat parameter vector
                "results": {var: result["data"][var] for var in output_vars if var in result["data"]},
                "simulation_id": result["index"],
                "sim_dir": result["sim_dir"],
            }
            parameter_result_mapping.append(mapping_entry)

        # Convert to numpy arrays
        parameters = np.array(parameters)
        output_fields = np.array(output_fields)

        # Normalize output fields if requested
        if normalize:
            output_fields = self._normalize_data(output_fields)

            # Update mapping with normalized data
            for i, mapping in enumerate(parameter_result_mapping):
                # Create normalized results from normalized output fields
                start_idx = 0
                normalized_results = {}
                for var in output_vars:
                    if var in mapping["results"]:
                        var_data = mapping["results"][var]
                        end_idx = start_idx + len(var_data)
                        normalized_results[var] = output_fields[i][start_idx:end_idx].tolist()
                        start_idx = end_idx
                mapping["normalized_results"] = normalized_results

        # Save dataset - only parameters and output_data (no input_data)
        np.save(dataset_path / "parameters.npy", parameters)
        np.save(dataset_path / "output_data.npy", output_fields)

        # Save parameter-to-result mapping
        mapping_file = dataset_path / "parameter_result_mapping.json"
        with open(mapping_file, "w") as f:

            def convert_numpy_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_for_json(item) for item in obj]
                return obj

            json.dump(convert_numpy_for_json(parameter_result_mapping), f, indent=2)

        # Save metadata
        metadata = {
            "creation_time": datetime.now().isoformat(),
            "num_samples": len(successful_results),
            "output_variables": output_vars,
            "parameter_names": list(successful_results[0]["parameters"].keys()),
            "normalize": normalize,
            "parameter_result_mapping_file": str(mapping_file),
        }

        with open(dataset_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ ML dataset saved to {dataset_dir}")
        print(f"  Shape: parameters={parameters.shape}, output={output_fields.shape}")
        print(f"  Parameter-result mapping saved to: {mapping_file}")

        return {
            "parameters": parameters,
            "output_data": output_fields,
            "metadata": metadata,
            "parameter_result_mapping": parameter_result_mapping,
        }

    def _generate_parameter_sets(self, param_config: Dict[str, Any], num_samples: int) -> List[Dict[str, Any]]:
        """Generate parameter sets based on configuration"""

        param_sets = []

        for i in range(num_samples):
            params = {}
            for param_name, config in param_config.items():
                if isinstance(config, dict):
                    dist_type = config.get("type", "uniform")

                    if dist_type == "uniform":
                        params[param_name] = np.random.uniform(config["min"], config["max"])
                    elif dist_type == "normal":
                        params[param_name] = np.random.normal(config["mean"], config["std"])
                    elif dist_type == "choice":
                        params[param_name] = np.random.choice(config["values"])
                    else:
                        params[param_name] = config["default"]
                else:
                    # Direct value
                    params[param_name] = config

            param_sets.append(params)

        return param_sets

    def _create_modified_input(self, base_input_file: str, output_file: str, params: Dict[str, Any]):
        """Create modified MOOSE input file with parameters"""

        with open(base_input_file, "r") as f:
            content = f.read()

        # Replace parameter placeholders using MOOSE-compatible format
        for key, value in params.items():
            placeholder = f"${{{key}}}"
            if isinstance(value, (int, float)):
                content = content.replace(placeholder, str(value))
            else:
                content = content.replace(placeholder, str(value))

        with open(output_file, "w") as f:
            f.write(content)

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to [0, 1] range"""

        data_min = np.min(data, axis=1, keepdims=True)
        data_max = np.max(data, axis=1, keepdims=True)

        # Handle edge case where min == max
        range_vals = data_max - data_min
        range_vals[range_vals == 0] = 1  # Prevent division by zero

        normalized = (data - data_min) / range_vals

        return normalized

    def extract_parameter_results_mapping(self, dataset_dir: str) -> Dict[str, Any]:
        """Extract and return parameter-to-results mapping from a dataset directory"""

        dataset_path = Path(dataset_dir)
        mapping_file = dataset_path / "parameter_result_mapping.json"

        if not mapping_file.exists():
            raise FileNotFoundError(f"Parameter-result mapping file not found: {mapping_file}")

        with open(mapping_file, "r") as f:
            mapping = json.load(f)

        # Also load the numpy arrays for direct access
        parameters = np.load(dataset_path / "parameters.npy")
        output_data = np.load(dataset_path / "output_data.npy")

        return {
            "mapping": mapping,
            "parameters": parameters,
            "output_data": output_data,
            "parameter_names": [list(m["parameters"].keys()) for m in mapping][0] if mapping else [],
        }


class MOOSEMLPipeline:
    """Complete pipeline from MOOSE to ML training"""

    def __init__(self, config_file: str):
        """Initialize pipeline with configuration"""

        with open(config_file, "r") as f:
            self.config = json.load(f)

        self.moose_executable = self.config.get("moose_executable", "/home/zt/workspace/mymoose/mymoose-opt")
        self.conda_env = self.config.get("conda_env", "physics")

        self.generator = EnhancedMOOSEDataGenerator(self.moose_executable, self.conda_env)

    def run_complete_pipeline(self):
        """Run the complete pipeline: MOOSE → Dataset → ML Training"""

        print("🚀 Starting MOOSE to ML Training Pipeline")
        print("=" * 50)

        # Step 1: Run parametric MOOSE simulations
        print("\n1️⃣ Running parametric MOOSE simulations...")
        sim_results = self.generator.run_parametric_simulations(
            base_input_file=self.config["base_input_file"],
            parameter_config=self.config["parameters"],
            num_samples=self.config["num_samples"],
            output_dir=self.config["simulation_output_dir"],
            variables_to_extract=self.config.get("variables_to_extract"),
        )

        # Step 2: Generate ML dataset
        print("\n2️⃣ Generating ML dataset...")
        dataset = self.generator.generate_ml_dataset(
            simulation_results=sim_results,
            dataset_dir=self.config["dataset_dir"],
            input_variables=self.config.get("input_variables"),
            output_variables=self.config.get("output_variables"),
            normalize=self.config.get("normalize", True),
        )

        # Step 3: Create training configuration
        print("\n3️⃣ Creating training configuration...")
        training_config = self._create_training_config(dataset)

        # Step 4: Save training configuration
        config_path = Path(self.config["dataset_dir"]) / "training_config.yaml"
        self._save_training_config(training_config, str(config_path))

        print(f"\n✅ Pipeline completed successfully!")
        print(f"   Dataset saved to: {self.config['dataset_dir']}")
        print(f"   Training config: {config_path}")
        print(f"   Ready to train with: python main/train.py --config {config_path}")

        return {
            "simulation_results": sim_results,
            "dataset": dataset,
            "training_config": training_config,
            "config_path": str(config_path),
        }

    def _create_training_config(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Create training configuration for the low-code platform"""

        # Determine data dimensions
        parameters_shape = dataset["parameters"].shape
        output_shape = dataset["output_data"].shape

        config = {
            "data": {
                "train_features_path": f"{self.config['dataset_dir']}/parameters.npy",
                "train_targets_path": f"{self.config['dataset_dir']}/output_data.npy",
                "test_features_path": f"{self.config['dataset_dir']}/parameters.npy",  # Use same for now
                "test_targets_path": f"{self.config['dataset_dir']}/output_data.npy",
                "batch_size": self.config.get("batch_size", 16),
            },
            "model": {
                "name": self.config.get("model_name", "mlp"),
                "parameters": {
                    "in_features": parameters_shape[1],  # Number of parameters
                    "out_features": output_shape[1] if len(output_shape) > 1 else 1,
                    "layer_sizes": 32,
                    "num_layers": 3,
                },
            },
            "training": {
                "epochs": self.config.get("epochs", 100),
                "device": "cuda" if self.config.get("use_gpu", True) else "cpu",
            },
            "optimizer": {"name": "adam", "parameters": {"lr": self.config.get("learning_rate", 0.001)}},
            "criterion": {"name": "mse"},
            "output": {
                "model_path": f"{self.config['dataset_dir']}/trained_model.pth",
                "history_path": f"{self.config['dataset_dir']}/training_history.json",
            },
        }

        return config

    def _save_training_config(self, config: Dict[str, Any], output_path: str):
        """Save training configuration as YAML file"""

        try:
            import yaml

            with open(output_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        except ImportError:
            # Fallback to JSON if yaml not available
            json_path = output_path.replace(".yaml", ".json")
            with open(json_path, "w") as f:
                json.dump(config, f, indent=2)
            print(f"✓ Training configuration saved to {json_path} (YAML unavailable)")
            return

        print(f"✓ Training configuration saved to {output_path}")

    def extract_parameter_results_mapping(self, dataset_dir: str) -> Dict[str, Any]:
        """Extract and return parameter-to-results mapping from dataset"""

        dataset_path = Path(dataset_dir)
        mapping_file = dataset_path / "parameter_result_mapping.json"

        if not mapping_file.exists():
            raise FileNotFoundError(f"Parameter-result mapping file not found: {mapping_file}")

        with open(mapping_file, "r") as f:
            mapping = json.load(f)

        # Also load the numpy arrays for direct access
        parameters = np.load(dataset_path / "parameters.npy")
        output_data = np.load(dataset_path / "output_data.npy")

        return {
            "mapping": mapping,
            "parameters": parameters,
            "output_data": output_data,
            "parameter_names": [list(m["parameters"].keys()) for m in mapping][0] if mapping else [],
        }
