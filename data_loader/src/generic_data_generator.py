#!/usr/bin/env python3
"""
Generic Data Generation Framework for Scientific Computing and ML Training

This framework provides a flexible pipeline for:
1. Parameter space definition
2. Simulation execution with various solvers
3. Data extraction from simulation outputs
4. Dataset generation for ML training
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ParameterSpace:
    """
    Generic parameter space definition

    This class handles parameter space definition and sampling for simulations.

    Configuration Parameters:
    -------------------------
    The parameter configuration should be a dictionary with the following structure:

    {
        "param1": {
            "type": "uniform",      # Type of distribution: uniform, normal, choice, fixed
            "min": 0.0,             # Minimum value (for uniform)
            "max": 1.0,             # Maximum value (for uniform)
            "required": False       # Whether this parameter is required
        },
        "param2": {
            "type": "normal",       # Normal distribution
            "mean": 0.0,            # Mean value
            "std": 1.0              # Standard deviation
        },
        "param3": {
            "type": "choice",       # Discrete choices
            "values": [1, 2, 3]     # List of possible values
        },
        "param4": {
            "type": "fixed",        # Fixed value
            "value": 42             # Fixed value for this parameter
        },
        "param5": 3.14              # Direct value assignment (treated as fixed)
    }
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize parameter space from configuration

        Args:
            config: Dictionary defining parameter ranges and distributions
        """
        self.config = config
        self.parameters = self._parse_config(config)

    def _parse_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate parameter configuration"""
        parameters = {}

        for param_name, param_config in config.items():
            # Validate parameter configuration
            if isinstance(param_config, dict):
                param_type = param_config.get("type", "uniform")
                param_required = param_config.get("required", False)

                # Validate required fields
                if param_required and "default" not in param_config:
                    raise ValueError(f"Required parameter '{param_name}' must have a default value")

                # Validate distribution type and required fields
                if param_type == "uniform":
                    if "min" not in param_config or "max" not in param_config:
                        raise ValueError(f"Uniform parameter '{param_name}' requires min and max values")
                elif param_type == "normal":
                    if "mean" not in param_config or "std" not in param_config:
                        raise ValueError(f"Normal parameter '{param_name}' requires mean and std values")
                elif param_type == "choice":
                    if "values" not in param_config or not param_config["values"]:
                        raise ValueError(f"Choice parameter '{param_name}' requires non-empty values list")
                elif param_type == "fixed":
                    if "value" not in param_config:
                        raise ValueError(f"Fixed parameter '{param_name}' requires a value")
                else:
                    raise ValueError(f"Unsupported parameter type: {param_type}")

                parameters[param_name] = param_config
            else:
                # Simple parameter with direct value
                parameters[param_name] = {"type": "fixed", "value": param_config}

        return parameters

    def generate_samples(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate parameter samples based on configuration"""
        samples = []

        for i in range(num_samples):
            sample = {}
            for param_name, param_config in self.parameters.items():
                param_type = param_config.get("type", "fixed")

                if param_type == "uniform":
                    sample[param_name] = np.random.uniform(param_config["min"], param_config["max"])
                elif param_type == "normal":
                    sample[param_name] = np.random.normal(param_config["mean"], param_config["std"])
                elif param_type == "choice":
                    sample[param_name] = np.random.choice(param_config["values"])
                elif param_type == "fixed":
                    sample[param_name] = param_config["value"]
                else:
                    raise ValueError(f"Unsupported parameter type: {param_type}")

            logger.debug(f"Generated parameter sample {i+1}: {sample}")
            samples.append(sample)

        logger.info(f"Generated {len(samples)} parameter samples")
        return samples


class SimulationRunner(ABC):
    """
    Abstract base class for simulation runners

    This class defines the interface for running simulations with different solvers.
    Subclasses should implement the run_simulation method for their specific solver.

    Configuration Parameters:
    -------------------------
    The configuration should be a dictionary with the following structure:

    {
        "working_dir": "/path/to/working/dir",  # Working directory for simulations
        "timeout": 600                          # Timeout for simulations in seconds
    }
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize simulation runner with configuration

        Args:
            config: Dictionary containing simulation-specific configuration
        """
        self.config = config
        self.working_dir = config.get("working_dir", os.getcwd())
        self.timeout = config.get("timeout", 600)  # Default 10 minutes

    @abstractmethod
    def run_simulation(self, input_file: str, output_dir: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a simulation with given parameters

        Args:
            input_file: Path to the base input file
            output_dir: Directory to run simulation in
            parameters: Dictionary of parameters to use in simulation

        Returns:
            Dictionary containing simulation results
        """
        pass

    def prepare_input_file(self, base_input: str, output_file: str, parameters: Dict[str, Any]):
        """
        Prepare input file with specific parameters

        Args:
            base_input: Path to base input file
            output_file: Path to write modified input file
            parameters: Dictionary of parameters to substitute
        """
        with open(base_input, "r") as f:
            content = f.read()

        # Replace parameter placeholders
        for key, value in parameters.items():
            placeholder = f"${{{key}}}"
            content = content.replace(placeholder, str(value))

        with open(output_file, "w") as f:
            f.write(content)

        logger.debug(f"Prepared input file at {output_file} with parameters {parameters}")

    def verify_environment(self):
        """Verify that the simulation environment is properly configured"""
        pass


class MOOSESimulationRunner(SimulationRunner):
    """
    Concrete implementation of SimulationRunner for MOOSE-based applications

    Configuration Parameters:
    -------------------------
    The configuration should include:

    {
        "moose_executable": "/path/to/moose_app",  # Path to MOOSE executable
        "working_dir": "/path/to/working/dir",     # Working directory for simulations
        "timeout": 600                             # Timeout for simulations in seconds
    }
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MOOSE simulation runner with configuration

        Args:
            config: Dictionary containing simulation-specific configuration
        """
        super().__init__(config)
        self.moose_executable = config.get("moose_executable")

        if not self.moose_executable:
            raise ValueError("MOOSE executable path must be specified in configuration")

    def verify_environment(self):
        """Verify that the MOOSE simulation environment is properly configured"""
        super().verify_environment()

        if not os.path.exists(self.moose_executable):
            raise ValueError(f"MOOSE executable not found at {self.moose_executable}")

        # Additional MOOSE-specific environment checks
        logger.info(f"Using MOOSE executable: {self.moose_executable}")

    def run_simulation(self, input_file: str, output_dir: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a MOOSE simulation with given parameters

        Args:
            input_file: Path to the base input file
            output_dir: Directory to run simulation in
            parameters: Dictionary of parameters to use in simulation

        Returns:
            Dictionary containing simulation results
        """
        logger.info(f"Running MOOSE simulation with parameters: {parameters}")

        # Prepare input file
        modified_input = os.path.join(output_dir, "input_modified.i")
        self.prepare_input_file(input_file, modified_input, parameters)

        # Prepare output file
        output_file = os.path.join(output_dir, "output")

        try:
            # Run MOOSE simulation
            command = [self.moose_executable, "-i", modified_input, "-o", output_file]

            logger.info(f"Executing command: {' '.join(command)}")

            # Run the simulation
            process = subprocess.run(command, cwd=output_dir, capture_output=True, text=True, timeout=self.timeout)

            # Check return code
            if process.returncode == 0:
                logger.info("Simulation completed successfully")
                status = "success"
                error = None
            else:
                logger.error("Simulation failed")
                status = "failed"
                error = process.stderr

            # Return results
            return {
                "status": status,
                "error": error,
                "stdout": process.stdout,
                "stderr": process.stderr,
                "returncode": process.returncode,
                "command": " ".join(command),
            }

        except subprocess.TimeoutExpired as e:
            logger.error(f"Simulation timed out after {self.timeout} seconds")
            return {"status": "failed", "error": str(e), "command": " ".join(command)}
        except Exception as e:
            logger.error(f"Unexpected error during simulation: {str(e)}")
            return {"status": "failed", "error": str(e), "command": " ".join(command)}


class DataExtractor(ABC):
    """
    Abstract base class for data extractors

    This class defines the interface for extracting data from simulation outputs.
    Subclasses should implement the extract_data method for their specific data format.

    Configuration Parameters:
    -------------------------
    The configuration is optional and depends on the specific extractor implementation.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize data extractor

        Args:
            config: Optional configuration for data extraction
        """
        self.config = config or {}
        self._verify_environment()

    def _verify_environment(self):
        """Verify that the data extraction environment is properly configured"""
        pass

    @abstractmethod
    def extract_data(self, file_path: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract data from simulation output file

        Args:
            file_path: Path to the simulation output file
            parameters: Optional parameters used in simulation

        Returns:
            Dictionary containing extracted data
        """
        pass


class ExodusDataExtractor(DataExtractor):
    """
    Concrete implementation of DataExtractor for Exodus output files

    This class extracts data from MOOSE/Exodus output files using the PyExodus library.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Exodus data extractor

        Args:
            config: Optional configuration for data extraction
        """
        try:
            import pyexodus

            self.pyexodus = pyexodus
        except ImportError:
            raise ImportError("PyExodus library is required for Exodus file extraction")

        super().__init__(config)

    def _verify_environment(self):
        """Verify that the Exodus data extraction environment is properly configured"""
        super()._verify_environment()

        # Verify that required libraries are available
        try:
            import pyexodus
        except ImportError:
            raise ImportError("PyExodus library is required for Exodus file extraction")

    def extract_data(self, file_path: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract data from Exodus output file

        Args:
            file_path: Path to the Exodus output file
            parameters: Optional parameters used in simulation

        Returns:
            Dictionary containing extracted data
        """
        logger.info(f"Extracting data from Exodus file: {file_path}")

        try:
            # Open Exodus file
            exodus = self.pyexodus.exodus(file_path, mode="r")

            # Extract basic information
            info = {
                "title": exodus.title(),
                "num_dim": exodus.num_dimensions(),
                "num_nodes": exodus.num_nodes(),
                "num_elems": exodus.num_elems(),
                "num_blocks": exodus.num_blocks(),
                "num_node_sets": exodus.num_node_sets(),
                "num_side_sets": exodus.num_side_sets(),
            }

            # Extract variable names
            var_names = exodus.get_all_var_names()
            info["variables"] = var_names

            # Extract variable values at last time step
            data = {}
            for var_type, vars in var_names.items():
                if vars:
                    values = exodus.get_all_var_values(var_type, exodus.num_times())
                    data[var_type] = dict(zip(vars, values))

            # Extract parameters if provided
            if parameters:
                data["parameters"] = parameters

            # Close Exodus file
            exodus.close()

            logger.info(f"Successfully extracted {len(data)} variables from Exodus file")
            return data

        except Exception as e:
            logger.error(f"Failed to extract data from Exodus file: {str(e)}")
            raise


class GenericDataGenerator:
    """
    Generic data generation framework

    This class orchestrates the complete data generation pipeline:
    1. Generate parameter samples
    2. Run simulations with different solvers
    3. Extract data from simulation outputs
    4. Generate ML-ready datasets

    Configuration Parameters:
    -------------------------
    The configuration should be a dictionary with the following structure:

    {
        "parameters": {                     # Parameter space configuration (see ParameterSpace)
            "param1": {
                "type": "uniform",
                "min": 0.0,
                "max": 1.0
            }
        },
        "num_samples": 10,                  # Number of parameter samples to generate
        "variables_to_extract": ["u", "v"], # Variables to extract from simulation outputs
        "normalize": True                   # Whether to normalize the output data
    }
    """

    def __init__(self, solver: SimulationRunner, extractor: DataExtractor, config: Dict[str, Any]):
        """
        Initialize generic data generator

        Args:
            solver: Simulation runner instance
            extractor: Data extractor instance
            config: Configuration dictionary
        """
        self.solver = solver
        self.extractor = extractor
        self.config = config
        self.parameter_space = ParameterSpace(config.get("parameters", {}))

    def run_simulations(self, base_input_file: str, output_dir: str) -> Dict[str, Any]:
        """
        Run parametric simulations and extract data

        Args:
            base_input_file: Path to the base input file
            output_dir: Directory to store simulation outputs

        Returns:
            Dictionary containing simulation results
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get parameters
        num_samples = self.config.get("num_samples", 10)
        variables_to_extract = self.config.get("variables_to_extract")

        # Generate parameter sets
        param_sets = self.parameter_space.generate_samples(num_samples)
        logger.info(f"Running {len(param_sets)} simulations")

        # Store simulation results
        results = []

        for i, params in enumerate(param_sets):
            logger.info(f"\nSimulation {i+1}/{len(param_sets)}")

            # Create simulation directory
            sim_dir = output_path / f"sim_{i:04d}"
            sim_dir.mkdir(exist_ok=True)

            # Run simulation
            try:
                result = self.solver.run_simulation(base_input_file, str(sim_dir), params)

                # Extract data if simulation was successful
                if result["status"] == "success":
                    # Find output file (implementation depends on solver)
                    output_files = list(sim_dir.glob("*"))
                    if output_files:
                        # Use first output file as default, subclasses can override this logic
                        output_file = str(output_files[0])
                        extracted_data = self.extractor.extract_data(output_file, params)

                        result.update({"data": extracted_data, "output_file": output_file})
                    else:
                        result["error"] = "No output file found"
                        result["status"] = "failed"
                        logger.error("No output file found")

                # Store result
                result["index"] = i
                result["sim_dir"] = str(sim_dir)
                results.append(result)

            except Exception as e:
                logger.error(f"Simulation {i+1} failed: {str(e)}")
                results.append(
                    {"index": i, "parameters": params, "sim_dir": str(sim_dir), "error": str(e), "status": "failed"}
                )

        # Save results metadata
        self._save_results(output_path, results)

        # Generate dataset
        successful_results = [r for r in results if r["status"] == "success"]
        logger.info(f"\nCompleted {len(successful_results)} out of {len(param_sets)} simulations")

        return {
            "results": results,
            "successful_count": len(successful_results),
            "total_count": len(param_sets),
            "output_dir": str(output_path),
        }

    def _save_results(self, output_path: Path, results: List[Dict[str, Any]]):
        """Save simulation results metadata"""

        # Convert numpy arrays to lists for JSON serialization
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

    def generate_ml_dataset(self, simulation_results: Dict[str, Any], dataset_dir: str) -> Dict[str, np.ndarray]:
        """
        Generate ML-ready dataset from simulation results

        Args:
            simulation_results: Dictionary containing simulation results
            dataset_dir: Directory to store dataset

        Returns:
            Dictionary containing ML dataset
        """
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
        input_variables = self.config.get("input_variables")
        output_variables = self.config.get("output_variables")

        logger.info(f"Processing {len(successful_results)} successful simulations")

        for result in successful_results:
            # Parameters
            params = list(result["parameters"].values())
            parameters.append(params)

            # Output fields
            if "data" in result:
                data = result["data"]

                # Extract output variables
                output_data = []
                if output_variables:
                    for var in output_variables:
                        if var in data:
                            if isinstance(data[var], np.ndarray):
                                output_data.extend(data[var])
                            else:
                                output_data.append(data[var])
                else:
                    # Fallback: extract all numerical values from data
                    for key, value in data.items():
                        if isinstance(value, (int, float, np.ndarray)):
                            if isinstance(value, np.ndarray):
                                output_data.extend(value)
                            else:
                                output_data.append(value)

                output_fields.append(output_data)

            # Create parameter-to-result mapping
            mapping_entry = {
                "parameters": result["parameters"],
                "parameter_vector": params,
                "results": result,
                "simulation_id": result["index"],
                "sim_dir": result["sim_dir"],
            }
            if "data" in result:
                mapping_entry["data"] = result["data"]

            parameter_result_mapping.append(mapping_entry)

        # Convert to numpy arrays
        parameters = np.array(parameters)
        output_fields = np.array(output_fields)

        # Normalize data if requested
        if self.config.get("normalize", True):
            output_fields = self._normalize_data(output_fields)

            # Update mapping with normalized data
            for i, mapping in enumerate(parameter_result_mapping):
                mapping["normalized_output"] = output_fields[i].tolist()

        # Save dataset
        np.save(dataset_path / "parameters.npy", parameters)
        np.save(dataset_path / "output_data.npy", output_fields)

        # Save parameter-to-result mapping
        mapping_file = dataset_path / "parameter_result_mapping.json"
        with open(mapping_file, "w") as f:

            def convert_numpy_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj) if isinstance(obj, np.floating) else int(obj)
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
            "output_variables": output_variables,
            "parameter_names": list(successful_results[0]["parameters"].keys()),
            "normalize": self.config.get("normalize", True),
            "parameter_result_mapping_file": str(mapping_file),
        }

        with open(dataset_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"ML dataset saved to {dataset_dir}")
        logger.info(f"Shape: parameters={parameters.shape}, output={output_fields.shape}")
        logger.info(f"Parameter-result mapping saved to: {mapping_file}")

        return {
            "parameters": parameters,
            "output_data": output_fields,
            "metadata": metadata,
            "parameter_result_mapping": parameter_result_mapping,
        }

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to [0, 1] range"""
        data_min = np.min(data, axis=1, keepdims=True)
        data_max = np.max(data, axis=1, keepdims=True)

        # Handle edge case where min == max
        range_vals = data_max - data_min
        range_vals[range_vals == 0] = 1  # Prevent division by zero

        normalized = (data - data_min) / range_vals
        return normalized


class GenericDataGenerationPipeline:
    """
    Generic pipeline from simulation to ML dataset

    This class provides a complete pipeline for data generation:
    1. Run simulations with various solvers
    2. Extract data from simulation outputs
    3. Generate ML-ready datasets
    4. Create training configurations

    Configuration Parameters:
    -------------------------
    The configuration should be a dictionary with the following structure:

    {
        "solver": "generic",                # Solver type (generic, moose, custom)
        "solver_config": {},                # Solver-specific configuration
        "extractor": "generic",             # Extractor type (generic, exodus, custom)
        "extractor_config": {},             # Extractor-specific configuration
        "base_input_file": "input.i",       # Base input file for simulations
        "simulation_output_dir": "sims",    # Directory to store simulation outputs
        "dataset_dir": "dataset",           # Directory to store ML dataset
        "parameters": {},                   # Parameter space configuration
        "num_samples": 10,                  # Number of parameter samples
        "variables_to_extract": ["u", "v"], # Variables to extract from outputs
        "normalize": True,                  # Whether to normalize output data
        "model_name": "mlp",                # Model name for training config
        "epochs": 100,                      # Number of training epochs
        "batch_size": 16,                   # Batch size for training
        "learning_rate": 0.001,             # Learning rate for training
        "use_gpu": True                     # Whether to use GPU for training
    }
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize generic data generation pipeline

        Args:
            config: Dictionary containing pipeline configuration
        """
        self.config = config
        self.solver = self._create_solver()
        self.extractor = self._create_extractor()
        self.data_generator = GenericDataGenerator(self.solver, self.extractor, self.config)

    def _create_solver(self) -> SimulationRunner:
        """Create appropriate solver based on configuration"""
        solver_type = self.config.get("solver", "generic")
        solver_config = self.config.get("solver_config", {})

        if solver_type == "custom":
            # Custom solver from configuration
            if "custom_solver_class" in solver_config:
                # Import and instantiate custom solver
                module_path, class_name = solver_config["custom_solver_class"].rsplit(".", 1)
                module = __import__(module_path, fromlist=[class_name])
                solver_class = getattr(module, class_name)

                if "custom_solver_config" in solver_config:
                    custom_config = solver_config["custom_solver_config"]
                else:
                    custom_config = {}

                return solver_class(custom_config)
        elif solver_type == "moose":
            return MOOSESimulationRunner(solver_config)
        else:
            return SimulationRunner(solver_config)

    def _create_extractor(self) -> DataExtractor:
        """Create appropriate extractor based on configuration"""
        extractor_type = self.config.get("extractor", "generic")
        extractor_config = self.config.get("extractor_config", {})

        if extractor_type == "custom":
            # Custom extractor from configuration
            if "custom_extractor_class" in extractor_config:
                # Import and instantiate custom extractor
                module_path, class_name = extractor_config["custom_extractor_class"].rsplit(".", 1)
                module = __import__(module_path, fromlist=[class_name])
                extractor_class = getattr(module, class_name)

                if "custom_extractor_config" in extractor_config:
                    custom_config = extractor_config["custom_extractor_config"]
                else:
                    custom_config = {}

                return extractor_class(custom_config)
        elif extractor_type == "exodus":
            return ExodusDataExtractor(extractor_config)
        else:
            return DataExtractor(extractor_config)

    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete data generation pipeline"""
        logger.info("🚀 Starting Generic Data Generation Pipeline")
        logger.info("=" * 50)

        # Verify environment before starting
        logger.info("\n🔍 Verifying environment...")
        self.solver.verify_environment()
        self.extractor._verify_environment()

        # 1. Run simulations
        logger.info("\n1️⃣ Running simulations...")
        sim_results = self.data_generator.run_simulations(
            base_input_file=self.config["base_input_file"], output_dir=self.config["simulation_output_dir"]
        )

        # 2. Generate ML dataset
        logger.info("\n2️⃣ Generating dataset...")
        dataset = self.data_generator.generate_ml_dataset(
            simulation_results=sim_results, dataset_dir=self.config["dataset_dir"]
        )

        # 3. Create training configuration
        logger.info("\n3️⃣ Creating training configuration...")
        training_config = self._create_training_config(dataset)

        # 4. Save training configuration
        config_path = Path(self.config["dataset_dir"]) / "training_config.yaml"
        self._save_training_config(training_config, str(config_path))

        logger.info("\n✅ Pipeline completed successfully!")
        logger.info(f"Dataset saved to: {self.config['dataset_dir']}")
        logger.info(f"Training config: {config_path}")
        logger.info(f"Ready to train with: python main/train.py --config {config_path}")

        return {
            "simulation_results": sim_results,
            "dataset": dataset,
            "training_config": training_config,
            "config_path": str(config_path),
        }

    def create_pipeline_example():
        """
        Example usage of the generic data generation pipeline.

        This example demonstrates how to use the framework with MOOSE simulations
        and Exodus file extraction.

        Example:
            >>> pipeline = create_pipeline_example()
            >>> results = pipeline.run_pipeline()
        """
        # Configuration for the pipeline
        config = {
            # MOOSE solver configuration
            "solver": "moose",
            "solver_config": {"moose_executable": "/path/to/your/moose_app", "working_dir": "./sims", "timeout": 600},
            # Exodus data extractor configuration
            "extractor": "exodus",
            "extractor_config": {},
            # Input file and directories
            "base_input_file": "input.i",
            "simulation_output_dir": "sims",
            "dataset_dir": "dataset",
            # Parameter space configuration
            "parameters": {
                "diffusivity": {"type": "uniform", "min": 0.1, "max": 1.0},
                "velocity_x": {"type": "uniform", "min": -1.0, "max": 1.0},
                "velocity_y": {"type": "uniform", "min": -1.0, "max": 1.0},
            },
            # Data generation parameters
            "num_samples": 5,
            "variables_to_extract": ["temperature"],
            "normalize": True,
            # Training configuration parameters
            "model_name": "mlp",
            "epochs": 50,
            "batch_size": 8,
            "learning_rate": 0.001,
            "use_gpu": True,
        }

        # Create and return the pipeline
        return GenericDataGenerationPipeline(config)

    def _create_training_config(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Create training configuration for the low-code platform"""
        # Determine data dimensions
        parameters_shape = dataset["parameters"].shape
        output_shape = dataset["output_data"].shape

        return {
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
                    "layer_sizes": self.config.get("layer_sizes", 32),
                    "num_layers": self.config.get("num_layers", 3),
                },
            },
            "training": {
                "epochs": self.config.get("epochs", 100),
                "device": "cuda" if self.config.get("use_gpu", True) else "cpu",
            },
            "optimizer": {
                "name": self.config.get("optimizer", "adam"),
                "parameters": {"lr": self.config.get("learning_rate", 0.001)},
            },
            "criterion": {"name": self.config.get("criterion", "mse")},
            "output": {
                "model_path": f"{self.config['dataset_dir']}/trained_model.pth",
                "history_path": f"{self.config['dataset_dir']}/training_history.json",
            },
        }

    def _save_training_config(self, config: Dict[str, Any], output_path: str):
        """Save training configuration"""
        try:
            import yaml

            with open(output_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        except ImportError:
            # Fallback to JSON if yaml not available
            with open(output_path.replace(".yaml", ".json"), "w") as f:
                json.dump(config, f, indent=2)
            logger.warning("YAML library not found, saved training configuration as JSON")

        logger.info(f"✓ Training configuration saved to {output_path}")
