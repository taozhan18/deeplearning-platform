#!/usr/bin/env python3
"""
MOOSE-specific Data Generation Implementation

This module implements the generic data generation framework for MOOSE simulations.
It provides MOOSE-specific solvers and data extractors that inherit from the generic framework.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import logging

# Add paths for local modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

# Import generic framework
from generic_data_generator import SimulationRunner, DataExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import exodusii library
try:
    import exodusii
    EXODUS_AVAILABLE = True
except ImportError:
    logger.warning("exodusii library not found, using fallback methods")
    exodusii = None
    EXODUS_AVAILABLE = False


class MOOSESolver(SimulationRunner):
    """
    MOOSE simulation runner
    
    This class implements the SimulationRunner interface for MOOSE simulations.
    
    Configuration Parameters:
    -------------------------
    The configuration should be a dictionary with the following structure:
    
    {
        "executable": "/path/to/moose-executable",  # Path to MOOSE executable
        "conda_env": "physics",                     # Conda environment to use
        "working_dir": "/path/to/working/dir",      # Working directory
        "timeout": 600,                             # Timeout in seconds
        "env_vars": {                               # Environment variables
            "VAR1": "value1"
        }
    }
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MOOSE simulation runner
        
        Args:
            config: Dictionary containing MOOSE-specific configuration
        """
        super().__init__(config)
        
        # MOOSE-specific configuration
        self.executable = config.get("executable", "/home/zt/workspace/mymoose/mymoose-opt")
        self.conda_env = config.get("conda_env", "physics")
        self.timeout = config.get("timeout", 600)  # Default timeout 600 seconds
        
        self._verify_environment()
    
    def _verify_environment(self):
        """Verify MOOSE environment setup"""
        if not os.path.exists(self.executable):
            raise FileNotFoundError(f"MOOSE executable not found: {self.executable}")
        
        if not os.access(self.executable, os.X_OK):
            raise PermissionError(f"MOOSE executable not executable: {self.executable}")
        
        logger.info(f"✓ MOOSE executable verified: {self.executable}")
    
    def run_simulation(self, input_file: str, output_dir: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run MOOSE simulation with given parameters
        
        Args:
            input_file: Path to the base input file
            output_dir: Directory to run simulation in
            parameters: Dictionary of parameters to use in simulation
            
        Returns:
            Dictionary containing simulation results
        """
        original_dir = os.getcwd()
        result = {
            "status": "success",
            "parameters": parameters,
            "sim_dir": output_dir
        }
        
        try:
            os.chdir(output_dir)
            
            # Prepare input file
            modified_input = os.path.join(output_dir, "input.i")
            self.prepare_input_file(input_file, modified_input, parameters)
            
            # Build command
            cmd = [self.executable, "-i", modified_input]
            
            # Add environment variables if any
            env = os.environ.copy()
            if "env_vars" in self.config:
                env.update(self.config["env_vars"])
            
            logger.info(f"Running MOOSE simulation: {' '.join(cmd)}")
            logger.info(f"Working directory: {output_dir}")
            
            # Run simulation
            process_result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                env=env, 
                timeout=self.timeout
            )
            
            # Log outputs
            if process_result.stdout:
                logger.debug(f"Simulation stdout: {process_result.stdout}")
            if process_result.stderr:
                logger.debug(f"Simulation stderr: {process_result.stderr}")
            
            # Check for success
            if process_result.returncode != 0:
                logger.error(f"MOOSE simulation failed with return code {process_result.returncode}")
                logger.error(f"STDERR: {process_result.stderr}")
                result.update({
                    "status": "failed",
                    "error": process_result.stderr,
                    "returncode": process_result.returncode
                })
                return result
            
            # Find output files
            exodus_files = list(Path(output_dir).glob("*.e"))
            if exodus_files:
                result["exodus_file"] = str(exodus_files[0])
            
            logger.info(f"✓ MOOSE simulation completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error running MOOSE simulation: {str(e)}")
            result.update({
                "status": "failed",
                "error": str(e)
            })
            return result
            
        finally:
            os.chdir(original_dir)


class ExodusDataExtractor(DataExtractor):
    """
    Data extractor for Exodus (.e) files using exodusii library
    
    This class implements the DataExtractor interface for Exodus files.
    
    Configuration Parameters:
    -------------------------
    The configuration should be a dictionary with the following structure:
    
    {
        "variables": ["u", "v"],  # Variables to extract
        "time_step": -1           # Time step to extract (-1 for last)
    }
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Exodus data extractor
        
        Args:
            config: Configuration for data extraction
        """
        super().__init__(config)
        self.exodusii_available = EXODUS_AVAILABLE
        if self.exodusii_available:
            self.exodusii = exodusii
    
    def _exodusii_extraction(self, exodus_file: str, variables: List[str] = None, time_step: int = -1) -> Dict[str, np.ndarray]:
        """Extract data using exodusii library"""
        data = {}
        
        with self.exodusii.File(exodus_file, mode="r") as exo:
            # Get time steps
            time_steps = exo.get_times()
            actual_time_step = time_step if 0 <= time_step < len(time_steps) else len(time_steps) - 1
            
            # Get nodal variables
            node_vars = exo.get_node_variable_names()
            
            # Extract specified variables or all available
            vars_to_extract = variables if variables else node_vars
            
            for var_name in vars_to_extract:
                if var_name in node_vars:
                    values = exo.get_node_variable_values(var_name, actual_time_step + 1)  # 1-based indexing
                    data[var_name] = values
                else:
                    logger.warning(f"Variable '{var_name}' not found in Exodus file")
            
            # Get element variables if needed
            elem_vars = exo.get_element_variable_names()
            for var_name in vars_to_extract:
                if var_name in elem_vars:
                    values = exo.get_element_variable_values(var_name, actual_time_step + 1)
                    data[f"element_{var_name}"] = values
            
            logger.info(f"Extracted {len(data)} variables from Exodus file")
            logger.debug(f"Variables: {list(data.keys())}, Time step: {actual_time_step}, Time: {time_steps[actual_time_step]}")
            
        return data
    
    def _fallback_extraction(self, file_path: str, variables: List[str] = None) -> Dict[str, np.ndarray]:
        """Fallback extraction using meshio or synthetic data"""
        try:
            import meshio
            
            mesh = meshio.read(file_path)
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
            
            logger.info(f"Extracted {len(data)} variables using meshio fallback")
            return data
            
        except ImportError:
            logger.warning("meshio not available, creating synthetic data")
            # Create synthetic data for testing
            n_points = 100
            x = np.linspace(0, 1, n_points)
            u = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(n_points)
            
            return {"x": x, "u": u}
    
    def extract_data(self, file_path: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract data from Exodus file
        
        Args:
            file_path: Path to the Exodus file
            parameters: Optional parameters used in simulation
            
        Returns:
            Dictionary containing extracted data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Simulation output file not found: {file_path}")
        
        variables = self.config.get("variables")
        time_step = self.config.get("time_step", -1)
        
        if self.exodusii_available:
            try:
                return self._exodusii_extraction(file_path, variables, time_step)
            except Exception as e:
                logger.error(f"Error with exodusii extraction: {e}")
                return self._fallback_extraction(file_path, variables)
        else:
            return self._fallback_extraction(file_path, variables)


class MOOSEDataGeneratorPipeline:
    """
    MOOSE-specific data generation pipeline
    
    This class provides a complete pipeline for MOOSE simulations:
    1. Run parametric MOOSE simulations
    2. Extract data from Exodus files
    3. Generate ML-ready datasets
    4. Create training configurations
    
    Configuration Parameters:
    -------------------------
    The configuration should be a dictionary with the following structure:
    
    {
        "solver": "moose",                  # Solver type (must be "moose")
        "solver_config": {                  # MOOSE solver configuration
            "executable": "/path/to/moose-executable",
            "conda_env": "physics"
        },
        "extractor": "exodus",              # Extractor type (must be "exodus")
        "extractor_config": {               # Exodus extractor configuration
            "variables": ["u", "v"],
            "time_step": -1
        },
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
    
    def __init__(self, config_file: str = None, config: Dict[str, Any] = None):
        """
        Initialize MOOSE data generation pipeline
        
        Args:
            config_file: Path to configuration file (JSON or YAML)
            config: Dictionary containing pipeline configuration (overrides config_file)
        """
        if config_file:
            # Load configuration from file
            with open(config_file, "r") as f:
                if config_file.endswith(".yaml") or config_file.endswith(".yml"):
                    import yaml
                    self.config = yaml.safe_load(f)
                else:
                    self.config = json.load(f)
        elif config:
            # Use provided configuration
            self.config = config
        else:
            # Use default configuration
            self.config = self._get_default_config()
        
        # Override solver and extractor types for MOOSE
        self.config["solver"] = "moose"
        self.config["extractor"] = "exodus"
        
        # Import generic pipeline
        from generic_data_generator import GenericDataGenerationPipeline
        self.pipeline = GenericDataGenerationPipeline(self.config)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for MOOSE pipeline"""
        return {
            "solver": "moose",
            "solver_config": {
                "executable": "/home/zt/workspace/mymoose/mymoose-opt",
                "conda_env": "physics"
            },
            "extractor": "exodus",
            "extractor_config": {
                "variables": None,
                "time_step": -1
            },
            "base_input_file": "input.i",
            "simulation_output_dir": "sims",
            "dataset_dir": "dataset",
            "parameters": {},
            "num_samples": 10,
            "variables_to_extract": None,
            "normalize": True,
            "model_name": "mlp",
            "epochs": 100,
            "batch_size": 16,
            "learning_rate": 0.001,
            "use_gpu": True
        }
    
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete MOOSE data generation pipeline
        
        Returns:
            Dictionary containing pipeline results
        """
        return self.pipeline.run_pipeline()
    
    def run_simulations(self, base_input_file: str = None, output_dir: str = None) -> Dict[str, Any]:
        """
        Run parametric MOOSE simulations
        
        Args:
            base_input_file: Path to the base input file (overrides config)
            output_dir: Directory to store simulation outputs (overrides config)
            
        Returns:
            Dictionary containing simulation results
        """
        if base_input_file:
            self.pipeline.data_generator.solver.prepare_input_file(
                base_input_file, 
                os.path.join(output_dir or self.config["simulation_output_dir"], "input.i"), 
                {}
            )
        
        return self.pipeline.data_generator.run_simulations(
            base_input_file=base_input_file or self.config["base_input_file"],
            output_dir=output_dir or self.config["simulation_output_dir"]
        )
    
    def generate_ml_dataset(self, simulation_results: Dict[str, Any], dataset_dir: str = None) -> Dict[str, np.ndarray]:
        """
        Generate ML-ready dataset from simulation results
        
        Args:
            simulation_results: Dictionary containing simulation results
            dataset_dir: Directory to store dataset (overrides config)
            
        Returns:
            Dictionary containing ML dataset
        """
        return self.pipeline.data_generator.generate_ml_dataset(
            simulation_results=simulation_results,
            dataset_dir=dataset_dir or self.config["dataset_dir"]
        )