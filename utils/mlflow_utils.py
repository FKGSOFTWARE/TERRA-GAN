"""
MLflow utility functions for consistent experiment tracking setup.

This module provides helper functions for MLflow configuration, management,
and cleanup to ensure consistent usage across the application.
"""

import os
import time
import logging
import subprocess
import shutil
from pathlib import Path
import mlflow
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

def initialize_mlflow(tracking_uri: str = "http://localhost:5000",
                     experiment_name: str = "dsm_inpainting_master") -> str:
    """
    Initialize MLflow with consistent settings.

    Args:
        tracking_uri: URI for MLflow tracking server
        experiment_name: Name of the experiment to use

    Returns:
        experiment_id: ID of the initialized experiment
    """
    # Configure MLflow - just once in a central location
    # os.environ["MLFLOW_TRACKING_URI"] = tracking_uri # NOT NEEDED.
    mlflow.set_tracking_uri(tracking_uri)

    # Get or create the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created new MLflow experiment: {experiment_name}")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing MLflow experiment: {experiment_name}")

    return experiment_id

def start_mlflow_server(port: int = 5000, log_file: Optional[Path] = None) -> Optional[int]:
    """
    Start MLflow server as a background process.  DEPRECATED. Use start_mlflow.sh

    Args:
        port: Port to run MLflow server on
        log_file: Optional path to log file

    Returns:
        process_id: PID of the started MLflow server, or None if failed
    """
    logger.warning("DEPRECATED: Use start_mlflow.sh to start the MLflow server.")
    return None

def stop_mlflow_server(pid: Optional[int] = None) -> bool:
    """
    Stop MLflow server process. DEPRECATED. Use start_mlflow.sh

    Args:
        pid: Optional process ID to kill specifically

    Returns:
        success: True if server was stopped successfully
    """
    logger.warning("DEPRECATED: Use start_mlflow.sh to stop the MLflow server.")
    return False

def clean_mlflow_directory(directory: str = "./mlruns",
                          keep_experiments: bool = True,
                          keep_metadata: bool = True) -> bool:
    """
    Clean up MLflow directory, removing empty files and unnecessary artifacts.

    Args:
        directory: Path to MLflow directory
        keep_experiments: Whether to keep experiment definitions
        keep_metadata: Whether to keep run metadata

    Returns:
        success: True if cleanup was successful
    """
    try:
        mlruns_dir = Path(directory)
        if not mlruns_dir.exists():
            logger.warning(f"MLflow directory {directory} does not exist")
            return False

        # Define directories to keep
        keep_dirs = set()
        if keep_experiments:
            # Keep experiment directories
            keep_dirs.update([
                d for d in mlruns_dir.glob("*")
                if d.is_dir() and not d.name.startswith(".")
            ])

        # Handle metadata files
        if keep_metadata:
            # Keep meta.yaml files
            for meta_file in mlruns_dir.glob("**/meta.yaml"):
                keep_dirs.add(meta_file.parent)

        # Remove empty files
        empty_files_count = 0
        for file_path in mlruns_dir.glob("**/*"):
            if file_path.is_file() and file_path.stat().st_size == 0:
                if any(str(file_path).startswith(str(d)) for d in keep_dirs):
                    # This is in a keep directory, check if it's a parameter file
                    if file_path.parent.name != "params":
                        file_path.unlink()
                        empty_files_count += 1
                else:
                    file_path.unlink()
                    empty_files_count += 1

        logger.info(f"Removed {empty_files_count} empty files from MLflow directory")
        return True

    except Exception as e:
        logger.error(f"Error cleaning MLflow directory: {str(e)}")
        return False

def reset_mlflow_environment(directory: str = "./mlruns",
                            experiment_name: str = "dsm_inpainting_master") -> bool:
    """
    Completely reset the MLflow environment.

    Args:
        directory: Path to MLflow directory
        experiment_name: Name of the experiment to recreate

    Returns:
        success: True if reset was successful
    """
    try:
        mlruns_dir = Path(directory)

        # Stop MLflow server if running.  This is redundant, but harmless
        stop_mlflow_server()

        # Back up the directory if it exists
        if mlruns_dir.exists():
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_dir = mlruns_dir.parent / f"mlruns_backup_{timestamp}"
            shutil.copytree(mlruns_dir, backup_dir)
            logger.info(f"Backed up MLflow directory to {backup_dir}")

            # Remove the original directory
            shutil.rmtree(mlruns_dir)
            logger.info(f"Removed MLflow directory {mlruns_dir}")

        # Create base structure
        mlruns_dir.mkdir(exist_ok=True)

        # Create experiment directory
        experiment_dir = mlruns_dir / experiment_name
        experiment_dir.mkdir(exist_ok=True)

        # Create required subdirectories
        for subdir in ["artifacts", "metrics", "params", "tags"]:
            (experiment_dir / subdir).mkdir(exist_ok=True)

        # Create meta.yaml
        current_time = int(time.time() * 1000)
        meta_content = {
            "artifact_location": f"file:{directory}/{experiment_name}",
            "creation_time": current_time,
            "experiment_id": experiment_name,
            "last_update_time": current_time,
            "lifecycle_stage": "active",
            "name": experiment_name
        }

        import yaml
        with open(experiment_dir / "meta.yaml", "w") as f:
            yaml.dump(meta_content, f)

        logger.info(f"Successfully reset MLflow environment with experiment {experiment_name}")
        return True

    except Exception as e:
        logger.error(f"Error resetting MLflow environment: {str(e)}")
        return False

def check_run_exists(run_id: str) -> bool:
    """
    Check if a run with the given ID exists.

    Args:
        run_id: MLflow run ID to check

    Returns:
        exists: True if run exists
    """
    try:
        run = mlflow.get_run(run_id)
        return run is not None
    except Exception:
        return False

def batch_log_metrics(metrics: Dict[str, Any],
                     step: Optional[int] = None,
                     run_id: Optional[str] = None) -> bool:
    """
    Log multiple metrics efficiently in a single operation.

    Args:
        metrics: Dictionary of metrics to log
        step: Optional step number
        run_id: Optional run ID to log to

    Returns:
        success: True if logging was successful
    """
    try:
        # Organize metrics by category
        categorized = {}
        for key, value in metrics.items():
            # Skip non-numeric values
            if not isinstance(value, (int, float)) and not (
                hasattr(value, "item") and callable(getattr(value, "item"))
            ):
                continue

            # Extract category from key
            if "." in key:
                category, metric_name = key.split(".", 1)
            else:
                category = "metrics"
                metric_name = key

            if category not in categorized:
                categorized[category] = {}

            # Convert to standard Python numeric type
            if hasattr(value, "item") and callable(getattr(value, "item")):
                value = value.item()

            categorized[category][f"{category}.{metric_name}"] = float(value)

        # Log metrics by category
        with mlflow.start_run(run_id=run_id, nested=True) as run: #Always nest.
            for metrics_dict in categorized.values():
                if metrics_dict:
                    mlflow.log_metrics(metrics_dict, step=step)

        return True

    except Exception as e:
        logger.error(f"Error batch logging metrics: {str(e)}")
        return False
