"""
Updated MLflow integration functions for main_pipeline.py.

These functions should replace or supplement the existing MLflow code
in main_pipeline.py to improve experiment tracking.
"""

import logging
import time
from pathlib import Path
import torch
from typing import Optional, Dict, Any, Union, Tuple

import mlflow
from utils.experiment_tracking import ExperimentTracker
from utils.mlflow_utils import initialize_mlflow, start_mlflow_server, stop_mlflow_server

logger = logging.getLogger(__name__)

# Global variables to track MLflow state
# MLFLOW_PID = None # No longer needed
EXPERIMENT_TRACKER = None

def setup_mlflow(config: Dict[str, Any],
                mode: str,
                log_dir: Path) -> Tuple[Optional[ExperimentTracker], str]:
    """
    Set up MLflow for the pipeline in a consistent way.

    Args:
        config: Configuration dictionary
        mode: Pipeline operation mode ('train', 'evaluate', 'human_guided_train')
        log_dir: Directory for logs

    Returns:
        tuple of (experiment_tracker, experiment_id)
    """
    global EXPERIMENT_TRACKER

    # Check if MLflow should be enabled
    tracking_enabled = config.get("experiment_tracking", {}).get("enabled", True)
    if not tracking_enabled:
        logger.info("MLflow experiment tracking is disabled in configuration")
        return None, ""

    try:
        # Get MLflow settings from config
        tracking_uri = config["experiment_tracking"].get("tracking_uri", "file:./mlruns")

        # Ensure tracking URI is correctly formatted
        if tracking_uri == "/mlruns":
            tracking_uri = "file:./mlruns"
        elif not tracking_uri.startswith("file:") and not tracking_uri.startswith("http:"):
            tracking_uri = f"file:{tracking_uri}"

        experiment_name = config["experiment_tracking"].get("experiment_name", "dsm_inpainting_master")

        # Initialize MLflow - THIS IS THE ONLY PLACE we call initialize_mlflow
        experiment_id = initialize_mlflow(tracking_uri, experiment_name)

        # Create experiment tracker if not already created
        if EXPERIMENT_TRACKER is None:
            EXPERIMENT_TRACKER = ExperimentTracker(
                experiment_name=experiment_name,
                tracking_uri=tracking_uri
            )
            logger.info(f"Created experiment tracker for {experiment_name}")

        return EXPERIMENT_TRACKER, experiment_id

    except Exception as e:
        logger.error(f"Failed to set up MLflow: {str(e)}")
        return None, ""

def cleanup_mlflow():
    """Clean up MLflow resources"""
    global EXPERIMENT_TRACKER # MLFLOW_PID no longer needed

    try:
        # End any active MLflow run
        if EXPERIMENT_TRACKER and EXPERIMENT_TRACKER.is_run_active:
            EXPERIMENT_TRACKER.end_run()
            logger.info("Ended active MLflow run")

        # Stop MLflow server if it was started - THIS IS HANDLED IN run_pipeline.sh
        # if MLFLOW_PID:
        #     success = stop_mlflow_server(MLFLOW_PID)
        #     if success:
        #         logger.info(f"Stopped MLflow server (PID: {MLFLOW_PID})")
        #     MLFLOW_PID = None
    except Exception as e:
        logger.error(f"Error cleaning up MLflow: {str(e)}")

def start_run_for_mode(mode: str, config: Dict[str, Any]) -> Optional[str]:
    """
    Start an appropriate MLflow run for the given mode.

    Args:
        mode: Pipeline operation mode
        config: Configuration dictionary

    Returns:
        run_id: ID of the started run, or None if failed
    """
    global EXPERIMENT_TRACKER
    if not EXPERIMENT_TRACKER:
        logger.warning("No experiment tracker available. Run start/setup failed previously.")
        return None  # Don't call setup_mlflow again.
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Create appropriate run based on mode
        if mode == "train":
            run_name = f"training_run_{timestamp}"
        elif mode == "evaluate":
            run_name = f"evaluation_run_{timestamp}"
        elif mode == "human_guided_train":
            run_name = f"human_guided_{timestamp}"
        else:
            run_name = f"run_{mode}_{timestamp}"

        # Start the run - ONLY CALL start_run here, not in training loops
        run = EXPERIMENT_TRACKER.start_run(run_name, config)
        if run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run: {run_id} ({run_name})")
            return run_id
        return None

    except Exception as e:
        logger.error(f"Failed to start MLflow run for {mode}: {str(e)}")
        return None

def log_model_safely(model: torch.nn.Module,
                    name: str,
                    metrics: Optional[Dict[str, Any]] = None,
                    save_path: Optional[Path] = None) -> bool:
    """
    Safely log a model to MLflow and optionally save it locally.

    Args:
        model: PyTorch model to log
        name: Name for the model
        metrics: Optional metrics to log with the model
        save_path: Optional path to save the model locally

    Returns:
        success: True if logging was successful
    """
    global EXPERIMENT_TRACKER

    if not EXPERIMENT_TRACKER:
        # If no experiment tracker but save path provided, just save locally
        if save_path:
            try:
                torch.save(model.state_dict(), save_path)
                logger.info(f"Saved model to {save_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to save model to {save_path}: {str(e)}")
                return False
        return False

    try:
        # Save original device
        original_device = next(model.parameters()).device

        # Log model to MLflow
        EXPERIMENT_TRACKER.log_model(model, name, metrics)
        logger.info(f"Logged model {name} to MLflow")

        # Optionally save locally
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved model to {save_path}")

        return True

    except Exception as e:
        logger.error(f"Failed to log model {name}: {str(e)}")

        # Attempt local save as fallback
        if save_path:
            try:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)
                logger.info(f"Saved model to {save_path} (fallback)")
                return True
            except Exception as nested_e:
                logger.error(f"Failed to save model to {save_path}: {str(nested_e)}")

        return False

def log_training_completion(results: Dict[str, Any]) -> bool:
    """
    Log training completion metrics.

    Args:
        results: Dictionary of training results

    Returns:
        success: True if logging was successful
    """
    global EXPERIMENT_TRACKER

    if not EXPERIMENT_TRACKER:
        return False

    try:
        # Convert results to appropriate format
        metrics = {}
        for key, value in results.items():
            if isinstance(value, (int, float)):
                metrics[f"training.{key}"] = value

        # Log metrics
        if metrics:
            EXPERIMENT_TRACKER.log_metrics(metrics)
            logger.info(f"Logged {len(metrics)} training completion metrics")

        return True

    except Exception as e:
        logger.error(f"Failed to log training completion: {str(e)}")
        return False
