# utils/experiment_tracking.py
import os
import time
import json
import yaml
import mlflow
import psutil
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import logging
from dataclasses import dataclass
import git
from contextlib import contextmanager

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec


@dataclass
class SystemMetrics:
    """Track system resource usage"""
    cpu_percent: float
    memory_percent: float
    gpu_memory_allocated: Optional[float] = None
    gpu_memory_cached: Optional[float] = None


class ExperimentTracker:
    """
    MLflow experiment tracker for managing ML experiment logging and model artifacts.

    This class handles:
    - Experiment initialization and management
    - Run tracking and metrics logging
    - Model architecture logging
    - System metrics tracking
    - Model artifact management
    """

    def __init__(self, experiment_name: str, tracking_uri: str = None):
        """
        Initialize MLflow experiment tracker

        Args:
            experiment_name: Name of the experiment
            tracking_uri: URI for MLflow tracking server. If None, uses local filesystem
        """
        self.logger = logging.getLogger(__name__)
        self.tracking_uri = tracking_uri or "file:./mlruns"
        self.experiment_name = experiment_name
        self.run = None
        self.start_time = None
        self.is_run_active = False
        self.run_id = None

        # Initialize temporary storage for metrics batching
        self.batch_metrics_buffer = {}
        self.last_metrics_flush_time = time.time()
        self.metrics_flush_interval = 5  # seconds

        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            self._initialize_mlflow_experiment()
        except Exception as e:
            self.logger.error(f"Failed to initialize experiment tracker: {str(e)}")
            raise

    def _initialize_mlflow_experiment(self) -> None:
        """Initialize or get existing MLflow experiment"""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(
                self.experiment_name
            )
            self.logger.info(f"Created new experiment: {self.experiment_name}")
        else:
            self.experiment_id = experiment.experiment_id
            self.logger.info(f"Using existing experiment: {self.experiment_name}")

    @contextmanager
    def _ensure_active_run(self, run_name: str = None):
        """Context manager to ensure an active run exists"""
        if not self.is_run_active:
            self.start_run(run_name or f"run_{int(time.time())}", {})
            created_new = True
        else:
            created_new = False

        try:
            yield
        finally:
            if created_new:
                self.end_run()

    def start_run(self,
                 run_name: str,
                 config: Dict[str, Any],
                 tags: Dict[str, str] = None) -> Optional[mlflow.ActiveRun]:
        """
        Start a new tracking run

        Args:
            run_name: Name for the new run
            config: Configuration dictionary to log as parameters
            tags: Optional dictionary of tags to attach to the run

        Returns:
            Active MLflow run or None if failed
        """
        if self.is_run_active:
            self.logger.info("Using existing MLflow run")
            return self.run

        try:
            git_tags = self._get_git_info()
            run_tags = {**(tags or {}), **git_tags}

            mlflow.set_experiment(self.experiment_name)
            self.run = mlflow.start_run(
                run_name=run_name,
                tags=run_tags
            )
            self.run_id = self.run.info.run_id
            self.is_run_active = True

            # Log flattened parameters in hierarchical structure
            flattened_params = self._flatten_dict(config)

            # Batch parameters by top-level category
            param_categories = {}
            for key, value in flattened_params.items():
                category = key.split('.')[0] if '.' in key else 'main'
                if category not in param_categories:
                    param_categories[category] = {}
                param_categories[category][key] = value

            # Log parameters by category to reduce individual file operations
            for params in param_categories.values():
                mlflow.log_params(params)

            # Log model architecture if available
            if "model" in config:
                self._log_model_architecture(config["model"])

            self.start_time = time.time()
            return self.run

        except Exception as e:
            self.logger.error(f"Failed to start run: {str(e)}")
            self.is_run_active = False
            self.run = None
            self.run_id = None
            raise

    def _get_git_info(self) -> Dict[str, str]:
        """
        Get git repository information

        Returns:
            Dictionary containing git commit, branch, and repo information
        """
        try:
            repo = git.Repo(search_parent_directories=True)
            return {
                "git_commit": repo.head.commit.hexsha,
                "git_branch": repo.active_branch.name,
                "git_repo": repo.remotes.origin.url if repo.remotes else "local"
            }
        except Exception as e:
            self.logger.warning(f"Could not get git info: {e}")
            return {}

    # Add the missing method
    def _calculate_l1_l2(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
        """
        Calculate L1 and L2 distances between predictions and targets.

        Args:
            pred: Model predictions
            target: Ground truth targets

        Returns:
            Tuple of (L1 distance, L2 distance)
        """
        try:
            l1_dist = torch.nn.functional.l1_loss(pred, target).item()
            l2_dist = torch.nn.functional.mse_loss(pred, target, reduction='mean').sqrt().item()
            return l1_dist, l2_dist
        except Exception as e:
            self.logger.error(f"Error calculating L1/L2 distances: {str(e)}")
            return 0.0, 0.0

    # Add the missing method
    def _calculate_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        try:
            mse = torch.nn.functional.mse_loss(pred, target)
            if mse == 0:
                return float('inf')
            max_pixel = 1.0
            psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
            return psnr.item()
        except Exception as e:
            self.logger.warning(f"Error calculating PSNR: {str(e)}")
            return 0.0

    # Add the missing method
    def _calculate_ssim(self, pred: torch.Tensor, target: torch.Tensor, window_size=11) -> float:
        """Calculate Structural Similarity Index"""
        try:
            C1 = (0.01 * 1.0) ** 2
            C2 = (0.03 * 1.0) ** 2

            mu1 = torch.nn.functional.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
            mu2 = torch.nn.functional.avg_pool2d(target, window_size, stride=1, padding=window_size//2)

            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2

            sigma1_sq = torch.nn.functional.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size//2) - mu1_sq
            sigma2_sq = torch.nn.functional.avg_pool2d(target * target, window_size, stride=1, padding=window_size//2) - mu2_sq
            sigma12 = torch.nn.functional.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean().item()
        except Exception as e:
            self.logger.warning(f"Error calculating SSIM: {str(e)}")
            return 0.0

    def log_training_batch(self,
                        pred: torch.Tensor,
                        target: torch.Tensor,
                        model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer,
                        batch_metrics: Dict[str, float],
                        step: int) -> None:
        """
        Log comprehensive batch-level metrics, batching them efficiently
        """
        if not self.is_run_active:
            with self._ensure_active_run():
                pass

        try:
            # Get standard training metrics
            batch_start_time = time.time()
            performance_metrics = self._calculate_performance_metrics(
                pred, target, model, optimizer, batch_start_time
            )

            # Clean and combine all metrics
            combined_metrics = {
                **self._clean_metrics(batch_metrics),
                **self._clean_metrics(performance_metrics),
                **self._get_system_metrics().__dict__
            }

            # Log boundary loss specifically if available
            if 'boundary_loss' in batch_metrics:
                combined_metrics['boundary_loss'] = batch_metrics['boundary_loss']

            # Log boundary metrics if available
            boundary_metrics = ['boundary_mse', 'boundary_psnr', 'boundary_gradient_diff']
            for metric in boundary_metrics:
                if metric in batch_metrics:
                    combined_metrics[metric] = batch_metrics[metric]

            # Add to metrics buffer with step information
            for k, v in combined_metrics.items():
                metric_key = f"batch.{k}"
                if metric_key not in self.batch_metrics_buffer:
                    self.batch_metrics_buffer[metric_key] = []
                self.batch_metrics_buffer[metric_key].append((step, v))

            # Flush metrics if enough time has passed
            current_time = time.time()
            if current_time - self.last_metrics_flush_time >= self.metrics_flush_interval:
                self._flush_metrics_buffer()
                self.last_metrics_flush_time = current_time

        except Exception as e:
            self.logger.error(f"Failed to log batch metrics: {str(e)}")

    def _flush_metrics_buffer(self):
        """Flush all buffered metrics to MLflow"""
        if not self.batch_metrics_buffer:
            return

        try:
            with mlflow.start_run(run_id=self.run_id, nested=True) as run: # Always nest.
                # Group metrics by step for efficient logging
                metrics_by_step = {}
                for metric_key, values in self.batch_metrics_buffer.items():
                    for step, value in values:
                        if step not in metrics_by_step:
                            metrics_by_step[step] = {}
                        metrics_by_step[step][metric_key] = value

                # Log metrics for each step
                for step, metrics in metrics_by_step.items():
                    mlflow.log_metrics(metrics, step=step)

                # Clear the buffer after successful logging
                self.batch_metrics_buffer = {}
        except Exception as e:
            self.logger.error(f"Failed to flush metrics buffer: {str(e)}")

    def log_validation_metrics(self,
                             model: torch.nn.Module,
                             val_loader: torch.utils.data.DataLoader,
                             device: torch.device,
                             step: int) -> Dict[str, float]:
        """
        Calculate and log validation metrics

        Args:
            model: The model to evaluate
            val_loader: DataLoader for validation data
            device: Device to run validation on
            step: Current training step

        Returns:
            Dictionary of calculated validation metrics
        """
        with self._ensure_active_run():
            try:
                model.eval()
                val_metrics = {
                    'val_psnr': 0,
                    'val_ssim': 0,
                    'val_l1': 0,
                    'val_l2': 0
                }

                with torch.no_grad():
                    for batch in val_loader:
                        pred = model(batch['image'].to(device))
                        target = batch['target'].to(device)

                        # Calculate metrics
                        val_metrics['val_psnr'] += self._calculate_psnr(pred, target)
                        val_metrics['val_ssim'] += self._calculate_ssim(pred, target)
                        l1, l2 = self._calculate_l1_l2(pred, target)
                        val_metrics['val_l1'] += l1
                        val_metrics['val_l2'] += l2

                # Average metrics
                num_batches = len(val_loader)
                val_metrics = {k: v/num_batches for k, v in val_metrics.items()}

                # Log validation metrics
                prefixed_metrics = {
                    f"validation.{k}": v for k, v in val_metrics.items()
                }

                mlflow.log_metrics(prefixed_metrics, step=step)
                return val_metrics

            except Exception as e:
                self.logger.error(f"Failed to log validation metrics: {str(e)}")
                return {}

    def log_metrics(self,
                   metrics: Dict[str, float],
                   step: Optional[int] = None) -> None:
        """
        Log metrics for current step

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for the metrics
        """
        with self._ensure_active_run():
            try:
                # Clean metrics first
                cleaned_metrics = self._clean_metrics(metrics)

                # Get system metrics
                sys_metrics = self._get_system_metrics()
                system_metrics = {
                    "system.cpu_percent": sys_metrics.cpu_percent,
                    "system.memory_percent": sys_metrics.memory_percent
                }

                if sys_metrics.gpu_memory_allocated is not None:
                    system_metrics.update({
                        "system.gpu_memory_allocated": sys_metrics.gpu_memory_allocated,
                        "system.gpu_memory_cached": sys_metrics.gpu_memory_cached
                    })

                # Combine all metrics
                all_metrics = {**cleaned_metrics, **system_metrics}
                mlflow.log_metrics(all_metrics, step=step)

            except Exception as e:
                self.logger.error(f"Failed to log metrics: {str(e)}")

    def _clean_metrics(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Convert tensor metrics to Python numeric types for MLflow logging.

        Args:
            metrics: Dictionary of metrics which may contain tensors

        Returns:
            Dictionary with all metrics converted to Python numeric types
        """
        clean_metrics = {}
        for k, v in metrics.items():
            try:
                if isinstance(v, torch.Tensor):
                    if v.numel() == 1:
                        clean_metrics[k] = v.item()
                elif isinstance(v, (int, float)):
                    clean_metrics[k] = float(v)
                elif isinstance(v, np.ndarray) and v.size == 1:
                    clean_metrics[k] = float(v.item())
            except Exception:
                continue
        return clean_metrics

    def _flatten_dict(self,
                     d: Dict,
                     parent_key: str = '',
                     sep: str = '.') -> Dict[str, str]:
        """
        Flatten nested dictionary for MLflow params

        Args:
            d: Dictionary to flatten
            parent_key: Prefix for flattened keys
            sep: Separator between nested keys

        Returns:
            Flattened dictionary with string values
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, str(v)))
        return dict(items)

    def _get_system_metrics(self) -> SystemMetrics:
        """
        Get current system resource usage

        Returns:
            SystemMetrics object containing CPU, memory, and GPU metrics
        """
        try:
            metrics = SystemMetrics(
                cpu_percent=psutil.cpu_percent(),
                memory_percent=psutil.virtual_memory().percent
            )

            # Get GPU metrics if available
            if torch.cuda.is_available():
                metrics.gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                metrics.gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB

            return metrics
        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {str(e)}")
            return SystemMetrics(cpu_percent=0.0, memory_percent=0.0)

    def _log_model_architecture(self, model):
        """
        Log model architecture details

        Args:
            model: Model to log architecture details for
        """
        if not isinstance(model, torch.nn.Module):
            return

        try:
            # Get model summary
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Use model class name to create unique parameter names
            model_type = model.__class__.__name__.lower()

            # Log as a single batch of parameters
            arch_params = {
                f"{model_type}.total_parameters": total_params,
                f"{model_type}.trainable_parameters": trainable_params,
                f"{model_type}.architecture": model.__class__.__name__
            }
            mlflow.log_params(arch_params)

            # Log model architecture as artifact
            arch_path = Path(f"{model_type}_architecture.txt")
            try:
                with open(arch_path, "w") as f:
                    f.write(str(model))
                    f.write(f"\nModel device: {next(model.parameters()).device}") # Add device information.

                mlflow.log_artifact(str(arch_path))
            finally:
                # Always clean up the temporary file
                if arch_path.exists():
                    arch_path.unlink()

        except Exception as e:
            self.logger.error(f"Failed to log model architecture: {str(e)}")

    def log_model(self, model, name: str, metrics: Dict[str, Any] = None, input_example=None) -> None:
        """
        Log a model checkpoint with optional metrics

        Args:
            model: The model to log
            name: Name for the logged model
            metrics: Optional dictionary of metrics to log with the model
            input_example: Optional input example for the model
        """
        with self._ensure_active_run():
            try:
                # Save original device and move model to CPU for logging
                original_device = next(model.parameters()).device
                model = model.cpu()

                try:
                    # Create a wrapper for the PConvUNet model to handle the mask input
                    if model.__class__.__name__ == 'PConvUNet':
                        # Create a simple wrapper to handle the input validation
                        class ModelWrapper(torch.nn.Module):
                            def __init__(self, model):
                                super().__init__()
                                self.model = model

                            def forward(self, x):
                                # Create a dummy mask of ones with the same batch size
                                mask = torch.ones_like(x)
                                return self.model(x, mask)

                        # Wrap the model
                        wrapped_model = ModelWrapper(model)

                        # Create a proper input example
                        if input_example is None:
                            input_example = np.zeros((1, 1, 512, 512), dtype=np.float32)

                        # Get torch requirement string with CUDA if available
                        if torch.cuda.is_available():
                            cuda_version = torch.version.cuda
                            torch_requirement = f'torch=={torch.__version__}+cu{cuda_version.replace(".", "")}'
                        else:
                            torch_requirement = f'torch=={torch.__version__}'

                        # Define pip requirements explicitly
                        pip_requirements = [
                            torch_requirement,
                            'numpy>=' + np.__version__,
                        ]

                        # Log the model relative to the artifacts with the wrapper
                        mlflow.pytorch.log_model(
                            pytorch_model=wrapped_model,
                            artifact_path=name,
                            input_example=input_example,
                            pip_requirements=pip_requirements
                        )
                    else:
                        # For other models, log normally
                        if input_example is None:
                            input_example = np.zeros((1, 1, 512, 512), dtype=np.float32)

                        # Get torch requirement string with CUDA if available
                        if torch.cuda.is_available():
                            cuda_version = torch.version.cuda
                            torch_requirement = f'torch=={torch.__version__}+cu{cuda_version.replace(".", "")}'
                        else:
                            torch_requirement = f'torch=={torch.__version__}'

                        # Define pip requirements explicitly
                        pip_requirements = [
                            torch_requirement,
                            'numpy>=' + np.__version__,
                        ]

                        # Log the model relative to the artifacts
                        mlflow.pytorch.log_model(
                            pytorch_model=model,
                            artifact_path=name,
                            input_example=input_example,
                            pip_requirements=pip_requirements
                        )

                    # Log metrics if provided
                    if metrics:
                        scalar_metrics = {
                            f"{name}.{k}": float(v)
                            for k, v in metrics.items()
                            if isinstance(v, (int, float)) or
                            (isinstance(v, torch.Tensor) and v.numel() == 1)
                        }
                        if scalar_metrics:
                            mlflow.log_metrics(scalar_metrics)

                    self.logger.info(f"Successfully logged model {name}")

                except Exception as e:
                    self.logger.error(f"Failed to log model artifact: {str(e)}")
                finally:
                    # Always restore model to original device
                    model = model.to(original_device)
                    if hasattr(model, 'train'):
                        model.train()  # Restore training mode if applicable

            except Exception as e:
                self.logger.error(f"Failed to log model: {str(e)}")
                # Ensure model returns to original device even in case of outer error
                if torch.cuda.is_available():
                    model = model.to(original_device)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log an artifact file or directory

        Args:
            local_path: Path to the local file or directory to log
            artifact_path: Optional path to use within the artifact directory
        """
        with self._ensure_active_run():
            try:
                mlflow.log_artifact(local_path, artifact_path)
            except Exception as e:
                self.logger.error(f"Failed to log artifact: {str(e)}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters for the current run

        Args:
            params: Dictionary of parameters to log
        """
        with self._ensure_active_run():
            try:
                # Flatten nested dictionaries for MLflow
                flat_params = self._flatten_dict(params)
                mlflow.log_params(flat_params)
            except Exception as e:
                self.logger.error(f"Failed to log parameters: {str(e)}")

    def end_run(self) -> None:
        """End current tracking run and log final metrics"""
        if not self.is_run_active:
            return

        try:
            # Flush any remaining metrics
            self._flush_metrics_buffer()

            # Log total run time
            if self.start_time is not None:
                duration = time.time() - self.start_time
                mlflow.log_metric("training_duration_seconds", duration)

            mlflow.end_run()
            self.logger.info(f"Ended MLflow run: {self.run_id}")
        except Exception as e:
            self.logger.error(f"Failed to end run cleanly: {str(e)}")
        finally:
            self.run = None
            self.run_id = None
            self.is_run_active = False
            self.start_time = None

    # Performance metric calculation methods
    def _calculate_performance_metrics(self, pred, target, model, optimizer, start_time):
        """Calculate comprehensive performance metrics"""
        metrics = {}

        # Basic metrics
        metrics['psnr'] = self._calculate_psnr(pred, target)
        metrics['ssim'] = self._calculate_ssim(pred, target)
        l1_dist, l2_dist = self._calculate_l1_l2(pred, target)
        metrics['l1_distance'] = l1_dist
        metrics['l2_distance'] = l2_dist

        # Optimizer metrics
        metrics.update(self._get_optimizer_metrics(optimizer))

        # Time metrics
        metrics['batch_time'] = time.time() - start_time

        return metrics

    def _get_optimizer_metrics(self, optimizer):
        """Get metrics from the optimizer"""
        try:
            metrics = {}
            for i, param_group in enumerate(optimizer.param_groups):
                metrics[f'lr_group_{i}'] = param_group['lr']
            return metrics
        except Exception as e:
            self.logger.warning(f"Error getting optimizer metrics: {str(e)}")
            return {}
