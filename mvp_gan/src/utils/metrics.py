import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np
from torchvision.transforms.functional import to_tensor
import psutil
import GPUtil
from datetime import datetime

class PerformanceMetrics:
    @staticmethod
    def calculate_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.item()

    @staticmethod
    def calculate_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
        """Calculate Structural Similarity Index"""
        C1 = (0.01 * 1.0) ** 2
        C2 = (0.03 * 1.0) ** 2

        mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
        mu2 = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean().item()

    @staticmethod
    def calculate_l1_l2(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
        """Calculate L1 and L2 distances"""
        l1_dist = F.l1_loss(pred, target).item()
        l2_dist = F.mse_loss(pred, target, reduction='mean').sqrt().item()
        return l1_dist, l2_dist

class TrainingMetrics:
    @staticmethod
    def calculate_gradient_norm(model: torch.nn.Module) -> Dict[str, float]:
        """Calculate gradient norms for model parameters"""
        total_norm = 0
        param_norms = {}

        for name, p in model.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                param_norms[f"grad_norm_{name}"] = param_norm.item()
                total_norm += param_norm.item() ** 2

        total_norm = total_norm ** 0.5
        param_norms['total_grad_norm'] = total_norm

        return param_norms

    @staticmethod
    def get_learning_rates(optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Get current learning rates"""
        return {f"lr_group_{i}": group['lr'] for i, group in enumerate(optimizer.param_groups)}

class ResourceMetrics:
    @staticmethod
    def get_gpu_metrics() -> Optional[Dict[str, float]]:
        """Get GPU utilization metrics"""
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return None

            gpu_metrics = {}
            for i, gpu in enumerate(gpus):
                gpu_metrics.update({
                    f"gpu_{i}_memory_used": gpu.memoryUsed,
                    f"gpu_{i}_memory_total": gpu.memoryTotal,
                    f"gpu_{i}_utilization": gpu.load * 100
                })
            return gpu_metrics
        except Exception:
            return None

    @staticmethod
    def get_cpu_metrics() -> Dict[str, float]:
        """Get CPU utilization metrics"""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent
        }

    @staticmethod
    def get_batch_timing(start_time: float) -> float:
        """Calculate batch processing time"""
        return datetime.now().timestamp() - start_time

class MetricsLogger:
    def __init__(self):
        self.performance_metrics = PerformanceMetrics()
        self.training_metrics = TrainingMetrics()
        self.resource_metrics = ResourceMetrics()

    def log_batch_metrics(self,
                         pred: torch.Tensor,
                         target: torch.Tensor,
                         model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer,
                         batch_start_time: float) -> Dict[str, float]:
        """Log comprehensive batch metrics"""
        metrics = {}

        # Performance metrics
        metrics['psnr'] = self.performance_metrics.calculate_psnr(pred, target)
        metrics['ssim'] = self.performance_metrics.calculate_ssim(pred, target)
        l1_dist, l2_dist = self.performance_metrics.calculate_l1_l2(pred, target)
        metrics['l1_distance'] = l1_dist
        metrics['l2_distance'] = l2_dist

        # Training metrics
        metrics.update(self.training_metrics.calculate_gradient_norm(model))
        metrics.update(self.training_metrics.get_learning_rates(optimizer))

        # Resource metrics
        gpu_metrics = self.resource_metrics.get_gpu_metrics()
        if gpu_metrics:
            metrics.update(gpu_metrics)
        metrics.update(self.resource_metrics.get_cpu_metrics())
        metrics['batch_time'] = self.resource_metrics.get_batch_timing(batch_start_time)

        return metrics

    def log_validation_metrics(self,
                             model: torch.nn.Module,
                             val_loader: torch.utils.data.DataLoader,
                             device: torch.device) -> Dict[str, float]:
        """Calculate validation metrics"""
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

                val_metrics['val_psnr'] += self.performance_metrics.calculate_psnr(pred, target)
                val_metrics['val_ssim'] += self.performance_metrics.calculate_ssim(pred, target)
                l1, l2 = self.performance_metrics.calculate_l1_l2(pred, target)
                val_metrics['val_l1'] += l1
                val_metrics['val_l2'] += l2

        # Average metrics
        num_batches = len(val_loader)
        return {k: v/num_batches for k, v in val_metrics.items()}
