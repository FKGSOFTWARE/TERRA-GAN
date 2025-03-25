import numpy as np
from dataclasses import dataclass
from pathlib import Path
import logging
import cv2
from typing import Dict, List, Optional, Tuple
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

@dataclass
class MaskMetrics:
    iou: float
    precision: float
    recall: float
    total_area: int
    feature_count: int
    average_feature_size: float

class MaskEvaluator:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def _identify_features(self, mask: np.ndarray) -> List:
        """Identify distinct features in the mask using connected components."""
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def evaluate_batch(self, gan_masks: List[np.ndarray],
                      human_masks: List[np.ndarray]) -> List[MaskMetrics]:
        """Evaluate a batch of masks."""
        return [self.calculate_metrics(gan, human)
                for gan, human in zip(gan_masks, human_masks)]

    def save_results(self, metrics: MaskMetrics, save_path: Path):
        """Save evaluation results."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'a') as f:
            metrics_dict = {k: v for k, v in metrics.__dict__.items()}
            f.write(f"{metrics_dict}\n")

    def _calculate_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.item()

    def _calculate_ssim(self, pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
        """Calculate Structural Similarity Index"""
        try:
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
        except Exception as e:
            logger.warning(f"Error calculating SSIM: {str(e)}")
            return 0.0

def calculate_boundary_quality(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                        boundary_width: int = 10) -> Dict[str, float]:
    """
    Calculate metrics that measure quality at the boundary between inpainted and original regions.
    Uses simple intensity differences instead of gradients to avoid dimension issues.
    """
    device = pred.device

    try:
        # Extract boundary (with safe operations)
        dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        eroded = 1 - F.max_pool2d(1 - mask, kernel_size=3, stride=1, padding=1)
        boundary = torch.clamp(dilated - eroded, 0.0, 1.0)

        # Handle empty boundary
        if torch.sum(boundary) < 1e-6:
            return {
                'boundary_mse': 0.0,
                'boundary_psnr': 0.0,
                'boundary_gradient_diff': 0.0
            }

        # MSE at boundary
        boundary_mse = torch.mean(((pred - target) * boundary) ** 2)

        # PSNR at boundary
        epsilon = 1e-6
        max_val = 1.0
        boundary_psnr = 10 * torch.log10(max_val**2 / (boundary_mse + epsilon))

        # Simple intensity variation as a proxy for gradient difference
        # This completely avoids the problematic convolution operations
        pred_diff = torch.abs(
            pred[:,:,1:,:] - pred[:,:,:-1,:]).mean() + torch.abs(
            pred[:,:,:,1:] - pred[:,:,:,:-1]).mean()

        target_diff = torch.abs(
            target[:,:,1:,:] - target[:,:,:-1,:]).mean() + torch.abs(
            target[:,:,:,1:] - target[:,:,:,:-1]).mean()

        boundary_gradient_diff = torch.abs(pred_diff - target_diff)

        return {
            'boundary_mse': boundary_mse.item(),
            'boundary_psnr': boundary_psnr.item(),
            'boundary_gradient_diff': boundary_gradient_diff.item()
        }

    except Exception as e:
        logger.error(f"Error calculating boundary quality: {str(e)}")
        return {
            'boundary_mse': 0.0,
            'boundary_psnr': 0.0,
            'boundary_gradient_diff': 0.0
        }

def calculate_metrics(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    """Calculate comprehensive metrics for evaluation"""
    metrics = {}

    # Standard metrics
    metrics['mse'] = F.mse_loss(pred, target).item()
    metrics['psnr'] = self._calculate_psnr(pred, target)
    metrics['ssim'] = self._calculate_ssim(pred, target)

    # Add boundary quality metrics
    boundary_width = self.config['evaluation']['metrics'].get('boundary_size', 10)
    boundary_metrics = calculate_boundary_quality(pred, target, mask, boundary_width)
    metrics.update(boundary_metrics)

    return metrics
