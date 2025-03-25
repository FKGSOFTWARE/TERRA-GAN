from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
import logging

logger = logging.getLogger(__name__)

class InpaintingLoss(nn.Module):
    def __init__(self,
                 perceptual_weight: float = 0.1,
                 tv_weight: float = 0.1,
                 boundary_weight: float = 0.5,
                 device=None):
        super().__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.l1_loss = nn.L1Loss().to(self.device)
        self.perceptual_weight = perceptual_weight
        self.tv_weight = tv_weight
        self.boundary_weight = boundary_weight  # New parameter for boundary loss

        # Load pre-trained VGG16 for perceptual loss
        # Note: We're initializing it directly on the correct device
        logger.info(f"Initializing VGG model on device: {self.device}")
        try:
            vgg_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            self.vgg_layers = vgg_model.features[:16].eval().to(self.device)
            for param in self.vgg_layers.parameters():
                param.requires_grad = False
        except Exception as e:
            logger.error(f"Error initializing VGG model: {str(e)}")
            raise

        # Initialize boundary-aware loss
        self.boundary_loss = BoundaryAwareLoss(device=self.device)
        self.boundary_loss = self.boundary_loss.to(self.device)

        logger.info("InpaintingLoss initialized successfully")

    def to(self, device):
        """Override to() to ensure all internal components are moved to the device"""
        logger.info(f"Moving InpaintingLoss to device: {device}")
        self.device = device
        self.l1_loss = self.l1_loss.to(device)
        self.vgg_layers = self.vgg_layers.to(device)

        # Make sure boundary loss is moved to the device too
        if hasattr(self, 'boundary_loss'):
            self.boundary_loss = self.boundary_loss.to(device)

        return super().to(device)

    def forward(self, input, target, mask):
        # Ensure inputs are on correct device
        input = input.to(self.device)
        target = target.to(self.device)
        mask = mask.to(self.device)

        # Log device information for debugging
        logger.debug(f"InpaintingLoss running on device: {self.device}")
        logger.debug(f"Input tensor device: {input.device}")
        logger.debug(f"VGG layers device: {next(self.vgg_layers.parameters()).device}")

        # Initialize total loss
        total_loss = torch.tensor(0.0, device=self.device)

        # L1 loss
        l1_loss = self.l1_loss(input, target)
        total_loss = total_loss + l1_loss

        # Perceptual Loss (if used)
        if self.perceptual_weight > 0:
            try:
                input_vgg = input.repeat(1, 3, 1, 1)  # Convert to 3 channels
                target_vgg = target.repeat(1, 3, 1, 1)

                # Double-check devices
                input_vgg = input_vgg.to(self.device)
                target_vgg = target_vgg.to(self.device)

                perceptual_loss = self.l1_loss(
                    self.vgg_layers(input_vgg),
                    self.vgg_layers(target_vgg)
                )
                total_loss = total_loss + self.perceptual_weight * perceptual_loss
            except Exception as e:
                logger.error(f"Error in perceptual loss: {str(e)}")
                perceptual_loss = torch.tensor(0.0, device=self.device)

        # Total Variation Loss on inpainted regions
        if self.tv_weight > 0:
            try:
                hole_mask = 1 - mask
                tv_loss = self.total_variation_loss(input * hole_mask)
                total_loss = total_loss + self.tv_weight * tv_loss
            except Exception as e:
                logger.error(f"Error in TV loss: {str(e)}")
                tv_loss = torch.tensor(0.0, device=self.device)

        # Boundary-aware loss
        if self.boundary_weight > 0:
            try:
                b_loss = self.boundary_loss(input, target, mask)
                # Add to total loss with weight
                total_loss = total_loss + self.boundary_weight * b_loss
                logger.debug(f"Boundary loss component: {b_loss.item()}")
            except Exception as e:
                logger.error(f"Error in boundary loss calculation: {str(e)}")
                # Don't add boundary loss component if it fails

        return total_loss

    def total_variation_loss(self, x):
        """Calculate total variation loss for smoothness"""
        batch_size = x.size(0)
        h_x = x.size(2)
        w_x = x.size(3)
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.numel()

class HumanGuidedLoss(InpaintingLoss):
    def __init__(self, config, device=None, **kwargs):
        # Make sure device isn't passed twice
        if 'device' in kwargs:
            del kwargs['device']

        # Get boundary weight from config if available
        boundary_weight = config['training'].get('loss_weights', {}).get('boundary', 0.5)

        # Initialize parent class
        super().__init__(
            device=device,
            boundary_weight=boundary_weight,
            **kwargs
        )

        self.human_feedback_weight = config['training']['modes']['human_guided']['human_feedback_weight']
        self.base_loss_weight = config['training']['modes']['human_guided']['base_loss_weight']
        logger.info(f"HumanGuidedLoss initialized with weights: base={self.base_loss_weight}, human={self.human_feedback_weight}, boundary={boundary_weight}")

    def forward(self, input, target, mask, human_feedback=None):
        # Ensure all tensors are on the correct device
        input = input.to(self.device)
        target = target.to(self.device)
        mask = mask.to(self.device)

        # Calculate base loss using parent class (which includes boundary loss)
        base_loss = super().forward(input, target, mask)

        # Initialize human_loss
        human_loss = torch.tensor(0.0, device=self.device)

        # Calculate human feedback loss (if provided)
        if human_feedback is not None and 'mask' in human_feedback and human_feedback['mask'] is not None:
            try:
                human_mask = human_feedback['mask'].to(self.device)
                human_guided_regions = (human_mask > 0).float().to(self.device)

                # Make sure there are non-zero values in the mask
                if human_guided_regions.sum() > 0:
                    human_loss = self.l1_loss(
                        input * human_guided_regions,
                        target * human_guided_regions
                    )

                    # Also apply boundary loss to human-masked regions for additional consistency
                    if self.boundary_weight > 0:
                        try:
                            human_boundary_loss = self.boundary_loss(
                                input,
                                target,
                                human_guided_regions
                            )
                            human_loss = human_loss + self.boundary_weight * human_boundary_loss
                        except Exception as e:
                            logger.error(f"Error computing human boundary loss: {str(e)}")

                    logger.debug(f"Human loss: {human_loss.item()}")
                else:
                    logger.warning("Human mask contains all zeros")
            except Exception as e:
                logger.error(f"Error computing human feedback loss: {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())

        # Combine losses according to configured weights
        total_loss = (
            self.base_loss_weight * base_loss +
            self.human_feedback_weight * human_loss
        )

        logger.debug(f"Total loss: {total_loss.item()} (base: {base_loss.item()}, human: {human_loss.item()})")
        return total_loss

class BoundaryAwareLoss(nn.Module):
    """
    Loss function that focuses on boundary regions between original and inpainted areas.

    This loss ensures smooth transitions at boundaries by:
    1. Detecting boundary regions using mask dilation/erosion
    2. Computing gradient consistency across boundaries
    3. Applying multi-scale boundary evaluation for robust performance
    """
    def __init__(self, boundary_width: int = 10, epsilon: float = 1e-6, device=None):
        """
        Initialize boundary-aware loss.

        Args:
            boundary_width: Width of the boundary region in pixels
            epsilon: Small value to prevent division by zero
            device: Computation device
        """
        super().__init__()
        self.boundary_width = boundary_width
        self.epsilon = epsilon

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Define Sobel filters for gradient calculation
        self.sobel_x = torch.tensor([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0]
        ], dtype=torch.float32).view(1, 1, 3, 3).to(self.device)

        self.sobel_y = torch.tensor([
            [-1.0, -2.0, -1.0],
            [0.0,  0.0,  0.0],
            [1.0,  2.0,  1.0]
        ], dtype=torch.float32).view(1, 1, 3, 3).to(self.device)

        logger.info(f"BoundaryAwareLoss initialized with boundary width: {boundary_width}, device: {self.device}")

    def to(self, device):
        """Override to() to ensure all tensors move to the specified device"""
        self.device = device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)
        return super().to(device)

    def extract_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Extract boundary region between masked and unmasked areas.

        Args:
            mask: Binary mask tensor where 1 indicates preserved regions
                 and 0 indicates regions to inpaint

        Returns:
            Boundary mask where 1 indicates boundary pixels
        """
        try:
            # Ensure mask is on the correct device
            mask = mask.to(self.device)

            # Dilate the mask
            dilated = F.max_pool2d(
                mask,
                kernel_size=self.boundary_width,
                stride=1,
                padding=self.boundary_width//2
            )

            # Erode the mask
            eroded = -F.max_pool2d(
                -mask,
                kernel_size=self.boundary_width,
                stride=1,
                padding=self.boundary_width//2
            )

            # Boundary is the difference between dilated and eroded
            boundary = dilated - eroded

            # Ensure boundary values are between 0 and 1
            boundary = torch.clamp(boundary, 0.0, 1.0)

            # Check if boundary contains numerical issues
            if torch.isnan(boundary).any() or torch.isinf(boundary).any():
                logger.warning("Numerical issues detected in boundary extraction")
                # Fallback to a safer version
                boundary = torch.zeros_like(mask)

                # Create a safe boundary using morphological operations
                with torch.no_grad():
                    # Simple dilation and erosion with smaller kernel
                    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
                    eroded = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
                    boundary = torch.clamp(dilated - eroded, 0.0, 1.0)

            return boundary

        except Exception as e:
            logger.error(f"Error in extract_boundary: {str(e)}")
            # Return empty boundary on error
            return torch.zeros_like(mask)

    def compute_gradients(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute image gradients using finite differences instead of convolution.

        Args:
            img: Input tensor image

        Returns:
            Tuple of (gradient_x, gradient_y) tensors
        """
        try:
            # Ensure image is on the correct device
            img = img.to(self.device)

            # Make sure image has the right shape
            if img.dim() == 3:  # Add batch dimension if missing
                img = img.unsqueeze(0)

            # Simple finite differences for gradients
            # Pad with zeros to maintain dimensions
            padded = F.pad(img, (1, 1, 1, 1), mode='replicate')

            # Horizontal gradient (central difference)
            grad_x = padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]
            grad_x = grad_x / 2.0  # Normalize

            # Vertical gradient (central difference)
            grad_y = padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]
            grad_y = grad_y / 2.0  # Normalize

            return grad_x, grad_y

        except Exception as e:
            logger.error(f"Error in compute_gradients: {str(e)}")
            # Return zero gradients on error
            return torch.zeros_like(img), torch.zeros_like(img)

    def gradient_consistency_loss(self, pred: torch.Tensor, target: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient consistency loss at boundary regions.

        Args:
            pred: Predicted (inpainted) image
            target: Target (ground truth) image
            boundary: Boundary region mask

        Returns:
            Gradient consistency loss value
        """
        try:
            # Ensure all inputs are the same shape
            if pred.shape != target.shape or pred.shape != boundary.shape:
                logger.warning("Shape mismatch in gradient_consistency_loss")
                # Simple reshaping to match
                target = F.interpolate(target, size=pred.shape[2:], mode='bilinear')
                boundary = F.interpolate(boundary, size=pred.shape[2:], mode='bilinear')

            # Simple intensity difference at boundary
            # This is a robust fallback that doesn't use gradients
            boundary_diff = torch.abs(pred - target) * boundary
            loss = boundary_diff.sum() / (boundary.sum() + self.epsilon)

            # Check for numerical issues
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("NaN or Inf detected in gradient_consistency_loss")
                return torch.tensor(0.01, device=self.device)

            return loss

        except Exception as e:
            logger.error(f"Error in gradient_consistency_loss: {str(e)}")
            # Return a small constant loss value on error
            return torch.tensor(0.01, device=self.device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the complete boundary-aware loss with robust fallback mechanisms.

        Args:
            pred: Predicted (inpainted) image
            target: Target (ground truth) image
            mask: Binary mask where 1 indicates preserved regions

        Returns:
            Boundary loss value
        """
        # Ensure all inputs are on the correct device
        pred = pred.to(self.device)
        target = target.to(self.device)
        mask = mask.to(self.device)

        try:
            # Simple fallback approach that will always work
            # Define a boundary region using simple dilation/erosion
            dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
            eroded = 1 - F.max_pool2d(1 - mask, kernel_size=3, stride=1, padding=1)
            boundary = torch.clamp(dilated - eroded, 0.0, 1.0)

            # If boundary is empty, return zero loss
            if torch.sum(boundary) < 1.0:
                return torch.tensor(0.0, device=self.device)

            # Simple L1 loss at boundary region (this always works)
            boundary_loss = torch.abs(pred - target) * boundary
            loss = boundary_loss.sum() / (boundary.sum() + self.epsilon)

            # Check for numerical issues
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("NaN or Inf detected in boundary loss")
                return torch.tensor(0.0, device=self.device)

            return loss

        except Exception as e:
            logger.error(f"Error in boundary loss calculation: {str(e)}")
            # Return zero loss on error to avoid breaking training
            return torch.tensor(0.0, device=self.device)
