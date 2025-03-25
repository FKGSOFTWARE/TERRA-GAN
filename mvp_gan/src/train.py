# src/train.py

from typing import Dict, Optional
import torch
import time
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import numpy as np

from mvp_gan import ExperimentTracker
from .models.generator import PConvUNet
from .models.discriminator import Discriminator
from .utils.dataset import InpaintingDataset
from .utils.losses import InpaintingLoss, HumanGuidedLoss
from .evaluation.metrics import MaskEvaluator

logger = logging.getLogger(__name__)

def train(img_dir: Path,
         mask_dir: Path,
         generator: Optional[PConvUNet] = None,
         discriminator: Optional[Discriminator] = None,
         optimizer_G: Optional[torch.optim.Optimizer] = None,
         optimizer_D: Optional[torch.optim.Optimizer] = None,
         checkpoint_path: Optional[Path] = None,
         config: Optional[Dict] = None,
         experiment_tracker = None,
         val_img_dir: Optional[Path] = None,
         val_mask_dir: Optional[Path] = None):
    """
    Train the GAN model with validation-based model selection and experiment tracking.

    Args:
        img_dir: Directory containing training images
        mask_dir: Directory containing training masks
        generator: Optional pre-initialized generator model
        discriminator: Optional pre-initialized discriminator model
        optimizer_G: Optional pre-initialized generator optimizer
        optimizer_D: Optional pre-initialized discriminator optimizer
        checkpoint_path: Path to save/load model checkpoints
        config: Optional configuration dictionary
        experiment_tracker: Optional experiment tracker
        val_img_dir: Optional directory containing validation images
        val_mask_dir: Optional directory containing validation masks
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use default config if none provided
    if config is None:
        config = {
            'training': {
                'batch_size': 2,
                'learning_rate': 2e-4,
                'epochs': 10,
                'loss_weights': {
                    'perceptual': 0.1,
                    'tv': 0.1,
                }
            }
        }

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    # Setup training dataset
    try:
        train_dataset = InpaintingDataset(img_dir=img_dir, mask_dir=mask_dir, transform=transform)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training'].get('batch_size', 2),
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=True if device.type == 'cuda' else False
        )
    except Exception as e:
        logger.error(f"Failed to initialize training dataset: {str(e)}")
        raise

    # Setup validation dataset if provided
    val_loader = None
    if val_img_dir and val_mask_dir:
        try:
            val_dataset = InpaintingDataset(img_dir=val_img_dir, mask_dir=val_mask_dir, transform=transform)
            val_loader = DataLoader(
                val_dataset,
                batch_size=config['training'].get('batch_size', 2),
                shuffle=False,  # No need to shuffle validation
                num_workers=0,
                pin_memory=True if device.type == 'cuda' else False
            )
            logger.info("Validation dataset initialized")
        except Exception as e:
            logger.error(f"Failed to initialize validation dataset: {str(e)}")
            val_loader = None

    # Model setup - now accept pre-initialized models
    if generator is None:
        generator = PConvUNet().to(device)
    if discriminator is None:
        discriminator = Discriminator().to(device)

        # Initialize losses
    criterion = InpaintingLoss(
        perceptual_weight=config['training']['loss_weights']['perceptual'],
        tv_weight=config['training']['loss_weights']['tv'],
        device=device
    )
    adversarial_loss = torch.nn.BCEWithLogitsLoss()

    # Load existing checkpoint if available and models not provided.
    # This has been updated so loading from a checkpoint is ONLY if a model
    # is not already provided to train().
    if checkpoint_path and checkpoint_path.exists() and generator is None:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
                generator.load_state_dict(checkpoint['generator_state_dict'])
                if 'discriminator_state_dict' in checkpoint: # Handle case if D is saved
                    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                logger.info(f"Loaded checkpoint from {checkpoint_path}")
            else:
                generator.load_state_dict(checkpoint)
                logger.info(f"Loaded generator-only checkpoint from {checkpoint_path}")

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            raise


    # Optimizers - now accept pre-initialized optimizers
    if optimizer_G is None:
        optimizer_G = torch.optim.Adam(
            generator.parameters(),
            lr=config['training'].get('learning_rate', 2e-4)
        )
    if optimizer_D is None:
        optimizer_D = torch.optim.Adam(
            discriminator.parameters(),
            lr=config['training'].get('learning_rate', 2e-4)
        )

    # Log model architectures if tracking enabled
    if experiment_tracker is not None:
        experiment_tracker._log_model_architecture(generator)
        # experiment_tracker._log_model_architecture(discriminator) # Don't log D


    best_val_loss = float('inf')
    best_train_loss = float('inf')
    start_time = time.time()

    for epoch in range(config['training'].get('epochs', 10)):
        # Training phase
        generator.train()
        discriminator.train()
        epoch_metrics = {
            'g_loss': 0.0,
            'd_loss': 0.0,
            'real_loss': 0.0,
            'fake_loss': 0.0,
        }

        # Only add boundary_loss key if boundary weight > 0
        boundary_weight = config['training']['loss_weights'].get('boundary', 0.0)
        if boundary_weight > 0:
            epoch_metrics['boundary_loss'] = 0.0

        epoch_start = time.time()

        for batch_idx, data in enumerate(train_loader):
            try:
                real_imgs = data['image'].to(device)
                masks = data['mask'].to(device)
                masked_imgs = real_imgs * masks

                # Train Generator
                optimizer_G.zero_grad()
                gen_imgs = generator(masked_imgs, masks)

                # Calculate generator loss - MODIFY THIS PART
                g_loss = criterion(gen_imgs, real_imgs, masks)

                # Extract boundary loss for logging (if available)
                boundary_loss = 0.0
                boundary_weight = config['training']['loss_weights'].get('boundary', 0.0)
                if boundary_weight > 0 and hasattr(criterion, 'boundary_loss') and criterion.boundary_weight > 0:
                    try:
                        boundary_loss = criterion.boundary_loss(gen_imgs, real_imgs, masks).item()
                        # Also add to epoch metrics to track average
                        epoch_metrics['boundary_loss'] += boundary_loss
                    except Exception as e:
                        logger.debug(f"Could not extract boundary loss: {e}")

                # Add adversarial loss component
                fake_validity = discriminator(gen_imgs)
                g_adv_loss = adversarial_loss(fake_validity, torch.ones_like(fake_validity, device=device))
                g_total_loss = g_loss + g_adv_loss

                g_total_loss.backward()
                optimizer_G.step()

                # Train Discriminator
                optimizer_D.zero_grad()
                real_validity = discriminator(real_imgs)
                fake_validity = discriminator(gen_imgs.detach())
                real_labels = torch.ones_like(real_validity, device=device)
                fake_labels = torch.zeros_like(fake_validity, device=device)
                real_loss = adversarial_loss(real_validity, real_labels)
                fake_loss = adversarial_loss(fake_validity, fake_labels)
                d_loss = 0.5 * (real_loss + fake_loss)
                d_loss.backward()
                optimizer_D.step()

                # Update metrics
                epoch_metrics['g_loss'] += g_total_loss.item()
                epoch_metrics['d_loss'] += d_loss.item()
                epoch_metrics['real_loss'] += real_loss.item()
                epoch_metrics['fake_loss'] += fake_loss.item()

                # Log batch metrics
                # Inside the training batch loop, add boundary metrics logging
                if experiment_tracker is not None and batch_idx % config['training'].get('log_interval', 10) == 0:
                    step = epoch * len(train_loader) + batch_idx

                    # Calculate boundary metrics if available
                    boundary_metrics = {}
                    try:
                        boundary_weight = config['training']['loss_weights'].get('boundary', 0.0)
                        if boundary_weight > 0:
                            from .evaluation.metrics import calculate_boundary_quality
                            boundary_metrics = calculate_boundary_quality(
                                gen_imgs.detach(),
                                real_imgs.detach(),
                                masks.detach()
                            )
                    except Exception as e:
                        logger.warning(f"Could not calculate boundary metrics: {e}")

                    batch_metrics = {
                        'g_loss': float(g_total_loss.item()),
                        'd_loss': float(d_loss.item()),
                        'real_loss': float(real_loss.item()),
                        'fake_loss': float(fake_loss.item()),
                    }

                    # Only add boundary metrics if boundary weight > 0
                    if boundary_weight > 0:
                        batch_metrics.update(boundary_metrics)
                        if 'boundary_loss' in locals() and boundary_loss > 0:
                            batch_metrics['boundary_loss'] = boundary_loss

                    experiment_tracker.log_training_batch(
                        pred=gen_imgs.detach(),
                        target=real_imgs.detach(),
                        model=generator,
                        optimizer=optimizer_G,
                        batch_metrics=batch_metrics,
                        step=step
                    )

            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                continue

        # Calculate epoch metrics
        num_batches = len(train_loader)
        epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        epoch_metrics['epoch_time'] = time.time() - epoch_start

        # Validation phase
        if val_loader is not None:
            generator.eval()
            val_g_loss = 0.0
            val_d_loss = 0.0

            with torch.no_grad():
                for val_batch in val_loader:
                    val_real_imgs = val_batch['image'].to(device)
                    val_masks = val_batch['mask'].to(device)
                    val_masked_imgs = val_real_imgs * val_masks

                    # Generate images
                    val_gen_imgs = generator(val_masked_imgs, val_masks)

                    # Calculate generator loss
                    val_g_total = criterion(val_gen_imgs, val_real_imgs, val_masks)
                    val_g_loss += val_g_total.item()

                    # Calculate discriminator loss for monitoring
                    val_real_validity = discriminator(val_real_imgs)
                    val_fake_validity = discriminator(val_gen_imgs)
                    val_d_real = adversarial_loss(val_real_validity, torch.ones_like(val_real_validity, device=device))
                    val_d_fake = adversarial_loss(val_fake_validity, torch.zeros_like(val_fake_validity, device=device))
                    val_d_loss += 0.5 * (val_d_real.item() + val_d_fake.item())

            val_g_loss /= len(val_loader)
            val_d_loss /= len(val_loader)

            # Log validation metrics
            if experiment_tracker is not None:
                val_metrics = {
                    'validation.g_loss': float(val_g_loss),
                    'validation.d_loss': float(val_d_loss)
                }
                experiment_tracker.log_metrics(val_metrics, step=epoch)

            # Save best model based on validation loss
            if val_g_loss < best_val_loss:
                best_val_loss = val_g_loss
                try:
                    checkpoint = {
                        'epoch': epoch,
                        'generator_state_dict': generator.state_dict(),
                        'discriminator_state_dict': discriminator.state_dict(), # Keep for now
                        'optimizer_G_state_dict': optimizer_G.state_dict(),
                        'optimizer_D_state_dict': optimizer_D.state_dict(),
                        'g_loss': float(epoch_metrics['g_loss']),
                        'd_loss': float(epoch_metrics['d_loss']),
                        'val_g_loss': float(val_g_loss),
                        'val_d_loss': float(val_d_loss),
                        'config': config
                    }
                    torch.save(checkpoint, checkpoint_path)

                    if experiment_tracker is not None:
                        scalar_metrics = {
                            'g_loss': float(epoch_metrics['g_loss']),
                            'd_loss': float(epoch_metrics['d_loss']),
                            'val_g_loss': float(val_g_loss),
                            'val_d_loss': float(val_d_loss),
                            'best_val_loss': float(best_val_loss),
                            'epoch': int(epoch)
                        }
                        generator.cpu()
                        if experiment_tracker is not None:
                            try:
                                # Move to CPU for logging
                                generator.cpu()

                                # Create a proper input example (single ndarray, not tuple)
                                input_example = np.zeros((1, 1, 512, 512), dtype=np.float32)

                                experiment_tracker.log_model(
                                    generator,
                                    "best_model_validation",
                                    metrics=scalar_metrics,
                                    input_example=input_example  # Pass this explicitly
                                )
                                # Move back to device
                                generator.to(device)
                            except Exception as e:
                                logger.error(f"Failed to log best model: {str(e)}")
                                generator.to(device)  # Ensure model returns to device

                    logger.info(f"Saved new best model with validation loss: {val_g_loss:.4f}")
                except Exception as e:
                    logger.error(f"Failed to save best model: {str(e)}")
        else:
            # If no validation set, fall back to training loss for model selection
            if epoch_metrics['g_loss'] < best_train_loss:
                best_train_loss = epoch_metrics['g_loss']
                try:
                    checkpoint = {
                        'epoch': epoch,
                        'generator_state_dict': generator.state_dict(),
                        'discriminator_state_dict': discriminator.state_dict(),  #Keep D
                        'optimizer_G_state_dict': optimizer_G.state_dict(),
                        'optimizer_D_state_dict': optimizer_D.state_dict(),
                        'g_loss': float(epoch_metrics['g_loss']),
                        'd_loss': float(epoch_metrics['d_loss']),
                        'config': config
                    }
                    torch.save(checkpoint, checkpoint_path)

                    if experiment_tracker is not None:
                        scalar_metrics = {
                            'g_loss': float(epoch_metrics['g_loss']),
                            'd_loss': float(epoch_metrics['d_loss']),
                            'best_loss': float(best_train_loss),
                            'epoch': int(epoch)
                        }
                        generator.cpu()
                        experiment_tracker.log_model(
                            generator,
                            "best_model_train",
                            metrics=scalar_metrics
                        )
                        generator.to(device)
                except Exception as e:
                    logger.error(f"Failed to save best model: {str(e)}")

        # Regular checkpoint save
        if epoch % config['training'].get('checkpoint_interval', 5) == 0:
            try:
                checkpoint_epoch_path = checkpoint_path.parent / f'checkpoint_epoch_{epoch}.pth'
                torch.save(checkpoint, checkpoint_epoch_path)
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {str(e)}")

        # Log epoch metrics
        if experiment_tracker is not None:
            clean_metrics = {
                'epoch.g_loss': float(epoch_metrics['g_loss']),
                'epoch.d_loss': float(epoch_metrics['d_loss']),
                'epoch.real_loss': float(epoch_metrics['real_loss']),
                'epoch.fake_loss': float(epoch_metrics['fake_loss']),
                'epoch.time': float(epoch_metrics['epoch_time'])
            }
            experiment_tracker.log_metrics(clean_metrics, step=epoch)

        # Log progress
        log_message = f"Epoch {epoch}: g_loss={epoch_metrics['g_loss']:.4f}, d_loss={epoch_metrics['d_loss']:.4f}"

        # Add boundary loss if it's enabled in the config
        boundary_weight = config['training']['loss_weights'].get('boundary', 0.0)
        if boundary_weight > 0 and 'boundary_loss' in epoch_metrics:
            log_message += f", boundary_loss={epoch_metrics['boundary_loss']:.4f}"

        if val_loader is not None:
            log_message += f", val_g_loss={val_g_loss:.4f}, val_d_loss={val_d_loss:.4f}"

        log_message += f", time={epoch_metrics['epoch_time']:.2f}s"
        logger.info(log_message)

    # End training
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f}s")

    # Log final metrics
    if experiment_tracker is not None:
        final_metrics = {
            'training.total_time': float(total_time),
            'training.best_train_loss': float(best_train_loss)
        }
        if val_loader is not None:
            final_metrics['training.best_val_loss'] = float(best_val_loss)
            final_metrics['training.validation_improvement'] = float(best_val_loss - val_g_loss)
        experiment_tracker.log_metrics(final_metrics)
        # experiment_tracker.end_run() # No longer needed

    return {
        'best_train_loss': best_train_loss,
        'best_val_loss': best_val_loss if val_loader is not None else None,
        'total_time': total_time,
        'final_epoch': epoch
    }
