"""
Human-guided training module for DSM inpainting models.

This module contains the HumanGuidedTrainer class which implements fine-tuning
of pre-trained models using human-annotated masks.
"""

import time
import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict
from torch.utils.data import DataLoader

from ..evaluation.metrics import MaskEvaluator
from ..utils.losses import HumanGuidedLoss

logger = logging.getLogger(__name__)

class HumanGuidedTrainer:
    """
    Trainer class for human-guided fine-tuning of DSM inpainting models.

    This class handles the training process using human annotations to fine-tune
    generative models for terrain inpainting.
    """

    def __init__(self, config: Dict, experiment_tracker=None):
        """
        Initialize the human-guided trainer.

        Args:
            config: Configuration dictionary containing training parameters
            experiment_tracker: Optional experiment tracking instance for metrics logging
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics_evaluator = MaskEvaluator(config)
        self.experiment_tracker = experiment_tracker

        logger.info(f"Initialized HumanGuidedTrainer with device: {self.device}")

    def train(self, generator: nn.Module, train_dataset, num_epochs: int, checkpoint_dir: Path):
        """
        Train the model with human guidance.

        Args:
            generator: Generator model to train
            train_dataset: Dataset with human annotations
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
        """
        try:
            # if self.experiment_tracker is not None:
            #     if not hasattr(self.experiment_tracker, 'run') or self.experiment_tracker.run is None:
            #         logger.warning("No active MLflow run found. Training metrics may not be logged.")

            # Ensure model is on the correct device
            generator = generator.to(self.device)
            logger.info(f"Model moved to device: {self.device}")

            # Initialize loss function and optimizer - EXPLICITLY with device
            criterion = HumanGuidedLoss(self.config, device=self.device)
            logger.info(f"Criterion initialized on device: {self.device}")
            logger.info(f"VGG model device: {next(criterion.vgg_layers.parameters()).device}")

            optimizer = torch.optim.Adam(
                generator.parameters(),
                lr=self.config['training']['modes']['human_guided']['learning_rate']
            )

            # Create data loader with appropriate settings
            pin_memory = self.device.type == 'cuda'  # Only pin memory if using CUDA
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['training']['modes']['human_guided']['batch_size'],
                shuffle=True,
                num_workers=0,
                pin_memory=False  # Always set to False to avoid pin memory errors
            )

            logger.info(f"Created DataLoader with {len(train_dataset)} samples")

            best_loss = float('inf')
            start_time = time.time()

            for epoch in range(num_epochs):
                generator.train()
                epoch_loss = 0.0
                epoch_start = time.time()
                batch_count = 0
                success_count = 0

                # In mvp_gan/src/training/human_guided_trainer.py

                # This would be inside the train method of the HumanGuidedTrainer class
                for batch_idx, batch in enumerate(train_loader):
                    try:
                        # Move input tensors to device
                        images = batch['image'].to(self.device)
                        masks = batch['mask'].to(self.device)
                        human_masks = batch.get('human_mask')

                        logger.debug(f"Batch {batch_idx} - Images device: {images.device}, Masks device: {masks.device}")

                        if human_masks is not None:
                            human_masks = human_masks.to(self.device)
                            logger.debug(f"Human masks device: {human_masks.device}")

                        # Apply mask to input images
                        masked_images = images * masks

                        # Generate output
                        generated = generator(masked_images, masks)
                        logger.debug(f"Generated output device: {generated.device}")

                        # Calculate loss
                        # Insert the new code block here (starts with the line below)
                        human_feedback = {'mask': human_masks} if human_masks is not None else None
                        loss = criterion(generated, images, masks, human_feedback)

                        # Log boundary components if applicable
                        if self.experiment_tracker is not None and batch_idx % 10 == 0:
                            try:
                                boundary_weight = self.config['training']['loss_weights'].get('boundary', 0.0)
                                if boundary_weight > 0:
                                    from ..evaluation.metrics import calculate_boundary_quality
                                    boundary_metrics = calculate_boundary_quality(
                                        generated.detach(),
                                        images.detach(),
                                        masks.detach()
                                    )

                                # Add these to the logged metrics
                                self.experiment_tracker.log_metrics({
                                    f'batch.boundary_{k}': v for k, v in boundary_metrics.items()
                                }, step=epoch * len(train_loader) + batch_idx)
                            except Exception as e:
                                logger.warning(f"Could not log boundary metrics: {e}")
                        # End of new code block

                        logger.debug(f"Loss value: {loss.item()}, device: {loss.device}")

                        # Only add to epoch loss if valid
                        if not torch.isnan(loss) and not torch.isinf(loss):
                            epoch_loss += loss.item()
                            success_count += 1

                        # Update generator
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        # Log batch metrics
                        if self.experiment_tracker is not None and batch_idx % self.config['training'].get('log_interval', 10) == 0:
                            step = epoch * len(train_loader) + batch_idx
                            self.experiment_tracker.log_training_batch(
                                pred=generated.detach(),
                                target=images.detach(),
                                model=generator,
                                optimizer=optimizer,
                                batch_metrics={'loss': float(loss.item())},
                                step=step
                            )

                        logger.debug(f"Processed batch {batch_idx}: loss={loss.item():.6f}")
                        batch_count += 1

                    except Exception as e:
                        logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                        import traceback
                        logger.debug(traceback.format_exc())
                        continue

                # Calculate epoch metrics
                avg_epoch_loss = epoch_loss / max(1, success_count) if epoch_loss > 0 else 0.0
                epoch_time = time.time() - epoch_start

                # Log epoch metrics
                if self.experiment_tracker is not None:
                    self.experiment_tracker.log_metrics({
                        'epoch.loss': float(avg_epoch_loss),
                        'epoch.time': float(epoch_time),
                        'epoch.success_rate': float(success_count / max(1, batch_count))
                    }, step=epoch)

                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': generator.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': float(avg_epoch_loss),
                    'config': self.config
                }

                try:
                    # Save regular checkpoint
                    checkpoint_path = checkpoint_dir / f'generator_epoch_{epoch}.pth'
                    torch.save(checkpoint, checkpoint_path)
                    logger.debug(f"Saved checkpoint to {checkpoint_path}")

                    # Save best model if current loss is best
                    if avg_epoch_loss < best_loss and avg_epoch_loss > 0:
                        best_loss = avg_epoch_loss
                        best_model_path = checkpoint_dir / 'best_model.pth'
                        torch.save(checkpoint, best_model_path)
                        logger.info(f"New best model saved with loss: {best_loss:.6f}")

                        if self.experiment_tracker is not None:
                            try:
                                # Move to CPU for logging
                                generator.cpu()
                                self.experiment_tracker.log_model(
                                    generator,
                                    "best_human_guided_model",
                                    metrics={'loss': float(best_loss)}
                                )
                                # Move back to device
                                generator.to(self.device)
                            except Exception as e:
                                logger.error(f"Failed to log best model: {str(e)}")
                                generator.to(self.device)  # Ensure model returns to device

                except Exception as e:
                    logger.error(f"Failed to save checkpoint: {str(e)}")

                logger.info(
                    f"Epoch {epoch}: loss={avg_epoch_loss:.6f}, "
                    f"success_rate={success_count}/{batch_count}, "
                    f"time={epoch_time:.2f}s"
                )

            # End training
            total_time = time.time() - start_time
            logger.info(f"Human-guided training completed in {total_time:.2f}s")

            if self.experiment_tracker is not None:
                self.experiment_tracker.log_metrics({
                    'training.total_time': float(total_time),
                    'training.best_loss': float(best_loss)
                })

            return {
                'best_loss': best_loss,
                'total_time': total_time,
                'final_epoch': epoch,
                'success': True
            }

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            if self.experiment_tracker is not None:
                self.experiment_tracker.end_run()
            return {
                'best_loss': float('inf'),  # Or some other default value
                'total_time': 0,  # Or a calculated partial time
                'final_epoch': 0, # or the current epoch
                'success': False # Return False in case of failure
            }
