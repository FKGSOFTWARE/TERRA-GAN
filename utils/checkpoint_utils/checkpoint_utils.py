import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

def validate_checkpoint(path: Path) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Validate a checkpoint file and return its contents if valid.

    Args:
        path: Path to checkpoint file

    Returns:
        Tuple of (is_valid, checkpoint_dict)
    """
    try:
        if not path.exists():
            logger.error(f"Checkpoint file not found: {path}")
            return False, None

        checkpoint = torch.load(path, map_location='cpu')

        # Validate checkpoint structure
        required_keys = {'epoch', 'generator_state_dict', 'optimizer_G_state_dict'}
        if isinstance(checkpoint, dict):
            missing_keys = required_keys - set(checkpoint.keys())
            if missing_keys:
                logger.error(f"Checkpoint missing required keys: {missing_keys}")
                return False, None

            # Additional validation could be added here
            return True, checkpoint
        else:
            # Handle legacy format (just state dict)
            logger.warning("Legacy checkpoint format detected")
            return True, {'generator_state_dict': checkpoint}

    except Exception as e:
        logger.error(f"Failed to validate checkpoint: {str(e)}")
        return False, None

def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> bool:
    """
    Safely load a checkpoint into model and optimizer.

    Args:
        path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into

    Returns:
        bool: Whether loading was successful
    """
    is_valid, checkpoint = validate_checkpoint(path)
    if not is_valid:
        return False

    try:
        if 'generator_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['generator_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        if optimizer is not None and 'optimizer_G_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_G_state_dict'])

        return True
    except Exception as e:
        logger.error(f"Failed to load checkpoint contents: {str(e)}")
        return False

def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
                   epoch: Optional[int] = None, **kwargs) -> bool:
    """
    Safely save a checkpoint with model and optimizer state.

    Args:
        path: Path to save checkpoint to
        model: Model to save
        optimizer: Optional optimizer to save
        epoch: Optional epoch number
        **kwargs: Additional items to save in checkpoint

    Returns:
        bool: Whether saving was successful
    """
    try:
        checkpoint = {
            'generator_state_dict': model.state_dict(),
            'epoch': epoch if epoch is not None else 0,
            **kwargs
        }

        if optimizer is not None:
            checkpoint['optimizer_G_state_dict'] = optimizer.state_dict()

        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save atomically using temporary file
        temp_path = path.with_suffix('.tmp')
        torch.save(checkpoint, temp_path)
        temp_path.replace(path)

        return True
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {str(e)}")
        if temp_path.exists():
            temp_path.unlink()
        return False
