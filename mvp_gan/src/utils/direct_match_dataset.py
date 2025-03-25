import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

class DirectMatchDataset(Dataset):
    """Dataset that loads directly from matched file paths"""

    def __init__(self, matched_pairs, transform=None):
        self.matched_pairs = matched_pairs

        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        logger.info(f"Initialized DirectMatchDataset with {len(matched_pairs)} samples")

    def __len__(self):
        return len(self.matched_pairs)

    def __getitem__(self, idx):
        try:
            pair = self.matched_pairs[idx]

            # Load all images directly
            image = Image.open(pair['image_path']).convert('L')
            system_mask = Image.open(pair['system_mask_path']).convert('L')
            human_mask = Image.open(pair['human_mask_path']).convert('L')

            # CRITICAL: Ensure all images are resized to the expected format (512x512)
            image = image.resize((512, 512), Image.BILINEAR)
            system_mask = system_mask.resize((512, 512), Image.NEAREST)
            human_mask = human_mask.resize((512, 512), Image.NEAREST)

            # Apply transformations
            image_tensor = self.transform(image)
            system_mask_tensor = self.transform(system_mask)
            human_mask_tensor = self.transform(human_mask)

            # Ensure masks are binary (0 or 1)
            system_mask_tensor = (system_mask_tensor > 0.5).float()
            human_mask_tensor = (human_mask_tensor > 0.5).float()

            # Validate tensor dimensions (good practice for debugging)
            expected_shape = (1, 512, 512)
            if (image_tensor.shape != expected_shape or
                system_mask_tensor.shape != expected_shape or
                human_mask_tensor.shape != expected_shape):
                logger.warning(f"Shape mismatch: image={image_tensor.shape}, "
                            f"system_mask={system_mask_tensor.shape}, "
                            f"human_mask={human_mask_tensor.shape}")

            # Add validation for zero masks
            if human_mask_tensor.sum() == 0:
                logger.warning(f"Human mask for tile {pair['tile_name']} contains all zeros")

            return {
                'image': image_tensor,
                'mask': system_mask_tensor,
                'human_mask': human_mask_tensor,
                'tile_name': pair['tile_name']
            }

        except Exception as e:
            logger.error(f"Error loading item at index {idx}: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            # Return fallback tensors
            empty_tensor = torch.zeros((1, 512, 512), dtype=torch.float32)
            return {
                'image': empty_tensor,
                'mask': empty_tensor,
                'human_mask': empty_tensor,
                'tile_name': "error"
            }
