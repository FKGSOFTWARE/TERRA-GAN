import cv2
import torch
from pathlib import Path
import logging
import shutil
from torchvision import transforms
from PIL import Image
import numpy as np
import json

logger = logging.getLogger(__name__)

class AnnotationValidator:
    """
    Validates and filters annotations for consistent sizes and binarizes masks.
    """
    def __init__(self,
                 target_size=(512, 512),
                 max_size_difference_percent=10,
                 resize_mode='strict'):
        """
        Initialize the annotation validator.

        Args:
            target_size: Tuple (height, width) of the expected image size
            max_size_difference_percent: Maximum allowed percentage difference
            resize_mode: 'strict' to skip mismatched images, 'resize' to force resize all
        """
        self.target_size = target_size
        self.max_size_difference = max_size_difference_percent / 100
        self.resize_mode = resize_mode
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])

    def check_image_size(self, image_path):
        """
        Check if an image fits within the allowed size range.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple (is_valid, dimensions) - Boolean validity and actual dimensions
        """
        try:
            img = Image.open(image_path)
            width, height = img.size

            # Check if dimensions are exactly the target size
            if (height, width) == self.target_size:
                return True, (height, width)

            # Check if dimensions are within the allowed difference
            h_target, w_target = self.target_size
            h_diff = abs(height - h_target) / h_target
            w_diff = abs(width - w_target) / w_target

            is_valid = h_diff <= self.max_size_difference and w_diff <= self.max_size_difference
            return is_valid, (height, width)

        except Exception as e:
            logger.error(f"Error checking image size for {image_path}: {str(e)}")
            return False, None

    def validate_and_filter_pairs(self, human_masks, system_masks, output_dir):
        """
        Validate annotation pairs, binarize masks, and copy to output directory.
        """
        # Create output directories
        img_dir = Path(output_dir) / "images"
        mask_dir = Path(output_dir) / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            "total_pairs": 0,
            "valid_pairs": 0,
            "invalid_human": 0,
            "invalid_system": 0,
            "resized_pairs": 0,
            "skipped_pairs": 0,
            "size_mismatches": [],
            "non_binary_human": 0,  # NEW: Count non-binary human masks
            "non_binary_system": 0, # NEW: Count non-binary system masks
            "file_mapping": {}      # NEW: Map validation indices to original filenames
        }

        # Create mapping from filenames to human masks
        human_mask_map = {}
        for mask_file in human_masks:
            # Extract the identifier part (e.g., "nj0957")
            parts = mask_file.stem.split('_')
            if len(parts) >= 2:
                base_name = parts[1]  # Should be "nj0957"
                human_mask_map[base_name] = mask_file

        # Create mapping from filenames to system masks
        system_mask_map = {}
        for mask_file in system_masks:
            # Extract base name from e.g., "nj0957_mask_resized.png"
            base_name = mask_file.stem.replace("_mask_resized", "")
            system_mask_map[base_name] = mask_file

        logger.info(f"Found {len(human_mask_map)} human annotations and {len(system_mask_map)} system masks")

        # Find matching pairs and validate them
        valid_pairs = 0
        for base_name, human_mask in human_mask_map.items():
            if base_name in system_mask_map:
                stats["total_pairs"] += 1
                system_mask = system_mask_map[base_name]

                # Check both masks for valid sizes
                human_valid, human_size = self.check_image_size(human_mask)
                system_valid, system_size = self.check_image_size(system_mask)

                # Record size information for invalid images
                if not human_valid or not system_valid:
                    stats["size_mismatches"].append({
                        "base_name": base_name,
                        "human_size": human_size,
                        "system_size": system_size,
                        "target_size": self.target_size
                    })

                if not human_valid:
                    stats["invalid_human"] += 1
                    logger.warning(f"Human annotation '{base_name}' has invalid dimensions {human_size}")

                if not system_valid:
                    stats["invalid_system"] += 1
                    logger.warning(f"System mask '{base_name}' has invalid dimensions {system_size}")

                # Process based on resize mode
                if self.resize_mode == 'strict' and (not human_valid or not system_valid):
                    logger.info(f"Skipping annotation pair for '{base_name}' due to size mismatch")
                    stats["skipped_pairs"] += 1
                    continue
                elif self.resize_mode == 'resize':
                    # Resize both to target size
                    img_out_path = img_dir / f"{valid_pairs:04d}.png"
                    mask_out_path = mask_dir / f"{valid_pairs:04d}.png"

                    try:
                        # --- Process Human Mask ---
                        img = Image.open(human_mask).convert('L')  # Ensure grayscale
                        if self.resize_mode == 'resize':
                            img = img.resize(self.target_size[::-1], Image.BILINEAR)  # PIL uses (width, height)

                        # Binarize the human mask *after* resizing (if resizing)
                        img_array = np.array(img)
                        if not np.isin(img_array, [0, 255]).all():
                            stats["non_binary_human"] += 1
                            logger.warning(f"Human mask {human_mask} is not binary. Binarizing...")
                            img_array = (img_array > 127).astype(np.uint8) * 255  # Threshold at 127

                        Image.fromarray(img_array).save(img_out_path)  # Save as 8-bit grayscale PNG

                        # --- Process System Mask ---
                        mask = Image.open(system_mask).convert('L')
                        if self.resize_mode == 'resize':
                            mask = mask.resize(self.target_size[::-1], Image.NEAREST)  # Use NEAREST for masks!

                        # Binarize the system mask after resizing
                        mask_array = np.array(mask)
                        if not np.isin(mask_array, [0, 255]).all():
                            stats["non_binary_system"] += 1
                            logger.warning(f"System mask {system_mask} is not binary. Binarizing...")
                            mask_array = (mask_array > 127).astype(np.uint8) * 255  # Threshold at 127

                        Image.fromarray(mask_array).save(mask_out_path)

                        # Store mapping for tracking
                        stats["file_mapping"][str(valid_pairs)] = str(human_mask)

                        if self.resize_mode == 'resize':
                            stats["resized_pairs"] += 1
                        valid_pairs += 1

                    except Exception as e:
                        logger.error(f"Error processing annotation pair for '{base_name}': {str(e)}")
                        continue
                else:
                    # Both are valid and we're using strict mode
                    img_out_path = img_dir / f"{valid_pairs:04d}.png"
                    mask_out_path = mask_dir / f"{valid_pairs:04d}.png"

                    try:
                        # Process human mask (copy and binarize)
                        img = Image.open(human_mask).convert('L')
                        img_array = np.array(img)
                        if not np.isin(img_array, [0, 255]).all():
                            stats["non_binary_human"] += 1
                            logger.warning(f"Human mask {human_mask} is not binary. Binarizing...")
                            img_array = (img_array > 127).astype(np.uint8) * 255
                        Image.fromarray(img_array).save(img_out_path)

                        # Process system mask (copy and binarize)
                        mask = Image.open(system_mask).convert('L')
                        mask_array = np.array(mask)
                        if not np.isin(mask_array, [0, 255]).all():
                            stats["non_binary_system"] += 1
                            logger.warning(f"System mask {system_mask} is not binary. Binarizing...")
                            mask_array = (mask_array > 127).astype(np.uint8) * 255
                        Image.fromarray(mask_array).save(mask_out_path)

                        # Store mapping for tracking
                        stats["file_mapping"][str(valid_pairs)] = str(human_mask)

                        valid_pairs += 1
                    except Exception as e:
                        logger.error(f"Error copying annotation pair for '{base_name}': {str(e)}")
                        continue

                stats["valid_pairs"] += 1

        # Save metadata for mapping indices to original files
        with open(Path(output_dir) / "validation_metadata.json", 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Processed {stats['total_pairs']} annotation pairs")
        logger.info(f"Valid pairs: {stats['valid_pairs']}")
        logger.info(f"Resized pairs: {stats['resized_pairs']}")
        logger.info(f"Skipped pairs due to size issues: {stats['skipped_pairs']}")
        logger.info(f"Non-binary human masks found and binarized: {stats['non_binary_human']}")
        logger.info(f"Non-binary system masks found and binarized: {stats['non_binary_system']}")

        return stats

def validate_annotations(human_annotations_dir, system_masks_dir, output_dir,
                         target_size=(512, 512), resize_mode='resize'):
    """
    Validate and prepare annotations for training by ensuring consistent sizes.

    Args:
        human_annotations_dir: Directory containing human annotation masks
        system_masks_dir: Directory containing system-generated masks
        output_dir: Directory to save prepared files to
        target_size: Target size for all images (height, width)
        resize_mode: 'strict' to skip mismatched images, 'resize' to force resize all

    Returns:
        int: Number of valid annotation pairs
    """
    # Create validator
    validator = AnnotationValidator(
        target_size=target_size,
        resize_mode=resize_mode
    )

    # Get file lists
    human_masks = list(Path(human_annotations_dir).glob("*.png"))
    system_masks = list(Path(system_masks_dir).glob("*_mask_resized.png"))

    # Validate and copy valid pairs
    stats = validator.validate_and_filter_pairs(
        human_masks=human_masks,
        system_masks=system_masks,
        output_dir=output_dir
    )

    return stats["valid_pairs"]
