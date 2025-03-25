import logging
import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import yaml

logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml"""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def match_human_and_system_masks(grid_square: str) -> List[Dict]:
    """Match human annotations with corresponding system masks based on tile names"""
    config = load_config()

    # Define paths for human and system masks
    human_annotation_dir = Path(f"data/output/{grid_square}/human_annotation_masks")
    system_masks_dir = Path(f"data/processed_data/{grid_square}/test/masks")

    # Ensure directories exist
    if not human_annotation_dir.exists():
        logger.error(f"Human annotation directory not found: {human_annotation_dir}")
        return []

    if not system_masks_dir.exists():
        logger.error(f"System masks directory not found: {system_masks_dir}")
        return []

    # Create dictionaries to store files by their tile identifiers
    system_mask_dict = {}
    human_mask_dict = {}

    # Process system masks
    for mask_path in system_masks_dir.glob("*_mask_resized.png"):
        # Extract tile name (e.g., "nm4927")
        tile_name = mask_path.stem.replace("_mask_resized", "").lower()
        system_mask_dict[tile_name] = mask_path

    logger.info(f"Found {len(system_mask_dict)} system masks in {system_masks_dir}")

    # Process human annotations
    for annot_path in human_annotation_dir.glob("*.png"):
        # Parse filename to extract tile name
        filename_parts = annot_path.stem.split('_')
        for part in filename_parts:
            # Look for parts that match the pattern like "nm4927"
            if len(part) >= 6 and part[:2].isalpha() and part[2:].isdigit():
                tile_name = part.lower()
                human_mask_dict[tile_name] = annot_path
                break

    logger.info(f"Found {len(human_mask_dict)} human annotations in {human_annotation_dir}")

    # Find matches
    matched_pairs = []
    for tile_name in set(system_mask_dict.keys()) & set(human_mask_dict.keys()):
        system_mask = system_mask_dict[tile_name]
        human_mask = human_mask_dict[tile_name]

        # Get the original image too
        image_dir = Path(f"data/processed_data/{grid_square}/test/images")
        image_path = image_dir / f"{tile_name}.png"

        if image_path.exists():
            matched_pairs.append({
                'tile_name': tile_name,
                'image_path': image_path,
                'system_mask_path': system_mask,
                'human_mask_path': human_mask
            })
        else:
            logger.warning(f"Image not found for tile {tile_name}")

    logger.info(f"Found {len(matched_pairs)} matching pairs out of {len(system_mask_dict)} system masks and {len(human_mask_dict)} human masks")

    # Log information about the first few matches for debugging
    for i, pair in enumerate(matched_pairs[:3]):
        logger.info(f"Match {i+1}: {pair['tile_name']}")
        logger.info(f"  Image: {pair['image_path']}")
        logger.info(f"  System mask: {pair['system_mask_path']}")
        logger.info(f"  Human mask: {pair['human_mask_path']}")

    return matched_pairs

def fetch_annotations_for_grid(grid_square: str, portal_client) -> Optional[Path]:
    """Fetch human annotations for a specific grid square from the portal"""
    # Define the target directory for human annotations
    human_annotation_dir = Path(f"data/output/{grid_square}/human_annotation_masks")
    human_annotation_dir.mkdir(parents=True, exist_ok=True)

    # Use the portal client to fetch annotations - this now returns paths directly in the target dir
    annotation_paths = portal_client.fetch_annotations(grid_square)

    if annotation_paths and len(annotation_paths) > 0:
        logger.info(f"Downloaded {len(annotation_paths)} annotations to {human_annotation_dir}")
        return human_annotation_dir
    else:
        logger.error(f"No annotations found for grid square {grid_square}")
        return None

def validate_dataset(dataset) -> bool:
    """Validate that the dataset contains usable human mask data"""
    empty_masks = 0
    total_samples = len(dataset)

    # Skip validation for empty datasets
    if total_samples == 0:
        logger.error("Dataset is empty")
        return False

    # Check first few samples
    for i in range(min(10, total_samples)):
        sample = dataset[i]
        if sample['human_mask'].sum() == 0:
            empty_masks += 1

    # If all checked samples have empty masks, check the entire dataset
    if empty_masks == min(10, total_samples):
        empty_masks = 0
        for i in range(total_samples):
            sample = dataset[i]
            if sample['human_mask'].sum() == 0:
                empty_masks += 1

    # Log and return validation result
    if empty_masks == total_samples:
        logger.error("All human masks are empty. Cannot proceed with training.")
        return False
    elif empty_masks > 0:
        empty_percent = (empty_masks / total_samples) * 100
        logger.warning(f"{empty_masks}/{total_samples} ({empty_percent:.1f}%) human masks contain all zeros")
        # Proceed with warning
        return True
    else:
        logger.info("All human masks contain valid data")
        return True
