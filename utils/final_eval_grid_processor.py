#!/usr/bin/env python3
"""
Final Evaluation Grid Processor

This script prepares the test grid (NS83) for evaluation by:
1. Moving DEM files from raw directory to test/images
2. Moving mask files from raw directory to test/masks
3. Ensuring proper directory structure exists
"""

import os
import shutil
import argparse
import logging
from pathlib import Path
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_test_directory(grid_square):
    """Ensure test directory structure exists for the grid square."""
    config = load_config()
    processed_dir = Path(config['data']['processed_dir']) / grid_square

    # Create test directory structure if it doesn't exist
    test_images_dir = processed_dir / "test" / "images"
    test_masks_dir = processed_dir / "test" / "masks"

    test_images_dir.mkdir(parents=True, exist_ok=True)
    test_masks_dir.mkdir(parents=True, exist_ok=True)

    return processed_dir, test_images_dir, test_masks_dir

def process_raw_files(grid_square):
    """
    Process raw files for the evaluation grid:
    - Move DEM files to test/images
    - Move mask files to test/masks
    """
    processed_dir, test_images_dir, test_masks_dir = setup_test_directory(grid_square)
    raw_dir = processed_dir / "raw"

    if not raw_dir.exists():
        logger.error(f"Raw directory not found for {grid_square}: {raw_dir}")
        return False

    # Count files before processing
    raw_file_count = len(list(raw_dir.glob("*.png")))
    logger.info(f"Found {raw_file_count} raw files to process for {grid_square}")

    # Counter for moved files
    moved_images = 0
    moved_masks = 0

    # Process each file in raw directory
    for file_path in raw_dir.glob("*.png"):
        # Skip files that are already processed
        if "_mask_" in file_path.name:
            # This is a mask file - copy to test/masks directory
            dest_path = test_masks_dir / file_path.name
            try:
                shutil.copy2(file_path, dest_path)
                moved_masks += 1
            except Exception as e:
                logger.error(f"Error copying mask file {file_path}: {e}")
        else:
            # This is a DEM file - copy to test/images directory
            dest_path = test_images_dir / file_path.name
            try:
                shutil.copy2(file_path, dest_path)
                moved_images += 1
            except Exception as e:
                logger.error(f"Error copying image file {file_path}: {e}")

    logger.info(f"Moved {moved_images} image files to {test_images_dir}")
    logger.info(f"Moved {moved_masks} mask files to {test_masks_dir}")

    # Verify files were moved successfully
    image_count = len(list(test_images_dir.glob("*.png")))
    mask_count = len(list(test_masks_dir.glob("*.png")))

    logger.info(f"Final count: {image_count} images and {mask_count} masks in test directories")

    if image_count == 0 or mask_count == 0:
        logger.warning(f"One or more test directories is empty for {grid_square}")
        return False

    return True

def main():
    """Main function for processing evaluation grid."""
    parser = argparse.ArgumentParser(description="Process final evaluation grid for testing")
    parser.add_argument("--grid", type=str, default="NS83", help="Grid square to process (default: NS83)")
    parser.add_argument("--force", action="store_true", help="Force reprocessing even if files exist")

    args = parser.parse_args()
    grid_square = args.grid

    logger.info(f"Processing evaluation grid: {grid_square}")

    # Process raw files and move to test directories
    success = process_raw_files(grid_square)

    if success:
        logger.info(f"Successfully prepared {grid_square} for evaluation")
    else:
        logger.error(f"Failed to prepare {grid_square} for evaluation")

if __name__ == "__main__":
    main()
