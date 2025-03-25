import re
import shutil
import logging
import cv2
from pathlib import Path
from typing import Optional

from utils.data_extraction import extract_relevant_folders, convert_dem_asc_to_png
from utils.mask_processing.core import MaskProcessor, downscale_and_match_mask
from utils.mask_processing.visualization import visualize_masks
from utils.data_splitting import GeographicalDataHandler
from utils.path_handling.path_utils import PathManager
import yaml

# Configure logging
logger = logging.getLogger(__name__)

def process_zip_for_parent(
    zip_file_path: Path,
    parent_grid: str,
    mode: str,
    config_dict: dict
) -> bool:
    """
    Process a zip file containing grid square data.

    Args:
        zip_file_path: Path to the zip file
        parent_grid: Grid square identifier (e.g., 'NJ05')
        mode: Processing mode ('train' or 'evaluate')
        config_dict: Configuration dictionary

    Returns:
        bool: True if processing was successful
    """
    parent_grid = parent_grid.upper()
    pm = PathManager(config_dict)

    # Create parent folder structure
    paths = pm.create_parent_structure(parent_grid)

    # Setup processing directories
    extracted_dir = Path(config_dict['data']['raw_dir']) / f"{parent_grid}_extracted"
    processed_dir = Path(config_dict['data']['processed_dir']) / parent_grid / "raw"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Extract zip contents
    logger.info(f"Extracting {parent_grid} data from {zip_file_path}")
    if not extract_relevant_folders(zip_file_path, extracted_dir):
        logger.error(f"Failed to extract {zip_file_path}")
        return False

    # Locate required directories
    dsm_dir = next(extracted_dir.glob("getmapping-dsm-2000*"), None)
    rgb_dir = next(extracted_dir.glob("getmapping_rgb_25cm*"), None)
    if not (dsm_dir and rgb_dir):
        logger.error(f"{parent_grid}: Required directories not found")
        return False

    # Initialize processors
    mask_processor = MaskProcessor(config_dict['mask_processing'])
    grid_handler = GeographicalDataHandler(parent_grid=parent_grid,
                                         root_dir=Path(config_dict['data']['processed_dir']))

    processed_count = 0
    error_count = 0

    # Process each DSM and RGB pair
    for dsm_file in dsm_dir.glob("**/*.asc"):
        base_name = dsm_file.stem.split("_")[0].lower()
        rgb_file = next(rgb_dir.glob(f"**/{base_name}*.jpg"), None)

        if not rgb_file:
            logger.warning(f"No matching RGB file for {base_name}")
            continue

        try:
            # Get child paths
            child_paths = pm.get_paths_for_child(parent_grid, base_name)

            # Convert DSM to PNG
            if not convert_dem_asc_to_png(dsm_file, child_paths['raw']):
                continue

            # Generate and process masks
            masks = mask_processor.process_image(rgb_file)
            combined_mask = mask_processor.combine_masks(masks)
            cv2.imwrite(str(child_paths['mask']), combined_mask)

            # Create visualization if enabled
            if config_dict['mask_processing']['visualization']['enabled']:
                viz_path = paths['visualization'] / f"{base_name}_masks.png"
                rgb_image = cv2.imread(str(rgb_file))
                visualize_masks(masks, viz_path, rgb_image)

            # Register tile with grid handler
            match = re.match(r"^[a-z]{2}(\d{2})(\d{2})$", base_name.lower())
            if match:
                x_val = int(match.group(1))
                y_val = int(match.group(2))
                grid_handler.add_tile(child_paths['raw'], x_val, y_val)
            else:
                logger.warning(f"Could not parse x,y from {base_name}")
                continue

            processed_count += 1
            logger.info(f"Processed {parent_grid}/{base_name}")

        except Exception as e:
            logger.error(f"Error processing {base_name}: {str(e)}")
            error_count += 1
            continue

    # Log processing results
    logger.info(f"Completed {parent_grid} processing: {processed_count} successful, {error_count} failed")

    # Cleanup extracted files if configured
    if config_dict.get('cleanup_extracted', True):
        try:
            shutil.rmtree(extracted_dir, ignore_errors=True)
            logger.info(f"Cleaned up extracted folder for {parent_grid}")
        except Exception as e:
            logger.warning(f"Failed to cleanup extracted_dir for {parent_grid}: {e}")

    return processed_count > 0
