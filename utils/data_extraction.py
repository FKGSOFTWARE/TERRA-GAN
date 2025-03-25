import zipfile
import logging
import shutil
import cv2
from pathlib import Path
import numpy as np
from PIL import Image
import yaml
import os
from typing import Optional, Tuple

from utils.path_handling.path_utils import PathManager
from utils.mask_processing.core import MaskProcessor
from utils.mask_processing.visualization import visualize_masks

# Load configurations
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Initialize global constants
RAW_DATA_DIR = Path(config['data']['raw_dir'])
PROCESSED_DATA_DIR = Path(config['data']['processed_dir'])
TARGET_FOLDERS = ("getmapping-dsm-2000", "getmapping_rgb_25cm")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_relevant_folders(input_zip_path: Path, extract_to: Path, target_folders: Tuple[str, ...] = TARGET_FOLDERS) -> bool:
    """
    Extracts only the specified directories from a zip file.

    Args:
        input_zip_path: Path to the zip file
        extract_to: Directory where files will be extracted
        target_folders: Folder names to look for within the zip file

    Returns:
        bool: True if extraction was successful
    """
    logger.info(f"Extracting from {input_zip_path} to {extract_to}")
    try:
        with zipfile.ZipFile(input_zip_path, 'r') as zip_ref:
            extracted_files = []
            for file in zip_ref.namelist():
                if any(target in file for target in target_folders):
                    zip_ref.extract(file, extract_to)
                    extracted_files.append(file)

        if not extracted_files:
            logger.warning("No matching folders found in zip file")
            return False

        logger.info(f"Successfully extracted {len(extracted_files)} files")
        return True
    except Exception as e:
        logger.error(f"Failed to extract {input_zip_path}: {str(e)}")
        return False

def convert_dem_asc_to_png(asc_file_path: Path, png_file_path: Path, default_no_data: int = -9999) -> bool:
    """
    Converts a DEM .asc file to a grayscale .png image, normalizing data for GAN compatibility.

    Args:
        asc_file_path: Path to the .asc file
        png_file_path: Output path for the .png file
        default_no_data: Default value to treat as no data if not specified in file

    Returns:
        bool: True if conversion was successful
    """
    try:
        # Read header
        header = {}
        with open(asc_file_path, 'r') as file:
            for _ in range(6):
                key, value = file.readline().strip().split()
                header[key] = float(value) if '.' in value else int(value)

        # Load and process DEM data
        no_data_value = header.get('NODATA_value', default_no_data)
        data = np.loadtxt(asc_file_path, skiprows=6)
        data[data == no_data_value] = np.nan

        # Normalize data
        valid_data = data[~np.isnan(data)]
        if len(valid_data) == 0:
            logger.warning(f"No valid data in {asc_file_path}")
            return False

        data_min, data_max = np.nanmin(data), np.nanmax(data)
        if data_min == data_max:
            logger.warning(f"Flat elevation data in {asc_file_path}")
            normalized_data = np.zeros_like(data)
        else:
            normalized_data = 255 * (data - data_min) / (data_max - data_min)

        # Replace NaNs and ensure uint8 type
        normalized_data = np.nan_to_num(normalized_data, nan=0)
        normalized_data = normalized_data.astype(np.uint8)

        # Create output directory if needed
        png_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as PNG
        img = Image.fromarray(normalized_data, mode='L')
        img = img.resize((512, 512), Image.BILINEAR)
        img.save(png_file_path)

        logger.info(f"Successfully converted {asc_file_path.name} to PNG")
        return True

    except Exception as e:
        logger.error(f"Error converting {asc_file_path} to PNG: {str(e)}")
        return False

def process_parent_grid(zip_file_path: Path, path_manager: PathManager) -> Optional[str]:
    """
    Process a parent grid square zip file.

    Args:
        zip_file_path: Path to the zip file
        path_manager: PathManager instance

    Returns:
        str or None: Parent grid reference if successful, None otherwise
    """
    try:
        # Extract parent grid reference
        parent_grid = path_manager.get_parent_from_zip(zip_file_path)
        logger.info(f"Processing parent grid: {parent_grid}")

        # Create directory structure
        paths = path_manager.create_parent_structure(parent_grid)

        # Setup extract directory
        extract_dir = RAW_DATA_DIR / f"{parent_grid}_extracted"
        if not extract_relevant_folders(zip_file_path, extract_dir):
            return None

        return parent_grid

    except Exception as e:
        logger.error(f"Failed to process parent grid from {zip_file_path}: {str(e)}")
        return None

def cleanup_extracted_files(parent_grid: str) -> None:
    """
    Clean up extracted files for a parent grid square.

    Args:
        parent_grid: Parent grid square identifier
    """
    extracted_dir = RAW_DATA_DIR / f"{parent_grid}_extracted"
    if extracted_dir.exists():
        try:
            shutil.rmtree(extracted_dir)
            logger.info(f"Cleaned up extracted files for {parent_grid}")
        except Exception as e:
            logger.error(f"Failed to cleanup extracted files for {parent_grid}: {str(e)}")
