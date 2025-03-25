import cv2
import numpy as np
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum
import yaml

from .processors.road_processor import RoadProcessor
from .processors.building_processor import BuildingProcessor
from .processors.vegetation_processor import VegetationProcessor
from .processors.field_processor import FieldProcessor

# Configure logging
logger = logging.getLogger(__name__)

class MaskType(Enum):
    """Enum for different types of masks"""
    ROADS = "roads"
    BUILDINGS = "buildings"
    VEGETATION = "vegetation"
    FIELDS = "fields"
    COMBINED = "combined"

@dataclass
class ProcessingParams:
    """Parameters for mask processing"""
    # General parameters
    blur_kernel_size: Tuple[int, int] = (5, 5)
    blur_sigma: int = 0

    # Road detection
    road_canny_low: int = 50
    road_canny_high: int = 150
    road_hough_threshold: int = 50
    road_hough_min_length: int = 50
    road_hough_max_gap: int = 10
    road_dilation_kernel: int = 3

    # Building detection
    building_area_threshold: int = 500
    building_rect_tolerance: float = 0.2
    building_edge_threshold: int = 30

    # Vegetation detection
    veg_green_threshold: float = 1.3
    veg_saturation_threshold: int = 50
    veg_value_threshold: int = 50

    # Field detection
    field_min_area: int = 5000
    field_texture_threshold: float = 0.4
    field_homogeneity_threshold: float = 0.7

class MaskProcessor:
    """Main class for processing aerial imagery into semantic masks"""

    def __init__(self, config: dict):
        """
        Initialize the mask processor with configuration.

        Args:
            config: Dictionary containing configuration for all processors
        """
        self.config = config
        self.params = ProcessingParams()

        # Initialize individual processors
        self.road_processor = RoadProcessor(config['roads'])
        self.building_processor = BuildingProcessor(config['buildings'])
        self.vegetation_processor = VegetationProcessor(config['vegetation'])
        self.field_processor = FieldProcessor(config['fields'])

        logger.info("Initialized MaskProcessor with all components")

    def combine_masks(self, masks: Dict[MaskType, np.ndarray], invert_output: bool = True) -> np.ndarray:
        """
        Combine individual masks into a single binary mask.

        Args:
            masks: Dictionary of mask type to mask array
            invert_output: If True, invert the final mask so white areas (255)
                        are regions to inpaint and black areas (0) are preserved

        Returns:
            Combined binary mask
        """
        try:
            # Get reference dimensions from first mask
            reference_mask = next(iter(masks.values()))
            height, width = reference_mask.shape[:2]

            # Initialize combined mask
            combined = np.zeros((height, width), dtype=np.uint8)

            # Resize all masks to match reference dimensions
            resized_masks = {}
            for mask_type, mask in masks.items():
                if mask.shape[:2] != (height, width):
                    resized_mask = cv2.resize(mask, (width, height),
                                        interpolation=cv2.INTER_NEAREST)
                else:
                    resized_mask = mask
                resized_masks[mask_type] = resized_mask

            # Combine masks with priority order
            priority_order = [
                MaskType.BUILDINGS,
                MaskType.ROADS,
                MaskType.VEGETATION,
                MaskType.FIELDS
            ]

            for mask_type in priority_order:
                if mask_type in resized_masks:
                    mask = resized_masks[mask_type]
                    # Ensure the mask is binary
                    binary_mask = (mask > 127).astype(np.uint8) * 255
                    # Combine using bitwise OR
                    combined = cv2.bitwise_or(combined, binary_mask)

            # Invert the mask if requested (so white becomes the inpainting area)
            if invert_output:
                combined = 255 - combined

            return combined

        except Exception as e:
            logger.error(f"Error in mask combination: {str(e)}")
            return np.zeros((height, width), dtype=np.uint8)

    def process_image(self, image_path: Path) -> Dict[MaskType, np.ndarray]:
        """
        Process an aerial image and generate all masks.

        Args:
            image_path: Path to the input image

        Returns:
            Dictionary containing all generated masks

        Raises:
            ValueError: If image cannot be read
            Exception: For other processing errors
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")

            logger.info(f"Processing image: {image_path}")

            # Generate individual masks
            masks = {
                MaskType.ROADS: self.road_processor.detect(image),
                MaskType.BUILDINGS: self.building_processor.detect(image),
                MaskType.VEGETATION: self.vegetation_processor.detect(image),
                MaskType.FIELDS: self.field_processor.detect(image)
            }

            # Generate combined mask
            masks[MaskType.COMBINED] = self.combine_masks(masks)

            logger.info("Successfully generated all masks")
            return masks

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            raise

def create_processor(config_path: str = "config.yaml") -> MaskProcessor:
    """
    Factory function to create a MaskProcessor instance.

    Args:
        config_path: Path to configuration file

    Returns:
        Configured MaskProcessor instance
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return MaskProcessor(config['mask_processing'])
    except Exception as e:
        logger.error(f"Error creating MaskProcessor: {str(e)}")
        raise

def downscale_and_match_mask(mask, dem_image_path):
    """
    Downscale and match mask dimensions to DEM image.

    Args:
        mask: Either a numpy array or a Path to the mask image
        dem_image_path: Path to the DEM image
    """
    logging.info(f"Downscaling mask to match DEM {dem_image_path}")

    # Handle mask input
    if isinstance(mask, (str, Path)):
        mask_array = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        if mask_array is None:
            raise ValueError(f"Unable to read the mask image at {mask}")
    else:
        mask_array = mask  # Assume it's a numpy array

    # Read DEM image
    dem_image = cv2.imread(str(dem_image_path), cv2.IMREAD_GRAYSCALE)
    if dem_image is None:
        raise ValueError(f"Unable to read the DEM image at {dem_image_path}")

    # Resize mask
    dem_height, dem_width = dem_image.shape[:2]
    mask_resized = cv2.resize(mask_array, (dem_width, dem_height), interpolation=cv2.INTER_NEAREST)

    # Binarize the mask *after* resizing
    mask_resized = (mask_resized > 127).astype(np.uint8) * 255

    # Create output path and save
    if isinstance(dem_image_path, str):
        dem_image_path = Path(dem_image_path)
    output_path = dem_image_path.parent / f"{dem_image_path.stem}_mask_resized.png"
    cv2.imwrite(str(output_path), mask_resized)
    return output_path

def process_parent_images_in_parallel(self, image_paths: List[Path], invert_masks: bool = True) -> Dict:
    """
    Process multiple images in parallel to generate all masks.

    Args:
        image_paths: List of paths to the input images
        invert_masks: If True, invert final masks so white areas are for inpainting

    Returns:
        Dictionary mapping image paths to their generated masks
    """
    from utils.parallel_processing import process_images_in_parallel

    # Use process_images_in_parallel to process all images
    # Pass invert_masks parameter to each processing task
    results = process_images_in_parallel(
        image_paths,
        lambda path: self.process_image(path, invert_masks=invert_masks)
    )

    # Organize results by image path
    mask_results = {}
    for result in results:
        if result:  # Skip None results (errors)
            image_path, masks = result
            mask_results[image_path] = masks

    return mask_results

def process_image(self, image_path: Path, invert_masks: bool = True) -> Dict[MaskType, np.ndarray]:
    """
    Process an aerial image and generate all masks.

    Args:
        image_path: Path to the input image
        invert_masks: If True, invert the final combined mask so white areas are for inpainting

    Returns:
        Dictionary containing all generated masks

    Raises:
        ValueError: If image cannot be read
        Exception: For other processing errors
    """
    try:
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")

        logger.info(f"Processing image: {image_path}")

        # Generate individual masks
        masks = {
            MaskType.ROADS: self.road_processor.detect(image),
            MaskType.BUILDINGS: self.building_processor.detect(image),
            MaskType.VEGETATION: self.vegetation_processor.detect(image),
            MaskType.FIELDS: self.field_processor.detect(image)
        }

        # Generate combined mask with option to invert
        masks[MaskType.COMBINED] = self.combine_masks(masks, invert_output=invert_masks)

        logger.info("Successfully generated all masks")
        return masks

    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        raise
