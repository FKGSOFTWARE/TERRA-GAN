import cv2
import numpy as np
import logging

class BuildingProcessor:
    """Temporary simplified processor for building detection."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Currently returns an empty mask.
        """
        # Create empty mask matching image dimensions
        if len(image.shape) == 3:
            height, width = image.shape[:2]
        else:
            height, width = image.shape

        return np.zeros((height, width), dtype=np.uint8)
