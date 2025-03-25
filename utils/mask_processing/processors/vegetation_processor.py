import cv2
import numpy as np
import logging

class VegetationProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def detect(self, image: np.ndarray) -> np.ndarray:
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Extract individual channels
            h, s, v = cv2.split(hsv)

            # Create vegetation mask using color thresholds
            mask = cv2.inRange(hsv,
                             np.array([30, 40, 40]),  # Lower green bound
                             np.array([90, 255, 255]))  # Upper green bound

            # Calculate ExG (Excess Green Index)
            b, g, r = cv2.split(image)
            g = g.astype(float)
            r = r.astype(float)
            b = b.astype(float)

            exg = 2 * g - r - b
            exg_normalized = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX)
            _, exg_mask = cv2.threshold(exg_normalized.astype(np.uint8),
                                      127, 255, cv2.THRESH_BINARY)

            # Combine masks
            combined_mask = cv2.bitwise_and(mask, exg_mask)

            # Remove small areas and fill holes
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Find and filter contours by area
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)

            clean_mask = np.zeros_like(mask)
            for contour in contours:
                if cv2.contourArea(contour) > self.config['min_area']:
                    cv2.drawContours(clean_mask, [contour], -1, 255, -1)

            return clean_mask

        except Exception as e:
            self.logger.error(f"Error in vegetation detection: {str(e)}")
            return np.zeros_like(image[:,:,0])
