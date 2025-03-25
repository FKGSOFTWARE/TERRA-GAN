import cv2
import numpy as np
import logging

class FieldProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def detect(self, image: np.ndarray) -> np.ndarray:
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:,:,0]

            # Apply bilateral filter to smooth while preserving edges
            smoothed = cv2.bilateralFilter(l_channel, 9, 75, 75)

            # Adaptive thresholding to identify potential field areas
            binary = cv2.adaptiveThreshold(smoothed, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 3)

            # Remove vegetation areas
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            veg_mask = cv2.inRange(hsv,
                                 np.array([35, 50, 50]),
                                 np.array([85, 255, 255]))
            binary = cv2.bitwise_and(binary, cv2.bitwise_not(veg_mask))

            # Clean up mask
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Find and filter contours by area
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)

            # Create clean mask with only large areas
            clean_mask = np.zeros_like(mask)
            for contour in contours:
                if cv2.contourArea(contour) > self.config['min_area']:
                    cv2.drawContours(clean_mask, [contour], -1, 255, -1)

            return clean_mask

        except Exception as e:
            self.logger.error(f"Error in field detection: {str(e)}")
            return np.zeros_like(image[:,:,0])
