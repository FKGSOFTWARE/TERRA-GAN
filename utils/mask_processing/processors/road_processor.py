import cv2
import numpy as np
import logging

class RoadProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def detect(self, image: np.ndarray) -> np.ndarray:
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)

            # Apply bilateral filter to reduce noise while preserving edges
            blurred = cv2.bilateralFilter(enhanced, 9, 75, 75)

            # Edge detection with higher thresholds
            edges = cv2.Canny(blurred,
                            self.config['canny_low'],
                            self.config['canny_high'])

            # Create mask excluding vegetation areas
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv,
                                   np.array([35, 50, 50]),
                                   np.array([85, 255, 255]))
            edges = cv2.bitwise_and(edges, cv2.bitwise_not(green_mask))

            # Line detection with strict parameters
            lines = cv2.HoughLinesP(edges,
                                  rho=1,
                                  theta=np.pi/180,
                                  threshold=self.config['hough_threshold'],
                                  minLineLength=self.config['hough_min_length'],
                                  maxLineGap=self.config['hough_max_gap'])

            # Create road mask
            mask = np.zeros_like(gray)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.abs(np.degrees(np.arctan2(y2-y1, x2-x1)))

                    # Filter lines by angle (near horizontal or vertical)
                    if (angle < 20 or abs(angle - 90) < 20 or
                        abs(angle - 180) < 20):
                        cv2.line(mask, (x1, y1), (x2, y2), 255,
                               self.config['line_thickness'])

            # Clean up mask
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            return mask

        except Exception as e:
            self.logger.error(f"Error in road detection: {str(e)}")
            return np.zeros_like(gray)
