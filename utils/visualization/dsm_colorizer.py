import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

class DSMColorizer:
    """Handles recoloring of DSM outputs using OS UK standard elevation color palette"""

    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # OS UK terrain colors from lowest to highest elevation
        # Based on standard topographic color scheme
        self.colors = [
            '#0C6B58',  # Deep green
            '#2E8B57',  # Sea green
            '#90EE90',  # Light green
            '#F4D03F',  # Yellow
            '#E67E22',  # Orange
            '#CB4335',  # Red
            '#6E2C00',  # Brown
            '#FFFFFF',  # White (peaks)
        ]

        # Create custom colormap
        self.colormap = plt.cm.colors.LinearSegmentedColormap.from_list(
            'osuk_terrain', self.colors)

    def recolor_all(self):
        """Process all inpainted PNGs in input directory"""
        for img_path in self.input_dir.glob("*_inpainted.png"):
            self.recolor_dsm(img_path)

    def recolor_dsm(self, img_path: Path):
        """Recolor single DSM image using OS UK elevation palette"""
        # Read grayscale image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        # Normalize values to 0-1
        normalized = img.astype(float) / 255

        # Apply colormap
        colored = self.colormap(normalized)
        colored = (colored[:, :, :3] * 255).astype(np.uint8)

        # Save colored version
        output_path = self.output_dir / f"{img_path.stem}_colored.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
        return output_path
