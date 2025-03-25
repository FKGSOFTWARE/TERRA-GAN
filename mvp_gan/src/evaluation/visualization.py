import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class ResultVisualizer:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def visualize_comparison(self,
                           original: np.ndarray,
                           inpainted: np.ndarray,
                           gan_mask: np.ndarray,
                           human_mask: np.ndarray,
                           save_path: Path):
        """Create visual comparison of results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        axes[0,0].imshow(original, cmap='gray')
        axes[0,0].set_title('Original')

        axes[0,1].imshow(inpainted, cmap='gray')
        axes[0,1].set_title('Inpainted')

        axes[1,0].imshow(gan_mask, cmap='gray')
        axes[1,0].set_title('GAN Mask')

        axes[1,1].imshow(human_mask, cmap='gray')
        axes[1,1].set_title('Human Annotation')

        plt.savefig(save_path)
        plt.close()
