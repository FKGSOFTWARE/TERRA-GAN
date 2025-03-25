import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict
from .core import MaskType

def visualize_masks(masks: Dict[MaskType, np.ndarray],
                   output_path: Path,
                   original_image: np.ndarray = None) -> None:
    """
    Visualize all masks with clear labeling.
    """
    # Create figure with subplots
    n_total = len(masks) + (1 if original_image is not None else 0)
    n_rows = (n_total + 2) // 3  # Ensure at most 3 columns
    n_cols = min(3, n_total)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    # Plot original image if provided
    idx = 0
    if original_image is not None:
        axes[idx].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[idx].set_title('Original Image')
        axes[idx].axis('off')
        idx += 1

    # Plot each mask with distinctive colormaps
    colormaps = {
        MaskType.ROADS: ('Reds', 'Roads'),
        MaskType.BUILDINGS: ('Blues', 'Buildings'),
        MaskType.VEGETATION: ('Greens', 'Vegetation'),
        MaskType.FIELDS: ('YlOrBr', 'Fields'),
        MaskType.COMBINED: ('gray', 'Combined')
    }

    for mask_type, mask in masks.items():
        cmap, title = colormaps[mask_type]
        axes[idx].imshow(mask, cmap=cmap)
        axes[idx].set_title(title)
        axes[idx].axis('off')
        idx += 1

    # Disable any unused subplots
    for i in range(idx, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path))
    plt.close()
