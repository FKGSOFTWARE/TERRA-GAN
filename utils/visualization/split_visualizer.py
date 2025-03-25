import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
import json
import yaml

logger = logging.getLogger(__name__)

class SplitVisualizer:
    """Visualizes the geographical split of data tiles"""

    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.processed_dir = Path(self.config['data']['processed_dir'])
        self.metadata_dir = self.processed_dir / 'metadata'

        # Define colors for each split type
        self.colors = {
            'train': '#2ECC71',  # Green
            'val': '#F1C40F',    # Yellow
            'test': '#E74C3C'    # Red
        }

    def load_split_data(self) -> dict:
        """Load split assignments from metadata"""
        try:
            split_path = self.metadata_dir / 'split_mapping.json'
            with open(split_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load split mapping: {str(e)}")
            return {}

    def visualize_splits(self):
        """Create and save visualization of the geographical splits"""
        # Load split data
        split_data = self.load_split_data()
        if not split_data:
            logger.error("No split data found to visualize")
            return

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(15, 15))

        # Process coordinates and create grid
        coords = [tuple(map(int, k.split(','))) for k in split_data.keys()]
        min_x = min(x for x, _ in coords)
        max_x = max(x for x, _ in coords)
        min_y = min(y for _, y in coords)
        max_y = max(y for _, y in coords)

        # Plot each tile
        for coord_str, split_type in split_data.items():
            x, y = map(int, coord_str.split(','))

            # Create tile rectangle
            rect = plt.Rectangle(
                (x - min_x, y - min_y),
                1, 1,
                facecolor=self.colors[split_type],
                edgecolor='black',
                alpha=0.7
            )
            ax.add_patch(rect)

            # Add tile identifier text
            tile_id = f"nj{x:02d}{y:02d}"
            ax.text(
                x - min_x + 0.5,
                y - min_y + 0.5,
                tile_id,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=8
            )

        # Set plot limits and aspect ratio
        ax.set_xlim(-0.1, max_x - min_x + 1.1)
        ax.set_ylim(-0.1, max_y - min_y + 1.1)
        ax.set_aspect('equal')

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)

        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', alpha=0.7, label=split)
            for split, color in self.colors.items()
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        # Add title
        plt.title('Geographical Data Splits', pad=20)

        # Save figure
        output_path = self.metadata_dir / 'geographical_splits.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Split visualization saved to {output_path}")

def create_split_visualization(config_path: str = "config.yaml"):
    """Helper function to create and save split visualization"""
    visualizer = SplitVisualizer(config_path)
    visualizer.visualize_splits()
