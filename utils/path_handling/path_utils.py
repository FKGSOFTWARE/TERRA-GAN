from pathlib import Path
import re
from typing import Dict
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class OutputPathConfig:
    """Configuration for output path structure"""
    base_dir: Path
    models_dir: Path
    inpainted_dir: str
    colored_dir: str
    visualization_dir: str
    masks_dir: str

class PathManager:
    def __init__(self, config: dict):
        self.config = config
        self.base_output_dir = Path(config['data']['output_dir'])
        self.base_processed_dir = Path(config['data']['processed_dir'])
        self.models_dir = Path(config['data']['models_dir'])

    def get_parent_from_zip(self, zip_path: Path) -> str:
        """Extract parent grid square from zip filename"""
        name = zip_path.stem.upper()
        if not self._validate_parent_grid(name):
            raise ValueError(f"Invalid parent grid square format: {name}")
        return name

    def _validate_parent_grid(self, grid_ref: str) -> bool:
        """
        Validate parent grid square format (e.g., 'NJ05', 'NH70')
        Any two letters followed by two digits
        """
        if not grid_ref:
            return False
        return (len(grid_ref) == 4 and
                grid_ref[:2].isalpha() and
                grid_ref[2:].isdigit())

    def _validate_child_grid(self, child_ref: str) -> bool:
        """
        Validate child grid reference format.
        Checks for format XXNNNN where XX is any two letters and NNNN are any four digits.

        Args:
            child_ref: Child grid reference (e.g., 'nj0957', 'nh7102', etc.)

        Returns:
            bool: True if valid format
        """
        if not child_ref:
            return False

        # Check format: any two letters followed by 4 digits
        pattern = re.compile(r'^[a-z]{2}\d{4}$', re.IGNORECASE)
        return bool(pattern.match(child_ref))

    def create_parent_structure(self, parent_grid: str) -> Dict[str, Path]:
        """Create complete directory structure for a parent grid square"""
        # Create processed data structure
        processed_parent = self.base_processed_dir / parent_grid
        for subdir in self.config['data']['parent_structure']['processed']:
            (processed_parent / subdir).mkdir(parents=True, exist_ok=True)

        # Create output structure
        output_parent = self.base_output_dir / parent_grid
        for subdir in self.config['data']['parent_structure']['output']:
            (output_parent / subdir).mkdir(parents=True, exist_ok=True)

        return {
            'processed': processed_parent,
            'processed_raw': processed_parent / 'raw',
            'processed_metadata': processed_parent / 'metadata',
            'output': output_parent,
            'output_inpainted': output_parent / 'inpainted',
            'output_colored': output_parent / 'colored',
            'visualization': output_parent / 'visualization',
            'masks': output_parent / 'masks'
        }

    def get_paths_for_child(self, parent_grid: str, child_name: str) -> Dict[str, Path]:
        """
        Get all relevant paths for a child grid square.

        Args:
            parent_grid: Parent grid square (e.g., 'NJ05')
            child_name: Child grid reference (e.g., 'nj0957')

        Returns:
            Dictionary of paths for the child
        """
        if not self._validate_child_grid(child_name):
            raise ValueError(f"Invalid child grid format: {child_name}")

        base_paths = self.create_parent_structure(parent_grid)
        return {
            'raw': base_paths['processed_raw'] / f"{child_name}.png",
            'mask': base_paths['processed_raw'] / f"{child_name}_mask_resized.png",
            'inpainted': base_paths['output_inpainted'] / f"{child_name}_inpainted.png",
            'colored': base_paths['output_colored'] / f"{child_name}_colored.png"
        }
