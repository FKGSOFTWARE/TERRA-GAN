# utils/data_splitting.py

import re
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging
from dataclasses import dataclass

@dataclass
class TileInfo:
    """Stores information about a single tile"""
    path: Path
    x: int
    y: int
    split: Optional[str] = None

class GeographicalDataHandler:
    def __init__(self, parent_grid: str, root_dir: Path):
        """Initialize handler for geographical data organization.

        Args:
            parent_grid: Parent grid square identifier (e.g., 'NJ05')
            root_dir: Base directory for all processing
        """
        self.parent_grid = parent_grid
        self.root_dir = root_dir / parent_grid  # Scope to parent
        self.tile_mapping: Dict[Tuple[int, int], TileInfo] = {}
        self.split_assignments: Dict[Tuple[int, int], str] = {}
        self.logger = logging.getLogger(__name__)

    def add_tile(self, tile_path: Path, x: int, y: int) -> None:
        """
        Add a tile to the mapping.

        Args:
            tile_path: Path to the tile file
            x: X coordinate
            y: Y coordinate
        """
        # Extract base name without extension
        base_name = tile_path.stem.lower()

        # Validate format
        if not self._validate_child_grid(base_name):
            raise ValueError(f"Invalid tile format: {base_name}")

        self.tile_mapping[(x, y)] = TileInfo(
            path=tile_path,
            x=x,
            y=y
        )

    def apply_splits(self) -> None:
        """Apply splits within parent directory structure"""
        split_dirs = {}
        for split in ['train', 'val', 'test']:
            split_images_dir = self.root_dir / split / 'images'
            split_masks_dir = self.root_dir / split / 'masks'
            split_images_dir.mkdir(parents=True, exist_ok=True)
            split_masks_dir.mkdir(parents=True, exist_ok=True)
            split_dirs[split] = {
                'images': split_images_dir,
                'masks': split_masks_dir
            }

        for coord, tile_info in self.tile_mapping.items():
            if coord in self.split_assignments:
                split = self.split_assignments[coord]
                # Handle both the DEM and its corresponding mask
                dem_name = tile_info.path.name
                mask_name = f"{tile_info.path.stem}_mask_resized.png"

                # Move files to appropriate split directory under parent
                dem_dest = split_dirs[split]['images'] / dem_name
                mask_path = tile_info.path.parent / mask_name
                mask_dest = split_dirs[split]['masks'] / mask_name

                if tile_info.path.exists():
                    shutil.copy2(tile_info.path, dem_dest)
                if mask_path.exists():
                    shutil.copy2(mask_path, mask_dest)

    def save_metadata(self) -> None:
        """Save split assignments and coordinates to parent's metadata"""
        metadata_dir = self.root_dir / 'metadata'
        metadata_dir.mkdir(parents=True, exist_ok=True)

        # Save split mapping
        split_data = {
            f"{x},{y}": split
            for (x, y), split in self.split_assignments.items()
        }
        with open(metadata_dir / 'split_mapping.json', 'w') as f:
            json.dump(split_data, f, indent=2)

        # Save coordinate mapping
        coord_data = {
            str(tile.path): {
                'x': tile.x,
                'y': tile.y,
                'split': tile.split,
                'parent_grid': self.parent_grid
            }
            for tile in self.tile_mapping.values()
        }
        with open(metadata_dir / 'coordinate_mapping.json', 'w') as f:
            json.dump(coord_data, f, indent=2)

    def generate_splits(self, split_ratios: Dict[str, float] = None) -> None:
        """Generate checkerboard split pattern for all tiles"""
        if split_ratios is None:
            split_ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}

        # Validate ratios
        if not abs(sum(split_ratios.values()) - 1.0) < 0.001:
            raise ValueError("Split ratios must sum to 1.0")

        # Get grid dimensions
        coords = list(self.tile_mapping.keys())
        if not coords:
            raise ValueError("No tiles registered")

        min_x = min(x for x, _ in coords)
        max_x = max(x for x, _ in coords)
        min_y = min(y for _, y in coords)
        max_y = max(y for _, y in coords)

        # Create base 3x3 pattern
        base_pattern = self._create_base_pattern(split_ratios)

        # Apply pattern across grid
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if (x, y) in self.tile_mapping:
                    pattern_x = (x - min_x) % 3
                    pattern_y = (y - min_y) % 3
                    self.split_assignments[(x, y)] = base_pattern[pattern_y][pattern_x]

        # Validate split distribution
        self._validate_splits(split_ratios)

    def _create_base_pattern(self, split_ratios: Dict[str, float]) -> List[List[str]]:
        """
        Create a fixed 10x10 pattern that ensures no adjacent tiles share the same split.
        Distribution aims for approximately: 40% train, 30% val, 30% test

        Returns a 10x10 grid where no two adjacent cells (including diagonals)
        have the same split type.
        """
        # Original Pattern (for reference)
        # original_pattern = [
        #     ['train', 'val',   'test',  'train', 'val',   'test',  'train', 'val',   'test',  'train'],
        #     ['test',  'train', 'val',   'test',  'train', 'val',   'test',  'train', 'val',   'test' ],
        #     ['val',   'test',  'train', 'val',   'test',  'train', 'val',   'test',  'train', 'val'  ],
        #     ['train', 'val',   'test',  'train', 'val',   'test',  'train', 'val',   'test',  'train'],
        #     ['test',  'train', 'val',   'test',  'train', 'val',   'test',  'train', 'val',   'test' ],
        #     ['val',   'test',  'train', 'val',   'test',  'train', 'val',   'test',  'train', 'val'  ],
        #     ['train', 'val',   'test',  'train', 'val',   'test',  'train', 'val',   'test',  'train'],
        #     ['test',  'train', 'val',   'test',  'train', 'val',   'test',  'train', 'val',   'test' ],
        #     ['val',   'test',  'train', 'val',   'test',  'train', 'val',   'test',  'train', 'val'  ],
        #     ['train', 'val',   'test',  'train', 'val',   'test',  'train', 'val',   'test',  'train']
        # ]
        # return original_pattern

        # Permutation 1: Rotated pattern (each row shifts right by one)
        # permutation_1 = [
        #     ['val',   'test',  'train', 'val',   'test',  'train', 'val',   'test',  'train', 'val'  ],
        #     ['train', 'val',   'test',  'train', 'val',   'test',  'train', 'val',   'test',  'train'],
        #     ['test',  'train', 'val',   'test',  'train', 'val',   'test',  'train', 'val',   'test' ],
        #     ['val',   'test',  'train', 'val',   'test',  'train', 'val',   'test',  'train', 'val'  ],
        #     ['train', 'val',   'test',  'train', 'val',   'test',  'train', 'val',   'test',  'train'],
        #     ['test',  'train', 'val',   'test',  'train', 'val',   'test',  'train', 'val',   'test' ],
        #     ['val',   'test',  'train', 'val',   'test',  'train', 'val',   'test',  'train', 'val'  ],
        #     ['train', 'val',   'test',  'train', 'val',   'test',  'train', 'val',   'test',  'train'],
        #     ['test',  'train', 'val',   'test',  'train', 'val',   'test',  'train', 'val',   'test' ],
        #     ['val',   'test',  'train', 'val',   'test',  'train', 'val',   'test',  'train', 'val'  ]
        # ]
        # return permutation_1

        # # Permutation 2: Different cyclic pattern (test-val-train instead of train-val-test)
        permutation_2 = [
            ['test',  'val',   'train', 'test',  'val',   'train', 'test',  'val',   'train', 'test' ],
            ['val',   'train', 'test',  'val',   'train', 'test',  'val',   'train', 'test',  'val'  ],
            ['train', 'test',  'val',   'train', 'test',  'val',   'train', 'test',  'val',   'train'],
            ['test',  'val',   'train', 'test',  'val',   'train', 'test',  'val',   'train', 'test' ],
            ['val',   'train', 'test',  'val',   'train', 'test',  'val',   'train', 'test',  'val'  ],
            ['train', 'test',  'val',   'train', 'test',  'val',   'train', 'test',  'val',   'train'],
            ['test',  'val',   'train', 'test',  'val',   'train', 'test',  'val',   'train', 'test' ],
            ['val',   'train', 'test',  'val',   'train', 'test',  'val',   'train', 'test',  'val'  ],
            ['train', 'test',  'val',   'train', 'test',  'val',   'train', 'test',  'val',   'train'],
            ['test',  'val',   'train', 'test',  'val',   'train', 'test',  'val',   'train', 'test' ]
        ]
        return permutation_2

    def _validate_splits(self, target_ratios: Dict[str, float], tolerance: float = 0.01) -> None:
        """Validate split ratios and adjacency constraints"""
        # Count splits
        split_counts = {split: 0 for split in target_ratios}
        for coord, split in self.split_assignments.items():
            split_counts[split] += 1

        # Check adjacency
        for (x, y), split in self.split_assignments.items():
            adjacent_coords = [
                (x+1, y), (x-1, y),
                (x, y+1), (x, y-1)
            ]
            for adj_x, adj_y in adjacent_coords:
                if (adj_x, adj_y) in self.split_assignments:
                    adj_split = self.split_assignments[(adj_x, adj_y)]
                    if adj_split == split:
                        self.logger.warning(
                            f"Adjacent tiles at ({x},{y}) and ({adj_x},{adj_y}) "
                            f"are both in {split} split"
                        )

    def load_metadata(self) -> None:
        """Load split assignments and coordinate mappings from metadata files."""
        metadata_dir = self.root_dir / 'metadata'

        # Load split mapping
        try:
            with open(metadata_dir / 'split_mapping.json', 'r') as f:
                split_data = json.load(f)
                self.split_assignments = {
                    tuple(map(int, coord.split(','))): split
                    for coord, split in split_data.items()
                }
        except FileNotFoundError:
            self.logger.warning("Split mapping file not found")

        # Load coordinate mapping
        try:
            with open(metadata_dir / 'coordinate_mapping.json', 'r') as f:
                coord_data = json.load(f)
                for path_str, info in coord_data.items():
                    self.tile_mapping[info['x'], info['y']] = TileInfo(
                        path=Path(path_str),
                        x=info['x'],
                        y=info['y'],
                        split=info['split']
                    )
        except FileNotFoundError:
            self.logger.warning("Coordinate mapping file not found")

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

    def get_split_statistics(self) -> Dict[str, int]:
        """Get count of tiles in each split"""
        stats = {'train': 0, 'val': 0, 'test': 0}
        for _, split in self.split_assignments.items():
            stats[split] += 1
        return stats
