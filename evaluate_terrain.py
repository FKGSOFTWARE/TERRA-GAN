#!/usr/bin/env python3
"""
Revised Terrain Generation Evaluation Metrics Script

This script calculates various metrics to evaluate the quality of terrain generation:
1. IoU-based metrics: Precision, Recall, F1 Score
2. Largest unidentified area (km²) - largest contiguous AI-generated area that humans didn't detect
3. Percentage of undetected AI-generated terrain

Mask interpretation:
- Original masks: WHITE (1) = preserved areas, BLACK (0) = in-painted/AI-generated areas
- Annotation masks: WHITE (1) = areas humans flagged as AI-generated, BLACK (0) = areas humans thought were real

Usage:
    python evaluate_terrain.py --original-masks <dir> --final-annotations <dir> --output-file <json_path>
"""

import os
import json
import argparse
import numpy as np
import cv2
import re
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
from scipy import ndimage
from tqdm import tqdm


class TerrainEvaluator:
    def __init__(self, original_masks_dir, final_annotations_dir, resolution_meters=0.25, debug=False):
        """
        Initialize the evaluator with paths to original masks and human annotations.

        Args:
            original_masks_dir: Directory containing original inpainting masks (ground truth)
            final_annotations_dir: Directory containing human annotations of suspected AI-generated areas
            resolution_meters: Spatial resolution in meters per pixel (default: 0.25m)
            debug: Enable debug output
        """
        self.original_masks_dir = Path(original_masks_dir)
        self.final_annotations_dir = Path(final_annotations_dir)
        self.resolution_meters = resolution_meters
        self.debug = debug

        # Validate directories
        if not self.original_masks_dir.exists():
            raise FileNotFoundError(f"Original masks directory not found: {self.original_masks_dir}")
        if not self.final_annotations_dir.exists():
            raise FileNotFoundError(f"Final annotations directory not found: {self.final_annotations_dir}")

        # Get all files
        self.original_files = sorted(list(self.original_masks_dir.glob("*.png")))
        self.annotation_files = sorted(list(self.final_annotations_dir.glob("*.png")))

        if self.debug:
            print(f"Found {len(self.original_files)} original mask files")
            print(f"Found {len(self.annotation_files)} annotation files")

            if len(self.original_files) > 0:
                print(f"Example original mask filename: {self.original_files[0].name}")
            if len(self.annotation_files) > 0:
                print(f"Example annotation filename: {self.annotation_files[0].name}")

        self.metrics = {}

    def extract_tile_id(self, filename):
        """
        Extract the tile ID from a filename.

        For example:
        - NS83_ns8030_inpainted_colored_Zmlu_mask.png -> ns8030
        - ns8030_mask_resized.png -> ns8030
        """
        # Try to match the pattern in annotation filenames
        match = re.search(r'NS83_(ns\d+)_inpainted', filename)
        if match:
            return match.group(1)

        # Try to match the pattern in original mask filenames
        match = re.search(r'(ns\d+)_mask', filename)
        if match:
            return match.group(1)

        return None

    def find_matching_pairs(self):
        """Find matching pairs of original masks and annotations based on tile ID."""
        pairs = []

        # Create a dictionary of annotation files by tile ID
        annotation_dict = {}
        for anno_file in self.annotation_files:
            tile_id = self.extract_tile_id(anno_file.name)
            if tile_id:
                annotation_dict[tile_id] = anno_file

        # Find matching original mask files
        for orig_file in self.original_files:
            tile_id = self.extract_tile_id(orig_file.name)
            if tile_id and tile_id in annotation_dict:
                pairs.append({
                    'original_mask': orig_file,
                    'annotation': annotation_dict[tile_id],
                    'tile_id': tile_id
                })

        if self.debug:
            print(f"Found {len(pairs)} matching pairs")
            if len(pairs) > 0:
                print(f"Example pair: {pairs[0]['tile_id']}")
                print(f"  Original mask: {pairs[0]['original_mask'].name}")
                print(f"  Annotation: {pairs[0]['annotation'].name}")

        return pairs

    def calculate_iou(self, annotation_mask, ground_truth_mask):
        """
        Calculate Intersection over Union.

        Note: ground_truth_mask is inverted since BLACK (0) represents AI-generated areas
        in the original masks, while we want to measure agreement on AI-generated areas.
        """
        # Invert ground truth so 1 = AI-generated areas (what we're measuring)
        inverted_ground_truth = np.logical_not(ground_truth_mask)

        intersection = np.logical_and(annotation_mask, inverted_ground_truth).sum()
        union = np.logical_or(annotation_mask, inverted_ground_truth).sum()
        return intersection / union if union > 0 else 0.0

    def calculate_precision_recall_f1(self, annotation_mask, ground_truth_mask):
        """
        Calculate precision, recall, and F1 score.

        Note: ground_truth_mask is inverted since BLACK (0) represents AI-generated areas
        in the original masks, which is what we're trying to detect.
        """
        # Invert ground truth so 1 = AI-generated areas (what we're measuring)
        inverted_ground_truth = np.logical_not(ground_truth_mask)

        # Flatten masks for sklearn metrics
        anno_flat = annotation_mask.flatten()
        gt_flat = inverted_ground_truth.flatten()

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            gt_flat, anno_flat, average='binary', zero_division=0
        )

        return precision, recall, f1

    def calculate_largest_unidentified_area(self, annotation_mask, ground_truth_mask):
        """
        Calculate the largest contiguous area of AI-generated terrain that humans failed to identify.
        Unidentified is defined as where ground truth is 0 (BLACK, AI-generated) but human annotation
        is 0 (BLACK, not flagged).
        """
        # Invert ground truth so 1 = AI-generated areas
        inverted_ground_truth = np.logical_not(ground_truth_mask)

        # Identify unidentified regions (AI-generated but not flagged by humans)
        # This is where inverted_ground_truth is 1 (AI-generated) AND annotation_mask is 0 (not flagged)
        unidentified = np.logical_and(inverted_ground_truth, np.logical_not(annotation_mask))

        # Label connected components
        labeled, num_features = ndimage.label(unidentified)

        if num_features == 0:
            return 0.0

        # Calculate size of each connected component
        component_sizes = np.bincount(labeled.flatten())[1:]  # Skip background (0)
        largest_component_size = np.max(component_sizes) if len(component_sizes) > 0 else 0

        # Convert to square kilometers
        pixel_area_sq_m = self.resolution_meters ** 2
        largest_area_sq_km = (largest_component_size * pixel_area_sq_m) / 1_000_000

        return largest_area_sq_km

    def calculate_undetected_percentage(self, annotation_mask, ground_truth_mask):
        """
        Calculate what percentage of AI-generated terrain went undetected by humans.

        AI-generated terrain is BLACK (0) in the original masks.
        """
        # Invert ground truth so 1 = AI-generated areas
        inverted_ground_truth = np.logical_not(ground_truth_mask)

        # Where AI generated terrain was not flagged by humans
        # This is where inverted_ground_truth is 1 (AI-generated) AND annotation_mask is 0 (not flagged)
        undetected = np.logical_and(inverted_ground_truth, np.logical_not(annotation_mask))

        # Total AI-generated area
        total_ai_generated = np.sum(inverted_ground_truth)

        if total_ai_generated == 0:
            return 0.0

        return (np.sum(undetected) / total_ai_generated) * 100

    def evaluate_all(self):
        """Evaluate all matched pairs of original masks and annotations."""
        results = {
            'per_image': {},
            'aggregate': {
                'mean_iou': 0.0,
                'mean_precision': 0.0,
                'mean_recall': 0.0,
                'mean_f1': 0.0,
                'mean_largest_unidentified_area_sq_km': 0.0,
                'mean_undetected_percentage': 0.0,
                'total_images': 0
            }
        }

        # Find matching pairs
        pairs = self.find_matching_pairs()
        total_processed = 0

        # Process each pair
        for pair in tqdm(pairs, desc="Evaluating images"):
            try:
                # Load masks
                # WHITE (1) = preserved areas, BLACK (0) = in-painted/AI-generated areas
                orig_mask = cv2.imread(str(pair['original_mask']), cv2.IMREAD_GRAYSCALE) > 127

                # WHITE (1) = areas humans flagged as AI-generated, BLACK (0) = areas humans thought were real
                anno_mask = cv2.imread(str(pair['annotation']), cv2.IMREAD_GRAYSCALE) > 127

                # Ensure masks have same dimensions
                if orig_mask.shape != anno_mask.shape:
                    if self.debug:
                        print(f"Size mismatch for {pair['tile_id']}. Resizing annotation.")
                        print(f"  Original mask: {orig_mask.shape}")
                        print(f"  Annotation: {anno_mask.shape}")

                    # Resize annotation to match original mask
                    anno_mask = cv2.resize(anno_mask.astype(np.uint8),
                                       (orig_mask.shape[1], orig_mask.shape[0]),
                                       interpolation=cv2.INTER_NEAREST) > 0

                # Calculate metrics
                iou = self.calculate_iou(anno_mask, orig_mask)
                precision, recall, f1 = self.calculate_precision_recall_f1(anno_mask, orig_mask)
                largest_unidentified = self.calculate_largest_unidentified_area(anno_mask, orig_mask)
                undetected_pct = self.calculate_undetected_percentage(anno_mask, orig_mask)

                # Store individual results
                results['per_image'][pair['tile_id']] = {
                    'iou': float(iou),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'largest_unidentified_area_sq_km': float(largest_unidentified),
                    'undetected_percentage': float(undetected_pct)
                }

                # Accumulate for aggregate metrics
                results['aggregate']['mean_iou'] += iou
                results['aggregate']['mean_precision'] += precision
                results['aggregate']['mean_recall'] += recall
                results['aggregate']['mean_f1'] += f1
                results['aggregate']['mean_largest_unidentified_area_sq_km'] += largest_unidentified
                results['aggregate']['mean_undetected_percentage'] += undetected_pct

                total_processed += 1

            except Exception as e:
                if self.debug:
                    print(f"Error processing pair {pair['tile_id']}: {str(e)}")
                continue

        # Calculate means
        if total_processed > 0:
            for key in results['aggregate']:
                if key != 'total_images':
                    results['aggregate'][key] /= total_processed

        results['aggregate']['total_images'] = total_processed

        # Add additional aggregate data
        if total_processed > 0:
            # Find best and worst performing images by F1 score
            f1_scores = [(name, data['f1']) for name, data in results['per_image'].items()]
            best_image = max(f1_scores, key=lambda x: x[1])
            worst_image = min(f1_scores, key=lambda x: x[1])

            results['aggregate']['best_f1_image'] = {
                'name': best_image[0],
                'f1': best_image[1]
            }

            results['aggregate']['worst_f1_image'] = {
                'name': worst_image[0],
                'f1': worst_image[1]
            }

            # Calculate largest unidentified area across all images
            largest_unidentified_areas = [data['largest_unidentified_area_sq_km'] for data in results['per_image'].values()]
            results['aggregate']['max_unidentified_area_sq_km'] = max(largest_unidentified_areas)

        self.metrics = results
        return results

    def save_results(self, output_path):
        """Save evaluation results to a JSON file."""
        if not self.metrics:
            self.evaluate_all()

        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        print(f"Results saved to {output_path}")

    def get_summary(self):
        """Return a comprehensive, intuitive summary of the evaluation results."""
        if not self.metrics:
            self.evaluate_all()

        agg = self.metrics['aggregate']

        # Find the most and least convincing tiles (based on undetected percentage)
        per_image = self.metrics['per_image']
        tiles_by_deception = sorted(
            [(name, data['undetected_percentage']) for name, data in per_image.items()],
            key=lambda x: x[1],
            reverse=True
        )

        most_convincing = tiles_by_deception[0] if tiles_by_deception else ('none', 0)
        least_convincing = tiles_by_deception[-1] if tiles_by_deception else ('none', 0)

        # Calculate approximate football field equivalents (assuming 1 football field = 0.0053 km²)
        max_area_football_fields = round(agg['max_unidentified_area_sq_km'] / 0.0053)

        # Calculate detection rate (inverse of undetected percentage)
        detection_rate = 100 - agg['mean_undetected_percentage']

        # Calculate false positive rate (approx. 1 - precision)
        false_positive_rate = (1 - agg['mean_precision']) * 100

        # Create a visual bar for the deception success rate
        bar_length = 40
        filled_chars = round((agg['mean_undetected_percentage'] / 100) * bar_length)
        success_bar = '[' + '|' * filled_chars + '-' * (bar_length - filled_chars) + ']'

        return (
            f"===================================================================\n"
            f"                Terrain Generation Evaluation Summary\n"
            f"===================================================================\n"
            f"  Images evaluated: {agg['total_images']}\n"
            f"\n"
            f"  Traditional Metrics:\n"
            f"  ---------------------\n"
            f"  Mean IoU: {agg['mean_iou']:.4f}\n"
            f"  Mean Precision: {agg['mean_precision']:.4f}\n"
            f"  Mean Recall: {agg['mean_recall']:.4f}\n"
            f"  Mean F1 Score: {agg['mean_f1']:.4f}\n"
            f"\n"
            f"  Undetected AI-Generated Terrain Metrics:\n"
            f"  ------------------------------------\n"
            f"  Mean Largest Unidentified Area: {agg['mean_largest_unidentified_area_sq_km']:.4f} km²\n"
            f"  Mean Undetected Percentage: {agg['mean_undetected_percentage']:.2f}%\n"
            f"  Maximum Unidentified Area: {agg['max_unidentified_area_sq_km']:.4f} km²\n"
            f"\n"
            f"===================================================================\n"
            f"                     INTERPRETABLE METRICS\n"
            f"===================================================================\n"
            f"\n"
            f"  OVERALL DECEPTION SUCCESS: {agg['mean_undetected_percentage']:.1f}%\n"
            f"  ({agg['mean_undetected_percentage']:.1f}% of AI-generated terrain went completely undetected)\n"
            f"\n"
            f"  Detection Failure by Humans:\n"
            f"  - Most Convincing Tile: {most_convincing[0]} ({most_convincing[1]:.1f}% undetected)\n"
            f"  - Largest Undetected Area: {agg['max_unidentified_area_sq_km']:.4f} km²\n"
            f"    (equivalent to approximately {max_area_football_fields} football fields)\n"
            f"\n"
            f"  Human Detection Performance:\n"
            f"  - False Positives: {false_positive_rate:.1f}%\n"
            f"    (humans frequently misidentified real terrain as AI-generated)\n"
            f"  - Detection Rate: {detection_rate:.1f}%\n"
            f"    (humans only caught about {detection_rate:.1f}% of AI-generated terrain)\n"
            f"\n"
            f"  Success Visualization: {success_bar}\n"
            f"\n"
            f"  Most Successful Deceptions (Highest DSR):\n"
            + ''.join([f"  - {name}: {pct:.1f}% undetected\n" for name, pct in tiles_by_deception[:3]]) +
            f"\n"
            f"  Least Successful Deceptions (Lowest DSR):\n"
            + ''.join([f"  - {name}: {pct:.1f}% undetected\n" for name, pct in tiles_by_deception[-3:]]) +
            f"\n"
            f"  Note: Higher undetected values indicate more effective terrain generation\n"
            f"        (larger areas of AI-generated terrain that were not detected by humans)"
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate terrain generation quality")
    parser.add_argument("--original-masks", required=True, help="Directory with original inpainting mask files")
    parser.add_argument("--final-annotations", required=True, help="Directory with final human annotations")
    parser.add_argument("--output-file", default="evaluation_results.json", help="Output JSON file")
    parser.add_argument("--resolution", type=float, default=0.25, help="Spatial resolution in meters per pixel")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    # Run evaluation with updated parameter names
    evaluator = TerrainEvaluator(
        args.original_masks,
        args.final_annotations,
        resolution_meters=args.resolution,
        debug=args.debug
    )

    evaluator.evaluate_all()
    print(evaluator.get_summary())
    evaluator.save_results(args.output_file)


if __name__ == "__main__":
    main()
