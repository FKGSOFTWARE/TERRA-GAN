#!/usr/bin/env python3
"""
annotations_uploader.py - Script to upload annotation files to PythonAnywhere

This script takes a local directory containing annotation files and uploads them to
the PythonAnywhere server using the official API with correct multipart encoding.
"""

import requests
import os
import logging
import argparse
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PythonAnywhere API configuration
PYTHONANYWHERE_API_BASE = "https://www.pythonanywhere.com/api/v0"
PYTHONANYWHERE_USERNAME = "fkgsoftware"
PYTHONANYWHERE_API_TOKEN = "a4f5628b730ac605ff94bfbd11a7bd4551150621"
ANNOTATIONS_PATH = "/home/fkgsoftware/dem_eep_web/annotations"

def upload_file_to_pythonanywhere(local_file_path, remote_file_path):
    """
    Upload a file to PythonAnywhere using the files/path endpoint with multipart encoding.

    Args:
        local_file_path: Path to the local file to upload
        remote_file_path: Full path on PythonAnywhere where the file should be stored

    Returns:
        bool: True if upload was successful, False otherwise
    """
    url = f"{PYTHONANYWHERE_API_BASE}/user/{PYTHONANYWHERE_USERNAME}/files/path{remote_file_path}"
    headers = {'Authorization': f'Token {PYTHONANYWHERE_API_TOKEN}'}

    try:
        with open(local_file_path, 'rb') as f:
            # The key part is using 'content' as the field name in the multipart form
            files = {'content': (os.path.basename(local_file_path), f)}
            response = requests.post(url, headers=headers, files=files)

        if response.status_code == 200 or response.status_code == 201:
            logger.info(f"Successfully uploaded {os.path.basename(local_file_path)}")
            return True
        else:
            logger.error(f"Failed to upload {os.path.basename(local_file_path)}: {response.status_code} - {response.text[:100]}")
            return False
    except Exception as e:
        logger.error(f"Error uploading {os.path.basename(local_file_path)}: {str(e)}")
        return False

def upload_annotations(annotations_dir, grid_square, delay=0.5, dry_run=False):
    """
    Upload all annotation files from a directory to PythonAnywhere.

    Args:
        annotations_dir: Path to directory containing annotation files
        grid_square: Grid square identifier (e.g., NH70)
        delay: Delay between uploads in seconds
        dry_run: If True, only simulate the upload without actually sending files

    Returns:
        tuple: (success_count, failure_count)
    """
    annotations_dir = Path(annotations_dir)
    if not annotations_dir.exists():
        logger.error(f"Directory not found: {annotations_dir}")
        return 0, 0

    # Find PNG files
    annotation_files = list(annotations_dir.glob("*.png"))
    if not annotation_files:
        logger.error(f"No PNG files found in {annotations_dir}")
        return 0, 0

    # Filter by grid square if specified
    if grid_square:
        annotation_files = [f for f in annotation_files if grid_square.upper() in f.name.upper()]

    logger.info(f"Found {len(annotation_files)} annotation files for grid square {grid_square}")

    if dry_run:
        logger.info("DRY RUN - The following files would be uploaded:")
        for file in annotation_files:
            logger.info(f"  {file.name}")
        return len(annotation_files), 0

    # Upload files
    success_count = 0
    failure_count = 0

    for i, file_path in enumerate(annotation_files):
        # Construct remote path
        remote_path = f"{ANNOTATIONS_PATH}/{file_path.name}"

        # Upload the file
        if upload_file_to_pythonanywhere(file_path, remote_path):
            success_count += 1
        else:
            failure_count += 1

        # Log progress periodically
        if (i + 1) % 5 == 0 or (i + 1) == len(annotation_files):
            logger.info(f"Progress: {i + 1}/{len(annotation_files)} files processed")

        # Add delay between uploads to avoid overwhelming the server
        if i < len(annotation_files) - 1:
            time.sleep(delay)

    logger.info(f"Upload complete: {success_count} successful, {failure_count} failed")
    return success_count, failure_count

def main():
    parser = argparse.ArgumentParser(description="Upload annotation files to PythonAnywhere")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing annotation files")
    parser.add_argument("--grid", type=str, required=True, help="Grid square identifier (e.g., NH70)")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between uploads in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Only simulate the upload without sending files")

    args = parser.parse_args()

    upload_annotations(args.dir, args.grid, args.delay, args.dry_run)

if __name__ == "__main__":
    main()
