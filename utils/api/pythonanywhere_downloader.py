#!/usr/bin/env python3
"""
Script to download annotations from PythonAnywhere using the official API.
This script uses the files/tree and files/path endpoints to list and download files.
"""
import requests
import os
import logging
from pathlib import Path
import argparse
import yaml
import time
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PythonAnywhere API configuration
PYTHONANYWHERE_API_BASE = "https://www.pythonanywhere.com/api/v0"
PYTHONANYWHERE_USERNAME = "fkgsoftware"
PYTHONANYWHERE_API_TOKEN = "a4f5628b730ac605ff94bfbd11a7bd4551150621"
ANNOTATIONS_PATH = "/home/fkgsoftware/dem_eep_web/annotations"

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return None

def get_authorization_headers():
    """Get the authorization headers for PythonAnywhere API"""
    return {'Authorization': f'Token {PYTHONANYWHERE_API_TOKEN}'}

def list_files_in_directory(directory_path=ANNOTATIONS_PATH):
    """
    List all files in a directory on PythonAnywhere using the files/tree endpoint.

    Args:
        directory_path: Path to the directory on PythonAnywhere

    Returns:
        List of filenames (or None if an error occurred)
    """
    url = f"{PYTHONANYWHERE_API_BASE}/user/{PYTHONANYWHERE_USERNAME}/files/tree/?path={directory_path}"

    try:
        response = requests.get(
            url,
            headers=get_authorization_headers(),
            timeout=30
        )

        if response.status_code == 200:
            files = response.json()
            # Filter out directories (paths ending with /)
            file_paths = [f for f in files if not f.endswith('/')]
            logger.info(f"Found {len(file_paths)} files in {directory_path}")
            return file_paths
        else:
            logger.error(f"Failed to list files: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        return None

def download_file(file_path, output_dir):
    """
    Download a file from PythonAnywhere using the files/path endpoint.

    Args:
        file_path: Full path to the file on PythonAnywhere
        output_dir: Local directory to save the file

    Returns:
        True if successful, False otherwise
    """
    url = f"{PYTHONANYWHERE_API_BASE}/user/{PYTHONANYWHERE_USERNAME}/files/path{file_path}"

    try:
        response = requests.get(
            url,
            headers=get_authorization_headers(),
            timeout=30
        )

        if response.status_code == 200:
            # Extract the filename from the path
            filename = os.path.basename(file_path)
            output_path = os.path.join(output_dir, filename)

            # Save the file
            with open(output_path, 'wb') as f:
                f.write(response.content)

            logger.info(f"Downloaded {filename}")
            return True
        else:
            logger.error(f"Failed to download {file_path}: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        logger.error(f"Error downloading {file_path}: {str(e)}")
        return False

def download_annotations_for_grid(grid_square, output_dir):
    """
    Download all annotation files for a specific grid square.

    Args:
        grid_square: Grid square identifier (e.g., 'NH70')
        output_dir: Directory to save downloaded files

    Returns:
        tuple: (downloaded_count, failed_count)
    """
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List all files in the annotations directory
    all_files = list_files_in_directory(ANNOTATIONS_PATH)

    if not all_files:
        logger.error("Could not retrieve file list from PythonAnywhere")
        return 0, 0

    # Filter files by grid square pattern
    grid_files = [f for f in all_files if os.path.basename(f).startswith(f"{grid_square}_")]

    if not grid_files:
        logger.warning(f"No files found matching pattern '{grid_square}_*'")
        return 0, 0

    logger.info(f"Found {len(grid_files)} files for grid square {grid_square}")

    # Download each file
    downloaded = 0
    failed = 0

    for file_path in grid_files:
        if download_file(file_path, output_dir):
            downloaded += 1
        else:
            failed += 1

        # Add a small delay to avoid overwhelming the server
        time.sleep(0.5)

    logger.info(f"Downloaded {downloaded} files, failed to download {failed} files")
    return downloaded, failed

def main():
    parser = argparse.ArgumentParser(description="Download annotations from PythonAnywhere")
    parser.add_argument("--grid", type=str, help="Grid square identifier (e.g., NH70)")
    parser.add_argument("--output", type=str, help="Output directory (default: data/human_annotations)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if not config:
        sys.exit(1)

    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        # Default to human_annotations directory from config
        output_dir = config['data']['human_annotations_dir']

    # Get grid square from argument or user input
    grid_square = args.grid
    if not grid_square:
        grid_square = input("Enter grid square identifier (e.g., NH70): ")

    logger.info(f"Starting download for grid square {grid_square}")
    downloaded, failed = download_annotations_for_grid(grid_square, output_dir)

    if downloaded > 0:
        logger.info(f"Successfully downloaded {downloaded} files to {output_dir}")
        return output_dir, downloaded
    else:
        logger.error("No files were downloaded")
        return None, 0

if __name__ == "__main__":
    main()
