#!/usr/bin/env python3
"""
Script to clean up files on PythonAnywhere server using the official API.
Can delete annotations, images, or both based on specified parameters.
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

# Default paths on the server
ANNOTATIONS_PATH = "/home/fkgsoftware/dem_eep_web/annotations"
IMAGES_PATH = "/home/fkgsoftware/dem_eep_web/static/images"

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

def list_files_in_directory(directory_path):
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

def delete_file(file_path):
    """
    Delete a file from PythonAnywhere using the files/path endpoint.

    Args:
        file_path: Full path to the file on PythonAnywhere

    Returns:
        True if successful, False otherwise
    """
    url = f"{PYTHONANYWHERE_API_BASE}/user/{PYTHONANYWHERE_USERNAME}/files/path{file_path}"

    try:
        response = requests.delete(
            url,
            headers=get_authorization_headers(),
            timeout=30
        )

        if response.status_code == 204:  # 204 No Content is the success response for DELETE
            logger.info(f"Deleted {file_path}")
            return True
        else:
            logger.error(f"Failed to delete {file_path}: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        logger.error(f"Error deleting {file_path}: {str(e)}")
        return False

def delete_files_in_directory(directory_path, filter_prefix=None, dry_run=False):
    """
    Delete all files in a directory on PythonAnywhere, optionally filtering by prefix.

    Args:
        directory_path: Path to the directory on PythonAnywhere
        filter_prefix: Optional prefix to filter files (e.g., 'NH70_')
        dry_run: If True, only list files that would be deleted without actually deleting

    Returns:
        tuple: (deleted_count, failed_count)
    """
    # List all files in the directory
    all_files = list_files_in_directory(directory_path)

    if not all_files:
        logger.error(f"Could not retrieve file list from {directory_path}")
        return 0, 0

    # Filter files if a prefix is specified
    if filter_prefix:
        files_to_delete = [f for f in all_files if os.path.basename(f).startswith(filter_prefix)]
        logger.info(f"Found {len(files_to_delete)} files matching prefix '{filter_prefix}'")
    else:
        files_to_delete = all_files
        logger.info(f"Preparing to delete all {len(files_to_delete)} files in {directory_path}")

    if not files_to_delete:
        logger.warning(f"No files found to delete")
        return 0, 0

    # If this is a dry run, just list the files
    if dry_run:
        logger.info("DRY RUN - These files would be deleted:")
        for file_path in files_to_delete:
            logger.info(f"  {file_path}")
        return len(files_to_delete), 0

    # Ask for confirmation if not in script mode
    if sys.stdout.isatty():  # Check if running in an interactive terminal
        confirm = input(f"Are you sure you want to delete {len(files_to_delete)} files? (y/N): ")
        if confirm.lower() != 'y':
            logger.info("Operation cancelled by user")
            return 0, 0

    # Delete each file
    deleted = 0
    failed = 0

    for file_path in files_to_delete:
        if delete_file(file_path):
            deleted += 1
        else:
            failed += 1

        # Add a small delay to avoid overwhelming the server
        time.sleep(0.25)

    logger.info(f"Deleted {deleted} files, failed to delete {failed} files")
    return deleted, failed

def main():
    parser = argparse.ArgumentParser(description="Clean up files on PythonAnywhere server")
    parser.add_argument("--annotations", action="store_true", help="Delete files in annotations directory")
    parser.add_argument("--images", action="store_true", help="Delete files in images directory")
    parser.add_argument("--grid", type=str, help="Filter by grid square prefix (e.g., NH70)")
    parser.add_argument("--dry-run", action="store_true", help="List files that would be deleted without actually deleting")
    parser.add_argument("--annotations-path", type=str, default=ANNOTATIONS_PATH, help="Custom path for annotations directory")
    parser.add_argument("--images-path", type=str, default=IMAGES_PATH, help="Custom path for images directory")
    parser.add_argument("--all", action="store_true", help="Delete both annotations and images")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    # If no action is specified, show help
    if not (args.annotations or args.images or args.all):
        parser.print_help()
        return

    # Set directories to clean
    directories_to_clean = []

    if args.annotations or args.all:
        directories_to_clean.append(args.annotations_path)

    if args.images or args.all:
        directories_to_clean.append(args.images_path)

    # If force is set, skip all confirmations regardless of environment
        if args.force:
            # Simply proceed without any confirmation
            pass
        # Otherwise if in interactive mode, confirm with the user
        elif sys.stdout.isatty():
            confirm = input("Are you absolutely sure you want to proceed? (yes/NO): ")
            if confirm.lower() != 'yes':
                logger.info("Operation cancelled by user")
                return

    # Process each directory
    for directory in directories_to_clean:
        logger.info(f"Processing directory: {directory}")
        deleted, failed = delete_files_in_directory(
            directory,
            filter_prefix=args.grid,
            dry_run=args.dry_run
        )

        logger.info(f"Directory {directory}: {deleted} deleted, {failed} failed")

if __name__ == "__main__":
    main()
