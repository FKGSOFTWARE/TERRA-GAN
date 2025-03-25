#!/usr/bin/env python3
"""
Script to download all annotations from PythonAnywhere and save them to the annotations folder.

This script handles API throttling with proper retry logic and ensures all annotations are
downloaded without fail.
"""

import os
import sys
import time
import logging
import argparse
import yaml
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Set, Optional
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("download_annotations.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# PythonAnywhere API configuration
PYTHONANYWHERE_API_BASE = "https://www.pythonanywhere.com/api/v0"
PYTHONANYWHERE_USERNAME = "fkgsoftware"
PYTHONANYWHERE_API_TOKEN = "a4f5628b730ac605ff94bfbd11a7bd4551150621"
ANNOTATIONS_PATH = "/home/fkgsoftware/dem_eep_web/annotations"

# Default throttling parameters - will be updated from command line args if provided
DEFAULT_MIN_DELAY = 0.5  # Minimum delay between requests in seconds
DEFAULT_MAX_DELAY = 2.0  # Maximum delay between requests in seconds
DEFAULT_MAX_RETRIES = 5  # Maximum number of retry attempts
DEFAULT_BACKOFF_FACTOR = 1.5  # Exponential backoff factor
DEFAULT_BATCH_SIZE = 5  # Number of files to download in parallel
DEFAULT_PAUSE_AFTER = 10  # Pause after this many downloads

def load_config(config_path="config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}

def get_authorization_headers() -> dict:
    """Get the authorization headers for PythonAnywhere API"""
    return {'Authorization': f'Token {PYTHONANYWHERE_API_TOKEN}'}

def list_files_in_directory(directory_path=ANNOTATIONS_PATH, max_retries=DEFAULT_MAX_RETRIES) -> Optional[List[str]]:
    """
    List all files in a directory on PythonAnywhere using the files/tree endpoint.

    Implements retry logic with exponential backoff to handle throttling.

    Args:
        directory_path: Path to the directory on PythonAnywhere
        max_retries: Maximum number of retry attempts

    Returns:
        List of file paths (or None if an error occurred)
    """
    url = f"{PYTHONANYWHERE_API_BASE}/user/{PYTHONANYWHERE_USERNAME}/files/tree/?path={directory_path}"

    for attempt in range(max_retries + 1):
        try:
            # Add jitter to avoid synchronized requests
            delay = DEFAULT_MIN_DELAY + random.random() * (DEFAULT_MAX_DELAY - DEFAULT_MIN_DELAY)
            if attempt > 0:
                logger.info(f"Retry attempt {attempt}/{max_retries} after {delay:.2f}s delay")
                time.sleep(delay * (DEFAULT_BACKOFF_FACTOR ** attempt))  # Exponential backoff

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
            elif response.status_code == 429:  # Too Many Requests
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limited. Waiting {retry_after}s before retry.")
                time.sleep(retry_after)
            else:
                logger.error(f"Failed to list files (HTTP {response.status_code}): {response.text}")
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")

        # If we've reached max retries, return None
        if attempt == max_retries:
            logger.error(f"Max retries reached when listing files in {directory_path}")
            return None

    return None

def download_file(file_path: str, output_dir: Path, max_retries=DEFAULT_MAX_RETRIES) -> bool:
    """
    Download a file from PythonAnywhere using the files/path endpoint.

    Implements retry logic with exponential backoff to handle throttling.

    Args:
        file_path: Full path to the file on PythonAnywhere
        output_dir: Local directory to save the file
        max_retries: Maximum number of retry attempts

    Returns:
        True if successful, False otherwise
    """
    filename = os.path.basename(file_path)
    output_path = output_dir / filename

    # Skip if file already exists and has content
    if output_path.exists() and output_path.stat().st_size > 0:
        logger.debug(f"File already exists: {filename}")
        return True

    url = f"{PYTHONANYWHERE_API_BASE}/user/{PYTHONANYWHERE_USERNAME}/files/path{file_path}"

    for attempt in range(max_retries + 1):
        try:
            # Add jitter to avoid synchronized requests
            delay = DEFAULT_MIN_DELAY + random.random() * (DEFAULT_MAX_DELAY - DEFAULT_MIN_DELAY)
            if attempt > 0:
                logger.debug(f"Retry attempt {attempt}/{max_retries} for {filename} after {delay:.2f}s delay")
                time.sleep(delay * (DEFAULT_BACKOFF_FACTOR ** attempt))  # Exponential backoff

            response = requests.get(
                url,
                headers=get_authorization_headers(),
                timeout=30
            )

            if response.status_code == 200:
                # Save the file
                with open(output_path, 'wb') as f:
                    f.write(response.content)

                # Verify file was written successfully
                if output_path.exists() and output_path.stat().st_size > 0:
                    logger.info(f"Downloaded {filename} ({output_path.stat().st_size} bytes)")
                    return True
                else:
                    logger.warning(f"File {filename} was created but is empty or missing")
            elif response.status_code == 429:  # Too Many Requests
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limited when downloading {filename}. Waiting {retry_after}s before retry.")
                time.sleep(retry_after)
            else:
                logger.error(f"Failed to download {filename} (HTTP {response.status_code}): {response.text}")
        except Exception as e:
            logger.error(f"Error downloading {filename}: {str(e)}")

        # If we're not at max retries yet, continue
        if attempt < max_retries:
            continue

        # If we've reached max retries, return False
        logger.error(f"Max retries reached when downloading {filename}")
        return False

def download_files_batch(file_paths: List[str], output_dir: Path, batch_size=DEFAULT_BATCH_SIZE, max_retries=DEFAULT_MAX_RETRIES) -> Tuple[int, int]:
    """
    Download a batch of files using a thread pool to improve throughput while
    still respecting rate limits.

    Args:
        file_paths: List of file paths to download
        output_dir: Directory to save downloaded files
        batch_size: Number of files to download in parallel
        max_retries: Maximum number of retry attempts

    Returns:
        Tuple of (successful_count, failed_count)
    """
    successful = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        future_to_file = {
            executor.submit(download_file, file_path, output_dir, max_retries): file_path
            for file_path in file_paths
        }

        for future in future_to_file:
            file_path = future_to_file[future]
            try:
                if future.result():
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Exception when downloading {os.path.basename(file_path)}: {str(e)}")
                failed += 1

    return successful, failed

def download_all_annotations(output_dir: Path, grid_filter: Optional[str] = None,
                            batch_size=DEFAULT_BATCH_SIZE, max_retries=DEFAULT_MAX_RETRIES,
                            pause_after=DEFAULT_PAUSE_AFTER) -> Tuple[int, int, Set[str]]:
    """
    Download all annotations from PythonAnywhere.

    Args:
        output_dir: Directory to save downloaded files
        grid_filter: Optional filter to download only annotations for a specific grid
        batch_size: Number of files to download in parallel
        max_retries: Maximum number of retry attempts
        pause_after: Number of batches after which to pause

    Returns:
        Tuple of (successful_count, failed_count, grid_squares)
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # List all files in the annotations directory
    all_files = list_files_in_directory(ANNOTATIONS_PATH, max_retries)
    if not all_files:
        logger.error("Failed to list annotation files")
        return 0, 0, set()

    # Filter files if grid specified
    if grid_filter:
        filtered_files = [f for f in all_files if os.path.basename(f).startswith(f"{grid_filter}_")]
        logger.info(f"Filtered to {len(filtered_files)} files matching grid {grid_filter}")
        files_to_download = filtered_files
    else:
        files_to_download = all_files

    # Extract grid squares from filenames
    grid_squares = set()
    for file_path in files_to_download:
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        if len(parts) > 0:
            grid_squares.add(parts[0])

    logger.info(f"Found annotations for {len(grid_squares)} grid squares: {', '.join(sorted(grid_squares))}")

    # Track statistics
    total_files = len(files_to_download)
    successful = 0
    failed = 0

    # Process in smaller batches to avoid overwhelming the server
    for i in range(0, total_files, batch_size):
        batch = files_to_download[i:min(i+batch_size, total_files)]
        logger.info(f"Processing batch {i//batch_size + 1}/{(total_files + batch_size - 1)//batch_size} ({len(batch)} files)")

        batch_success, batch_failed = download_files_batch(batch, output_dir, batch_size, max_retries)
        successful += batch_success
        failed += batch_failed

        # Pause after processing PAUSE_AFTER files to avoid rate limiting
        if (i + batch_size) % (pause_after * batch_size) == 0 and i + batch_size < total_files:
            pause_time = 5 + random.random() * 5  # Random pause between 5-10 seconds
            logger.info(f"Pausing for {pause_time:.1f}s to avoid rate limiting...")
            time.sleep(pause_time)

    # Verify completion
    downloaded_files = list(output_dir.glob("*.png"))
    download_count = len(downloaded_files)

    logger.info(f"Download summary: {successful} successful, {failed} failed")
    logger.info(f"Files in output directory: {download_count}")

    # Check for missing files
    if download_count < total_files - failed:
        logger.warning(f"Some files may be missing: expected at least {total_files - failed}, found {download_count}")

        # Attempt to identify and retry missing files
        downloaded_filenames = {f.name for f in downloaded_files}
        expected_filenames = {os.path.basename(f) for f in files_to_download}
        missing_filenames = expected_filenames - downloaded_filenames

        if missing_filenames:
            logger.info(f"Attempting to download {len(missing_filenames)} missing files...")
            missing_files = [f for f in files_to_download if os.path.basename(f) in missing_filenames]

            # Retry with higher retry count and longer delays
            retry_success = 0
            for file_path in missing_files:
                if download_file(file_path, output_dir, max_retries * 2):
                    retry_success += 1

            logger.info(f"Retry results: {retry_success}/{len(missing_filenames)} files recovered")
            successful += retry_success

    return successful, failed, grid_squares

def main():
    parser = argparse.ArgumentParser(description="Download all annotations from PythonAnywhere")
    parser.add_argument("--output", type=str, default="annotations", help="Output directory (default: 'annotations' in current directory)")
    parser.add_argument("--grid", type=str, help="Filter by grid square (e.g., NH70)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--retry", type=int, default=DEFAULT_MAX_RETRIES, help=f"Maximum retry attempts (default: {DEFAULT_MAX_RETRIES})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch size for parallel downloads (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--pause-after", type=int, default=DEFAULT_PAUSE_AFTER, help=f"Pause after this many batches (default: {DEFAULT_PAUSE_AFTER})")

    args = parser.parse_args()

    # Get command-line parameters
    max_retries = args.retry
    batch_size = args.batch_size
    pause_after = args.pause_after

    # Get output directory
    output_dir = Path(args.output)

    # Load config
    config = load_config(args.config)

    logger.info(f"Starting download of all annotations to {output_dir}")
    logger.info(f"Configuration: max_retries={max_retries}, batch_size={batch_size}, pause_after={pause_after}")

    if args.grid:
        logger.info(f"Filtering annotations for grid square: {args.grid}")

    start_time = time.time()

    # Download all annotations
    successful, failed, grid_squares = download_all_annotations(
        output_dir,
        args.grid,
        batch_size=batch_size,
        max_retries=max_retries,
        pause_after=pause_after
    )

    duration = time.time() - start_time

    # Log summary
    logger.info(f"Download completed in {duration:.1f} seconds")
    logger.info(f"Total annotations: {successful + failed}")
    logger.info(f"Successfully downloaded: {successful}")
    logger.info(f"Failed to download: {failed}")
    logger.info(f"Grid squares found: {', '.join(sorted(grid_squares))}")

    # Final verification
    final_count = len(list(output_dir.glob("*.png")))
    logger.info(f"Final file count in {output_dir}: {final_count}")

    if failed > 0:
        sys.exit(1)
    else:
        logger.info("All annotations were downloaded successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
