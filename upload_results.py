# #!/usr/bin/env python3

# import argparse
# import logging
# from pathlib import Path
# import yaml
# import sys
# import time
# from typing import List, Optional

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


# def load_config():
#     """Load configuration from config.yaml"""
#     with open("config.yaml", "r") as f:
#         return yaml.safe_load(f)


# def upload_results(grid_square: str = None, retry_count: int = 3, chunk_size: int = 2):
#     """
#     Upload processed "colored" results to the portal.

#     Now expects output in data/output/<GridSquare>/colored/.
#     If --grid is not provided, we pick the first available grid folder with a "colored" subdir.

#     Args:
#         grid_square: Grid square identifier (e.g., "NJ05")
#         retry_count: Number of retry attempts if upload fails
#         chunk_size: Number of files to upload in each batch
#     """
#     config = load_config()

#     # Initialize portal client
#     from utils.api.portal_client import PortalClient
#     client = PortalClient(
#         base_url=config['portal']['base_url'],
#         api_key=config['portal']['api_key']
#     )

#     # Base output directory (e.g. data/output)
#     output_dir = Path(config['data']['output_dir'])

#     # If no grid square specified, scan data/output/ for subdirectories
#     # that contain a 'colored' folder. Pick the first one you find.
#     if grid_square is None:
#         candidate_grids = []
#         for subdir in output_dir.iterdir():
#             if subdir.is_dir() and (subdir / "colored").is_dir():
#                 candidate_grids.append(subdir.name)

#         if not candidate_grids:
#             logger.error("No processed results found. Please run processing first.")
#             return

#         grid_square = candidate_grids[0]
#         if len(candidate_grids) > 1:
#             logger.info(f"Multiple grid squares found. Using {grid_square}")

#     # Construct the new path: data/output/<GridSquare>/colored
#     colored_dir = output_dir / grid_square / "colored"

#     # Check if colored directory exists
#     if not colored_dir.exists():
#         logger.error(f"No colored output found for grid square {grid_square}.")
#         logger.error(f"Expected path: {colored_dir}")
#         logger.error("Please run the processing pipeline first.")
#         return

#     # Get all PNG files
#     image_paths = list(colored_dir.glob("*.png"))
#     if not image_paths:
#         logger.error(f"No PNG files found in {colored_dir}")
#         return

#     logger.info(f"Found {len(image_paths)} images to upload for {grid_square}")

#     # Try to upload with retries
#     for attempt in range(retry_count):
#         try:
#             # Break the upload into smaller chunks for better reliability
#             if len(image_paths) > chunk_size:
#                 logger.info(f"Uploading in chunks of {chunk_size} images")
#                 success = True

#                 # Process in chunks
#                 for i in range(0, len(image_paths), chunk_size):
#                     chunk = image_paths[i:i+chunk_size]
#                     logger.info(f"Uploading chunk {i//chunk_size + 1}/{(len(image_paths) + chunk_size - 1)//chunk_size} ({len(chunk)} files)")

#                     # Upload this chunk
#                     chunk_success = client.upload_batch(grid_square, chunk)
#                     if not chunk_success:
#                         logger.warning(f"Chunk {i//chunk_size + 1} upload failed, will retry")
#                         success = False
#                         break

#                     # Add a short delay between chunks
#                     if i + chunk_size < len(image_paths):
#                         time.sleep(2)

#                 if success:
#                     logger.info(f"Successfully uploaded {len(image_paths)} images for {grid_square}")
#                     return
#             else:
#                 # Upload all at once for small batches
#                 success = client.upload_batch(grid_square, image_paths)
#                 if success:
#                     logger.info(f"Successfully uploaded {len(image_paths)} images for {grid_square}")
#                     return

#             # If we're here, the upload failed
#             logger.warning(f"Upload attempt {attempt+1} failed, waiting before retry...")
#             time.sleep(5 * (attempt + 1))  # Increasing backoff

#         except Exception as e:
#             logger.error(f"Upload attempt {attempt+1} failed with error: {str(e)}")
#             if attempt < retry_count - 1:
#                 logger.info(f"Retrying in {5 * (attempt + 1)} seconds...")
#                 time.sleep(5 * (attempt + 1))
#             else:
#                 logger.error("Maximum retry attempts reached. Upload failed.")
#                 return

#     logger.error("All upload attempts failed.")


# def main():
#     parser = argparse.ArgumentParser(description="Upload processed results to the portal")
#     parser.add_argument("--grid", type=str, help="Grid square identifier (e.g., NJ05)", default=None)
#     parser.add_argument("--retry", type=int, help="Number of retry attempts", default=3)
#     parser.add_argument("--chunk-size", type=int, help="Number of files per upload batch", default=2)

#     args = parser.parse_args()
#     upload_results(args.grid, args.retry, args.chunk_size)


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
import yaml
import sys
import time
from typing import List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from config.yaml"""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def upload_results(grid_square: str = None, retry_count: int = 3, chunk_size: int = 2, experiment_path: str = None):
    """
    Upload processed "colored" results to the portal.

    Now expects output in data/output/<GridSquare>/colored/ or a custom experiment path.
    If --grid is not provided, we pick the first available grid folder with a "colored" subdir.

    Args:
        grid_square: Grid square identifier (e.g., "NJ05")
        retry_count: Number of retry attempts if upload fails
        chunk_size: Number of files to upload in each batch
        experiment_path: Optional path to experiment output directory
    """
    config = load_config()

    # Initialize portal client
    from utils.api.portal_client import PortalClient
    client = PortalClient(
        base_url=config['portal']['base_url'],
        api_key=config['portal']['api_key']
    )

    # Base output directory (either from config or experiment path)
    if experiment_path:
        output_dir = Path(experiment_path)
    else:
        output_dir = Path(config['data']['output_dir'])

    # If no grid square specified, scan output directory for subdirectories
    # that contain a 'colored' folder. Pick the first one you find.
    if grid_square is None:
        candidate_grids = []
        for subdir in output_dir.iterdir():
            if subdir.is_dir() and (subdir / "colored").is_dir():
                candidate_grids.append(subdir.name)

        if not candidate_grids:
            logger.error("No processed results found. Please run processing first.")
            return

        grid_square = candidate_grids[0]
        if len(candidate_grids) > 1:
            logger.info(f"Multiple grid squares found. Using {grid_square}")

    # Construct the path: output_dir/<GridSquare>/colored
    colored_dir = output_dir / grid_square / "colored"

    # Check if colored directory exists
    if not colored_dir.exists():
        logger.error(f"No colored output found for grid square {grid_square}.")
        logger.error(f"Expected path: {colored_dir}")
        logger.error("Please run the processing pipeline first.")
        return

    # Get all PNG files
    image_paths = list(colored_dir.glob("*.png"))
    if not image_paths:
        logger.error(f"No PNG files found in {colored_dir}")
        return

    logger.info(f"Found {len(image_paths)} images to upload for {grid_square}")

    # Try to upload with retries
    for attempt in range(retry_count):
        try:
            # Break the upload into smaller chunks for better reliability
            if len(image_paths) > chunk_size:
                logger.info(f"Uploading in chunks of {chunk_size} images")
                success = True

                # Process in chunks
                for i in range(0, len(image_paths), chunk_size):
                    chunk = image_paths[i:i+chunk_size]
                    logger.info(f"Uploading chunk {i//chunk_size + 1}/{(len(image_paths) + chunk_size - 1)//chunk_size} ({len(chunk)} files)")

                    # Upload this chunk
                    chunk_success = client.upload_batch(grid_square, chunk)
                    if not chunk_success:
                        logger.warning(f"Chunk {i//chunk_size + 1} upload failed, will retry")
                        success = False
                        break

                    # Add a short delay between chunks
                    if i + chunk_size < len(image_paths):
                        time.sleep(2)

                if success:
                    logger.info(f"Successfully uploaded {len(image_paths)} images for {grid_square}")
                    return
            else:
                # Upload all at once for small batches
                success = client.upload_batch(grid_square, image_paths)
                if success:
                    logger.info(f"Successfully uploaded {len(image_paths)} images for {grid_square}")
                    return

            # If we're here, the upload failed
            logger.warning(f"Upload attempt {attempt+1} failed, waiting before retry...")
            time.sleep(5 * (attempt + 1))  # Increasing backoff

        except Exception as e:
            logger.error(f"Upload attempt {attempt+1} failed with error: {str(e)}")
            if attempt < retry_count - 1:
                logger.info(f"Retrying in {5 * (attempt + 1)} seconds...")
                time.sleep(5 * (attempt + 1))
            else:
                logger.error("Maximum retry attempts reached. Upload failed.")
                return

    logger.error("All upload attempts failed.")


def main():
    parser = argparse.ArgumentParser(description="Upload processed results to the portal")
    parser.add_argument("--grid", type=str, help="Grid square identifier (e.g., NJ05)", default=None)
    parser.add_argument("--retry", type=int, help="Number of retry attempts", default=3)
    parser.add_argument("--chunk-size", type=int, help="Number of files per upload batch", default=2)
    parser.add_argument("--experiment", type=str, help="Path to experiment output directory", default=None)

    args = parser.parse_args()
    upload_results(args.grid, args.retry, args.chunk_size, args.experiment)


if __name__ == "__main__":
    main()
