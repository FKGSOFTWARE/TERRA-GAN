import os
import requests
import logging
import time
import io
from pathlib import Path
from typing import List, Optional, Dict, Union, Set
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class PortalClient:
    """
    Client for interacting with PythonAnywhere annotation portal.

    Enhanced with better error handling, retry logic, incremental uploads,
    and annotation deletion functionality.
    """

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        # Remove Content-Type from default headers to allow file uploads
        self.default_headers = {
            'Authorization': f'Bearer {api_key}'
        }
        self.logger = logging.getLogger(__name__)

        # Configure a requests Session with retry/backoff
        self.session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "DELETE"]
        )
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

    def _handle_response(self, response: requests.Response, operation: str) -> Dict:
        """Handle API responses and log/raise errors for non-2xx statuses."""
        try:
            response.raise_for_status()
            try:
                return response.json()
            except ValueError:
                # Not all responses will be JSON
                return {"status": "success", "text": response.text}
        except requests.exceptions.HTTPError as e:
            # If server says '429 Too Many Requests', wait the recommended time
            if response.status_code == 429:
                self.logger.warning("Rate limit exceeded, waiting before retry")
                time.sleep(int(response.headers.get('Retry-After', 60)))
                raise
            self.logger.error(f"{operation} failed: {str(e)}")
            self.logger.error(f"Response content: {response.text[:200]}...")
            raise
        except ValueError as e:
            # JSON parsing error
            self.logger.error(f"Invalid JSON response: {str(e)}")
            raise
        except Exception as e:
            # Catch-all for unexpected exceptions
            self.logger.error(f"Unexpected error during {operation}: {str(e)}")
            raise

    def upload_batch(self, grid_square: str, image_paths: List[Path]) -> bool:
        """
        Upload a batch of recolored DSM images as multipart/form-data.
        Uses smaller chunks to avoid server timeouts.

        - grid_square: e.g. "NJ05"
        - image_paths: list of .png or .jpg Paths to upload
        """
        endpoint = f"{self.base_url}/api/upload/{grid_square}"

        try:
            # Validate input paths
            valid_paths = [p for p in image_paths if p.exists() and p.suffix.lower() in ['.png', '.jpg']]
            if not valid_paths:
                raise ValueError("No valid image files provided for upload")

            self.logger.info(f"Uploading {len(valid_paths)} files for {grid_square}")

            # Use much smaller chunks to avoid overwhelming the server
            chunk_size = 2  # Only send 2 files at a time
            success_count = 0

            # Process files in smaller chunks
            for i in range(0, len(valid_paths), chunk_size):
                chunk = valid_paths[i:i+chunk_size]
                self.logger.info(f"Uploading chunk {i//chunk_size + 1}/{(len(valid_paths) + chunk_size - 1)//chunk_size}: {len(chunk)} files")

                # Prepare files for this chunk
                files = [
                    ('files', (path.name, open(path, 'rb'), 'image/png'))
                    for path in chunk
                ]

                try:
                    # Use headers without explicit Content-Type (requests will set it)
                    upload_headers = dict(self.default_headers)

                    # Set a longer timeout for uploads
                    response = self.session.post(
                        endpoint,
                        headers=upload_headers,
                        files=files,
                        timeout=60  # Longer timeout for uploads
                    )

                    self._handle_response(response, f"upload chunk {i//chunk_size + 1}")
                    success_count += len(chunk)

                    # Add a short delay between chunks to not overwhelm the server
                    time.sleep(1)

                except Exception as e:
                    self.logger.error(f"Error uploading chunk {i//chunk_size + 1}: {str(e)}")
                    # Continue with next chunk even if this one failed
                finally:
                    # Clean up opened file handles for this chunk
                    for f in files:
                        f[1][1].close()

            self.logger.info(f"Successfully uploaded {success_count}/{len(valid_paths)} files for {grid_square}")
            return success_count > 0

        except Exception as e:
            self.logger.error(f"Upload failed: {str(e)}")
            return False

    def fetch_annotations(self, grid_square: str) -> Optional[List[Path]]:
        """
        Fetch human annotations for a grid square using PythonAnywhere API.

        This implementation uses the PythonAnywhere API to access files directly.

        Args:
            grid_square: Grid square identifier (e.g., 'NH70')

        Returns:
            List of Paths where annotations were saved, or None if failed
        """
        try:
            # Import the PythonAnywhere-specific downloader
            from utils.api.pythonanywhere_downloader import download_annotations_for_grid

            # Create annotation directory - using the new structure
            annotation_dir = Path(f"data/output/{grid_square}/human_annotation_masks")
            annotation_dir.mkdir(parents=True, exist_ok=True)

            # Use the PythonAnywhere downloader - this already saves files to annotation_dir
            self.logger.info(f"Fetching annotations for grid square {grid_square} using PythonAnywhere API")
            downloaded, failed = download_annotations_for_grid(grid_square, str(annotation_dir))

            if downloaded > 0:
                # Get the list of downloaded files - no need to copy them again
                downloaded_files = list(annotation_dir.glob(f"{grid_square}_*.png"))
                self.logger.info(f"Retrieved {len(downloaded_files)} annotations for {grid_square}")
                return downloaded_files
            else:
                self.logger.warning(f"No annotations found for {grid_square}")
                return None

        except Exception as e:
            self.logger.error(f"Failed to fetch annotations: {str(e)}")
            return None

    def get_annotation_status(self, grid_square: str) -> Optional[Dict]:
        """
        Get the status of annotations for a grid square from the portal.
        """
        endpoint = f"{self.base_url}/api/status/{grid_square}"
        try:
            response = self.session.get(
                endpoint,
                headers=self.default_headers,
                timeout=15
            )
            return self._handle_response(response, "get status")
        except Exception as e:
            self.logger.error(f"Failed to get annotation status: {str(e)}")
            return None

    def submit_feedback(self, grid_square: str, feedback: Dict) -> bool:
        """
        Submit feedback on generated inpainting results as JSON
        to /api/feedback/<grid_square>.
        """
        endpoint = f"{self.base_url}/api/feedback/{grid_square}"

        try:
            response = self.session.post(
                endpoint,
                headers={**self.default_headers, 'Content-Type': 'application/json'},
                json=feedback,
                timeout=15
            )
            self._handle_response(response, "submit feedback")
            return True
        except Exception as e:
            self.logger.error(f"Failed to submit feedback: {str(e)}")
            return False

    def create_test_file(self, grid_square: str) -> bool:
        """
        Create a simple test file to verify upload functionality.
        This can be used to diagnose server issues.
        """
        from PIL import Image
        import numpy as np

        try:
            # Create a small test image
            test_img = Image.new('L', (100, 100), color=128)
            img_array = np.array(test_img)

            # Add some text/pattern to identify it
            img_array[40:60, 40:60] = 255  # Add a white square in the middle

            # Create test file in memory
            test_img = Image.fromarray(img_array)
            img_byte_arr = io.BytesIO()
            test_img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)

            # Prepare for upload
            files = [('files', (f'{grid_square}_test.png', img_byte_arr, 'image/png'))]
            endpoint = f"{self.base_url}/api/upload/{grid_square}"

            # Upload
            self.logger.info(f"Uploading test file to {endpoint}")
            response = self.session.post(
                endpoint,
                headers=self.default_headers,
                files=files,
                timeout=30
            )

            if response.status_code == 200:
                self.logger.info("Test file upload successful")
                return True
            else:
                self.logger.error(f"Test file upload failed: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            self.logger.error(f"Test file creation/upload failed: {str(e)}")
            return False

    def delete_annotation(self, grid_square: str, filename: str, confirm: bool = True) -> bool:
        """
        Delete a specific annotation file from the server.

        Args:
            grid_square: Grid square identifier (e.g., "NJ05")
            filename: Name of the annotation file to delete
            confirm: If True, requires confirmation before deleting

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            if confirm:
                # Log confirmation message - in a real app, might prompt user
                self.logger.info(f"Preparing to delete annotation: {filename}")

            endpoint = f"{self.base_url}/api/delete/{grid_square}/{filename}"

            response = self.session.delete(
                endpoint,
                headers=self.default_headers,
                timeout=15
            )

            result = self._handle_response(response, f"delete annotation {filename}")
            if result.get('status') == 'success':
                self.logger.info(f"Successfully deleted annotation: {filename}")
                return True
            else:
                self.logger.warning(f"Failed to delete annotation: {filename}. Server response: {result}")
                return False

        except Exception as e:
            self.logger.error(f"Error deleting annotation {filename}: {str(e)}")
            return False

    def delete_processed_annotations(self, grid_square: str, filenames: List[str],
                                    confirm: bool = True) -> Dict[str, List[str]]:
        """
        Delete multiple annotations after successful processing.

        Args:
            grid_square: Grid square identifier (e.g., "NJ05")
            filenames: List of annotation filenames to delete
            confirm: If True, requires confirmation before deleting

        Returns:
            Dict with lists of successful and failed deletions
        """
        if not filenames:
            self.logger.warning("No filenames provided for deletion")
            return {"deleted": [], "failed": []}

        # Check if we're in experiment mode and bypass confirmation
        experiment_mode = os.environ.get('EXPERIMENT_MODE') == 'true'
        if experiment_mode:
            confirm = False  # No confirmation needed in experiment mode

        if confirm and not experiment_mode:
            # In a real app, might prompt user with a dialog
            self.logger.info(f"Preparing to delete {len(filenames)} processed annotations for {grid_square}")
            user_confirm = input(f"Delete {len(filenames)} annotations? [y/N]: ").lower()
            if user_confirm != 'y':
                self.logger.info("Deletion cancelled by user")
                return {"deleted": [], "failed": filenames}

        # Use batch endpoint if available
        try:
            endpoint = f"{self.base_url}/api/delete-batch/{grid_square}"
            response = self.session.post(
                endpoint,
                headers={**self.default_headers, 'Content-Type': 'application/json'},
                json={"filenames": filenames},
                timeout=30
            )

            result = self._handle_response(response, "batch delete annotations")
            return {
                "deleted": result.get("deleted", []),
                "failed": result.get("failed", [])
            }

        except Exception as e:
            self.logger.error(f"Batch deletion failed: {str(e)}")

            # Fall back to individual deletions if batch fails
            successful = []
            failed = []

            for filename in filenames:
                if self.delete_annotation(grid_square, filename, confirm=False):
                    successful.append(filename)
                else:
                    failed.append(filename)

            self.logger.info(f"Individual deletions: {len(successful)} successful, {len(failed)} failed")
            return {
                "deleted": successful,
                "failed": failed
            }
