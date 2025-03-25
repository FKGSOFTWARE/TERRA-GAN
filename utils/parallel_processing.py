import concurrent.futures
import logging
import os
from functools import partial
from typing import List, Callable, Any, Optional, Dict
import threading

logger = logging.getLogger(__name__)

def process_images_in_parallel(image_paths: List,
                               processor_func: Callable,
                               max_workers: Optional[int] = None,
                               **kwargs) -> List:
    """
    Process multiple images in parallel using a thread pool.

    Args:
        image_paths: List of image paths to process
        processor_func: Function that processes a single image
        max_workers: Maximum number of worker threads (None = use CPU count)
        **kwargs: Additional arguments to pass to processor_func

    Returns:
        List of results from processing each image
    """
    # Use default max_workers if not specified
    if max_workers is None:
        # Use CPU count but cap at a reasonable number to avoid system overload
        max_workers = min(os.cpu_count() or 4, 8)
        logger.info(f"Using {max_workers} worker threads for parallel processing")

    # If we have additional arguments, create a partial function
    if kwargs:
        processing_func = partial(processor_func, **kwargs)
    else:
        processing_func = processor_func

    # Track successful and failed processing
    results = []
    success_count = 0
    error_count = 0
    error_lock = threading.Lock()

    # Use ThreadPoolExecutor for I/O-bound operations like image loading
    # Use ProcessPoolExecutor for CPU-bound operations (but beware of memory usage)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and create a future-to-path mapping for error reporting
        future_to_path = {executor.submit(processing_func, path): path for path in image_paths}

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                result = future.result()
                results.append(result)
                success_count += 1

                # Log progress periodically
                if success_count % 10 == 0:
                    logger.info(f"Successfully processed {success_count} images")

            except Exception as e:
                with error_lock:
                    error_count += 1
                    logger.error(f"Error processing {path}: {str(e)}")

    logger.info(f"Parallel processing complete: {success_count} succeeded, {error_count} failed")
    return results

def batch_process(items: List,
                  process_func: Callable,
                  batch_size: int = 4,
                  max_workers: Optional[int] = None,
                  **kwargs) -> List:
    """
    Process items in batches, with each batch processed in parallel.
    This is useful for memory-intensive operations where you don't want
    to load everything into memory at once.

    Args:
        items: List of items to process
        process_func: Function to process a batch of items
        batch_size: Number of items to process in each batch
        max_workers: Maximum number of worker threads
        **kwargs: Additional arguments to pass to process_func

    Returns:
        List of results from processing all batches
    """
    results = []
    total_items = len(items)

    for i in range(0, total_items, batch_size):
        batch = items[i:min(i + batch_size, total_items)]
        logger.info(f"Processing batch {i//batch_size + 1}/{(total_items + batch_size - 1)//batch_size} ({len(batch)} items)")

        batch_results = process_images_in_parallel(
            batch,
            process_func,
            max_workers=max_workers,
            **kwargs
        )

        results.extend(batch_results)

    return results
