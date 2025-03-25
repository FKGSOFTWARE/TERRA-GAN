# main_pipeline.py

import json
import shutil
import logging
import argparse
import os
from pathlib import Path
import time
import yaml
import re
from mvp_gan.src.evaluate import evaluate
from mvp_gan.src.train import train
import torch
from typing import Optional, Tuple
from mvp_gan.src.models.generator import PConvUNet
from mvp_gan.src.training.human_guided_trainer import HumanGuidedTrainer
from utils.visualization.dsm_colorizer import DSMColorizer
from utils.api.portal_client import PortalClient
from utils.data_splitting import GeographicalDataHandler
from utils.visualization.split_visualizer import create_split_visualization
from utils.path_handling.path_utils import PathManager
from utils.zip_handler import process_zip_for_parent
from mvp_gan.src.models.discriminator import Discriminator
from utils.main_pipeline_mlflow import (
    setup_mlflow, cleanup_mlflow, start_run_for_mode,
    log_model_safely, log_training_completion
)


def get_base_dir():
    """Determine base directory based on environment"""
    if os.environ.get("DOCKER_ENV"):
        # When running in Docker
        return Path("/app")
    else:
        # When running locally
        return Path(__file__).resolve().parent


# Set up base directory
BASE_DIR = get_base_dir()

def load_config() -> dict:
    """Load config.yaml and return as a dict."""
    config_path = Path("config.yaml")
    if not config_path.exists():
        logger.error("No config.yaml found in the current directory!")
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Initialize path manager
path_manager = PathManager(config)

# Setup logging
if os.environ.get("DOCKER_ENV"):
    LOG_DIR = Path("/app/logs")  # Use Path object
else:
    LOG_DIR = BASE_DIR / "logs" # Use Path object

LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)

base_path = Path("/app") if os.environ.get("DOCKER_ENV") else BASE_DIR

logger = logging.getLogger(__name__)

def get_portal_client(config) -> PortalClient:
    base_url = config.get("portal", {}).get("base_url", "")
    api_key = config.get("portal", {}).get("api_key", "")
    logger.info(f"Creating PortalClient with base_url={base_url!r} and a non-empty API key={bool(api_key)}")
    return PortalClient(base_url, api_key)


# Set up directories based on environment
RAW_DATA_DIR = base_path / config["data"]["raw_dir"]
PROCESSED_DATA_DIR = base_path / config["data"]["processed_dir"]
OUTPUT_DIR = path_manager.base_output_dir
GAN_IMAGES_DIR = base_path / config["data"]["gan_images_dir"]
GAN_MASKS_DIR = base_path / config["data"]["gan_masks_dir"]
CHECKPOINT_PATH = (
    base_path
    / config["evaluation"]["checkpoint_dir"]
    / config["evaluation"]["checkpoint_file"]
)

# Create necessary directories
for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    OUTPUT_DIR,
    GAN_IMAGES_DIR,
    GAN_MASKS_DIR,
    CHECKPOINT_PATH.parent,
]:
    directory.mkdir(parents=True, exist_ok=True)


def main():
    print("===== MAIN PIPELINE STARTING =====")
    print(f"Current working directory: {os.getcwd()}")
    parser = argparse.ArgumentParser(
        description="Pipeline for GAN-based DSM inpainting."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "human_guided_train"],
        default="evaluate",
        help="Choose operation mode",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--input_model",
        type=str,
        default=None,
        help="Path to the input model (.pth file) for training or evaluation."
    )
    # Add the new grid argument
    parser.add_argument(
        "--grid",
        type=str,
        default=None,
        help="Specific grid square to process (e.g., NH70)"
    )
    args = parser.parse_args()

    try:
        # Setup MLflow tracking
        experiment_tracker, experiment_id = setup_mlflow(config, args.mode, LOG_DIR)

        if args.mode == "train":
            logging.info("Running in training mode.")
            run_training_mode(args.input_model, experiment_tracker)  # Pass the tracker

        elif args.mode == "evaluate":
            logging.info("Running in evaluation mode.")
            # Pass the grid parameter here
            run_evaluation_mode(args.input_model, experiment_tracker, args.grid)

        elif args.mode == "human_guided_train":
            logging.info("Running in human-guided training mode.")
            run_human_guided_training_mode(args.input_model, experiment_tracker)  # Pass the tracker

    except Exception as e:
        logging.exception("An error occurred in main: %s", e)

    finally:
        # Clean up MLflow resources
        cleanup_mlflow()

def cleanup_training_artifacts():
    """Clean up temporary training artifacts but preserve MLflow data."""
    logging.info("Cleaning up training artifacts...")

    cleanup_dirs = [GAN_IMAGES_DIR, GAN_MASKS_DIR]

    for path in cleanup_dirs:
        if path.exists():
            try:
                for file in path.glob("*"):
                    file.unlink()
                logging.info(f"Cleaned up {path}")
            except Exception as e:
                logging.error(f"Failed to clean up {path}: {e}")

    # Ensure directories exist for next run
    for path in cleanup_dirs:
        path.mkdir(parents=True, exist_ok=True)


def run_training_mode(input_model_path: Optional[str] = None, experiment_tracker = None) -> bool:
    """
    Run training mode with a single master model for all parent grid squares.
    Each parent (e.g., NJ05) still maintains its own data structure.

    Args:
        input_model_path: Optional path to initial model weights

    Returns:
        bool: True if training completed successfully for at least one parent
    """
    # Start MLflow run for this mode
    run_id = start_run_for_mode("train", config)
    experiment_tracker = setup_mlflow(config, "train", LOG_DIR)[0]  # Get the tracker

    input_zip_folder = RAW_DATA_DIR / "input_zip_folder"
    zip_files = list(input_zip_folder.glob("*.zip"))

    if not zip_files:
        logging.error(
            "No zip files found in input_zip_folder. Please place the files and try again."
        )
        return False

    # Initialize master model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    master_generator = PConvUNet().to(device)
    master_discriminator = Discriminator().to(device)

    # Setup optimizers for master model
    optimizer_G = torch.optim.Adam(
        master_generator.parameters(),
        lr=config['training'].get('learning_rate', 2e-4)
    )
    optimizer_D = torch.optim.Adam(
        master_discriminator.parameters(),
        lr=config['training'].get('learning_rate', 2e-4)
    )

    # Create master checkpoint path
    master_checkpoint_path = CHECKPOINT_PATH.parent / "master_checkpoint.pth"

    # Define the baseline model path
    baseline_model_path = Path("_BASELINE_MODEL/BASELINE_MODEL.pth")

    # Determine which model to load based on priority
    if input_model_path:
        # User specified a model path, use that
        model_to_load = input_model_path
        logging.info(f"Using user-specified model: {model_to_load}")
    elif master_checkpoint_path.exists():
        # Master checkpoint exists, use that for continued training
        model_to_load = str(master_checkpoint_path)
        logging.info(f"Using existing master checkpoint: {model_to_load}")
    elif baseline_model_path.exists():
        # No master checkpoint yet, but baseline exists - use for fresh training
        model_to_load = str(baseline_model_path)
        logging.info(f"Starting fresh training with baseline model: {model_to_load}")
    else:
        # No models available at all
        model_to_load = None
        logging.info("No model specified and no baseline found. Starting with random weights.")

    # Load the selected model
    if model_to_load:
        logging.info(f"Loading model weights from: {model_to_load}")
        try:
            checkpoint = torch.load(model_to_load, map_location=device)
            if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
                master_generator.load_state_dict(checkpoint['generator_state_dict'])
                # Only load discriminator if it exists in the checkpoint
                if 'discriminator_state_dict' in checkpoint:
                    master_discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

                # Only load optimizer states if continuing from master checkpoint
                if model_to_load == str(master_checkpoint_path):
                    if 'optimizer_G_state_dict' in checkpoint:
                        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
                    if 'optimizer_D_state_dict' in checkpoint:
                        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            else:
                master_generator.load_state_dict(checkpoint)
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            logging.warning("Proceeding with randomly initialized model")

    successful_parents = []
    failed_parents = []
    master_metrics = {
        "best_train_loss": float('inf'),
        "best_val_loss": float('inf'),
        "total_time": 0.0,
        "final_epoch": 0,
        "processed_grids": []
    }

    # Process each parent grid square
    for zip_file_path in zip_files:
        parent_grid = None

        try:
            # Extract and validate parent grid square
            parent_grid = path_manager.get_parent_from_zip(zip_file_path)
            logging.info(f"Starting processing for parent grid square: {parent_grid}")

            # Create complete directory structure for this parent
            paths = path_manager.create_parent_structure(parent_grid)
            logging.info(f"Created directory structure for {parent_grid}")

            # Extract and process data (no change to this part)
            logging.info(f"Processing zip file for {parent_grid}")
            processed_dir = PROCESSED_DATA_DIR / parent_grid / "raw"

            # Process zip file first
            if not process_zip_for_parent(zip_file_path, parent_grid, mode="train", config_dict=config):
                raise ValueError(f"Failed to process zip file for {parent_grid}")

            # After successful zip processing, handle geographical splits
            if not processed_dir.exists():
                raise ValueError(f"Processed directory not found after extraction: {processed_dir}")

            logging.info(f"Setting up geographical splits for {parent_grid}")
            grid_handler = GeographicalDataHandler(
                parent_grid=parent_grid,
                root_dir=PROCESSED_DATA_DIR
            )

            # Register all processed files (no change to this part)
            pattern = re.compile(r"^[a-z]{2}(\d{2})(\d{2})\.png$", re.IGNORECASE)
            tiles_registered = 0

            for tile_path in processed_dir.glob("*.png"):
                if "_mask" in tile_path.name or "_combined" in tile_path.name:
                    continue

                match = pattern.match(tile_path.name)
                if match:
                    x = int(match.group(1))
                    y = int(match.group(2))
                    grid_handler.add_tile(tile_path, x, y)
                    tiles_registered += 1
                    logging.info(f"Registered tile {tile_path.name} at ({x}, {y})")

            if tiles_registered == 0:
                raise ValueError("No valid tiles found to register")

            # Generate and apply splits
            grid_handler.generate_splits()
            grid_handler.apply_splits()
            grid_handler.save_metadata()

            # Log split distribution
            splits = grid_handler.get_split_statistics()
            for split, count in splits.items():
                logging.info(f"{parent_grid} {split}: {count} tiles")

            # Setup training directories
            train_images_dir = paths["processed"] / "train" / "images"
            train_masks_dir = paths["processed"] / "train" / "masks"
            val_images_dir = paths["processed"] / "val" / "images"
            val_masks_dir = paths["processed"] / "val" / "masks"

            # Validate training data
            if not (train_images_dir.exists() and train_masks_dir.exists()):
                raise ValueError(f"Training directories not found for {parent_grid}")

            if not any(train_images_dir.iterdir()):
                raise ValueError(f"No training images found for {parent_grid}")

            # Run training on this parent grid using the master model
            logging.info(f"Training master model on {parent_grid}")
            train_result = train(
                img_dir=train_images_dir,
                mask_dir=train_masks_dir,
                generator=master_generator,
                discriminator=master_discriminator,
                optimizer_G=optimizer_G,
                optimizer_D=optimizer_D,
                checkpoint_path=master_checkpoint_path,
                config=config,
                experiment_tracker=experiment_tracker,
                val_img_dir=val_images_dir if val_images_dir.exists() else None,
                val_mask_dir=val_masks_dir if val_masks_dir.exists() else None,
            )

            # Update master metrics
            master_metrics["best_train_loss"] = min(master_metrics["best_train_loss"], train_result["best_train_loss"])
            if "best_val_loss" in train_result and train_result["best_val_loss"] is not None:
                master_metrics["best_val_loss"] = min(master_metrics["best_val_loss"], train_result["best_val_loss"])
            master_metrics["total_time"] += train_result["total_time"]
            master_metrics["final_epoch"] = train_result["final_epoch"]
            master_metrics["processed_grids"].append(parent_grid)

            successful_parents.append(parent_grid)
            logging.info(f"Successfully completed training on {parent_grid}")

        except Exception as e:
            logging.error(f"Failed processing for {parent_grid}: {str(e)}")
            failed_parents.append(parent_grid)
            continue

    # Save final master model
    try:
        # Save master checkpoint
        checkpoint = {
            'generator_state_dict': master_generator.state_dict(),
            'discriminator_state_dict': master_discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'config': config,
            'processed_grids': successful_parents,
            'metrics': master_metrics
        }
        torch.save(checkpoint, master_checkpoint_path)

        # Save to models directory with timestamp
        models_dir = path_manager.models_dir
        models_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_name = f"master_model_{timestamp}.pth"
        final_model_path = models_dir / model_name

        shutil.copy2(master_checkpoint_path, final_model_path)
        logging.info(f"Saved master model as {model_name}")

        # Save training metrics
        metrics_path = models_dir / f"master_metrics_{timestamp}.json"
        with open(metrics_path, "w") as f:
            json.dump(master_metrics, f, indent=2)

        # Log model safely
        log_model_safely(
            model=master_generator,
            name="master_model",
            metrics=master_metrics,
            save_path=None  # Already saved above
        )

        # Log training completion
        log_training_completion(master_metrics)

    except Exception as e:
        logging.error(f"Failed to save master model: {str(e)}")

    # Final summary
    logging.info("\n=== Training Summary ===")
    logging.info(
        f"Successfully processed: {len(successful_parents)} parent grid squares"
    )
    if successful_parents:
        logging.info(f"Successful grids: {', '.join(successful_parents)}")
    if failed_parents:
        logging.info(f"Failed grids: {', '.join(failed_parents)}")

    # Cleanup temporary files
    try:
        cleanup_training_artifacts()
    except Exception as e:
        logging.error(f"Failed to cleanup training artifacts: {str(e)}")

    return len(successful_parents) > 0

def run_evaluation_mode(input_model_path, experiment_tracker = None, target_grid = None):
    """
    Run evaluation mode using the master model.
    """
    # Start MLflow run for this mode
    run_id = start_run_for_mode("evaluate", config)

    # Use master checkpoint by default, or user-specified path
    master_checkpoint_path = CHECKPOINT_PATH.parent / "master_checkpoint.pth"
    model_path = input_model_path if input_model_path else master_checkpoint_path

    if not model_path.exists():
        logging.error(f"No model available for evaluation: {model_path}")
        return False

    try:
        # Get parent grid squares from processed data
        if target_grid and path_manager._validate_parent_grid(target_grid):
            # Use the specified grid if provided and valid
            parent_grids = [target_grid]
            if not (PROCESSED_DATA_DIR / target_grid).is_dir():
                logging.error(f"Specified grid {target_grid} not found in processed data")
                return False
        else:
            # Otherwise get all parent grid squares (with a more general pattern)
            parent_grids = [
                d.name
                for d in PROCESSED_DATA_DIR.glob("N[A-Z]*")
                if d.is_dir() and path_manager._validate_parent_grid(d.name)
            ]

        if not parent_grids:
            logging.error("No processed parent grid squares found")
            return False

        # Load the master model once
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = PConvUNet().to(device)

        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
                generator.load_state_dict(checkpoint['generator_state_dict'])
                logging.info(f"Loaded generator from {model_path}")
            else:
                generator.load_state_dict(checkpoint)
                logging.info(f"Loaded generator-only model from {model_path}")
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            return False

        for parent_grid in parent_grids:
            logging.info(f"Evaluating parent grid square: {parent_grid}")

            # Get parent paths
            paths = path_manager.create_parent_structure(parent_grid)
            test_images_dir = paths["processed"] / "test" / "images"
            test_masks_dir = paths["processed"] / "test" / "masks"

            if not (test_images_dir.exists() and test_masks_dir.exists()):
                logging.warning(
                    f"Test directories not found for {parent_grid}, skipping"
                )
                continue

            # Process each test image
            successful_inpaints = 0
            for img_path in test_images_dir.glob("*.png"):
                try:
                    mask_path = test_masks_dir / f"{img_path.stem}_mask_resized.png"
                    if not mask_path.exists():
                        continue

                    # Get output paths
                    inpainted_path = (
                        paths["output_inpainted"] / f"{img_path.stem}_inpainted.png"
                    )

                    # Run evaluation with the master model
                    evaluate(img_path, mask_path, generator, inpainted_path)
                    successful_inpaints += 1

                except Exception as e:
                    logging.error(f"Error processing {img_path.name}: {str(e)}")
                    continue

            # Recolor outputs
            if successful_inpaints > 0:
                try:
                    colorizer = DSMColorizer(
                        input_dir=paths["output_inpainted"],
                        output_dir=paths["output_colored"],
                    )
                    colorizer.recolor_all()
                    logging.info(
                        f"Recolored {successful_inpaints} outputs for {parent_grid}"
                    )
                except Exception as e:
                    logging.error(
                        f"Error during recoloring for {parent_grid}: {str(e)}"
                    )

            logging.info(
                f"Completed {parent_grid}: {successful_inpaints} files processed"
            )

        return True

    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        return False


def run_human_guided_training_mode(input_model_path, experiment_tracker=None):
    """
    Runs the human-guided training mode which incorporates human annotations
    from the portal into model training. This compares machine-generated masks
    with human-created annotation masks to fine-tune the model.
    """
    # Import new modules
    from utils.human_guided_helpers import match_human_and_system_masks, fetch_annotations_for_grid, validate_dataset
    from mvp_gan.src.utils.direct_match_dataset import DirectMatchDataset

    # Start MLflow run for human-guided training
    run_id = start_run_for_mode("human_guided_train", config)

    # Initialize trainer
    trainer = HumanGuidedTrainer(config)

    # Get portal client
    portal = get_portal_client(config)

    # Load the master model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Default to master checkpoint if not specified
    if not input_model_path:
        master_checkpoint_path = CHECKPOINT_PATH.parent / "master_checkpoint.pth"
        if master_checkpoint_path.exists():
            input_model_path = str(master_checkpoint_path)

    generator = PConvUNet().to(device)
    if input_model_path:
        try:
            checkpoint = torch.load(input_model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
                generator.load_state_dict(checkpoint['generator_state_dict'])
                logging.info(f"Loaded master model from {input_model_path}")
            else:
                generator.load_state_dict(checkpoint)
                logging.info(f"Loaded model from {input_model_path}")
        except Exception as e:
            logging.error(f"Failed to load model from {input_model_path}: {str(e)}")
            return

    # Check if in experiment mode
    experiment_mode = os.environ.get('EXPERIMENT_MODE') == 'true'

    # Get grid square from args, config, or prompt user
    args = globals().get('args', None)
    grid_square = None

    if args and hasattr(args, 'grid') and args.grid:
        grid_square = args.grid
    elif experiment_mode and not grid_square:
        # In experiment mode, try to determine grid from directory structure
        try:
            input_dir = Path("data/raw_data/input_zip_folder")
            for zip_file in input_dir.glob("*.zip"):
                grid_square = zip_file.stem
                if path_manager._validate_parent_grid(grid_square):
                    logger.info(f"Automatically determined grid square: {grid_square} from input directory")
                    break
        except Exception as e:
            logger.warning(f"Could not automatically determine grid square: {e}")

    # If grid_square is still None, get from config or prompt user
    if not grid_square:
        grid_square = config.get("grid_square")
        if not grid_square and not experiment_mode:
            grid_square = input("Enter grid square identifier (e.g., NM42): ")
        if not grid_square:
            grid_square = "NM42"  # Default if empty input

    logger.info(f"Using grid square: {grid_square}")

    # Create output directory structure if needed
    human_annotation_dir = Path(f"data/output/{grid_square}/human_annotation_masks")
    human_annotation_dir.mkdir(parents=True, exist_ok=True)

    # Track annotation filenames for deletion after successful training
    processed_annotations = []

    # Check if we should fetch annotations (always yes in experiment mode)
    fetch_annotations = True
    if not experiment_mode:
        fetch_annotations = input("Fetch latest annotations from portal? [y/N]: ").lower() == "y"

    if fetch_annotations:
        logger.info(f"Fetching annotations for grid square {grid_square}")
        # Fetch annotations using the helper function
        fetch_annotations_for_grid(grid_square, portal)

    # Match human annotations with system masks
    matched_pairs = match_human_and_system_masks(grid_square)

    if not matched_pairs:
        logger.error("No matching pairs found. Cannot proceed with training.")
        return

    # Create dataset directly from matched pairs
    dataset = DirectMatchDataset(matched_pairs)

    # Validate dataset contains usable human masks
    if not validate_dataset(dataset):
        logger.error("Dataset validation failed. Cannot proceed with training.")
        return

    # Train with human feedback
    training_success = False
    try:
        checkpoint_dir = Path(config["evaluation"]["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Get the list of human annotation filenames for potential deletion
        processed_annotations = [str(pair['human_mask_path'].name) for pair in matched_pairs]

        # Run the custom trainer
        training_result = trainer.train(
            generator=generator,
            train_dataset=dataset,
            num_epochs=config["training"]["modes"]["human_guided"]["epochs"],
            checkpoint_dir=checkpoint_dir,
        )

        # Check if training was successful
        training_success = training_result.get('success', False)
        if training_success:
            logger.info("Human-guided training completed successfully")
        else:
            logger.error("Human-guided training did not complete successfully")
            return

        # Log the trained model and metrics
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        training_metrics = {
            "best_loss": training_result.get("best_loss", 0),
            "total_time": training_result.get("total_time", 0),
            "grid_square": grid_square,
            "timestamp": timestamp
        }

        log_model_safely(
            model=generator,
            name="human_guided_model",
            metrics=training_metrics
        )

        # Save as master model
        checkpoint_dir = Path(config["evaluation"]["checkpoint_dir"])
        master_checkpoint_path = checkpoint_dir / "master_checkpoint.pth"

        # Save the updated master model
        try:
            # Get existing checkpoint if it exists to preserve other components
            if master_checkpoint_path.exists():
                existing_checkpoint = torch.load(master_checkpoint_path, map_location=device)
                # Update only the generator component
                existing_checkpoint['generator_state_dict'] = generator.state_dict()
                existing_checkpoint['human_guided_training_applied'] = True
                existing_checkpoint['human_guided_training_timestamp'] = timestamp
                existing_checkpoint['processed_annotations'] = processed_annotations
                torch.save(existing_checkpoint, master_checkpoint_path)
            else:
                # Create new checkpoint with just the generator
                torch.save({
                    'generator_state_dict': generator.state_dict(),
                    'human_guided_training_applied': True,
                    'human_guided_training_timestamp': timestamp,
                    'processed_annotations': processed_annotations
                }, master_checkpoint_path)

            logging.info(f"Saved human-guided master model to {master_checkpoint_path}")

            # Also save a timestamped copy in the models directory
            models_dir = Path(config["data"]["models_dir"])
            models_dir.mkdir(parents=True, exist_ok=True)
            model_name = f"master_model_human_guided_{timestamp}.pth"
            final_model_path = models_dir / model_name

            shutil.copy2(master_checkpoint_path, final_model_path)
            logging.info(f"Saved timestamped copy of master model to {final_model_path}")

        except Exception as e:
            logging.error(f"Failed to save human-guided master model: {str(e)}")
            training_success = False

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        training_success = False
        return

    # If training was successful, offer to delete processed annotations
    if training_success and processed_annotations:
        logger.info(f"Training completed successfully with {len(processed_annotations)} annotations")

        # Check config for auto-delete setting
        auto_delete = config.get("training", {}).get("modes", {}).get(
            "human_guided", {}).get("auto_delete_annotations", False)

        # In experiment mode, always delete annotations
        experiment_mode = os.environ.get('EXPERIMENT_MODE') == 'true'
        delete_confirmed = auto_delete or experiment_mode

        # Otherwise prompt for confirmation
        if not (auto_delete or experiment_mode):
            delete_confirmation = input(
                f"Do you want to delete {len(processed_annotations)} processed annotations? [y/N]: "
            ).lower()
            delete_confirmed = delete_confirmation == 'y'

        if delete_confirmed:
            logger.info(f"Deleting {len(processed_annotations)} processed annotations")
            deletion_result = portal.delete_processed_annotations(
                grid_square=grid_square,
                filenames=processed_annotations,
                confirm=True  # Always confirm even with auto-delete for safety
            )

            deleted_count = len(deletion_result.get("deleted", []))
            failed_count = len(deletion_result.get("failed", []))

            logger.info(f"Deletion complete: {deleted_count} deleted, {failed_count} failed")

            if failed_count > 0:
                logger.warning("Some annotations could not be deleted. They may require manual cleanup.")
                for failed_item in deletion_result.get("failed", []):
                    if isinstance(failed_item, dict):
                        logger.warning(f"Failed to delete {failed_item.get('filename')}: {failed_item.get('reason')}")
                    else:
                        logger.warning(f"Failed to delete {failed_item}")
        else:
            logger.info("Annotations were not deleted")

if __name__ == "__main__":
    main()
