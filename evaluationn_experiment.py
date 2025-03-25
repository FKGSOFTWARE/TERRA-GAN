# evaluation_experiment.py (updated for one-at-a-time fine tuning)

import os
import logging
import time
import subprocess
import argparse
import shutil
from pathlib import Path
import yaml
import mlflow
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("evaluation_experiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EvaluationExperiment:
    def __init__(self, config_path="config.yaml", experiment_name=None):
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.experiment_name = experiment_name or f"evaluation_experiment_{time.strftime('%Y%m%d_%H%M%S')}"

        # Define experiment-specific paths
        self.base_dir = Path("pipeline_mvp") if "pipeline_mvp" in os.getcwd() else Path(".")
        self.training_input_dir = self.base_dir / "data/raw_data/experiment_training_input"
        self.eval_input_dir = self.base_dir / "data/raw_data/experiment_human_eval_input"
        self.default_input_dir = self.base_dir / "data/raw_data/input_zip_folder"

        # Ensure experiment directories exist
        self.training_input_dir.mkdir(parents=True, exist_ok=True)
        self.eval_input_dir.mkdir(parents=True, exist_ok=True)

        # Define grids
        self.training_grids = ["NJ05", "NJ06", "NJ07", "NJ08", "NJ09"]  # Your 5 training grids
        self.evaluation_grid = "NJ10"  # Your separate evaluation grid

        # Verify zip files exist
        self.verify_zip_files()

        # Setup MLflow
        self.setup_mlflow()

    def verify_zip_files(self):
        """Verify that required zip files exist in the experiment directories"""
        missing_files = []

        # Check training grids
        for grid in self.training_grids:
            zip_path = self.training_input_dir / f"{grid}.zip"
            if not zip_path.exists():
                missing_files.append(str(zip_path))

        # Check evaluation grid
        eval_zip_path = self.eval_input_dir / f"{self.evaluation_grid}.zip"
        if not eval_zip_path.exists():
            missing_files.append(str(eval_zip_path))

        if missing_files:
            logger.error(f"Missing required zip files: {', '.join(missing_files)}")
            raise FileNotFoundError(f"Missing required zip files: {', '.join(missing_files)}")

    def setup_input_files(self, run_id, grid=None, include_eval=False):
        """
        Set up input files for the current run by copying the required
        zip files to the input_zip_folder

        Args:
            run_id: The current run ID
            grid: Optional specific grid to copy (for one-at-a-time fine-tuning)
            include_eval: Whether to include the evaluation grid
        """
        # Clear the default input folder
        for file in self.default_input_dir.glob("*.zip"):
            file.unlink()

        # Copy specific grid if provided, otherwise copy all training grids
        if grid:
            src = self.training_input_dir / f"{grid}.zip"
            dst = self.default_input_dir / f"{grid}.zip"
            shutil.copy2(src, dst)
            logger.info(f"Copied {src} to {dst}")
        else:
            # Copy all training grid files
            for grid in self.training_grids:
                src = self.training_input_dir / f"{grid}.zip"
                dst = self.default_input_dir / f"{grid}.zip"
                shutil.copy2(src, dst)
                logger.info(f"Copied {src} to {dst}")

        # Copy evaluation grid if requested
        if include_eval:
            src = self.eval_input_dir / f"{self.evaluation_grid}.zip"
            dst = self.default_input_dir / f"{self.evaluation_grid}.zip"
            shutil.copy2(src, dst)
            logger.info(f"Copied {src} to {dst}")

    def setup_mlflow(self):
        """Initialize MLflow for experiment tracking"""
        mlflow_uri = self.config["experiment_tracking"].get("tracking_uri", "file:./mlruns")
        mlflow.set_tracking_uri(mlflow_uri)

        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
            logger.info(f"Created new MLflow experiment: {self.experiment_name}")
        else:
            self.experiment_id = experiment.experiment_id
            logger.info(f"Using existing MLflow experiment: {self.experiment_name}")

    def run_training(self, run_id):
        """Run training mode on all training grids"""
        logger.info(f"Starting training for run {run_id}")

        # Setup input files for this run (all training grids)
        self.setup_input_files(run_id)

        # Start MLflow run
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=f"training_run_{run_id}") as run:
            mlflow.log_params({
                "run_id": run_id,
                "mode": "train",
                "grids": ",".join(self.training_grids)
            })

            # Log run start time
            start_time = time.time()

            # Run the training command
            try:
                result = subprocess.run(
                    ["python", "main_pipeline.py", "--mode", "train"],
                    check=True,
                    capture_output=True,
                    text=True
                )

                # Log the output
                mlflow.log_text(result.stdout, f"train_output_run_{run_id}.txt")
                if result.stderr:
                    mlflow.log_text(result.stderr, f"train_stderr_run_{run_id}.txt")

                # Log run duration
                duration = time.time() - start_time
                mlflow.log_metric("training_duration", duration)

                logger.info(f"Training completed successfully in {duration:.2f} seconds")
                return True

            except subprocess.CalledProcessError as e:
                logger.error(f"Training failed with exit code {e.returncode}")
                logger.error(f"Output: {e.stdout}")
                logger.error(f"Error: {e.stderr}")
                mlflow.log_text(e.stdout, f"train_output_run_{run_id}.txt")
                mlflow.log_text(e.stderr, f"train_error_run_{run_id}.txt")
                return False

    def run_evaluation(self, run_id, grid=None, include_eval=False):
        """
        Run evaluation mode and upload results to site

        Args:
            run_id: The current run ID
            grid: Optional specific grid to evaluate
            include_eval: Whether to include the evaluation grid
        """
        grid_label = grid if grid else "all_grids"
        logger.info(f"Starting evaluation for run {run_id}, grid {grid_label}")

        # Setup input files for this run
        self.setup_input_files(run_id, grid=grid, include_eval=include_eval)

        # Determine which grids will be evaluated
        grids_to_evaluate = []
        if grid:
            grids_to_evaluate = [grid]
        else:
            grids_to_evaluate = self.training_grids

        if include_eval:
            grids_to_evaluate.append(self.evaluation_grid)

        # Start MLflow run
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=f"evaluation_run_{run_id}_{grid_label}") as run:
            mlflow.log_params({
                "run_id": run_id,
                "mode": "evaluate",
                "grids": ",".join(grids_to_evaluate)
            })

            # Log run start time
            start_time = time.time()

            # Run evaluation
            try:
                result = subprocess.run(
                    ["python", "main_pipeline.py", "--mode", "evaluate"],
                    check=True,
                    capture_output=True,
                    text=True
                )

                # Log the output
                mlflow.log_text(result.stdout, f"evaluate_output_run_{run_id}_{grid_label}.txt")
                if result.stderr:
                    mlflow.log_text(result.stderr, f"evaluate_stderr_run_{run_id}_{grid_label}.txt")

                # Upload results for each grid
                upload_success = True
                for eval_grid in grids_to_evaluate:
                    try:
                        upload_result = subprocess.run(
                            ["python", "upload_results.py", "--grid", eval_grid],
                            check=True,
                            capture_output=True,
                            text=True
                        )
                        mlflow.log_text(upload_result.stdout, f"upload_{eval_grid}_run_{run_id}.txt")
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Upload failed for grid {eval_grid}: {e}")
                        upload_success = False

                # Log run duration
                duration = time.time() - start_time
                mlflow.log_metric("evaluation_duration", duration)

                logger.info(f"Evaluation and upload completed in {duration:.2f} seconds")
                return upload_success

            except subprocess.CalledProcessError as e:
                logger.error(f"Evaluation failed with exit code {e.returncode}")
                logger.error(f"Output: {e.stdout}")
                logger.error(f"Error: {e.stderr}")
                mlflow.log_text(e.stdout, f"evaluate_output_run_{run_id}_{grid_label}.txt")
                mlflow.log_text(e.stderr, f"evaluate_error_run_{run_id}_{grid_label}.txt")
                return False

    def wait_for_annotations(self, run_id, grid):
        """Prompt user and wait for human annotations for a specific grid"""
        logger.info(f"Waiting for human annotations for run {run_id}, grid {grid}")
        print(f"\n{'='*80}")
        print(f"RUN {run_id}: HUMAN ANNOTATION REQUIRED FOR {grid}")
        print(f"Please create annotations for grid: {grid}")
        print(f"When annotations are ready, press Enter to continue...")
        print(f"{'='*80}\n")

        input("Press Enter to continue when annotations are ready...")
        return True

    def run_human_guided(self, run_id, grid):
        """Run human-guided training with annotations for a specific grid"""
        logger.info(f"Starting human-guided training for run {run_id}, grid {grid}")

        # Setup input files for this run (just the current grid)
        self.setup_input_files(run_id, grid=grid)

        # Start MLflow run
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=f"human_guided_run_{run_id}_{grid}") as run:
            mlflow.log_params({
                "run_id": run_id,
                "mode": "human_guided_train",
                "grid": grid
            })

            # Log run start time
            start_time = time.time()

            # Run human-guided training
            try:
                result = subprocess.run(
                    ["python", "main_pipeline.py", "--mode", "human_guided_train"],
                    check=True,
                    capture_output=True,
                    text=True
                )

                # Log the output
                mlflow.log_text(result.stdout, f"human_guided_output_run_{run_id}_{grid}.txt")
                if result.stderr:
                    mlflow.log_text(result.stderr, f"human_guided_stderr_run_{run_id}_{grid}.txt")

                # Log run duration
                duration = time.time() - start_time
                mlflow.log_metric("human_guided_duration", duration)

                logger.info(f"Human-guided training completed in {duration:.2f} seconds")
                return True

            except subprocess.CalledProcessError as e:
                logger.error(f"Human-guided training failed with exit code {e.returncode}")
                logger.error(f"Output: {e.stdout}")
                logger.error(f"Error: {e.stderr}")
                mlflow.log_text(e.stdout, f"human_guided_output_run_{run_id}_{grid}.txt")
                mlflow.log_text(e.stderr, f"human_guided_error_run_{run_id}_{grid}.txt")
                return False

    def run_full_experiment(self, num_runs=5):
        """Run the full experiment with multiple training-evaluation-human guided cycles"""
        logger.info(f"Starting full experiment with {num_runs} runs")

        overall_start_time = time.time()

        for run_id in range(1, num_runs + 1):
            logger.info(f"Starting run {run_id}/{num_runs}")

            # Run training on all grids
            if not self.run_training(run_id):
                logger.error(f"Training failed for run {run_id}, stopping experiment")
                return False

            # Evaluate all grids together first
            if not self.run_evaluation(run_id):
                logger.error(f"Evaluation failed for run {run_id}, stopping experiment")
                return False

            # Process each grid one at a time for human-guided fine-tuning
            for grid in self.training_grids:
                logger.info(f"Processing grid {grid} for run {run_id}")

                # Evaluate this specific grid
                if not self.run_evaluation(run_id, grid=grid):
                    logger.error(f"Evaluation failed for run {run_id}, grid {grid}")
                    return False

                # Wait for human annotations for this grid
                if not self.wait_for_annotations(run_id, grid):
                    logger.error(f"Annotation process interrupted for run {run_id}, grid {grid}")
                    return False

                # Run human-guided training for this grid
                if not self.run_human_guided(run_id, grid):
                    logger.error(f"Human-guided training failed for run {run_id}, grid {grid}")
                    return False

                logger.info(f"Successfully completed processing for grid {grid} in run {run_id}")

            logger.info(f"Successfully completed run {run_id}/{num_runs}")

        # Log overall experiment completion
        total_duration = time.time() - overall_start_time
        logger.info(f"Full experiment completed in {total_duration:.2f} seconds")

        # Prepare final evaluation grid
        logger.info(f"Preparing final evaluation for grid {self.evaluation_grid}")
        self.prepare_final_evaluation()

        return True

    def prepare_final_evaluation(self):
        """Prepare the final evaluation grid for human annotators"""
        logger.info(f"Preparing evaluation grid {self.evaluation_grid} for human evaluation")

        # Run one final evaluation on the evaluation grid
        self.setup_input_files(run_id="final", include_eval=True)

        try:
            result = subprocess.run(
                ["python", "main_pipeline.py", "--mode", "evaluate"],
                check=True,
                capture_output=True,
                text=True
            )

            # Upload results for evaluation grid
            upload_result = subprocess.run(
                ["python", "upload_results.py", "--grid", self.evaluation_grid],
                check=True,
                capture_output=True,
                text=True
            )

            with mlflow.start_run(experiment_id=self.experiment_id, run_name=f"final_evaluation") as run:
                mlflow.log_params({
                    "grid": self.evaluation_grid,
                    "mode": "evaluate"
                })
                mlflow.log_text(result.stdout, "final_evaluation_output.txt")
                mlflow.log_text(upload_result.stdout, f"upload_{self.evaluation_grid}_final.txt")

        except subprocess.CalledProcessError as e:
            logger.error(f"Final evaluation failed: {e}")
            return False

        print(f"\n{'='*80}")
        print(f"FINAL EVALUATION GRID {self.evaluation_grid} PREPARED")
        print(f"The grid is now ready for final human evaluation")
        print(f"{'='*80}\n")

        return True

def main():
    parser = argparse.ArgumentParser(description="Automated evaluation experiment for bare earth generator")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--name", type=str, default=None, help="Name for the experiment")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs to perform")
    args = parser.parse_args()

    experiment = EvaluationExperiment(config_path=args.config, experiment_name=args.name)
    experiment.run_full_experiment(num_runs=args.runs)

if __name__ == "__main__":
    main()
