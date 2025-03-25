# Project Snapshot

## result_metrics_statistical_significance.py
```python
#!/usr/bin/env python3
"""
Statistical Significance Testing for Terrain Generation Experiments

This script performs statistical significance testing between experiment results:

1. Loads metrics from multiple experiment JSON files
2. Performs statistical hypothesis testing (t-tests, Mann-Whitney U, etc.)
3. Calculates effect sizes to quantify the magnitude of differences
4. Outputs comprehensive results in JSON format

Usage:
    python statistical_significance.py --experiments <exp1.json> <exp2.json> ...
                                      [--output <output.json>]
                                      [--significance-level 0.05]
                                      [--paired]
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
from scipy import stats
import warnings

# Configure warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


class StatisticalTester:
    """Statistical significance tester for comparing terrain generation experiments"""

    def __init__(self, experiments: List[Dict], experiment_names: List[str] = None,
                 output_file: str = None, alpha: float = 0.05, paired: bool = False):
        """
        Initialize the tester with experiment data

        Args:
            experiments: List of experiment data dictionaries (loaded from JSON)
            experiment_names: Optional names for experiments (defaults to exp1, exp2, etc.)
            output_file: Path to save JSON results
            alpha: Significance level (default: 0.05)
            paired: Whether to use paired tests (default: False)
        """
        self.experiments = experiments
        self.experiment_names = experiment_names or [f"exp{i+1}" for i in range(len(experiments))]

        if len(self.experiment_names) != len(self.experiments):
            raise ValueError("Number of experiment names must match number of experiments")

        self.output_file = output_file or f"stat_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        self.alpha = alpha
        self.paired = paired

        # Extract per-image metrics for statistical testing
        self.per_image_data = {}
        self.extract_per_image_data()

        # Results storage
        self.test_results = {}
        self.effect_sizes = {}
        self.descriptive_stats = None

    def extract_per_image_data(self):
        """Extract per-image metrics from each experiment for statistical testing"""
        for i, exp in enumerate(self.experiments):
            exp_name = self.experiment_names[i]

            # Check if experiment has per-image data
            if 'per_image' not in exp:
                print(f"Warning: Experiment {exp_name} does not have per-image data. Statistical tests may be limited.")
                continue

            # Extract metrics for each image
            per_image = exp['per_image']
            metrics = {}

            # Find all metrics in the first image
            if per_image:
                first_image = next(iter(per_image.values()))
                metric_names = list(first_image.keys())

                # Initialize metric lists
                for metric in metric_names:
                    metrics[metric] = []

                # Collect metric values for all images
                for img_id, img_metrics in per_image.items():
                    for metric in metric_names:
                        if metric in img_metrics:
                            # Ensure the value is numeric
                            try:
                                value = float(img_metrics[metric])
                                metrics[metric].append(value)
                            except (ValueError, TypeError):
                                continue

            self.per_image_data[exp_name] = metrics

    def calculate_descriptive_statistics(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Calculate descriptive statistics for all metrics across experiments"""
        stats_data = {}

        for exp_name, metrics in self.per_image_data.items():
            stats_data[exp_name] = {}

            for metric_name, values in metrics.items():
                if not values:
                    continue

                values_array = np.array(values)
                stats_data[exp_name][metric_name] = {
                    'mean': float(np.mean(values_array)),
                    'median': float(np.median(values_array)),
                    'std_dev': float(np.std(values_array)),
                    'min': float(np.min(values_array)),
                    'max': float(np.max(values_array)),
                    'count': int(len(values_array))
                }

        self.descriptive_stats = stats_data
        return stats_data

    def run_statistical_tests(self):
        """Run statistical tests for all metrics between all experiment pairs"""
        if len(self.experiments) < 2:
            print("Need at least 2 experiments to run statistical tests")
            return

        # Get all experiment pairs
        exp_pairs = []
        for i in range(len(self.experiments)):
            for j in range(i+1, len(self.experiments)):
                exp_pairs.append((self.experiment_names[i], self.experiment_names[j]))

        # Run tests for all metrics between all pairs
        for exp1_name, exp2_name in exp_pairs:
            if exp1_name not in self.per_image_data or exp2_name not in self.per_image_data:
                continue

            exp1_metrics = self.per_image_data[exp1_name]
            exp2_metrics = self.per_image_data[exp2_name]

            # Find common metrics
            common_metrics = set(exp1_metrics.keys()) & set(exp2_metrics.keys())

            # Initialize results for this pair
            pair_key = f"{exp1_name}_vs_{exp2_name}"
            self.test_results[pair_key] = {}
            self.effect_sizes[pair_key] = {}

            # Test each metric
            for metric in common_metrics:
                values1 = exp1_metrics[metric]
                values2 = exp2_metrics[metric]

                # Skip if insufficient data
                if len(values1) < 2 or len(values2) < 2:
                    continue

                # Check if this can be a paired test
                paired = self.paired
                if paired and len(values1) != len(values2):
                    print(f"Warning: Cannot use paired test for {metric} - unequal sample sizes")
                    paired = False

                # Run t-test
                try:
                    t_stat, p_value = self._run_t_test(values1, values2, paired)
                    self.test_results[pair_key][f"{metric}_t_test"] = {
                        'statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < self.alpha
                    }
                except Exception as e:
                    print(f"Warning: t-test failed for {metric}: {e}")

                # Run Mann-Whitney U test (non-parametric alternative)
                try:
                    u_stat, p_value = self._run_mann_whitney(values1, values2)
                    self.test_results[pair_key][f"{metric}_mann_whitney"] = {
                        'statistic': float(u_stat),
                        'p_value': float(p_value),
                        'significant': p_value < self.alpha
                    }
                except Exception as e:
                    print(f"Warning: Mann-Whitney test failed for {metric}: {e}")

                # Calculate effect size (Cohen's d)
                try:
                    effect_size = self._calculate_cohens_d(values1, values2)
                    self.effect_sizes[pair_key][metric] = float(effect_size)
                except Exception as e:
                    print(f"Warning: Effect size calculation failed for {metric}: {e}")

    def _run_t_test(self, values1, values2, paired=False):
        """Run t-test (paired or unpaired) between two sets of values"""
        if paired:
            return stats.ttest_rel(values1, values2)
        else:
            return stats.ttest_ind(values1, values2, equal_var=False)  # Welch's t-test

    def _run_mann_whitney(self, values1, values2):
        """Run Mann-Whitney U test between two sets of values"""
        return stats.mannwhitneyu(values1, values2, alternative='two-sided')

    def _calculate_cohens_d(self, values1, values2):
        """Calculate Cohen's d effect size"""
        n1, n2 = len(values1), len(values2)
        mean1, mean2 = np.mean(values1), np.mean(values2)
        var1, var2 = np.var(values1, ddof=1), np.var(values2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        # Cohen's d
        if pooled_std == 0:
            return 0  # No effect if no variation
        else:
            return (mean1 - mean2) / pooled_std

    def interpret_effect_size(self, d):
        """Interpret Cohen's d effect size"""
        d = abs(d)  # Use absolute value for interpretation
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    def create_summary_report(self):
        """Create a comprehensive summary of all statistical tests"""
        if not self.test_results:
            self.run_statistical_tests()

        # Calculate descriptive statistics if not already done
        if self.descriptive_stats is None:
            self.calculate_descriptive_statistics()

        # Create a summary of test results
        test_summary = {}
        for pair_key, tests in self.test_results.items():
            exp1, exp2 = pair_key.split('_vs_')

            if pair_key not in test_summary:
                test_summary[pair_key] = {}

            for test_key, result in tests.items():
                # Extract metric name and test type
                parts = test_key.rsplit('_', 2)
                if len(parts) >= 2:
                    metric = parts[0]
                    test_type = '_'.join(parts[1:])
                else:
                    metric = test_key
                    test_type = "unknown"

                # Get effect size
                effect_size = self.effect_sizes.get(pair_key, {}).get(metric, float('nan'))
                effect_interp = self.interpret_effect_size(effect_size) if not np.isnan(effect_size) else "unknown"

                # Get means for each experiment
                mean1 = np.mean(self.per_image_data[exp1].get(metric, [0]))
                mean2 = np.mean(self.per_image_data[exp2].get(metric, [0]))

                # Create metric entry if it doesn't exist
                if metric not in test_summary[pair_key]:
                    test_summary[pair_key][metric] = {}

                # Add test results
                test_summary[pair_key][metric][test_type] = {
                    'p_value': float(result['p_value']),
                    'significant': bool(result['significant']),
                    'statistic': float(result['statistic']),
                    'mean_1': float(mean1),
                    'mean_2': float(mean2),
                    'difference': float(mean1 - mean2),
                    'effect_size': float(effect_size) if not np.isnan(effect_size) else None,
                    'effect_interpretation': effect_interp
                }

        # Compile final results
        final_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'significance_level': self.alpha,
                'paired_tests': self.paired,
                'experiments': self.experiment_names
            },
            'descriptive_statistics': self.descriptive_stats,
            'test_results': test_summary
        }

        return final_results

    def save_results(self):
        """Save the results to a JSON file"""
        results = self.create_summary_report()

        with open(self.output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {self.output_file}")
        return self.output_file


def load_experiment_file(file_path):
    """Load an experiment results file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading experiment file {file_path}: {e}")
        return None


def extract_experiment_name(file_path):
    """Extract experiment name from file path"""
    path = Path(file_path)
    stem = path.stem

    # Remove common suffixes
    for suffix in ['_terrain_metrics', '_results', '_evaluation']:
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]

    # Extract experiment name - prefer names with EXPERIMENT prefix
    if 'EXPERIMENT' in stem:
        parts = stem.split('_')
        exp_idx = [i for i, part in enumerate(parts) if 'EXPERIMENT' in part]
        if exp_idx:
            idx = exp_idx[0]
            # Include the experiment number if available
            if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                return f"{parts[idx]}_{parts[idx + 1]}"
            return parts[idx]

    return stem


def main():
    parser = argparse.ArgumentParser(description="Statistical significance testing for terrain generation experiments")
    parser.add_argument("--experiments", nargs='+', required=True, help="Paths to experiment JSON files")
    parser.add_argument("--names", nargs='+', help="Names for experiments (default: derived from filenames)")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    parser.add_argument("--significance-level", type=float, default=0.05, help="Significance level (default: 0.05)")
    parser.add_argument("--paired", action='store_true', help="Use paired tests where possible")

    args = parser.parse_args()

    # Validate arguments
    if args.names and len(args.names) != len(args.experiments):
        parser.error("Number of names must match number of experiments")

    # Load experiment data
    experiments = []
    experiment_names = []

    for i, exp_file in enumerate(args.experiments):
        exp_data = load_experiment_file(exp_file)
        if exp_data:
            experiments.append(exp_data)

            # Use provided name or extract from filename
            if args.names and i < len(args.names):
                name = args.names[i]
            else:
                name = extract_experiment_name(exp_file)

            experiment_names.append(name)
            print(f"Loaded experiment '{name}' from {exp_file}")

    if len(experiments) < 2:
        parser.error("Need at least 2 valid experiment files for comparison")

    # Create tester and run analysis
    tester = StatisticalTester(
        experiments=experiments,
        experiment_names=experiment_names,
        output_file=args.output,
        alpha=args.significance_level,
        paired=args.paired
    )

    tester.run_statistical_tests()
    tester.save_results()


if __name__ == "__main__":
    main()
```

## run_experiment.sh
```bash
#!/bin/bash
#
# run_experiment.sh - Orchestrate the complete ML pipeline experiment
#
# This script automates the entire process:
# 1. Clean environment
# 2. Run training on all grids sequentially
# 3. Run evaluation, upload results, and human-guided training for each grid
# 4. Perform final evaluation on separate test grid

set -e  # Exit on any error

export EXPERIMENT_MODE=true

# Configuration
TRAINING_DIR="./data/raw_data/experiment_training_input"
EVAL_DIR="./data/raw_data/experiment_human_eval_input"
INPUT_DIR="./data/raw_data/input_zip_folder"
LOGS_DIR="./logs"
TIMEOUT_HOURS=48  # Human annotation timeout
TRAINING_SUCCESS=false
EVAL_SUCCESS=false
HG_SUCCESS=false
FINAL_EVAL_SUCCESS=false

# Create necessary directories
mkdir -p "$TRAINING_DIR"
mkdir -p "$EVAL_DIR"
mkdir -p "$INPUT_DIR"
mkdir -p "$LOGS_DIR"

# Check if directories are accessible
if [ ! -w "$INPUT_DIR" ]; then
    echo "ERROR: Cannot write to input directory: $INPUT_DIR"
    exit 1
fi

# Timestamp for logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOGS_DIR/experiment_run_${TIMESTAMP}.log"

# Log function
log() {
    local message="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $message" | tee -a "$LOG_FILE"
}

# Function to properly stop any running MLflow servers
ensure_mlflow_server_stopped() {
    log "Ensuring MLflow server is stopped..."

    # More aggressive approach to find and kill all MLflow processes
    pkill -9 -f "mlflow server" || true
    sleep 2

    # Try multiple approaches to ensure processes are killed
    MLFLOW_PIDS=$(pgrep -f "mlflow server" || echo "")
    if [ -n "$MLFLOW_PIDS" ]; then
        log "Found running MLflow processes: $MLFLOW_PIDS. Terminating..."
        kill -9 $MLFLOW_PIDS 2>/dev/null || true
        sleep 2  # Give processes time to terminate

        # Verify processes were killed
        REMAINING=$(pgrep -f "mlflow server" || echo "")
        if [ -n "$REMAINING" ]; then
            log "WARNING: Failed to kill MLflow processes: $REMAINING"
        fi
    else
        log "No running MLflow server found."
    fi

    # Check process using port 5000
    PORT_PROCESS=$(lsof -i :5000 -t 2>/dev/null || echo "")
    if [ -n "$PORT_PROCESS" ]; then
        log "Killing process using port 5000: $PORT_PROCESS"
        kill -9 $PORT_PROCESS 2>/dev/null || true
        sleep 1
    fi

    # Final verification
    if netstat -tuln 2>/dev/null | grep -q ":5000 "; then
        log "WARNING: Port 5000 is still in use. Waiting 10 seconds..."
        sleep 10

        # Check again
        if netstat -tuln 2>/dev/null | grep -q ":5000 "; then
            log "ERROR: Port 5000 is still in use after waiting. Proceeding anyway..."
        else
            log "Port 5000 is now free"
        fi
    else
        log "Port 5000 is now free"
    fi
}

# Add a custom wrapper for run_pipeline.sh to use our port
run_pipeline_with_port() {
    mode="$1"
    grid_square="${2:-}"
    extra_args="${3:-}"

    # Add grid parameter if provided
    if [ -n "$grid_square" ]; then
        extra_args="$extra_args --grid $grid_square"
    fi

    # Run the pipeline script
    ./run_pipeline.sh "$mode" $extra_args
    return $?
}

# Add this function to run_experiment.sh
clean_server_images() {
    grid_square="$1"
    log "Cleaning up images on server for grid square: $grid_square"

    # Use pythonanywhere_cleanup.py to remove images with force flag
    python utils/api/pythonanywhere_cleanup.py --images --grid "$grid_square" --force

    if [ $? -eq 0 ]; then
        log "Successfully cleaned up server images for $grid_square"
    else
        log "WARNING: Failed to clean up server images for $grid_square"
    fi
}

# Add this function to run_experiment.sh
clean_server_annotations() {
    grid_square="$1"
    log "Cleaning up annotations on server for grid square: $grid_square"

    # Use pythonanywhere_cleanup.py to remove annotations with force flag
    python utils/api/pythonanywhere_cleanup.py --annotations --grid "$grid_square" --force

    if [ $? -eq 0 ]; then
        log "Successfully cleaned up server annotations for $grid_square"
    else
        log "WARNING: Failed to clean up server annotations for $grid_square"
    fi
}

# Add this function to run_experiment.sh
organize_annotation_files() {
    grid_square="$1"
    log "Organizing annotation files for $grid_square"

    # Source directory from validation process
    validated_dir="data/processed_data/validated_annotations"
    # Target directory in grid output
    target_dir="data/output/$grid_square/annotations"

    # Create target directory if it doesn't exist
    mkdir -p "$target_dir"

    # Copy validated files to the grid's output directory
    if [ -d "$validated_dir" ]; then
        cp -r "$validated_dir"/* "$target_dir"/ 2>/dev/null || true
        log "Copied validated annotations to $target_dir"
    else
        log "WARNING: Validated annotations directory not found: $validated_dir"
    fi
}

# Create an alternative MLflow port option
MLFLOW_PORT=5000
if netstat -tuln 2>/dev/null | grep -q ":5000 "; then
    log "WARNING: Default port 5000 is in use. Trying alternative port 6000..."
    MLFLOW_PORT=6000
fi

# Keep track of the alternative port throughout the script
export MLFLOW_PORT

# Get experiment name from user
read -p "Enter experiment name: " EXPERIMENT_NAME
if [ -z "$EXPERIMENT_NAME" ]; then
    EXPERIMENT_NAME="ml_experiment_${TIMESTAMP}"
    log "Using default experiment name: $EXPERIMENT_NAME"
else
    log "Using experiment name: $EXPERIMENT_NAME"
fi

# Create experiment folder
EXPERIMENT_DIR="./${EXPERIMENT_NAME}"
mkdir -p "$EXPERIMENT_DIR"
log "Created experiment directory: $EXPERIMENT_DIR"

# Step 0: Clean environment
log "Starting environment cleanup..."

# More aggressive process cleanup
log "Killing any existing MLflow processes..."
ensure_mlflow_server_stopped

# Run MLflow reset
log "Running reset_mlflow.sh..."
./reset_mlflow.sh

# Clean specific directories but preserve experiment inputs
log "Cleaning output directories..."
# Clean output directory
rm -rf "./data/output"/* 2>/dev/null
# Don't clean processed data as it's needed for evaluation
# rm -rf "./data/processed_data"/* 2>/dev/null
# Clean input_zip_folder (will be populated with specific zips later)
rm -rf "$INPUT_DIR"/* 2>/dev/null
# Clean gan training directories
rm -rf "./mvp_gan/data/train/images"/* 2>/dev/null
rm -rf "./mvp_gan/data/train/masks"/* 2>/dev/null

# Clean logs directory but keep recent logs
log "Cleaning old logs..."
find "$LOGS_DIR" -name "*.log" -type f -mtime +7 -delete 2>/dev/null || true

# Clean PythonAnywhere resources
log "Running cleanup for PythonAnywhere resources..."
python utils/api/pythonanywhere_cleanup.py --annotations --images --force

log "Environment cleanup completed"

# Get list of training zip files
TRAINING_ZIPS=()
for file in "$TRAINING_DIR"/*.zip; do
    if [ -f "$file" ]; then
        TRAINING_ZIPS+=("$file")
    fi
done

if [ ${#TRAINING_ZIPS[@]} -eq 0 ]; then
    log "ERROR: No zip files found in $TRAINING_DIR"
    exit 1
fi
log "Found ${#TRAINING_ZIPS[@]} training zip files: ${TRAINING_ZIPS[@]##*/}"

# Step 1-3: Initial training on all grids
log "Starting initial training phase for all grids..."
log "Training zips to process: ${TRAINING_ZIPS[@]##*/}"

for i in "${!TRAINING_ZIPS[@]}"; do
    ZIP="${TRAINING_ZIPS[$i]}"
    GRID_NAME=$(basename "$ZIP" .zip)

    log "Processing grid $((i+1))/${#TRAINING_ZIPS[@]}: $GRID_NAME (file: $ZIP)"

    # Clear input folder completely
    log "Clearing input folder..."
    rm -rf "$INPUT_DIR"/*
    mkdir -p "$INPUT_DIR"  # Recreate it in case it was deleted
    cp "$ZIP" "$INPUT_DIR/"
    log "Copied $(basename "$ZIP") to input folder"

    # Make sure MLflow is stopped before starting a new run
    ensure_mlflow_server_stopped

    # Run training
    log "Starting training for $GRID_NAME..."
    if run_pipeline_with_port train 2>&1 | tee -a "$LOG_FILE"; then
        log "Training completed successfully for $GRID_NAME"
        TRAINING_SUCCESS=true
    else
        log "ERROR: Training failed for $GRID_NAME"
        TRAINING_SUCCESS=false
    fi

    # Ensure MLflow is stopped after the run
    ensure_mlflow_server_stopped

    # Save a copy of the model after each training
    MODELS_DIR="./data/output/models"
    if [ -d "$MODELS_DIR" ]; then
        LATEST_MODEL=$(ls -1t "$MODELS_DIR"/master_model_*.pth 2>/dev/null | head -n 1)
        if [ -n "$LATEST_MODEL" ]; then
            MODEL_COPY="$EXPERIMENT_DIR/${GRID_NAME}_initial_training.pth"
            cp "$LATEST_MODEL" "$MODEL_COPY"
            log "Saved model copy to $MODEL_COPY"
        else
            log "WARNING: No model found to copy after training"
        fi
    else
        log "WARNING: Models directory not found: $MODELS_DIR"
    fi
done

log "Initial training phase completed for all grids"

# Steps 4-8: Evaluation, human annotation, and human-guided training for each grid
log "Starting evaluation and human-guided training phase..."

for i in "${!TRAINING_ZIPS[@]}"; do
    ZIP="${TRAINING_ZIPS[$i]}"
    GRID_NAME=$(basename "$ZIP" .zip)

    log "Processing grid $((i+1))/${#TRAINING_ZIPS[@]} for evaluation and human-guided training: $GRID_NAME"

    # Clear input folder completely
    log "Clearing input folder..."
    rm -rf "$INPUT_DIR"/*
    mkdir -p "$INPUT_DIR"  # Recreate it in case it was deleted
    cp "$ZIP" "$INPUT_DIR/"
    log "Copied $(basename "$ZIP") to input folder"

    # Ensure we preserve the processed_data for evaluation
    log "Ensuring processed data is available for evaluation..."
    GRID_PROCESSED_DIR="./data/processed_data/$GRID_NAME"
    if [ ! -d "$GRID_PROCESSED_DIR" ] || [ -z "$(ls -A "$GRID_PROCESSED_DIR" 2>/dev/null)" ]; then
        log "WARNING: Processed data for $GRID_NAME is missing or empty. Re-processing zip file..."
        # Need to run training first to ensure proper processing
        log "Running training with --process-only flag..."
        run_pipeline_with_port train "--process-only" 2>&1 | tee -a "$LOG_FILE" || true

        # Check if processed data is now available
        if [ ! -d "$GRID_PROCESSED_DIR" ] || [ -z "$(ls -A "$GRID_PROCESSED_DIR" 2>/dev/null)" ]; then
            log "CRITICAL: Still missing processed data after re-processing. Trying full training instead..."
            run_pipeline_with_port train 2>&1 | tee -a "$LOG_FILE" || true
        fi
    fi

    # Verify we have the necessary structure
    if [ ! -d "$GRID_PROCESSED_DIR/test" ]; then
        log "WARNING: Missing test directory structure. Creating it..."
        mkdir -p "$GRID_PROCESSED_DIR/test/images"
        mkdir -p "$GRID_PROCESSED_DIR/test/masks"
    fi

    # Ensure MLflow is stopped before starting a new run
    ensure_mlflow_server_stopped

    clean_server_images "$GRID_NAME"

    # Run evaluation
    log "Starting evaluation for $GRID_NAME..."
    if run_pipeline_with_port evaluate "$GRID_NAME" 2>&1 | tee -a "$LOG_FILE"; then
        log "Evaluation completed successfully for $GRID_NAME"
        EVAL_SUCCESS=true
    else
        log "ERROR: Evaluation failed for $GRID_NAME"
        EVAL_SUCCESS=false
    fi

    # Ensure MLflow is stopped after the run
    ensure_mlflow_server_stopped

    # Upload results
    log "Uploading results for $GRID_NAME..."
    if python upload_results.py --grid "$GRID_NAME" 2>&1 | tee -a "$LOG_FILE"; then
        log "Results uploaded successfully for $GRID_NAME"
    else
        log "WARNING: Results upload may have failed for $GRID_NAME"
    fi

    # Wait for human annotations
    log "Waiting for human annotations for $GRID_NAME..."
    echo "=================================================================="
    echo "HUMAN ANNOTATIONS REQUIRED FOR $GRID_NAME"
    echo "Please complete annotations for $GRID_NAME and press Enter when ready"
    echo "Timeout: $TIMEOUT_HOURS hours"
    echo "=================================================================="

    read -t $((TIMEOUT_HOURS * 3600)) -p "Press Enter when annotations are ready..."
    READ_STATUS=$?

    if [ $READ_STATUS -ne 0 ]; then
        log "TIMEOUT: Human annotation period expired for $GRID_NAME"
        # Continue to next grid if timeout occurs
        continue
    fi

    log "Human annotations confirmed for $GRID_NAME"

    # Ensure MLflow is stopped before starting a new run
    ensure_mlflow_server_stopped

    # Run human-guided training
    log "Starting human-guided training for $GRID_NAME..."
    if run_pipeline_with_port human_guided "$GRID_NAME" 2>&1 | tee -a "$LOG_FILE"; then
        log "Human-guided training completed successfully for $GRID_NAME"
        HG_SUCCESS=true
    else
        log "ERROR: Human-guided training failed for $GRID_NAME"
        HG_SUCCESS=false
    fi

    # Ensure MLflow is stopped after the run
    ensure_mlflow_server_stopped

    if [ "$HG_SUCCESS" = "true" ]; then
        organize_annotation_files "$GRID_NAME"
        clean_server_annotations "$GRID_NAME"
        log "Post-training cleanup completed for $GRID_NAME"
    fi

    # Save a copy of the model after human-guided training
    MODELS_DIR="./data/output/models"
    if [ -d "$MODELS_DIR" ]; then
        LATEST_MODEL=$(ls -1t "$MODELS_DIR"/master_model_human_guided_*.pth 2>/dev/null | head -n 1)
        if [ -n "$LATEST_MODEL" ]; then
            MODEL_COPY="$EXPERIMENT_DIR/${GRID_NAME}_human_guided.pth"
            cp "$LATEST_MODEL" "$MODEL_COPY"
            log "Saved human-guided model copy to $MODEL_COPY"
        else
            log "WARNING: No human-guided model found to copy"
        fi
    else
        log "WARNING: Models directory not found: $MODELS_DIR"
    fi
done

log "Evaluation and human-guided training phase completed for all grids"

# Step 9-10: Final evaluation on separate test grid
log "Starting final evaluation on test grid..."

# A more robust way to find the NS83 test grid
TEST_ZIP=""
if [ -f "$EVAL_DIR/NS83.zip" ]; then
    TEST_ZIP="$EVAL_DIR/NS83.zip"
else
    # Try a case-insensitive search
    for file in "$EVAL_DIR"/*.zip; do
        FILENAME=$(basename "$file" | tr '[:upper:]' '[:lower:]')
        if [[ "$FILENAME" == "ns83.zip" ]]; then
            TEST_ZIP="$file"
            break
        fi
    done
fi

if [ -z "$TEST_ZIP" ]; then
    log "ERROR: Test grid NS83.zip not found in $EVAL_DIR"
    exit 1
fi

# Clear input folder completely
log "Clearing input folder for final evaluation..."
rm -rf "$INPUT_DIR"/*
mkdir -p "$INPUT_DIR"  # Recreate it in case it was deleted
cp "$TEST_ZIP" "$INPUT_DIR/"
log "Copied $(basename "$TEST_ZIP") to input folder"

# Process test grid NS83 from zip file
log "Processing test grid NS83 from zip file..."
# First use the zip handler to extract and process files
python -c "
import yaml
import logging
from pathlib import Path
from utils.zip_handler import process_zip_for_parent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Define paths
zip_file_path = Path('$INPUT_DIR/NS83.zip')
parent_grid = 'NS83'

# Process the zip file
logger.info(f'Processing test grid: {parent_grid}')
success = process_zip_for_parent(zip_file_path, parent_grid, mode='evaluate', config_dict=config)

if success:
    logger.info(f'Successfully processed {parent_grid}')
else:
    logger.error(f'Failed to process {parent_grid}')
" 2>&1 | tee -a "$LOG_FILE"

PROCESS_STATUS=$?
if [ $PROCESS_STATUS -ne 0 ]; then
    log "ERROR: Failed to process test grid NS83"
    exit 1
else
    log "Successfully processed test grid NS83"
fi

# Now prepare the test directories with our special processor
log "Preparing NS83 test directories for evaluation..."
python utils/final_eval_grid_processor.py 2>&1 | tee -a "$LOG_FILE"

PREP_STATUS=$?
if [ $PREP_STATUS -ne 0 ]; then
    log "ERROR: Failed to prepare test directories for NS83"
    exit 1
else
    log "Successfully prepared test directories for NS83"
fi

# Run the processing script
python process_test_grid.py 2>&1 | tee -a "$LOG_FILE"
PROCESS_STATUS=$?
if [ $PROCESS_STATUS -ne 0 ]; then
    log "ERROR: Failed to process test grid NS83"
    exit 1
else
    log "Successfully processed test grid NS83"
fi

# Clean up the temporary script
rm -f process_test_grid.py

# Ensure MLflow is stopped before starting a new run
ensure_mlflow_server_stopped

# Run final evaluation
log "Running final evaluation on NS83..."
if run_pipeline_with_port evaluate 2>&1 | tee -a "$LOG_FILE"; then
    log "Final evaluation completed successfully for NS83"
    FINAL_EVAL_SUCCESS=true
else
    log "ERROR: Final evaluation failed for NS83"
    FINAL_EVAL_SUCCESS=false
fi

# Ensure MLflow is stopped after the run
ensure_mlflow_server_stopped

# Upload final results
log "Uploading final results for NS83..."
if python upload_results.py --grid "NS83" 2>&1 | tee -a "$LOG_FILE"; then
    log "Final results uploaded successfully for NS83"
else
    log "WARNING: Final results upload may have failed for NS83"
fi

# Create final results directory
FINAL_RESULTS_DIR="$EXPERIMENT_DIR/final_results"
mkdir -p "$FINAL_RESULTS_DIR"

# Copy output files to final results directory
log "Copying final results to $FINAL_RESULTS_DIR..."
if [ -d "./data/output/NS83" ]; then
    cp -r ./data/output/NS83/* "$FINAL_RESULTS_DIR/" 2>/dev/null || true
    log "Copied output files to final results folder"
else
    log "WARNING: No output directory found for NS83"
fi

# Copy latest metrics if available
LATEST_METRICS=$(ls -1t ./data/output/models/master_metrics_*.json 2>/dev/null | head -n 1)
if [ -n "$LATEST_METRICS" ]; then
    cp "$LATEST_METRICS" "$FINAL_RESULTS_DIR/"
    log "Copied metrics to final results folder"
else
    log "WARNING: No metrics file found to copy"
fi

# Clean up at the end of the script
log "Experiment completed!"
log "Final results available in: $FINAL_RESULTS_DIR"

# Final summary
OVERALL_SUCCESS="SUCCESS"
if [ "$TRAINING_SUCCESS" != "true" ]; then
    OVERALL_SUCCESS="WITH ERRORS (training phase)"
fi

if [ "$FINAL_EVAL_SUCCESS" != "true" ]; then
    OVERALL_SUCCESS="WITH ERRORS (evaluation phase)"
fi

echo "=================================================================="
echo "EXPERIMENT COMPLETED: $EXPERIMENT_NAME - $OVERALL_SUCCESS"
echo "Final results are available in: $FINAL_RESULTS_DIR"
echo "=================================================================="

# Ensure MLflow is completely stopped at the end
ensure_mlflow_server_stopped
```

## upload_results.py
```python
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


def upload_results(grid_square: str = None, retry_count: int = 3, chunk_size: int = 2):
    """
    Upload processed "colored" results to the portal.

    Now expects output in data/output/<GridSquare>/colored/.
    If --grid is not provided, we pick the first available grid folder with a "colored" subdir.

    Args:
        grid_square: Grid square identifier (e.g., "NJ05")
        retry_count: Number of retry attempts if upload fails
        chunk_size: Number of files to upload in each batch
    """
    config = load_config()

    # Initialize portal client
    from utils.api.portal_client import PortalClient
    client = PortalClient(
        base_url=config['portal']['base_url'],
        api_key=config['portal']['api_key']
    )

    # Base output directory (e.g. data/output)
    output_dir = Path(config['data']['output_dir'])

    # If no grid square specified, scan data/output/ for subdirectories
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

    # Construct the new path: data/output/<GridSquare>/colored
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

    args = parser.parse_args()
    upload_results(args.grid, args.retry, args.chunk_size)


if __name__ == "__main__":
    main()
```

## road_processor.py
```python
import cv2
import numpy as np
import logging

class RoadProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def detect(self, image: np.ndarray) -> np.ndarray:
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)

            # Apply bilateral filter to reduce noise while preserving edges
            blurred = cv2.bilateralFilter(enhanced, 9, 75, 75)

            # Edge detection with higher thresholds
            edges = cv2.Canny(blurred,
                            self.config['canny_low'],
                            self.config['canny_high'])

            # Create mask excluding vegetation areas
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv,
                                   np.array([35, 50, 50]),
                                   np.array([85, 255, 255]))
            edges = cv2.bitwise_and(edges, cv2.bitwise_not(green_mask))

            # Line detection with strict parameters
            lines = cv2.HoughLinesP(edges,
                                  rho=1,
                                  theta=np.pi/180,
                                  threshold=self.config['hough_threshold'],
                                  minLineLength=self.config['hough_min_length'],
                                  maxLineGap=self.config['hough_max_gap'])

            # Create road mask
            mask = np.zeros_like(gray)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.abs(np.degrees(np.arctan2(y2-y1, x2-x1)))

                    # Filter lines by angle (near horizontal or vertical)
                    if (angle < 20 or abs(angle - 90) < 20 or
                        abs(angle - 180) < 20):
                        cv2.line(mask, (x1, y1), (x2, y2), 255,
                               self.config['line_thickness'])

            # Clean up mask
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            return mask

        except Exception as e:
            self.logger.error(f"Error in road detection: {str(e)}")
            return np.zeros_like(gray)
```

## vegetation_processor.py
```python
import cv2
import numpy as np
import logging

class VegetationProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def detect(self, image: np.ndarray) -> np.ndarray:
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Extract individual channels
            h, s, v = cv2.split(hsv)

            # Create vegetation mask using color thresholds
            mask = cv2.inRange(hsv,
                             np.array([30, 40, 40]),  # Lower green bound
                             np.array([90, 255, 255]))  # Upper green bound

            # Calculate ExG (Excess Green Index)
            b, g, r = cv2.split(image)
            g = g.astype(float)
            r = r.astype(float)
            b = b.astype(float)

            exg = 2 * g - r - b
            exg_normalized = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX)
            _, exg_mask = cv2.threshold(exg_normalized.astype(np.uint8),
                                      127, 255, cv2.THRESH_BINARY)

            # Combine masks
            combined_mask = cv2.bitwise_and(mask, exg_mask)

            # Remove small areas and fill holes
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Find and filter contours by area
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)

            clean_mask = np.zeros_like(mask)
            for contour in contours:
                if cv2.contourArea(contour) > self.config['min_area']:
                    cv2.drawContours(clean_mask, [contour], -1, 255, -1)

            return clean_mask

        except Exception as e:
            self.logger.error(f"Error in vegetation detection: {str(e)}")
            return np.zeros_like(image[:,:,0])
```

## building_processor.py
```python
import cv2
import numpy as np
import logging

class BuildingProcessor:
    """Temporary simplified processor for building detection."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Currently returns an empty mask.
        """
        # Create empty mask matching image dimensions
        if len(image.shape) == 3:
            height, width = image.shape[:2]
        else:
            height, width = image.shape

        return np.zeros((height, width), dtype=np.uint8)
```

## field_processor.py
```python
import cv2
import numpy as np
import logging

class FieldProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def detect(self, image: np.ndarray) -> np.ndarray:
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:,:,0]

            # Apply bilateral filter to smooth while preserving edges
            smoothed = cv2.bilateralFilter(l_channel, 9, 75, 75)

            # Adaptive thresholding to identify potential field areas
            binary = cv2.adaptiveThreshold(smoothed, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 3)

            # Remove vegetation areas
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            veg_mask = cv2.inRange(hsv,
                                 np.array([35, 50, 50]),
                                 np.array([85, 255, 255]))
            binary = cv2.bitwise_and(binary, cv2.bitwise_not(veg_mask))

            # Clean up mask
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Find and filter contours by area
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)

            # Create clean mask with only large areas
            clean_mask = np.zeros_like(mask)
            for contour in contours:
                if cv2.contourArea(contour) > self.config['min_area']:
                    cv2.drawContours(clean_mask, [contour], -1, 255, -1)

            return clean_mask

        except Exception as e:
            self.logger.error(f"Error in field detection: {str(e)}")
            return np.zeros_like(image[:,:,0])
```

## core.py
```python
import cv2
import numpy as np
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum
import yaml

from .processors.road_processor import RoadProcessor
from .processors.building_processor import BuildingProcessor
from .processors.vegetation_processor import VegetationProcessor
from .processors.field_processor import FieldProcessor

# Configure logging
logger = logging.getLogger(__name__)

class MaskType(Enum):
    """Enum for different types of masks"""
    ROADS = "roads"
    BUILDINGS = "buildings"
    VEGETATION = "vegetation"
    FIELDS = "fields"
    COMBINED = "combined"

@dataclass
class ProcessingParams:
    """Parameters for mask processing"""
    # General parameters
    blur_kernel_size: Tuple[int, int] = (5, 5)
    blur_sigma: int = 0

    # Road detection
    road_canny_low: int = 50
    road_canny_high: int = 150
    road_hough_threshold: int = 50
    road_hough_min_length: int = 50
    road_hough_max_gap: int = 10
    road_dilation_kernel: int = 3

    # Building detection
    building_area_threshold: int = 500
    building_rect_tolerance: float = 0.2
    building_edge_threshold: int = 30

    # Vegetation detection
    veg_green_threshold: float = 1.3
    veg_saturation_threshold: int = 50
    veg_value_threshold: int = 50

    # Field detection
    field_min_area: int = 5000
    field_texture_threshold: float = 0.4
    field_homogeneity_threshold: float = 0.7

class MaskProcessor:
    """Main class for processing aerial imagery into semantic masks"""

    def __init__(self, config: dict):
        """
        Initialize the mask processor with configuration.

        Args:
            config: Dictionary containing configuration for all processors
        """
        self.config = config
        self.params = ProcessingParams()

        # Initialize individual processors
        self.road_processor = RoadProcessor(config['roads'])
        self.building_processor = BuildingProcessor(config['buildings'])
        self.vegetation_processor = VegetationProcessor(config['vegetation'])
        self.field_processor = FieldProcessor(config['fields'])

        logger.info("Initialized MaskProcessor with all components")

    def combine_masks(self, masks: Dict[MaskType, np.ndarray], invert_output: bool = True) -> np.ndarray:
        """
        Combine individual masks into a single binary mask.

        Args:
            masks: Dictionary of mask type to mask array
            invert_output: If True, invert the final mask so white areas (255)
                        are regions to inpaint and black areas (0) are preserved

        Returns:
            Combined binary mask
        """
        try:
            # Get reference dimensions from first mask
            reference_mask = next(iter(masks.values()))
            height, width = reference_mask.shape[:2]

            # Initialize combined mask
            combined = np.zeros((height, width), dtype=np.uint8)

            # Resize all masks to match reference dimensions
            resized_masks = {}
            for mask_type, mask in masks.items():
                if mask.shape[:2] != (height, width):
                    resized_mask = cv2.resize(mask, (width, height),
                                        interpolation=cv2.INTER_NEAREST)
                else:
                    resized_mask = mask
                resized_masks[mask_type] = resized_mask

            # Combine masks with priority order
            priority_order = [
                MaskType.BUILDINGS,
                MaskType.ROADS,
                MaskType.VEGETATION,
                MaskType.FIELDS
            ]

            for mask_type in priority_order:
                if mask_type in resized_masks:
                    mask = resized_masks[mask_type]
                    # Ensure the mask is binary
                    binary_mask = (mask > 127).astype(np.uint8) * 255
                    # Combine using bitwise OR
                    combined = cv2.bitwise_or(combined, binary_mask)

            # Invert the mask if requested (so white becomes the inpainting area)
            if invert_output:
                combined = 255 - combined

            return combined

        except Exception as e:
            logger.error(f"Error in mask combination: {str(e)}")
            return np.zeros((height, width), dtype=np.uint8)

    def process_image(self, image_path: Path) -> Dict[MaskType, np.ndarray]:
        """
        Process an aerial image and generate all masks.

        Args:
            image_path: Path to the input image

        Returns:
            Dictionary containing all generated masks

        Raises:
            ValueError: If image cannot be read
            Exception: For other processing errors
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")

            logger.info(f"Processing image: {image_path}")

            # Generate individual masks
            masks = {
                MaskType.ROADS: self.road_processor.detect(image),
                MaskType.BUILDINGS: self.building_processor.detect(image),
                MaskType.VEGETATION: self.vegetation_processor.detect(image),
                MaskType.FIELDS: self.field_processor.detect(image)
            }

            # Generate combined mask
            masks[MaskType.COMBINED] = self.combine_masks(masks)

            logger.info("Successfully generated all masks")
            return masks

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            raise

def create_processor(config_path: str = "config.yaml") -> MaskProcessor:
    """
    Factory function to create a MaskProcessor instance.

    Args:
        config_path: Path to configuration file

    Returns:
        Configured MaskProcessor instance
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return MaskProcessor(config['mask_processing'])
    except Exception as e:
        logger.error(f"Error creating MaskProcessor: {str(e)}")
        raise

def downscale_and_match_mask(mask, dem_image_path):
    """
    Downscale and match mask dimensions to DEM image.

    Args:
        mask: Either a numpy array or a Path to the mask image
        dem_image_path: Path to the DEM image
    """
    logging.info(f"Downscaling mask to match DEM {dem_image_path}")

    # Handle mask input
    if isinstance(mask, (str, Path)):
        mask_array = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        if mask_array is None:
            raise ValueError(f"Unable to read the mask image at {mask}")
    else:
        mask_array = mask  # Assume it's a numpy array

    # Read DEM image
    dem_image = cv2.imread(str(dem_image_path), cv2.IMREAD_GRAYSCALE)
    if dem_image is None:
        raise ValueError(f"Unable to read the DEM image at {dem_image_path}")

    # Resize mask
    dem_height, dem_width = dem_image.shape[:2]
    mask_resized = cv2.resize(mask_array, (dem_width, dem_height), interpolation=cv2.INTER_NEAREST)

    # Binarize the mask *after* resizing
    mask_resized = (mask_resized > 127).astype(np.uint8) * 255

    # Create output path and save
    if isinstance(dem_image_path, str):
        dem_image_path = Path(dem_image_path)
    output_path = dem_image_path.parent / f"{dem_image_path.stem}_mask_resized.png"
    cv2.imwrite(str(output_path), mask_resized)
    return output_path

def process_parent_images_in_parallel(self, image_paths: List[Path], invert_masks: bool = True) -> Dict:
    """
    Process multiple images in parallel to generate all masks.

    Args:
        image_paths: List of paths to the input images
        invert_masks: If True, invert final masks so white areas are for inpainting

    Returns:
        Dictionary mapping image paths to their generated masks
    """
    from utils.parallel_processing import process_images_in_parallel

    # Use process_images_in_parallel to process all images
    # Pass invert_masks parameter to each processing task
    results = process_images_in_parallel(
        image_paths,
        lambda path: self.process_image(path, invert_masks=invert_masks)
    )

    # Organize results by image path
    mask_results = {}
    for result in results:
        if result:  # Skip None results (errors)
            image_path, masks = result
            mask_results[image_path] = masks

    return mask_results

def process_image(self, image_path: Path, invert_masks: bool = True) -> Dict[MaskType, np.ndarray]:
    """
    Process an aerial image and generate all masks.

    Args:
        image_path: Path to the input image
        invert_masks: If True, invert the final combined mask so white areas are for inpainting

    Returns:
        Dictionary containing all generated masks

    Raises:
        ValueError: If image cannot be read
        Exception: For other processing errors
    """
    try:
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")

        logger.info(f"Processing image: {image_path}")

        # Generate individual masks
        masks = {
            MaskType.ROADS: self.road_processor.detect(image),
            MaskType.BUILDINGS: self.building_processor.detect(image),
            MaskType.VEGETATION: self.vegetation_processor.detect(image),
            MaskType.FIELDS: self.field_processor.detect(image)
        }

        # Generate combined mask with option to invert
        masks[MaskType.COMBINED] = self.combine_masks(masks, invert_output=invert_masks)

        logger.info("Successfully generated all masks")
        return masks

    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        raise
```

## annotation_validator.py
```python
import cv2
import torch
from pathlib import Path
import logging
import shutil
from torchvision import transforms
from PIL import Image
import numpy as np
import json

logger = logging.getLogger(__name__)

class AnnotationValidator:
    """
    Validates and filters annotations for consistent sizes and binarizes masks.
    """
    def __init__(self,
                 target_size=(512, 512),
                 max_size_difference_percent=10,
                 resize_mode='strict'):
        """
        Initialize the annotation validator.

        Args:
            target_size: Tuple (height, width) of the expected image size
            max_size_difference_percent: Maximum allowed percentage difference
            resize_mode: 'strict' to skip mismatched images, 'resize' to force resize all
        """
        self.target_size = target_size
        self.max_size_difference = max_size_difference_percent / 100
        self.resize_mode = resize_mode
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])

    def check_image_size(self, image_path):
        """
        Check if an image fits within the allowed size range.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple (is_valid, dimensions) - Boolean validity and actual dimensions
        """
        try:
            img = Image.open(image_path)
            width, height = img.size

            # Check if dimensions are exactly the target size
            if (height, width) == self.target_size:
                return True, (height, width)

            # Check if dimensions are within the allowed difference
            h_target, w_target = self.target_size
            h_diff = abs(height - h_target) / h_target
            w_diff = abs(width - w_target) / w_target

            is_valid = h_diff <= self.max_size_difference and w_diff <= self.max_size_difference
            return is_valid, (height, width)

        except Exception as e:
            logger.error(f"Error checking image size for {image_path}: {str(e)}")
            return False, None

    def validate_and_filter_pairs(self, human_masks, system_masks, output_dir):
        """
        Validate annotation pairs, binarize masks, and copy to output directory.
        """
        # Create output directories
        img_dir = Path(output_dir) / "images"
        mask_dir = Path(output_dir) / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            "total_pairs": 0,
            "valid_pairs": 0,
            "invalid_human": 0,
            "invalid_system": 0,
            "resized_pairs": 0,
            "skipped_pairs": 0,
            "size_mismatches": [],
            "non_binary_human": 0,  # NEW: Count non-binary human masks
            "non_binary_system": 0, # NEW: Count non-binary system masks
            "file_mapping": {}      # NEW: Map validation indices to original filenames
        }

        # Create mapping from filenames to human masks
        human_mask_map = {}
        for mask_file in human_masks:
            # Extract the identifier part (e.g., "nj0957")
            parts = mask_file.stem.split('_')
            if len(parts) >= 2:
                base_name = parts[1]  # Should be "nj0957"
                human_mask_map[base_name] = mask_file

        # Create mapping from filenames to system masks
        system_mask_map = {}
        for mask_file in system_masks:
            # Extract base name from e.g., "nj0957_mask_resized.png"
            base_name = mask_file.stem.replace("_mask_resized", "")
            system_mask_map[base_name] = mask_file

        logger.info(f"Found {len(human_mask_map)} human annotations and {len(system_mask_map)} system masks")

        # Find matching pairs and validate them
        valid_pairs = 0
        for base_name, human_mask in human_mask_map.items():
            if base_name in system_mask_map:
                stats["total_pairs"] += 1
                system_mask = system_mask_map[base_name]

                # Check both masks for valid sizes
                human_valid, human_size = self.check_image_size(human_mask)
                system_valid, system_size = self.check_image_size(system_mask)

                # Record size information for invalid images
                if not human_valid or not system_valid:
                    stats["size_mismatches"].append({
                        "base_name": base_name,
                        "human_size": human_size,
                        "system_size": system_size,
                        "target_size": self.target_size
                    })

                if not human_valid:
                    stats["invalid_human"] += 1
                    logger.warning(f"Human annotation '{base_name}' has invalid dimensions {human_size}")

                if not system_valid:
                    stats["invalid_system"] += 1
                    logger.warning(f"System mask '{base_name}' has invalid dimensions {system_size}")

                # Process based on resize mode
                if self.resize_mode == 'strict' and (not human_valid or not system_valid):
                    logger.info(f"Skipping annotation pair for '{base_name}' due to size mismatch")
                    stats["skipped_pairs"] += 1
                    continue
                elif self.resize_mode == 'resize':
                    # Resize both to target size
                    img_out_path = img_dir / f"{valid_pairs:04d}.png"
                    mask_out_path = mask_dir / f"{valid_pairs:04d}.png"

                    try:
                        # --- Process Human Mask ---
                        img = Image.open(human_mask).convert('L')  # Ensure grayscale
                        if self.resize_mode == 'resize':
                            img = img.resize(self.target_size[::-1], Image.BILINEAR)  # PIL uses (width, height)

                        # Binarize the human mask *after* resizing (if resizing)
                        img_array = np.array(img)
                        if not np.isin(img_array, [0, 255]).all():
                            stats["non_binary_human"] += 1
                            logger.warning(f"Human mask {human_mask} is not binary. Binarizing...")
                            img_array = (img_array > 127).astype(np.uint8) * 255  # Threshold at 127

                        Image.fromarray(img_array).save(img_out_path)  # Save as 8-bit grayscale PNG

                        # --- Process System Mask ---
                        mask = Image.open(system_mask).convert('L')
                        if self.resize_mode == 'resize':
                            mask = mask.resize(self.target_size[::-1], Image.NEAREST)  # Use NEAREST for masks!

                        # Binarize the system mask after resizing
                        mask_array = np.array(mask)
                        if not np.isin(mask_array, [0, 255]).all():
                            stats["non_binary_system"] += 1
                            logger.warning(f"System mask {system_mask} is not binary. Binarizing...")
                            mask_array = (mask_array > 127).astype(np.uint8) * 255  # Threshold at 127

                        Image.fromarray(mask_array).save(mask_out_path)

                        # Store mapping for tracking
                        stats["file_mapping"][str(valid_pairs)] = str(human_mask)

                        if self.resize_mode == 'resize':
                            stats["resized_pairs"] += 1
                        valid_pairs += 1

                    except Exception as e:
                        logger.error(f"Error processing annotation pair for '{base_name}': {str(e)}")
                        continue
                else:
                    # Both are valid and we're using strict mode
                    img_out_path = img_dir / f"{valid_pairs:04d}.png"
                    mask_out_path = mask_dir / f"{valid_pairs:04d}.png"

                    try:
                        # Process human mask (copy and binarize)
                        img = Image.open(human_mask).convert('L')
                        img_array = np.array(img)
                        if not np.isin(img_array, [0, 255]).all():
                            stats["non_binary_human"] += 1
                            logger.warning(f"Human mask {human_mask} is not binary. Binarizing...")
                            img_array = (img_array > 127).astype(np.uint8) * 255
                        Image.fromarray(img_array).save(img_out_path)

                        # Process system mask (copy and binarize)
                        mask = Image.open(system_mask).convert('L')
                        mask_array = np.array(mask)
                        if not np.isin(mask_array, [0, 255]).all():
                            stats["non_binary_system"] += 1
                            logger.warning(f"System mask {system_mask} is not binary. Binarizing...")
                            mask_array = (mask_array > 127).astype(np.uint8) * 255
                        Image.fromarray(mask_array).save(mask_out_path)

                        # Store mapping for tracking
                        stats["file_mapping"][str(valid_pairs)] = str(human_mask)

                        valid_pairs += 1
                    except Exception as e:
                        logger.error(f"Error copying annotation pair for '{base_name}': {str(e)}")
                        continue

                stats["valid_pairs"] += 1

        # Save metadata for mapping indices to original files
        with open(Path(output_dir) / "validation_metadata.json", 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Processed {stats['total_pairs']} annotation pairs")
        logger.info(f"Valid pairs: {stats['valid_pairs']}")
        logger.info(f"Resized pairs: {stats['resized_pairs']}")
        logger.info(f"Skipped pairs due to size issues: {stats['skipped_pairs']}")
        logger.info(f"Non-binary human masks found and binarized: {stats['non_binary_human']}")
        logger.info(f"Non-binary system masks found and binarized: {stats['non_binary_system']}")

        return stats

def validate_annotations(human_annotations_dir, system_masks_dir, output_dir,
                         target_size=(512, 512), resize_mode='resize'):
    """
    Validate and prepare annotations for training by ensuring consistent sizes.

    Args:
        human_annotations_dir: Directory containing human annotation masks
        system_masks_dir: Directory containing system-generated masks
        output_dir: Directory to save prepared files to
        target_size: Target size for all images (height, width)
        resize_mode: 'strict' to skip mismatched images, 'resize' to force resize all

    Returns:
        int: Number of valid annotation pairs
    """
    # Create validator
    validator = AnnotationValidator(
        target_size=target_size,
        resize_mode=resize_mode
    )

    # Get file lists
    human_masks = list(Path(human_annotations_dir).glob("*.png"))
    system_masks = list(Path(system_masks_dir).glob("*_mask_resized.png"))

    # Validate and copy valid pairs
    stats = validator.validate_and_filter_pairs(
        human_masks=human_masks,
        system_masks=system_masks,
        output_dir=output_dir
    )

    return stats["valid_pairs"]
```

## visualization.py
```python
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict
from .core import MaskType

def visualize_masks(masks: Dict[MaskType, np.ndarray],
                   output_path: Path,
                   original_image: np.ndarray = None) -> None:
    """
    Visualize all masks with clear labeling.
    """
    # Create figure with subplots
    n_total = len(masks) + (1 if original_image is not None else 0)
    n_rows = (n_total + 2) // 3  # Ensure at most 3 columns
    n_cols = min(3, n_total)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    # Plot original image if provided
    idx = 0
    if original_image is not None:
        axes[idx].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[idx].set_title('Original Image')
        axes[idx].axis('off')
        idx += 1

    # Plot each mask with distinctive colormaps
    colormaps = {
        MaskType.ROADS: ('Reds', 'Roads'),
        MaskType.BUILDINGS: ('Blues', 'Buildings'),
        MaskType.VEGETATION: ('Greens', 'Vegetation'),
        MaskType.FIELDS: ('YlOrBr', 'Fields'),
        MaskType.COMBINED: ('gray', 'Combined')
    }

    for mask_type, mask in masks.items():
        cmap, title = colormaps[mask_type]
        axes[idx].imshow(mask, cmap=cmap)
        axes[idx].set_title(title)
        axes[idx].axis('off')
        idx += 1

    # Disable any unused subplots
    for i in range(idx, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path))
    plt.close()
```

## human_guided_helpers.py
```python
import logging
import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import yaml

logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml"""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def match_human_and_system_masks(grid_square: str) -> List[Dict]:
    """Match human annotations with corresponding system masks based on tile names"""
    config = load_config()

    # Define paths for human and system masks
    human_annotation_dir = Path(f"data/output/{grid_square}/human_annotation_masks")
    system_masks_dir = Path(f"data/processed_data/{grid_square}/test/masks")

    # Ensure directories exist
    if not human_annotation_dir.exists():
        logger.error(f"Human annotation directory not found: {human_annotation_dir}")
        return []

    if not system_masks_dir.exists():
        logger.error(f"System masks directory not found: {system_masks_dir}")
        return []

    # Create dictionaries to store files by their tile identifiers
    system_mask_dict = {}
    human_mask_dict = {}

    # Process system masks
    for mask_path in system_masks_dir.glob("*_mask_resized.png"):
        # Extract tile name (e.g., "nm4927")
        tile_name = mask_path.stem.replace("_mask_resized", "").lower()
        system_mask_dict[tile_name] = mask_path

    logger.info(f"Found {len(system_mask_dict)} system masks in {system_masks_dir}")

    # Process human annotations
    for annot_path in human_annotation_dir.glob("*.png"):
        # Parse filename to extract tile name
        filename_parts = annot_path.stem.split('_')
        for part in filename_parts:
            # Look for parts that match the pattern like "nm4927"
            if len(part) >= 6 and part[:2].isalpha() and part[2:].isdigit():
                tile_name = part.lower()
                human_mask_dict[tile_name] = annot_path
                break

    logger.info(f"Found {len(human_mask_dict)} human annotations in {human_annotation_dir}")

    # Find matches
    matched_pairs = []
    for tile_name in set(system_mask_dict.keys()) & set(human_mask_dict.keys()):
        system_mask = system_mask_dict[tile_name]
        human_mask = human_mask_dict[tile_name]

        # Get the original image too
        image_dir = Path(f"data/processed_data/{grid_square}/test/images")
        image_path = image_dir / f"{tile_name}.png"

        if image_path.exists():
            matched_pairs.append({
                'tile_name': tile_name,
                'image_path': image_path,
                'system_mask_path': system_mask,
                'human_mask_path': human_mask
            })
        else:
            logger.warning(f"Image not found for tile {tile_name}")

    logger.info(f"Found {len(matched_pairs)} matching pairs out of {len(system_mask_dict)} system masks and {len(human_mask_dict)} human masks")

    # Log information about the first few matches for debugging
    for i, pair in enumerate(matched_pairs[:3]):
        logger.info(f"Match {i+1}: {pair['tile_name']}")
        logger.info(f"  Image: {pair['image_path']}")
        logger.info(f"  System mask: {pair['system_mask_path']}")
        logger.info(f"  Human mask: {pair['human_mask_path']}")

    return matched_pairs

def fetch_annotations_for_grid(grid_square: str, portal_client) -> Optional[Path]:
    """Fetch human annotations for a specific grid square from the portal"""
    # Define the target directory for human annotations
    human_annotation_dir = Path(f"data/output/{grid_square}/human_annotation_masks")
    human_annotation_dir.mkdir(parents=True, exist_ok=True)

    # Use the portal client to fetch annotations - this now returns paths directly in the target dir
    annotation_paths = portal_client.fetch_annotations(grid_square)

    if annotation_paths and len(annotation_paths) > 0:
        logger.info(f"Downloaded {len(annotation_paths)} annotations to {human_annotation_dir}")
        return human_annotation_dir
    else:
        logger.error(f"No annotations found for grid square {grid_square}")
        return None

def validate_dataset(dataset) -> bool:
    """Validate that the dataset contains usable human mask data"""
    empty_masks = 0
    total_samples = len(dataset)

    # Skip validation for empty datasets
    if total_samples == 0:
        logger.error("Dataset is empty")
        return False

    # Check first few samples
    for i in range(min(10, total_samples)):
        sample = dataset[i]
        if sample['human_mask'].sum() == 0:
            empty_masks += 1

    # If all checked samples have empty masks, check the entire dataset
    if empty_masks == min(10, total_samples):
        empty_masks = 0
        for i in range(total_samples):
            sample = dataset[i]
            if sample['human_mask'].sum() == 0:
                empty_masks += 1

    # Log and return validation result
    if empty_masks == total_samples:
        logger.error("All human masks are empty. Cannot proceed with training.")
        return False
    elif empty_masks > 0:
        empty_percent = (empty_masks / total_samples) * 100
        logger.warning(f"{empty_masks}/{total_samples} ({empty_percent:.1f}%) human masks contain all zeros")
        # Proceed with warning
        return True
    else:
        logger.info("All human masks contain valid data")
        return True
```

## final_eval_grid_processor.py
```python
#!/usr/bin/env python3
"""
Final Evaluation Grid Processor

This script prepares the test grid (NS83) for evaluation by:
1. Moving DEM files from raw directory to test/images
2. Moving mask files from raw directory to test/masks
3. Ensuring proper directory structure exists
"""

import os
import shutil
import argparse
import logging
from pathlib import Path
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_test_directory(grid_square):
    """Ensure test directory structure exists for the grid square."""
    config = load_config()
    processed_dir = Path(config['data']['processed_dir']) / grid_square

    # Create test directory structure if it doesn't exist
    test_images_dir = processed_dir / "test" / "images"
    test_masks_dir = processed_dir / "test" / "masks"

    test_images_dir.mkdir(parents=True, exist_ok=True)
    test_masks_dir.mkdir(parents=True, exist_ok=True)

    return processed_dir, test_images_dir, test_masks_dir

def process_raw_files(grid_square):
    """
    Process raw files for the evaluation grid:
    - Move DEM files to test/images
    - Move mask files to test/masks
    """
    processed_dir, test_images_dir, test_masks_dir = setup_test_directory(grid_square)
    raw_dir = processed_dir / "raw"

    if not raw_dir.exists():
        logger.error(f"Raw directory not found for {grid_square}: {raw_dir}")
        return False

    # Count files before processing
    raw_file_count = len(list(raw_dir.glob("*.png")))
    logger.info(f"Found {raw_file_count} raw files to process for {grid_square}")

    # Counter for moved files
    moved_images = 0
    moved_masks = 0

    # Process each file in raw directory
    for file_path in raw_dir.glob("*.png"):
        # Skip files that are already processed
        if "_mask_" in file_path.name:
            # This is a mask file - copy to test/masks directory
            dest_path = test_masks_dir / file_path.name
            try:
                shutil.copy2(file_path, dest_path)
                moved_masks += 1
            except Exception as e:
                logger.error(f"Error copying mask file {file_path}: {e}")
        else:
            # This is a DEM file - copy to test/images directory
            dest_path = test_images_dir / file_path.name
            try:
                shutil.copy2(file_path, dest_path)
                moved_images += 1
            except Exception as e:
                logger.error(f"Error copying image file {file_path}: {e}")

    logger.info(f"Moved {moved_images} image files to {test_images_dir}")
    logger.info(f"Moved {moved_masks} mask files to {test_masks_dir}")

    # Verify files were moved successfully
    image_count = len(list(test_images_dir.glob("*.png")))
    mask_count = len(list(test_masks_dir.glob("*.png")))

    logger.info(f"Final count: {image_count} images and {mask_count} masks in test directories")

    if image_count == 0 or mask_count == 0:
        logger.warning(f"One or more test directories is empty for {grid_square}")
        return False

    return True

def main():
    """Main function for processing evaluation grid."""
    parser = argparse.ArgumentParser(description="Process final evaluation grid for testing")
    parser.add_argument("--grid", type=str, default="NS83", help="Grid square to process (default: NS83)")
    parser.add_argument("--force", action="store_true", help="Force reprocessing even if files exist")

    args = parser.parse_args()
    grid_square = args.grid

    logger.info(f"Processing evaluation grid: {grid_square}")

    # Process raw files and move to test directories
    success = process_raw_files(grid_square)

    if success:
        logger.info(f"Successfully prepared {grid_square} for evaluation")
    else:
        logger.error(f"Failed to prepare {grid_square} for evaluation")

if __name__ == "__main__":
    main()
```

## gan_inpainting.py
```python
import logging
from pathlib import Path
from mvp_gan.src.evaluate import evaluate  # Import the GAN evaluation function

def inpaint_with_gan(dem_image_path, mask_path, output_dir, checkpoint_path):
    """
    Inpaint using GAN and save the output.

    Parameters:
    - dem_image_path: Path to the DEM image.
    - mask_path: Path to the binary mask.
    - output_dir: Directory where the inpainted image will be saved.
    - checkpoint_path: Path to the GAN model checkpoint.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    inpainted_image_path = output_dir / f"{dem_image_path.stem}_inpainted.png"
    evaluate(dem_image_path, mask_path, checkpoint_path, inpainted_image_path)
    logging.info(f"Inpainted image saved to {inpainted_image_path}")
    return inpainted_image_path
```

## data_extraction.py
```python
import zipfile
import logging
import shutil
import cv2
from pathlib import Path
import numpy as np
from PIL import Image
import yaml
import os
from typing import Optional, Tuple

from utils.path_handling.path_utils import PathManager
from utils.mask_processing.core import MaskProcessor
from utils.mask_processing.visualization import visualize_masks

# Load configurations
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Initialize global constants
RAW_DATA_DIR = Path(config['data']['raw_dir'])
PROCESSED_DATA_DIR = Path(config['data']['processed_dir'])
TARGET_FOLDERS = ("getmapping-dsm-2000", "getmapping_rgb_25cm")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_relevant_folders(input_zip_path: Path, extract_to: Path, target_folders: Tuple[str, ...] = TARGET_FOLDERS) -> bool:
    """
    Extracts only the specified directories from a zip file.

    Args:
        input_zip_path: Path to the zip file
        extract_to: Directory where files will be extracted
        target_folders: Folder names to look for within the zip file

    Returns:
        bool: True if extraction was successful
    """
    logger.info(f"Extracting from {input_zip_path} to {extract_to}")
    try:
        with zipfile.ZipFile(input_zip_path, 'r') as zip_ref:
            extracted_files = []
            for file in zip_ref.namelist():
                if any(target in file for target in target_folders):
                    zip_ref.extract(file, extract_to)
                    extracted_files.append(file)

        if not extracted_files:
            logger.warning("No matching folders found in zip file")
            return False

        logger.info(f"Successfully extracted {len(extracted_files)} files")
        return True
    except Exception as e:
        logger.error(f"Failed to extract {input_zip_path}: {str(e)}")
        return False

def convert_dem_asc_to_png(asc_file_path: Path, png_file_path: Path, default_no_data: int = -9999) -> bool:
    """
    Converts a DEM .asc file to a grayscale .png image, normalizing data for GAN compatibility.

    Args:
        asc_file_path: Path to the .asc file
        png_file_path: Output path for the .png file
        default_no_data: Default value to treat as no data if not specified in file

    Returns:
        bool: True if conversion was successful
    """
    try:
        # Read header
        header = {}
        with open(asc_file_path, 'r') as file:
            for _ in range(6):
                key, value = file.readline().strip().split()
                header[key] = float(value) if '.' in value else int(value)

        # Load and process DEM data
        no_data_value = header.get('NODATA_value', default_no_data)
        data = np.loadtxt(asc_file_path, skiprows=6)
        data[data == no_data_value] = np.nan

        # Normalize data
        valid_data = data[~np.isnan(data)]
        if len(valid_data) == 0:
            logger.warning(f"No valid data in {asc_file_path}")
            return False

        data_min, data_max = np.nanmin(data), np.nanmax(data)
        if data_min == data_max:
            logger.warning(f"Flat elevation data in {asc_file_path}")
            normalized_data = np.zeros_like(data)
        else:
            normalized_data = 255 * (data - data_min) / (data_max - data_min)

        # Replace NaNs and ensure uint8 type
        normalized_data = np.nan_to_num(normalized_data, nan=0)
        normalized_data = normalized_data.astype(np.uint8)

        # Create output directory if needed
        png_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as PNG
        img = Image.fromarray(normalized_data, mode='L')
        img = img.resize((512, 512), Image.BILINEAR)
        img.save(png_file_path)

        logger.info(f"Successfully converted {asc_file_path.name} to PNG")
        return True

    except Exception as e:
        logger.error(f"Error converting {asc_file_path} to PNG: {str(e)}")
        return False

def process_parent_grid(zip_file_path: Path, path_manager: PathManager) -> Optional[str]:
    """
    Process a parent grid square zip file.

    Args:
        zip_file_path: Path to the zip file
        path_manager: PathManager instance

    Returns:
        str or None: Parent grid reference if successful, None otherwise
    """
    try:
        # Extract parent grid reference
        parent_grid = path_manager.get_parent_from_zip(zip_file_path)
        logger.info(f"Processing parent grid: {parent_grid}")

        # Create directory structure
        paths = path_manager.create_parent_structure(parent_grid)

        # Setup extract directory
        extract_dir = RAW_DATA_DIR / f"{parent_grid}_extracted"
        if not extract_relevant_folders(zip_file_path, extract_dir):
            return None

        return parent_grid

    except Exception as e:
        logger.error(f"Failed to process parent grid from {zip_file_path}: {str(e)}")
        return None

def cleanup_extracted_files(parent_grid: str) -> None:
    """
    Clean up extracted files for a parent grid square.

    Args:
        parent_grid: Parent grid square identifier
    """
    extracted_dir = RAW_DATA_DIR / f"{parent_grid}_extracted"
    if extracted_dir.exists():
        try:
            shutil.rmtree(extracted_dir)
            logger.info(f"Cleaned up extracted files for {parent_grid}")
        except Exception as e:
            logger.error(f"Failed to cleanup extracted files for {parent_grid}: {str(e)}")
```

## parallel_processing.py
```python
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
```

## dsm_colorizer.py
```python
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

class DSMColorizer:
    """Handles recoloring of DSM outputs using OS UK standard elevation color palette"""

    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # OS UK terrain colors from lowest to highest elevation
        # Based on standard topographic color scheme
        self.colors = [
            '#0C6B58',  # Deep green
            '#2E8B57',  # Sea green
            '#90EE90',  # Light green
            '#F4D03F',  # Yellow
            '#E67E22',  # Orange
            '#CB4335',  # Red
            '#6E2C00',  # Brown
            '#FFFFFF',  # White (peaks)
        ]

        # Create custom colormap
        self.colormap = plt.cm.colors.LinearSegmentedColormap.from_list(
            'osuk_terrain', self.colors)

    def recolor_all(self):
        """Process all inpainted PNGs in input directory"""
        for img_path in self.input_dir.glob("*_inpainted.png"):
            self.recolor_dsm(img_path)

    def recolor_dsm(self, img_path: Path):
        """Recolor single DSM image using OS UK elevation palette"""
        # Read grayscale image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        # Normalize values to 0-1
        normalized = img.astype(float) / 255

        # Apply colormap
        colored = self.colormap(normalized)
        colored = (colored[:, :, :3] * 255).astype(np.uint8)

        # Save colored version
        output_path = self.output_dir / f"{img_path.stem}_colored.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
        return output_path
```

## experiment_tracking.py
```python
# utils/experiment_tracking.py
import os
import time
import json
import yaml
import mlflow
import psutil
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import logging
from dataclasses import dataclass
import git
from contextlib import contextmanager

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec


@dataclass
class SystemMetrics:
    """Track system resource usage"""
    cpu_percent: float
    memory_percent: float
    gpu_memory_allocated: Optional[float] = None
    gpu_memory_cached: Optional[float] = None


class ExperimentTracker:
    """
    MLflow experiment tracker for managing ML experiment logging and model artifacts.

    This class handles:
    - Experiment initialization and management
    - Run tracking and metrics logging
    - Model architecture logging
    - System metrics tracking
    - Model artifact management
    """

    def __init__(self, experiment_name: str, tracking_uri: str = None):
        """
        Initialize MLflow experiment tracker

        Args:
            experiment_name: Name of the experiment
            tracking_uri: URI for MLflow tracking server. If None, uses local filesystem
        """
        self.logger = logging.getLogger(__name__)
        self.tracking_uri = tracking_uri or "file:./mlruns"
        self.experiment_name = experiment_name
        self.run = None
        self.start_time = None
        self.is_run_active = False
        self.run_id = None

        # Initialize temporary storage for metrics batching
        self.batch_metrics_buffer = {}
        self.last_metrics_flush_time = time.time()
        self.metrics_flush_interval = 5  # seconds

        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            self._initialize_mlflow_experiment()
        except Exception as e:
            self.logger.error(f"Failed to initialize experiment tracker: {str(e)}")
            raise

    def _initialize_mlflow_experiment(self) -> None:
        """Initialize or get existing MLflow experiment"""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(
                self.experiment_name
            )
            self.logger.info(f"Created new experiment: {self.experiment_name}")
        else:
            self.experiment_id = experiment.experiment_id
            self.logger.info(f"Using existing experiment: {self.experiment_name}")

    @contextmanager
    def _ensure_active_run(self, run_name: str = None):
        """Context manager to ensure an active run exists"""
        if not self.is_run_active:
            self.start_run(run_name or f"run_{int(time.time())}", {})
            created_new = True
        else:
            created_new = False

        try:
            yield
        finally:
            if created_new:
                self.end_run()

    def start_run(self,
                 run_name: str,
                 config: Dict[str, Any],
                 tags: Dict[str, str] = None) -> Optional[mlflow.ActiveRun]:
        """
        Start a new tracking run

        Args:
            run_name: Name for the new run
            config: Configuration dictionary to log as parameters
            tags: Optional dictionary of tags to attach to the run

        Returns:
            Active MLflow run or None if failed
        """
        if self.is_run_active:
            self.logger.info("Using existing MLflow run")
            return self.run

        try:
            git_tags = self._get_git_info()
            run_tags = {**(tags or {}), **git_tags}

            mlflow.set_experiment(self.experiment_name)
            self.run = mlflow.start_run(
                run_name=run_name,
                tags=run_tags
            )
            self.run_id = self.run.info.run_id
            self.is_run_active = True

            # Log flattened parameters in hierarchical structure
            flattened_params = self._flatten_dict(config)

            # Batch parameters by top-level category
            param_categories = {}
            for key, value in flattened_params.items():
                category = key.split('.')[0] if '.' in key else 'main'
                if category not in param_categories:
                    param_categories[category] = {}
                param_categories[category][key] = value

            # Log parameters by category to reduce individual file operations
            for params in param_categories.values():
                mlflow.log_params(params)

            # Log model architecture if available
            if "model" in config:
                self._log_model_architecture(config["model"])

            self.start_time = time.time()
            return self.run

        except Exception as e:
            self.logger.error(f"Failed to start run: {str(e)}")
            self.is_run_active = False
            self.run = None
            self.run_id = None
            raise

    def _get_git_info(self) -> Dict[str, str]:
        """
        Get git repository information

        Returns:
            Dictionary containing git commit, branch, and repo information
        """
        try:
            repo = git.Repo(search_parent_directories=True)
            return {
                "git_commit": repo.head.commit.hexsha,
                "git_branch": repo.active_branch.name,
                "git_repo": repo.remotes.origin.url if repo.remotes else "local"
            }
        except Exception as e:
            self.logger.warning(f"Could not get git info: {e}")
            return {}

    # Add the missing method
    def _calculate_l1_l2(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
        """
        Calculate L1 and L2 distances between predictions and targets.

        Args:
            pred: Model predictions
            target: Ground truth targets

        Returns:
            Tuple of (L1 distance, L2 distance)
        """
        try:
            l1_dist = torch.nn.functional.l1_loss(pred, target).item()
            l2_dist = torch.nn.functional.mse_loss(pred, target, reduction='mean').sqrt().item()
            return l1_dist, l2_dist
        except Exception as e:
            self.logger.error(f"Error calculating L1/L2 distances: {str(e)}")
            return 0.0, 0.0

    # Add the missing method
    def _calculate_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        try:
            mse = torch.nn.functional.mse_loss(pred, target)
            if mse == 0:
                return float('inf')
            max_pixel = 1.0
            psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
            return psnr.item()
        except Exception as e:
            self.logger.warning(f"Error calculating PSNR: {str(e)}")
            return 0.0

    # Add the missing method
    def _calculate_ssim(self, pred: torch.Tensor, target: torch.Tensor, window_size=11) -> float:
        """Calculate Structural Similarity Index"""
        try:
            C1 = (0.01 * 1.0) ** 2
            C2 = (0.03 * 1.0) ** 2

            mu1 = torch.nn.functional.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
            mu2 = torch.nn.functional.avg_pool2d(target, window_size, stride=1, padding=window_size//2)

            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2

            sigma1_sq = torch.nn.functional.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size//2) - mu1_sq
            sigma2_sq = torch.nn.functional.avg_pool2d(target * target, window_size, stride=1, padding=window_size//2) - mu2_sq
            sigma12 = torch.nn.functional.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean().item()
        except Exception as e:
            self.logger.warning(f"Error calculating SSIM: {str(e)}")
            return 0.0

    def log_training_batch(self,
                        pred: torch.Tensor,
                        target: torch.Tensor,
                        model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer,
                        batch_metrics: Dict[str, float],
                        step: int) -> None:
        """
        Log comprehensive batch-level metrics, batching them efficiently
        """
        if not self.is_run_active:
            with self._ensure_active_run():
                pass

        try:
            # Get standard training metrics
            batch_start_time = time.time()
            performance_metrics = self._calculate_performance_metrics(
                pred, target, model, optimizer, batch_start_time
            )

            # Clean and combine all metrics
            combined_metrics = {
                **self._clean_metrics(batch_metrics),
                **self._clean_metrics(performance_metrics),
                **self._get_system_metrics().__dict__
            }

            # Log boundary loss specifically if available
            if 'boundary_loss' in batch_metrics:
                combined_metrics['boundary_loss'] = batch_metrics['boundary_loss']

            # Log boundary metrics if available
            boundary_metrics = ['boundary_mse', 'boundary_psnr', 'boundary_gradient_diff']
            for metric in boundary_metrics:
                if metric in batch_metrics:
                    combined_metrics[metric] = batch_metrics[metric]

            # Add to metrics buffer with step information
            for k, v in combined_metrics.items():
                metric_key = f"batch.{k}"
                if metric_key not in self.batch_metrics_buffer:
                    self.batch_metrics_buffer[metric_key] = []
                self.batch_metrics_buffer[metric_key].append((step, v))

            # Flush metrics if enough time has passed
            current_time = time.time()
            if current_time - self.last_metrics_flush_time >= self.metrics_flush_interval:
                self._flush_metrics_buffer()
                self.last_metrics_flush_time = current_time

        except Exception as e:
            self.logger.error(f"Failed to log batch metrics: {str(e)}")

    def _flush_metrics_buffer(self):
        """Flush all buffered metrics to MLflow"""
        if not self.batch_metrics_buffer:
            return

        try:
            with mlflow.start_run(run_id=self.run_id, nested=True) as run: # Always nest.
                # Group metrics by step for efficient logging
                metrics_by_step = {}
                for metric_key, values in self.batch_metrics_buffer.items():
                    for step, value in values:
                        if step not in metrics_by_step:
                            metrics_by_step[step] = {}
                        metrics_by_step[step][metric_key] = value

                # Log metrics for each step
                for step, metrics in metrics_by_step.items():
                    mlflow.log_metrics(metrics, step=step)

                # Clear the buffer after successful logging
                self.batch_metrics_buffer = {}
        except Exception as e:
            self.logger.error(f"Failed to flush metrics buffer: {str(e)}")

    def log_validation_metrics(self,
                             model: torch.nn.Module,
                             val_loader: torch.utils.data.DataLoader,
                             device: torch.device,
                             step: int) -> Dict[str, float]:
        """
        Calculate and log validation metrics

        Args:
            model: The model to evaluate
            val_loader: DataLoader for validation data
            device: Device to run validation on
            step: Current training step

        Returns:
            Dictionary of calculated validation metrics
        """
        with self._ensure_active_run():
            try:
                model.eval()
                val_metrics = {
                    'val_psnr': 0,
                    'val_ssim': 0,
                    'val_l1': 0,
                    'val_l2': 0
                }

                with torch.no_grad():
                    for batch in val_loader:
                        pred = model(batch['image'].to(device))
                        target = batch['target'].to(device)

                        # Calculate metrics
                        val_metrics['val_psnr'] += self._calculate_psnr(pred, target)
                        val_metrics['val_ssim'] += self._calculate_ssim(pred, target)
                        l1, l2 = self._calculate_l1_l2(pred, target)
                        val_metrics['val_l1'] += l1
                        val_metrics['val_l2'] += l2

                # Average metrics
                num_batches = len(val_loader)
                val_metrics = {k: v/num_batches for k, v in val_metrics.items()}

                # Log validation metrics
                prefixed_metrics = {
                    f"validation.{k}": v for k, v in val_metrics.items()
                }

                mlflow.log_metrics(prefixed_metrics, step=step)
                return val_metrics

            except Exception as e:
                self.logger.error(f"Failed to log validation metrics: {str(e)}")
                return {}

    def log_metrics(self,
                   metrics: Dict[str, float],
                   step: Optional[int] = None) -> None:
        """
        Log metrics for current step

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for the metrics
        """
        with self._ensure_active_run():
            try:
                # Clean metrics first
                cleaned_metrics = self._clean_metrics(metrics)

                # Get system metrics
                sys_metrics = self._get_system_metrics()
                system_metrics = {
                    "system.cpu_percent": sys_metrics.cpu_percent,
                    "system.memory_percent": sys_metrics.memory_percent
                }

                if sys_metrics.gpu_memory_allocated is not None:
                    system_metrics.update({
                        "system.gpu_memory_allocated": sys_metrics.gpu_memory_allocated,
                        "system.gpu_memory_cached": sys_metrics.gpu_memory_cached
                    })

                # Combine all metrics
                all_metrics = {**cleaned_metrics, **system_metrics}
                mlflow.log_metrics(all_metrics, step=step)

            except Exception as e:
                self.logger.error(f"Failed to log metrics: {str(e)}")

    def _clean_metrics(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Convert tensor metrics to Python numeric types for MLflow logging.

        Args:
            metrics: Dictionary of metrics which may contain tensors

        Returns:
            Dictionary with all metrics converted to Python numeric types
        """
        clean_metrics = {}
        for k, v in metrics.items():
            try:
                if isinstance(v, torch.Tensor):
                    if v.numel() == 1:
                        clean_metrics[k] = v.item()
                elif isinstance(v, (int, float)):
                    clean_metrics[k] = float(v)
                elif isinstance(v, np.ndarray) and v.size == 1:
                    clean_metrics[k] = float(v.item())
            except Exception:
                continue
        return clean_metrics

    def _flatten_dict(self,
                     d: Dict,
                     parent_key: str = '',
                     sep: str = '.') -> Dict[str, str]:
        """
        Flatten nested dictionary for MLflow params

        Args:
            d: Dictionary to flatten
            parent_key: Prefix for flattened keys
            sep: Separator between nested keys

        Returns:
            Flattened dictionary with string values
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, str(v)))
        return dict(items)

    def _get_system_metrics(self) -> SystemMetrics:
        """
        Get current system resource usage

        Returns:
            SystemMetrics object containing CPU, memory, and GPU metrics
        """
        try:
            metrics = SystemMetrics(
                cpu_percent=psutil.cpu_percent(),
                memory_percent=psutil.virtual_memory().percent
            )

            # Get GPU metrics if available
            if torch.cuda.is_available():
                metrics.gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                metrics.gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB

            return metrics
        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {str(e)}")
            return SystemMetrics(cpu_percent=0.0, memory_percent=0.0)

    def _log_model_architecture(self, model):
        """
        Log model architecture details

        Args:
            model: Model to log architecture details for
        """
        if not isinstance(model, torch.nn.Module):
            return

        try:
            # Get model summary
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Use model class name to create unique parameter names
            model_type = model.__class__.__name__.lower()

            # Log as a single batch of parameters
            arch_params = {
                f"{model_type}.total_parameters": total_params,
                f"{model_type}.trainable_parameters": trainable_params,
                f"{model_type}.architecture": model.__class__.__name__
            }
            mlflow.log_params(arch_params)

            # Log model architecture as artifact
            arch_path = Path(f"{model_type}_architecture.txt")
            try:
                with open(arch_path, "w") as f:
                    f.write(str(model))
                    f.write(f"\nModel device: {next(model.parameters()).device}") # Add device information.

                mlflow.log_artifact(str(arch_path))
            finally:
                # Always clean up the temporary file
                if arch_path.exists():
                    arch_path.unlink()

        except Exception as e:
            self.logger.error(f"Failed to log model architecture: {str(e)}")

    def log_model(self, model, name: str, metrics: Dict[str, Any] = None, input_example=None) -> None:
        """
        Log a model checkpoint with optional metrics

        Args:
            model: The model to log
            name: Name for the logged model
            metrics: Optional dictionary of metrics to log with the model
            input_example: Optional input example for the model
        """
        with self._ensure_active_run():
            try:
                # Save original device and move model to CPU for logging
                original_device = next(model.parameters()).device
                model = model.cpu()

                try:
                    # Create a wrapper for the PConvUNet model to handle the mask input
                    if model.__class__.__name__ == 'PConvUNet':
                        # Create a simple wrapper to handle the input validation
                        class ModelWrapper(torch.nn.Module):
                            def __init__(self, model):
                                super().__init__()
                                self.model = model

                            def forward(self, x):
                                # Create a dummy mask of ones with the same batch size
                                mask = torch.ones_like(x)
                                return self.model(x, mask)

                        # Wrap the model
                        wrapped_model = ModelWrapper(model)

                        # Create a proper input example
                        if input_example is None:
                            input_example = np.zeros((1, 1, 512, 512), dtype=np.float32)

                        # Get torch requirement string with CUDA if available
                        if torch.cuda.is_available():
                            cuda_version = torch.version.cuda
                            torch_requirement = f'torch=={torch.__version__}+cu{cuda_version.replace(".", "")}'
                        else:
                            torch_requirement = f'torch=={torch.__version__}'

                        # Define pip requirements explicitly
                        pip_requirements = [
                            torch_requirement,
                            'numpy>=' + np.__version__,
                        ]

                        # Log the model relative to the artifacts with the wrapper
                        mlflow.pytorch.log_model(
                            pytorch_model=wrapped_model,
                            artifact_path=name,
                            input_example=input_example,
                            pip_requirements=pip_requirements
                        )
                    else:
                        # For other models, log normally
                        if input_example is None:
                            input_example = np.zeros((1, 1, 512, 512), dtype=np.float32)

                        # Get torch requirement string with CUDA if available
                        if torch.cuda.is_available():
                            cuda_version = torch.version.cuda
                            torch_requirement = f'torch=={torch.__version__}+cu{cuda_version.replace(".", "")}'
                        else:
                            torch_requirement = f'torch=={torch.__version__}'

                        # Define pip requirements explicitly
                        pip_requirements = [
                            torch_requirement,
                            'numpy>=' + np.__version__,
                        ]

                        # Log the model relative to the artifacts
                        mlflow.pytorch.log_model(
                            pytorch_model=model,
                            artifact_path=name,
                            input_example=input_example,
                            pip_requirements=pip_requirements
                        )

                    # Log metrics if provided
                    if metrics:
                        scalar_metrics = {
                            f"{name}.{k}": float(v)
                            for k, v in metrics.items()
                            if isinstance(v, (int, float)) or
                            (isinstance(v, torch.Tensor) and v.numel() == 1)
                        }
                        if scalar_metrics:
                            mlflow.log_metrics(scalar_metrics)

                    self.logger.info(f"Successfully logged model {name}")

                except Exception as e:
                    self.logger.error(f"Failed to log model artifact: {str(e)}")
                finally:
                    # Always restore model to original device
                    model = model.to(original_device)
                    if hasattr(model, 'train'):
                        model.train()  # Restore training mode if applicable

            except Exception as e:
                self.logger.error(f"Failed to log model: {str(e)}")
                # Ensure model returns to original device even in case of outer error
                if torch.cuda.is_available():
                    model = model.to(original_device)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log an artifact file or directory

        Args:
            local_path: Path to the local file or directory to log
            artifact_path: Optional path to use within the artifact directory
        """
        with self._ensure_active_run():
            try:
                mlflow.log_artifact(local_path, artifact_path)
            except Exception as e:
                self.logger.error(f"Failed to log artifact: {str(e)}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters for the current run

        Args:
            params: Dictionary of parameters to log
        """
        with self._ensure_active_run():
            try:
                # Flatten nested dictionaries for MLflow
                flat_params = self._flatten_dict(params)
                mlflow.log_params(flat_params)
            except Exception as e:
                self.logger.error(f"Failed to log parameters: {str(e)}")

    def end_run(self) -> None:
        """End current tracking run and log final metrics"""
        if not self.is_run_active:
            return

        try:
            # Flush any remaining metrics
            self._flush_metrics_buffer()

            # Log total run time
            if self.start_time is not None:
                duration = time.time() - self.start_time
                mlflow.log_metric("training_duration_seconds", duration)

            mlflow.end_run()
            self.logger.info(f"Ended MLflow run: {self.run_id}")
        except Exception as e:
            self.logger.error(f"Failed to end run cleanly: {str(e)}")
        finally:
            self.run = None
            self.run_id = None
            self.is_run_active = False
            self.start_time = None

    # Performance metric calculation methods
    def _calculate_performance_metrics(self, pred, target, model, optimizer, start_time):
        """Calculate comprehensive performance metrics"""
        metrics = {}

        # Basic metrics
        metrics['psnr'] = self._calculate_psnr(pred, target)
        metrics['ssim'] = self._calculate_ssim(pred, target)
        l1_dist, l2_dist = self._calculate_l1_l2(pred, target)
        metrics['l1_distance'] = l1_dist
        metrics['l2_distance'] = l2_dist

        # Optimizer metrics
        metrics.update(self._get_optimizer_metrics(optimizer))

        # Time metrics
        metrics['batch_time'] = time.time() - start_time

        return metrics

    def _get_optimizer_metrics(self, optimizer):
        """Get metrics from the optimizer"""
        try:
            metrics = {}
            for i, param_group in enumerate(optimizer.param_groups):
                metrics[f'lr_group_{i}'] = param_group['lr']
            return metrics
        except Exception as e:
            self.logger.warning(f"Error getting optimizer metrics: {str(e)}")
            return {}
```

## data_splitting.py
```python
# utils/data_splitting.py

import re
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging
from dataclasses import dataclass

@dataclass
class TileInfo:
    """Stores information about a single tile"""
    path: Path
    x: int
    y: int
    split: Optional[str] = None

class GeographicalDataHandler:
    def __init__(self, parent_grid: str, root_dir: Path):
        """Initialize handler for geographical data organization.

        Args:
            parent_grid: Parent grid square identifier (e.g., 'NJ05')
            root_dir: Base directory for all processing
        """
        self.parent_grid = parent_grid
        self.root_dir = root_dir / parent_grid  # Scope to parent
        self.tile_mapping: Dict[Tuple[int, int], TileInfo] = {}
        self.split_assignments: Dict[Tuple[int, int], str] = {}
        self.logger = logging.getLogger(__name__)

    def add_tile(self, tile_path: Path, x: int, y: int) -> None:
        """
        Add a tile to the mapping.

        Args:
            tile_path: Path to the tile file
            x: X coordinate
            y: Y coordinate
        """
        # Extract base name without extension
        base_name = tile_path.stem.lower()

        # Validate format
        if not self._validate_child_grid(base_name):
            raise ValueError(f"Invalid tile format: {base_name}")

        self.tile_mapping[(x, y)] = TileInfo(
            path=tile_path,
            x=x,
            y=y
        )

    def apply_splits(self) -> None:
        """Apply splits within parent directory structure"""
        split_dirs = {}
        for split in ['train', 'val', 'test']:
            split_images_dir = self.root_dir / split / 'images'
            split_masks_dir = self.root_dir / split / 'masks'
            split_images_dir.mkdir(parents=True, exist_ok=True)
            split_masks_dir.mkdir(parents=True, exist_ok=True)
            split_dirs[split] = {
                'images': split_images_dir,
                'masks': split_masks_dir
            }

        for coord, tile_info in self.tile_mapping.items():
            if coord in self.split_assignments:
                split = self.split_assignments[coord]
                # Handle both the DEM and its corresponding mask
                dem_name = tile_info.path.name
                mask_name = f"{tile_info.path.stem}_mask_resized.png"

                # Move files to appropriate split directory under parent
                dem_dest = split_dirs[split]['images'] / dem_name
                mask_path = tile_info.path.parent / mask_name
                mask_dest = split_dirs[split]['masks'] / mask_name

                if tile_info.path.exists():
                    shutil.copy2(tile_info.path, dem_dest)
                if mask_path.exists():
                    shutil.copy2(mask_path, mask_dest)

    def save_metadata(self) -> None:
        """Save split assignments and coordinates to parent's metadata"""
        metadata_dir = self.root_dir / 'metadata'
        metadata_dir.mkdir(parents=True, exist_ok=True)

        # Save split mapping
        split_data = {
            f"{x},{y}": split
            for (x, y), split in self.split_assignments.items()
        }
        with open(metadata_dir / 'split_mapping.json', 'w') as f:
            json.dump(split_data, f, indent=2)

        # Save coordinate mapping
        coord_data = {
            str(tile.path): {
                'x': tile.x,
                'y': tile.y,
                'split': tile.split,
                'parent_grid': self.parent_grid
            }
            for tile in self.tile_mapping.values()
        }
        with open(metadata_dir / 'coordinate_mapping.json', 'w') as f:
            json.dump(coord_data, f, indent=2)

    def generate_splits(self, split_ratios: Dict[str, float] = None) -> None:
        """Generate checkerboard split pattern for all tiles"""
        if split_ratios is None:
            split_ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}

        # Validate ratios
        if not abs(sum(split_ratios.values()) - 1.0) < 0.001:
            raise ValueError("Split ratios must sum to 1.0")

        # Get grid dimensions
        coords = list(self.tile_mapping.keys())
        if not coords:
            raise ValueError("No tiles registered")

        min_x = min(x for x, _ in coords)
        max_x = max(x for x, _ in coords)
        min_y = min(y for _, y in coords)
        max_y = max(y for _, y in coords)

        # Create base 3x3 pattern
        base_pattern = self._create_base_pattern(split_ratios)

        # Apply pattern across grid
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if (x, y) in self.tile_mapping:
                    pattern_x = (x - min_x) % 3
                    pattern_y = (y - min_y) % 3
                    self.split_assignments[(x, y)] = base_pattern[pattern_y][pattern_x]

        # Validate split distribution
        self._validate_splits(split_ratios)

    def _create_base_pattern(self, split_ratios: Dict[str, float]) -> List[List[str]]:
        """
        Create a fixed 10x10 pattern that ensures no adjacent tiles share the same split.
        Distribution aims for approximately: 40% train, 30% val, 30% test

        Returns a 10x10 grid where no two adjacent cells (including diagonals)
        have the same split type.
        """
        # Original Pattern (for reference)
        original_pattern = [
            ['train', 'val',   'test',  'train', 'val',   'test',  'train', 'val',   'test',  'train'],
            ['test',  'train', 'val',   'test',  'train', 'val',   'test',  'train', 'val',   'test' ],
            ['val',   'test',  'train', 'val',   'test',  'train', 'val',   'test',  'train', 'val'  ],
            ['train', 'val',   'test',  'train', 'val',   'test',  'train', 'val',   'test',  'train'],
            ['test',  'train', 'val',   'test',  'train', 'val',   'test',  'train', 'val',   'test' ],
            ['val',   'test',  'train', 'val',   'test',  'train', 'val',   'test',  'train', 'val'  ],
            ['train', 'val',   'test',  'train', 'val',   'test',  'train', 'val',   'test',  'train'],
            ['test',  'train', 'val',   'test',  'train', 'val',   'test',  'train', 'val',   'test' ],
            ['val',   'test',  'train', 'val',   'test',  'train', 'val',   'test',  'train', 'val'  ],
            ['train', 'val',   'test',  'train', 'val',   'test',  'train', 'val',   'test',  'train']
        ]
        return original_pattern

        # Permutation 1: Rotated pattern (each row shifts right by one)
        # permutation_1 = [
        #     ['val',   'test',  'train', 'val',   'test',  'train', 'val',   'test',  'train', 'val'  ],
        #     ['train', 'val',   'test',  'train', 'val',   'test',  'train', 'val',   'test',  'train'],
        #     ['test',  'train', 'val',   'test',  'train', 'val',   'test',  'train', 'val',   'test' ],
        #     ['val',   'test',  'train', 'val',   'test',  'train', 'val',   'test',  'train', 'val'  ],
        #     ['train', 'val',   'test',  'train', 'val',   'test',  'train', 'val',   'test',  'train'],
        #     ['test',  'train', 'val',   'test',  'train', 'val',   'test',  'train', 'val',   'test' ],
        #     ['val',   'test',  'train', 'val',   'test',  'train', 'val',   'test',  'train', 'val'  ],
        #     ['train', 'val',   'test',  'train', 'val',   'test',  'train', 'val',   'test',  'train'],
        #     ['test',  'train', 'val',   'test',  'train', 'val',   'test',  'train', 'val',   'test' ],
        #     ['val',   'test',  'train', 'val',   'test',  'train', 'val',   'test',  'train', 'val'  ]
        # ]
        # return permutation_1

        # # Permutation 2: Different cyclic pattern (test-val-train instead of train-val-test)
        # permutation_2 = [
        #     ['test',  'val',   'train', 'test',  'val',   'train', 'test',  'val',   'train', 'test' ],
        #     ['val',   'train', 'test',  'val',   'train', 'test',  'val',   'train', 'test',  'val'  ],
        #     ['train', 'test',  'val',   'train', 'test',  'val',   'train', 'test',  'val',   'train'],
        #     ['test',  'val',   'train', 'test',  'val',   'train', 'test',  'val',   'train', 'test' ],
        #     ['val',   'train', 'test',  'val',   'train', 'test',  'val',   'train', 'test',  'val'  ],
        #     ['train', 'test',  'val',   'train', 'test',  'val',   'train', 'test',  'val',   'train'],
        #     ['test',  'val',   'train', 'test',  'val',   'train', 'test',  'val',   'train', 'test' ],
        #     ['val',   'train', 'test',  'val',   'train', 'test',  'val',   'train', 'test',  'val'  ],
        #     ['train', 'test',  'val',   'train', 'test',  'val',   'train', 'test',  'val',   'train'],
        #     ['test',  'val',   'train', 'test',  'val',   'train', 'test',  'val',   'train', 'test' ]
        # ]
        # return permutation_2

    def _validate_splits(self, target_ratios: Dict[str, float], tolerance: float = 0.01) -> None:
        """Validate split ratios and adjacency constraints"""
        # Count splits
        split_counts = {split: 0 for split in target_ratios}
        for coord, split in self.split_assignments.items():
            split_counts[split] += 1

        # Check adjacency
        for (x, y), split in self.split_assignments.items():
            adjacent_coords = [
                (x+1, y), (x-1, y),
                (x, y+1), (x, y-1)
            ]
            for adj_x, adj_y in adjacent_coords:
                if (adj_x, adj_y) in self.split_assignments:
                    adj_split = self.split_assignments[(adj_x, adj_y)]
                    if adj_split == split:
                        self.logger.warning(
                            f"Adjacent tiles at ({x},{y}) and ({adj_x},{adj_y}) "
                            f"are both in {split} split"
                        )

    def load_metadata(self) -> None:
        """Load split assignments and coordinate mappings from metadata files."""
        metadata_dir = self.root_dir / 'metadata'

        # Load split mapping
        try:
            with open(metadata_dir / 'split_mapping.json', 'r') as f:
                split_data = json.load(f)
                self.split_assignments = {
                    tuple(map(int, coord.split(','))): split
                    for coord, split in split_data.items()
                }
        except FileNotFoundError:
            self.logger.warning("Split mapping file not found")

        # Load coordinate mapping
        try:
            with open(metadata_dir / 'coordinate_mapping.json', 'r') as f:
                coord_data = json.load(f)
                for path_str, info in coord_data.items():
                    self.tile_mapping[info['x'], info['y']] = TileInfo(
                        path=Path(path_str),
                        x=info['x'],
                        y=info['y'],
                        split=info['split']
                    )
        except FileNotFoundError:
            self.logger.warning("Coordinate mapping file not found")

    def _validate_child_grid(self, child_ref: str) -> bool:
        """
        Validate child grid reference format.
        Checks for format XXNNNN where XX is any two letters and NNNN are any four digits.

        Args:
            child_ref: Child grid reference (e.g., 'nj0957', 'nh7102', etc.)

        Returns:
            bool: True if valid format
        """
        if not child_ref:
            return False

        # Check format: any two letters followed by 4 digits
        pattern = re.compile(r'^[a-z]{2}\d{4}$', re.IGNORECASE)
        return bool(pattern.match(child_ref))

    def get_split_statistics(self) -> Dict[str, int]:
        """Get count of tiles in each split"""
        stats = {'train': 0, 'val': 0, 'test': 0}
        for _, split in self.split_assignments.items():
            stats[split] += 1
        return stats
```

## zip_handler.py
```python
import re
import shutil
import logging
import cv2
from pathlib import Path
from typing import Optional

from utils.data_extraction import extract_relevant_folders, convert_dem_asc_to_png
from utils.mask_processing.core import MaskProcessor, downscale_and_match_mask
from utils.mask_processing.visualization import visualize_masks
from utils.data_splitting import GeographicalDataHandler
from utils.path_handling.path_utils import PathManager
import yaml

# Configure logging
logger = logging.getLogger(__name__)

def process_zip_for_parent(
    zip_file_path: Path,
    parent_grid: str,
    mode: str,
    config_dict: dict
) -> bool:
    """
    Process a zip file containing grid square data.

    Args:
        zip_file_path: Path to the zip file
        parent_grid: Grid square identifier (e.g., 'NJ05')
        mode: Processing mode ('train' or 'evaluate')
        config_dict: Configuration dictionary

    Returns:
        bool: True if processing was successful
    """
    parent_grid = parent_grid.upper()
    pm = PathManager(config_dict)

    # Create parent folder structure
    paths = pm.create_parent_structure(parent_grid)

    # Setup processing directories
    extracted_dir = Path(config_dict['data']['raw_dir']) / f"{parent_grid}_extracted"
    processed_dir = Path(config_dict['data']['processed_dir']) / parent_grid / "raw"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Extract zip contents
    logger.info(f"Extracting {parent_grid} data from {zip_file_path}")
    if not extract_relevant_folders(zip_file_path, extracted_dir):
        logger.error(f"Failed to extract {zip_file_path}")
        return False

    # Locate required directories
    dsm_dir = next(extracted_dir.glob("getmapping-dsm-2000*"), None)
    rgb_dir = next(extracted_dir.glob("getmapping_rgb_25cm*"), None)
    if not (dsm_dir and rgb_dir):
        logger.error(f"{parent_grid}: Required directories not found")
        return False

    # Initialize processors
    mask_processor = MaskProcessor(config_dict['mask_processing'])
    grid_handler = GeographicalDataHandler(parent_grid=parent_grid,
                                         root_dir=Path(config_dict['data']['processed_dir']))

    processed_count = 0
    error_count = 0

    # Process each DSM and RGB pair
    for dsm_file in dsm_dir.glob("**/*.asc"):
        base_name = dsm_file.stem.split("_")[0].lower()
        rgb_file = next(rgb_dir.glob(f"**/{base_name}*.jpg"), None)

        if not rgb_file:
            logger.warning(f"No matching RGB file for {base_name}")
            continue

        try:
            # Get child paths
            child_paths = pm.get_paths_for_child(parent_grid, base_name)

            # Convert DSM to PNG
            if not convert_dem_asc_to_png(dsm_file, child_paths['raw']):
                continue

            # Generate and process masks
            masks = mask_processor.process_image(rgb_file)
            combined_mask = mask_processor.combine_masks(masks)
            cv2.imwrite(str(child_paths['mask']), combined_mask)

            # Create visualization if enabled
            if config_dict['mask_processing']['visualization']['enabled']:
                viz_path = paths['visualization'] / f"{base_name}_masks.png"
                rgb_image = cv2.imread(str(rgb_file))
                visualize_masks(masks, viz_path, rgb_image)

            # Register tile with grid handler
            match = re.match(r"^[a-z]{2}(\d{2})(\d{2})$", base_name.lower())
            if match:
                x_val = int(match.group(1))
                y_val = int(match.group(2))
                grid_handler.add_tile(child_paths['raw'], x_val, y_val)
            else:
                logger.warning(f"Could not parse x,y from {base_name}")
                continue

            processed_count += 1
            logger.info(f"Processed {parent_grid}/{base_name}")

        except Exception as e:
            logger.error(f"Error processing {base_name}: {str(e)}")
            error_count += 1
            continue

    # Log processing results
    logger.info(f"Completed {parent_grid} processing: {processed_count} successful, {error_count} failed")

    # Cleanup extracted files if configured
    if config_dict.get('cleanup_extracted', True):
        try:
            shutil.rmtree(extracted_dir, ignore_errors=True)
            logger.info(f"Cleaned up extracted folder for {parent_grid}")
        except Exception as e:
            logger.warning(f"Failed to cleanup extracted_dir for {parent_grid}: {e}")

    return processed_count > 0
```

## path_utils.py
```python
from pathlib import Path
import re
from typing import Dict
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class OutputPathConfig:
    """Configuration for output path structure"""
    base_dir: Path
    models_dir: Path
    inpainted_dir: str
    colored_dir: str
    visualization_dir: str
    masks_dir: str

class PathManager:
    def __init__(self, config: dict):
        self.config = config
        self.base_output_dir = Path(config['data']['output_dir'])
        self.base_processed_dir = Path(config['data']['processed_dir'])
        self.models_dir = Path(config['data']['models_dir'])

    def get_parent_from_zip(self, zip_path: Path) -> str:
        """Extract parent grid square from zip filename"""
        name = zip_path.stem.upper()
        if not self._validate_parent_grid(name):
            raise ValueError(f"Invalid parent grid square format: {name}")
        return name

    def _validate_parent_grid(self, grid_ref: str) -> bool:
        """
        Validate parent grid square format (e.g., 'NJ05', 'NH70')
        Any two letters followed by two digits
        """
        if not grid_ref:
            return False
        return (len(grid_ref) == 4 and
                grid_ref[:2].isalpha() and
                grid_ref[2:].isdigit())

    def _validate_child_grid(self, child_ref: str) -> bool:
        """
        Validate child grid reference format.
        Checks for format XXNNNN where XX is any two letters and NNNN are any four digits.

        Args:
            child_ref: Child grid reference (e.g., 'nj0957', 'nh7102', etc.)

        Returns:
            bool: True if valid format
        """
        if not child_ref:
            return False

        # Check format: any two letters followed by 4 digits
        pattern = re.compile(r'^[a-z]{2}\d{4}$', re.IGNORECASE)
        return bool(pattern.match(child_ref))

    def create_parent_structure(self, parent_grid: str) -> Dict[str, Path]:
        """Create complete directory structure for a parent grid square"""
        # Create processed data structure
        processed_parent = self.base_processed_dir / parent_grid
        for subdir in self.config['data']['parent_structure']['processed']:
            (processed_parent / subdir).mkdir(parents=True, exist_ok=True)

        # Create output structure
        output_parent = self.base_output_dir / parent_grid
        for subdir in self.config['data']['parent_structure']['output']:
            (output_parent / subdir).mkdir(parents=True, exist_ok=True)

        return {
            'processed': processed_parent,
            'processed_raw': processed_parent / 'raw',
            'processed_metadata': processed_parent / 'metadata',
            'output': output_parent,
            'output_inpainted': output_parent / 'inpainted',
            'output_colored': output_parent / 'colored',
            'visualization': output_parent / 'visualization',
            'masks': output_parent / 'masks'
        }

    def get_paths_for_child(self, parent_grid: str, child_name: str) -> Dict[str, Path]:
        """
        Get all relevant paths for a child grid square.

        Args:
            parent_grid: Parent grid square (e.g., 'NJ05')
            child_name: Child grid reference (e.g., 'nj0957')

        Returns:
            Dictionary of paths for the child
        """
        if not self._validate_child_grid(child_name):
            raise ValueError(f"Invalid child grid format: {child_name}")

        base_paths = self.create_parent_structure(parent_grid)
        return {
            'raw': base_paths['processed_raw'] / f"{child_name}.png",
            'mask': base_paths['processed_raw'] / f"{child_name}_mask_resized.png",
            'inpainted': base_paths['output_inpainted'] / f"{child_name}_inpainted.png",
            'colored': base_paths['output_colored'] / f"{child_name}_colored.png"
        }
```

## test_paths.py
```python
from pathlib import Path
import yaml
from path_utils import PathManager

def test_path_manager():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize PathManager
    pm = PathManager(config)

    # Test directory creation
    paths = pm.create_output_structure("NJ05")

    # Print created paths
    for key, path in paths.items():
        print(f"{key}: {path}")
        print(f"Exists: {path.exists()}")

if __name__ == "__main__":
    test_path_manager()
```

## main_pipeline_mlflow.py
```python
"""
Updated MLflow integration functions for main_pipeline.py.

These functions should replace or supplement the existing MLflow code
in main_pipeline.py to improve experiment tracking.
"""

import logging
import time
from pathlib import Path
import torch
from typing import Optional, Dict, Any, Union, Tuple

import mlflow
from utils.experiment_tracking import ExperimentTracker
from utils.mlflow_utils import initialize_mlflow, start_mlflow_server, stop_mlflow_server

logger = logging.getLogger(__name__)

# Global variables to track MLflow state
# MLFLOW_PID = None # No longer needed
EXPERIMENT_TRACKER = None

def setup_mlflow(config: Dict[str, Any],
                mode: str,
                log_dir: Path) -> Tuple[Optional[ExperimentTracker], str]:
    """
    Set up MLflow for the pipeline in a consistent way.

    Args:
        config: Configuration dictionary
        mode: Pipeline operation mode ('train', 'evaluate', 'human_guided_train')
        log_dir: Directory for logs

    Returns:
        tuple of (experiment_tracker, experiment_id)
    """
    global EXPERIMENT_TRACKER

    # Check if MLflow should be enabled
    tracking_enabled = config.get("experiment_tracking", {}).get("enabled", True)
    if not tracking_enabled:
        logger.info("MLflow experiment tracking is disabled in configuration")
        return None, ""

    try:
        # Get MLflow settings from config
        tracking_uri = config["experiment_tracking"].get("tracking_uri", "file:./mlruns")

        # Ensure tracking URI is correctly formatted
        if tracking_uri == "/mlruns":
            tracking_uri = "file:./mlruns"
        elif not tracking_uri.startswith("file:") and not tracking_uri.startswith("http:"):
            tracking_uri = f"file:{tracking_uri}"

        experiment_name = config["experiment_tracking"].get("experiment_name", "dsm_inpainting_master")

        # Initialize MLflow - THIS IS THE ONLY PLACE we call initialize_mlflow
        experiment_id = initialize_mlflow(tracking_uri, experiment_name)

        # Create experiment tracker if not already created
        if EXPERIMENT_TRACKER is None:
            EXPERIMENT_TRACKER = ExperimentTracker(
                experiment_name=experiment_name,
                tracking_uri=tracking_uri
            )
            logger.info(f"Created experiment tracker for {experiment_name}")

        return EXPERIMENT_TRACKER, experiment_id

    except Exception as e:
        logger.error(f"Failed to set up MLflow: {str(e)}")
        return None, ""

def cleanup_mlflow():
    """Clean up MLflow resources"""
    global EXPERIMENT_TRACKER # MLFLOW_PID no longer needed

    try:
        # End any active MLflow run
        if EXPERIMENT_TRACKER and EXPERIMENT_TRACKER.is_run_active:
            EXPERIMENT_TRACKER.end_run()
            logger.info("Ended active MLflow run")

        # Stop MLflow server if it was started - THIS IS HANDLED IN run_pipeline.sh
        # if MLFLOW_PID:
        #     success = stop_mlflow_server(MLFLOW_PID)
        #     if success:
        #         logger.info(f"Stopped MLflow server (PID: {MLFLOW_PID})")
        #     MLFLOW_PID = None
    except Exception as e:
        logger.error(f"Error cleaning up MLflow: {str(e)}")

def start_run_for_mode(mode: str, config: Dict[str, Any]) -> Optional[str]:
    """
    Start an appropriate MLflow run for the given mode.

    Args:
        mode: Pipeline operation mode
        config: Configuration dictionary

    Returns:
        run_id: ID of the started run, or None if failed
    """
    global EXPERIMENT_TRACKER
    if not EXPERIMENT_TRACKER:
        logger.warning("No experiment tracker available. Run start/setup failed previously.")
        return None  # Don't call setup_mlflow again.
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Create appropriate run based on mode
        if mode == "train":
            run_name = f"training_run_{timestamp}"
        elif mode == "evaluate":
            run_name = f"evaluation_run_{timestamp}"
        elif mode == "human_guided_train":
            run_name = f"human_guided_{timestamp}"
        else:
            run_name = f"run_{mode}_{timestamp}"

        # Start the run - ONLY CALL start_run here, not in training loops
        run = EXPERIMENT_TRACKER.start_run(run_name, config)
        if run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run: {run_id} ({run_name})")
            return run_id
        return None

    except Exception as e:
        logger.error(f"Failed to start MLflow run for {mode}: {str(e)}")
        return None

def log_model_safely(model: torch.nn.Module,
                    name: str,
                    metrics: Optional[Dict[str, Any]] = None,
                    save_path: Optional[Path] = None) -> bool:
    """
    Safely log a model to MLflow and optionally save it locally.

    Args:
        model: PyTorch model to log
        name: Name for the model
        metrics: Optional metrics to log with the model
        save_path: Optional path to save the model locally

    Returns:
        success: True if logging was successful
    """
    global EXPERIMENT_TRACKER

    if not EXPERIMENT_TRACKER:
        # If no experiment tracker but save path provided, just save locally
        if save_path:
            try:
                torch.save(model.state_dict(), save_path)
                logger.info(f"Saved model to {save_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to save model to {save_path}: {str(e)}")
                return False
        return False

    try:
        # Save original device
        original_device = next(model.parameters()).device

        # Log model to MLflow
        EXPERIMENT_TRACKER.log_model(model, name, metrics)
        logger.info(f"Logged model {name} to MLflow")

        # Optionally save locally
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved model to {save_path}")

        return True

    except Exception as e:
        logger.error(f"Failed to log model {name}: {str(e)}")

        # Attempt local save as fallback
        if save_path:
            try:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)
                logger.info(f"Saved model to {save_path} (fallback)")
                return True
            except Exception as nested_e:
                logger.error(f"Failed to save model to {save_path}: {str(nested_e)}")

        return False

def log_training_completion(results: Dict[str, Any]) -> bool:
    """
    Log training completion metrics.

    Args:
        results: Dictionary of training results

    Returns:
        success: True if logging was successful
    """
    global EXPERIMENT_TRACKER

    if not EXPERIMENT_TRACKER:
        return False

    try:
        # Convert results to appropriate format
        metrics = {}
        for key, value in results.items():
            if isinstance(value, (int, float)):
                metrics[f"training.{key}"] = value

        # Log metrics
        if metrics:
            EXPERIMENT_TRACKER.log_metrics(metrics)
            logger.info(f"Logged {len(metrics)} training completion metrics")

        return True

    except Exception as e:
        logger.error(f"Failed to log training completion: {str(e)}")
        return False
```


## checkpoint_utils.py
```python
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

def validate_checkpoint(path: Path) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Validate a checkpoint file and return its contents if valid.

    Args:
        path: Path to checkpoint file

    Returns:
        Tuple of (is_valid, checkpoint_dict)
    """
    try:
        if not path.exists():
            logger.error(f"Checkpoint file not found: {path}")
            return False, None

        checkpoint = torch.load(path, map_location='cpu')

        # Validate checkpoint structure
        required_keys = {'epoch', 'generator_state_dict', 'optimizer_G_state_dict'}
        if isinstance(checkpoint, dict):
            missing_keys = required_keys - set(checkpoint.keys())
            if missing_keys:
                logger.error(f"Checkpoint missing required keys: {missing_keys}")
                return False, None

            # Additional validation could be added here
            return True, checkpoint
        else:
            # Handle legacy format (just state dict)
            logger.warning("Legacy checkpoint format detected")
            return True, {'generator_state_dict': checkpoint}

    except Exception as e:
        logger.error(f"Failed to validate checkpoint: {str(e)}")
        return False, None

def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> bool:
    """
    Safely load a checkpoint into model and optimizer.

    Args:
        path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into

    Returns:
        bool: Whether loading was successful
    """
    is_valid, checkpoint = validate_checkpoint(path)
    if not is_valid:
        return False

    try:
        if 'generator_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['generator_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        if optimizer is not None and 'optimizer_G_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_G_state_dict'])

        return True
    except Exception as e:
        logger.error(f"Failed to load checkpoint contents: {str(e)}")
        return False

def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
                   epoch: Optional[int] = None, **kwargs) -> bool:
    """
    Safely save a checkpoint with model and optimizer state.

    Args:
        path: Path to save checkpoint to
        model: Model to save
        optimizer: Optional optimizer to save
        epoch: Optional epoch number
        **kwargs: Additional items to save in checkpoint

    Returns:
        bool: Whether saving was successful
    """
    try:
        checkpoint = {
            'generator_state_dict': model.state_dict(),
            'epoch': epoch if epoch is not None else 0,
            **kwargs
        }

        if optimizer is not None:
            checkpoint['optimizer_G_state_dict'] = optimizer.state_dict()

        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save atomically using temporary file
        temp_path = path.with_suffix('.tmp')
        torch.save(checkpoint, temp_path)
        temp_path.replace(path)

        return True
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {str(e)}")
        if temp_path.exists():
            temp_path.unlink()
        return False
```

## pythonanywhere_cleanup.py
```python
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
```

## portal_client.py
```python
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
```

## pythonanywhere_downloader.py
```python
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
```

## mlflow_utils.py
```python
"""
MLflow utility functions for consistent experiment tracking setup.

This module provides helper functions for MLflow configuration, management,
and cleanup to ensure consistent usage across the application.
"""

import os
import time
import logging
import subprocess
import shutil
from pathlib import Path
import mlflow
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

def initialize_mlflow(tracking_uri: str = "http://localhost:5000",
                     experiment_name: str = "dsm_inpainting_master") -> str:
    """
    Initialize MLflow with consistent settings.

    Args:
        tracking_uri: URI for MLflow tracking server
        experiment_name: Name of the experiment to use

    Returns:
        experiment_id: ID of the initialized experiment
    """
    # Configure MLflow - just once in a central location
    # os.environ["MLFLOW_TRACKING_URI"] = tracking_uri # NOT NEEDED.
    mlflow.set_tracking_uri(tracking_uri)

    # Get or create the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created new MLflow experiment: {experiment_name}")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing MLflow experiment: {experiment_name}")

    return experiment_id

def start_mlflow_server(port: int = 5000, log_file: Optional[Path] = None) -> Optional[int]:
    """
    Start MLflow server as a background process.  DEPRECATED. Use start_mlflow.sh

    Args:
        port: Port to run MLflow server on
        log_file: Optional path to log file

    Returns:
        process_id: PID of the started MLflow server, or None if failed
    """
    logger.warning("DEPRECATED: Use start_mlflow.sh to start the MLflow server.")
    return None

def stop_mlflow_server(pid: Optional[int] = None) -> bool:
    """
    Stop MLflow server process. DEPRECATED. Use start_mlflow.sh

    Args:
        pid: Optional process ID to kill specifically

    Returns:
        success: True if server was stopped successfully
    """
    logger.warning("DEPRECATED: Use start_mlflow.sh to stop the MLflow server.")
    return False

def clean_mlflow_directory(directory: str = "./mlruns",
                          keep_experiments: bool = True,
                          keep_metadata: bool = True) -> bool:
    """
    Clean up MLflow directory, removing empty files and unnecessary artifacts.

    Args:
        directory: Path to MLflow directory
        keep_experiments: Whether to keep experiment definitions
        keep_metadata: Whether to keep run metadata

    Returns:
        success: True if cleanup was successful
    """
    try:
        mlruns_dir = Path(directory)
        if not mlruns_dir.exists():
            logger.warning(f"MLflow directory {directory} does not exist")
            return False

        # Define directories to keep
        keep_dirs = set()
        if keep_experiments:
            # Keep experiment directories
            keep_dirs.update([
                d for d in mlruns_dir.glob("*")
                if d.is_dir() and not d.name.startswith(".")
            ])

        # Handle metadata files
        if keep_metadata:
            # Keep meta.yaml files
            for meta_file in mlruns_dir.glob("**/meta.yaml"):
                keep_dirs.add(meta_file.parent)

        # Remove empty files
        empty_files_count = 0
        for file_path in mlruns_dir.glob("**/*"):
            if file_path.is_file() and file_path.stat().st_size == 0:
                if any(str(file_path).startswith(str(d)) for d in keep_dirs):
                    # This is in a keep directory, check if it's a parameter file
                    if file_path.parent.name != "params":
                        file_path.unlink()
                        empty_files_count += 1
                else:
                    file_path.unlink()
                    empty_files_count += 1

        logger.info(f"Removed {empty_files_count} empty files from MLflow directory")
        return True

    except Exception as e:
        logger.error(f"Error cleaning MLflow directory: {str(e)}")
        return False

def reset_mlflow_environment(directory: str = "./mlruns",
                            experiment_name: str = "dsm_inpainting_master") -> bool:
    """
    Completely reset the MLflow environment.

    Args:
        directory: Path to MLflow directory
        experiment_name: Name of the experiment to recreate

    Returns:
        success: True if reset was successful
    """
    try:
        mlruns_dir = Path(directory)

        # Stop MLflow server if running.  This is redundant, but harmless
        stop_mlflow_server()

        # Back up the directory if it exists
        if mlruns_dir.exists():
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_dir = mlruns_dir.parent / f"mlruns_backup_{timestamp}"
            shutil.copytree(mlruns_dir, backup_dir)
            logger.info(f"Backed up MLflow directory to {backup_dir}")

            # Remove the original directory
            shutil.rmtree(mlruns_dir)
            logger.info(f"Removed MLflow directory {mlruns_dir}")

        # Create base structure
        mlruns_dir.mkdir(exist_ok=True)

        # Create experiment directory
        experiment_dir = mlruns_dir / experiment_name
        experiment_dir.mkdir(exist_ok=True)

        # Create required subdirectories
        for subdir in ["artifacts", "metrics", "params", "tags"]:
            (experiment_dir / subdir).mkdir(exist_ok=True)

        # Create meta.yaml
        current_time = int(time.time() * 1000)
        meta_content = {
            "artifact_location": f"file:{directory}/{experiment_name}",
            "creation_time": current_time,
            "experiment_id": experiment_name,
            "last_update_time": current_time,
            "lifecycle_stage": "active",
            "name": experiment_name
        }

        import yaml
        with open(experiment_dir / "meta.yaml", "w") as f:
            yaml.dump(meta_content, f)

        logger.info(f"Successfully reset MLflow environment with experiment {experiment_name}")
        return True

    except Exception as e:
        logger.error(f"Error resetting MLflow environment: {str(e)}")
        return False

def check_run_exists(run_id: str) -> bool:
    """
    Check if a run with the given ID exists.

    Args:
        run_id: MLflow run ID to check

    Returns:
        exists: True if run exists
    """
    try:
        run = mlflow.get_run(run_id)
        return run is not None
    except Exception:
        return False

def batch_log_metrics(metrics: Dict[str, Any],
                     step: Optional[int] = None,
                     run_id: Optional[str] = None) -> bool:
    """
    Log multiple metrics efficiently in a single operation.

    Args:
        metrics: Dictionary of metrics to log
        step: Optional step number
        run_id: Optional run ID to log to

    Returns:
        success: True if logging was successful
    """
    try:
        # Organize metrics by category
        categorized = {}
        for key, value in metrics.items():
            # Skip non-numeric values
            if not isinstance(value, (int, float)) and not (
                hasattr(value, "item") and callable(getattr(value, "item"))
            ):
                continue

            # Extract category from key
            if "." in key:
                category, metric_name = key.split(".", 1)
            else:
                category = "metrics"
                metric_name = key

            if category not in categorized:
                categorized[category] = {}

            # Convert to standard Python numeric type
            if hasattr(value, "item") and callable(getattr(value, "item")):
                value = value.item()

            categorized[category][f"{category}.{metric_name}"] = float(value)

        # Log metrics by category
        with mlflow.start_run(run_id=run_id, nested=True) as run: #Always nest.
            for metrics_dict in categorized.values():
                if metrics_dict:
                    mlflow.log_metrics(metrics_dict, step=step)

        return True

    except Exception as e:
        logger.error(f"Error batch logging metrics: {str(e)}")
        return False
```

## start_mlflow.sh
```bash
#!/bin/bash
#
# start_mlflow.sh - Improved script to start MLflow tracking server
#
# This script properly manages MLflow server processes and provides
# better logging and error handling.

# Configuration (Allow port to be passed as an argument)
MLFLOW_PORT="${1:-5000}"  # Use argument if provided, otherwise default to 5000
MLFLOW_HOST="0.0.0.0"
MLFLOW_WORKERS=4
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/mlflow_${TIMESTAMP}.log"
PIDFILE="${LOG_DIR}/mlflow.pid"

# Log messages with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Make sure logs directory exists
mkdir -p "$LOG_DIR"
chmod -R 755 "$LOG_DIR"

# Check if mlruns directory exists and create it if needed
if [ ! -d "./mlruns" ]; then
    mkdir -p "./mlruns"
    mkdir -p "./mlruns/models"
    mkdir -p "./mlruns/.trash"
    chmod -R 755 "./mlruns"
fi

# Kill any existing MLflow processes
stop_mlflow() {
    if [ -f "$PIDFILE" ]; then
        local pid=$(cat "$PIDFILE")
        log "Stopping MLflow server (PID: $pid)"
        kill -TERM "$pid" 2>/dev/null || true # Graceful termination
        wait "$pid" 2>/dev/null || true  # Wait for process to exit
        rm -f "$PIDFILE"
        log "MLflow server stopped."
    else
        log "No MLflow server PID file found. Checking for running processes..."
        local pids=$(pgrep -f "mlflow.*server")
        if [ -n "$pids" ]; then
            log "Found running MLflow processes: $pids. Terminating..."
            kill -TERM $pids 2>/dev/null || true
        else
            log "No running MLflow server found."
        fi
    fi
}

# Check if port is already in use
check_port() {
    local port="$1"
    if [ -n "$(netstat -tuln 2>/dev/null | grep ":$port ")" ]; then
        log "Error: Port $port is already in use. Please choose a different port."
        exit 1
    fi
}

# Start MLflow server
start_mlflow() {
    local port="$1"
    check_port "$port"

    log "Starting MLflow server on $MLFLOW_HOST:$port (logging to $LOG_FILE)"

    nohup mlflow server \
        --host "$MLFLOW_HOST" \
        --port "$port" \
        --backend-store-uri "file:./mlruns" \
        --default-artifact-root "file:./mlruns" \
        --workers $MLFLOW_WORKERS > "$LOG_FILE" 2>&1 &

    local pid=$!
    echo "$pid" > "$PIDFILE"
    chmod 644 "$PIDFILE"
    log "MLflow server started with PID: $pid"
    log "Server UI is available at http://$MLFLOW_HOST:$port"
}

stop_mlflow #Always stop existing server
start_mlflow "$MLFLOW_PORT"
```

## plot_research_metrics.py
```python
#!/usr/bin/env python3
"""
MLflow Research Metrics Visualization

Creates publication-quality visualizations for research purposes with:
- Normalized timeline (t=0 at first run)
- Separate files for each metric
- Raw data points without aggregation
- Clear phase transitions

Usage:
    python plot_research_metrics.py --experiment-name <name> --tracking-uri <uri> --output-dir <dir>
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import re

# Set plot style for research/publication quality
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

class ResearchMetricVisualizer:
    def __init__(self, experiment_name, tracking_uri=None, output_dir=None):
        """
        Initialize the research metric visualizer.

        Args:
            experiment_name: Name of the MLflow experiment to visualize
            tracking_uri: MLflow tracking URI
            output_dir: Directory to save visualization outputs
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "file:./mlruns"
        self.output_dir = Path(output_dir or f"research_viz_{experiment_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data structures
        self.runs_df = None
        self.runs_by_phase = None
        self.metrics_data = None
        self.t0 = None  # Start time reference for normalization

        # Configure MLflow client
        mlflow.set_tracking_uri(self.tracking_uri)

        # Get experiment
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        if not self.experiment:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        print(f"Found experiment '{experiment_name}' with ID: {self.experiment.experiment_id}")

    def load_runs_data(self):
        """Load runs data and prepare for visualization with normalized timeline."""
        # Get all runs for the experiment
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=["attribute.start_time ASC"]
        )

        if runs.empty:
            raise ValueError(f"No runs found for experiment '{self.experiment_name}'")

        # Clean up the DataFrame
        self.runs_df = runs.copy()

        # Parse timestamps
        self.runs_df['start_time_dt'] = pd.to_datetime(self.runs_df['start_time'], unit='ms')
        self.runs_df['end_time_dt'] = pd.to_datetime(self.runs_df['end_time'], unit='ms')

        # Set t0 as the earliest start time
        self.t0 = self.runs_df['start_time_dt'].min()

        # Calculate elapsed time in seconds from t0
        self.runs_df['elapsed_seconds'] = (self.runs_df['start_time_dt'] - self.t0).dt.total_seconds()

        # Calculate run duration
        self.runs_df['duration_seconds'] = (self.runs_df['end_time_dt'] - self.runs_df['start_time_dt']).dt.total_seconds()

        # Categorize runs by phase based on run name pattern
        self.runs_df['phase'] = 'unknown'

        # Use run name to categorize when available
        if 'tags.mlflow.runName' in self.runs_df.columns:
            # Training runs
            mask_training = self.runs_df['tags.mlflow.runName'].str.contains('train', case=False, na=False)
            self.runs_df.loc[mask_training, 'phase'] = 'training'

            # Evaluation runs
            mask_eval = self.runs_df['tags.mlflow.runName'].str.contains('eval', case=False, na=False)
            self.runs_df.loc[mask_eval, 'phase'] = 'evaluation'

            # Human-guided runs
            mask_human = self.runs_df['tags.mlflow.runName'].str.contains('human|guided', case=False, na=False)
            self.runs_df.loc[mask_human, 'phase'] = 'human_guided'

        # Group runs by phase
        self.runs_by_phase = {
            'training': self.runs_df[self.runs_df['phase'] == 'training'],
            'evaluation': self.runs_df[self.runs_df['phase'] == 'evaluation'],
            'human_guided': self.runs_df[self.runs_df['phase'] == 'human_guided'],
            'unknown': self.runs_df[self.runs_df['phase'] == 'unknown']
        }

        # Log run counts by phase
        for phase, df in self.runs_by_phase.items():
            print(f"Found {len(df)} {phase} runs")

        return self.runs_df

    def extract_metrics_data(self):
        """
        Extract metrics data and steps to create time series for each metric.
        This extracts raw individual data points rather than aggregates.
        """
        if self.runs_df is None:
            self.load_runs_data()

        # Collect all metrics data
        metrics_data = []

        # Get metrics from MLflow API for each run
        for idx, row in self.runs_df.iterrows():
            run_id = row['run_id']
            phase = row['phase']
            run_start_time = row['start_time_dt']
            elapsed_seconds = row['elapsed_seconds']

            try:
                # Get metrics history
                client = mlflow.tracking.MlflowClient()
                metrics_history = client.get_metric_history(run_id, '_step')

                # If no steps found, try to extract metrics directly from the DataFrame
                if not metrics_history:
                    metrics_cols = [col for col in self.runs_df.columns if col.startswith('metrics.')]

                    for col in metrics_cols:
                        metric_name = col.replace('metrics.', '')
                        value = row[col]

                        if pd.notna(value):
                            metrics_data.append({
                                'run_id': run_id,
                                'phase': phase,
                                'metric_name': metric_name,
                                'value': value,
                                'step': 0,  # Default step
                                'start_time': run_start_time,
                                'elapsed_seconds': elapsed_seconds
                            })
                else:
                    # Get all metrics for this run
                    for metric_name in set([m.key for m in client.get_metric_history(run_id, '*')]):
                        if metric_name == '_step':
                            continue

                        metric_history = client.get_metric_history(run_id, metric_name)

                        for metric in metric_history:
                            # Convert timestamp to relative seconds
                            metric_time = pd.to_datetime(metric.timestamp, unit='ms')
                            metric_elapsed = (metric_time - self.t0).total_seconds()

                            metrics_data.append({
                                'run_id': run_id,
                                'phase': phase,
                                'metric_name': metric_name,
                                'value': metric.value,
                                'step': metric.step,
                                'start_time': metric_time,
                                'elapsed_seconds': metric_elapsed
                            })
            except Exception as e:
                print(f"Error getting metrics for run {run_id}: {e}")

        # Convert to DataFrame
        self.metrics_data = pd.DataFrame(metrics_data)

        if self.metrics_data.empty:
            print("Warning: No metrics data found. Falling back to run-level metrics.")
            # Create metrics data from run-level metrics
            metrics_cols = [col for col in self.runs_df.columns if col.startswith('metrics.')]
            metrics_data = []

            for idx, row in self.runs_df.iterrows():
                for col in metrics_cols:
                    if pd.notna(row[col]):
                        metrics_data.append({
                            'run_id': row['run_id'],
                            'phase': row['phase'],
                            'metric_name': col.replace('metrics.', ''),
                            'value': row[col],
                            'step': 0,  # No step information
                            'start_time': row['start_time_dt'],
                            'elapsed_seconds': row['elapsed_seconds']
                        })

            self.metrics_data = pd.DataFrame(metrics_data)

        print(f"Extracted {len(self.metrics_data)} metric data points")
        return self.metrics_data

    def plot_metric_by_time(self, metric_name, figsize=(10, 6)):
        """
        Create a separate plot for a specific metric over normalized time.

        Args:
            metric_name: Name of the metric to plot
            figsize: Figure size (width, height) in inches
        """
        if self.metrics_data is None:
            self.extract_metrics_data()

        # Filter data for this metric
        metric_data = self.metrics_data[self.metrics_data['metric_name'] == metric_name]

        if metric_data.empty:
            print(f"No data found for metric '{metric_name}'")
            return None

        # Create figure with proper size for publication
        fig, ax = plt.subplots(figsize=figsize, dpi=150)

        # Phase styles for clear distinction
        phase_styles = {
            'training': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'label': 'Training'},
            'evaluation': {'color': '#2ca02c', 'marker': 's', 'linestyle': '--', 'label': 'Evaluation'},
            'human_guided': {'color': '#d62728', 'marker': '^', 'linestyle': '-.', 'label': 'Human-Guided'},
            'unknown': {'color': '#7f7f7f', 'marker': 'x', 'linestyle': ':', 'label': 'Unknown'}
        }

        # Plot by phase with consistent ordering to ensure proper legend
        handles = []
        labels = []

        for phase in ['training', 'evaluation', 'human_guided', 'unknown']:
            phase_data = metric_data[metric_data['phase'] == phase]

            if not phase_data.empty:
                style = phase_styles[phase]

                # Sort by elapsed time
                phase_data = phase_data.sort_values('elapsed_seconds')

                # Plot individual points with connecting lines
                line, = ax.plot(phase_data['elapsed_seconds'], phase_data['value'],
                          marker=style['marker'], color=style['color'],
                          linestyle=style['linestyle'], label=style['label'],
                          markersize=6, markeredgewidth=1, markeredgecolor='black',
                          alpha=0.8)

                handles.append(line)
                labels.append(style['label'])

        # Add clear phase transition markers
        if not self.runs_df.empty:
            phase_changes = []
            sorted_runs = self.runs_df.sort_values('elapsed_seconds')

            if len(sorted_runs) > 1:
                for i in range(1, len(sorted_runs)):
                    if sorted_runs.iloc[i-1]['phase'] != sorted_runs.iloc[i]['phase']:
                        transition_time = sorted_runs.iloc[i]['elapsed_seconds']
                        phase_changes.append((transition_time, sorted_runs.iloc[i]['phase']))

            # Plot phase transitions
            for t, phase in phase_changes:
                style = phase_styles.get(phase, phase_styles['unknown'])
                ax.axvline(x=t, color=style['color'], linestyle='--', alpha=0.7,
                          linewidth=2)

                # Add text label for phase transition
                y_pos = ax.get_ylim()[0] + 0.9 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                ax.text(t + 10, y_pos, f"{phase.replace('_', ' ').title()} →",
                      ha='left', va='top', fontsize=10, color=style['color'],
                      bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

        # Add descriptive labels
        ax.set_xlabel('Time from Experiment Start (seconds)')
        ax.set_ylabel(metric_name)

        # Add properly formatted title
        metric_title = metric_name.replace('_', ' ').title()
        plt.title(f"{metric_title} Over Time", fontweight='bold')

        # Add clear legend outside plot
        if handles:
            ax.legend(handles, labels, loc='best', frameon=True, framealpha=0.9)

        # Format axes for readability
        ax.grid(True, linestyle='--', alpha=0.7)

        # Add experiment info
        plt.text(0.01, 0.01, f"Experiment: {self.experiment_name}",
                transform=ax.transAxes, fontsize=8, alpha=0.7)

        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        plt.text(0.99, 0.01, f"Generated: {timestamp}",
                transform=ax.transAxes, fontsize=8,
                ha='right', alpha=0.7)

        # Tight layout for better spacing
        plt.tight_layout()

        # Save figure with high quality and clear naming
        safe_metric_name = re.sub(r'[^\w\-_]', '_', metric_name)
        output_path = self.output_dir / f"{safe_metric_name}_time_series.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved {metric_name} plot to {output_path}")

        return fig, ax

    def create_all_metric_plots(self):
        """Generate separate plots for all available metrics."""
        if self.metrics_data is None:
            self.extract_metrics_data()

        # Get unique metrics
        unique_metrics = self.metrics_data['metric_name'].unique()

        # Create metrics subfolder for organization
        metrics_dir = self.output_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)

        # Store original output dir and set to metrics subfolder
        original_output_dir = self.output_dir
        self.output_dir = metrics_dir

        # Track metrics for summary
        metric_plots = []

        # Plot each metric separately
        for metric in unique_metrics:
            try:
                fig, ax = self.plot_metric_by_time(metric)
                plt.close(fig)  # Close to save memory
                metric_plots.append(metric)
            except Exception as e:
                print(f"Error plotting metric {metric}: {e}")

        # Restore original output directory
        self.output_dir = original_output_dir

        # Create summary index
        with open(self.output_dir / "metrics_index.txt", "w") as f:
            f.write(f"Metrics visualized for experiment: {self.experiment_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total metrics: {len(metric_plots)}\n\n")

            for i, metric in enumerate(sorted(metric_plots), 1):
                f.write(f"{i}. {metric}\n")

        print(f"Created plots for {len(metric_plots)} metrics in {metrics_dir}")

    def plot_loss_metrics(self):
        """Create plots specifically for loss-related metrics."""
        if self.metrics_data is None:
            self.extract_metrics_data()

        # Filter for loss-related metrics
        loss_metrics = [m for m in self.metrics_data['metric_name'].unique()
                       if 'loss' in m.lower()]

        if not loss_metrics:
            print("No loss-related metrics found")
            return []

        # Create loss subfolder
        loss_dir = self.output_dir / "loss_metrics"
        loss_dir.mkdir(exist_ok=True)

        # Store original output dir and set to loss subfolder
        original_output_dir = self.output_dir
        self.output_dir = loss_dir

        # Plot each loss metric
        loss_plots = []
        for metric in loss_metrics:
            try:
                fig, ax = self.plot_metric_by_time(metric)
                plt.close(fig)  # Close to save memory
                loss_plots.append(metric)
            except Exception as e:
                print(f"Error plotting loss metric {metric}: {e}")

        # Restore original output directory
        self.output_dir = original_output_dir

        print(f"Created plots for {len(loss_plots)} loss metrics in {loss_dir}")
        return loss_plots

    def create_metrics_table(self):
        """Create a detailed CSV table of metrics data for further analysis."""
        if self.metrics_data is None:
            self.extract_metrics_data()

        # Save full metrics data for reference
        output_path = self.output_dir / "all_metrics_data.csv"
        self.metrics_data.to_csv(output_path, index=False)

        # Create pivot table by phase and metric for summary statistics
        pivot_df = self.metrics_data.pivot_table(
            values='value',
            index=['phase', 'metric_name'],
            aggfunc=['mean', 'min', 'max', 'count']
        ).reset_index()

        # Flatten multi-level columns
        pivot_df.columns = ['_'.join(col).strip('_') for col in pivot_df.columns.values]

        # Save summary table
        summary_path = self.output_dir / "metrics_summary_by_phase.csv"
        pivot_df.to_csv(summary_path, index=False)

        print(f"Saved detailed metrics data to {output_path}")
        print(f"Saved metrics summary to {summary_path}")

        return output_path, summary_path

    def create_full_research_report(self):
        """Generate complete research-focused analysis with all visualizations."""
        # Load all data
        self.load_runs_data()

        # Extract detailed metrics
        self.extract_metrics_data()

        # Create plots for all metrics
        self.create_all_metric_plots()

        # Create special loss metric plots
        self.plot_loss_metrics()

        # Create detailed metrics tables
        self.create_metrics_table()

        # Save run information
        runs_path = self.output_dir / "experiment_runs_info.csv"
        self.runs_df.to_csv(runs_path, index=False)

        print(f"Full research report generated in {self.output_dir}")

        # Create simple HTML index for easier navigation
        self._create_html_index()

    def _create_html_index(self):
        """Create a simple HTML index for easier browsing of outputs."""
        index_path = self.output_dir / "index.html"

        with open(index_path, "w") as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Research Metrics: {self.experiment_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        .section {{ margin-bottom: 30px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}
        .metric-card {{ border: 1px solid #ddd; padding: 10px; border-radius: 5px; }}
        img {{ max-width: 100%; height: auto; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Research Metrics Visualization</h1>
    <p>Experiment: <strong>{self.experiment_name}</strong></p>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="section">
        <h2>Loss Metrics</h2>
        <p>Dedicated visualizations for loss-related metrics:</p>
        <div class="metrics-grid">
""")

            # Add loss metrics
            loss_dir = self.output_dir / "loss_metrics"
            if loss_dir.exists():
                for img_file in sorted(loss_dir.glob("*_time_series.png")):
                    metric_name = img_file.stem.replace("_time_series", "").replace("_", " ").title()
                    rel_path = img_file.relative_to(self.output_dir)
                    f.write(f"""
            <div class="metric-card">
                <h3>{metric_name}</h3>
                <a href="{rel_path}"><img src="{rel_path}" alt="{metric_name}"></a>
            </div>
""")

            f.write("""
        </div>
    </div>

    <div class="section">
        <h2>All Metrics</h2>
        <p>Complete set of metric visualizations:</p>
        <div class="metrics-grid">
""")

            # Add all metrics
            metrics_dir = self.output_dir / "metrics"
            if metrics_dir.exists():
                for img_file in sorted(metrics_dir.glob("*_time_series.png")):
                    metric_name = img_file.stem.replace("_time_series", "").replace("_", " ").title()
                    rel_path = img_file.relative_to(self.output_dir)
                    f.write(f"""
            <div class="metric-card">
                <h3>{metric_name}</h3>
                <a href="{rel_path}"><img src="{rel_path}" alt="{metric_name}"></a>
            </div>
""")

            f.write("""
        </div>
    </div>

    <div class="section">
        <h2>Data Tables</h2>
        <ul>
""")

            # Add links to CSV files
            for csv_file in self.output_dir.glob("*.csv"):
                rel_path = csv_file.relative_to(self.output_dir)
                f.write(f'            <li><a href="{rel_path}">{csv_file.name}</a></li>\n')

            f.write("""
        </ul>
    </div>
</body>
</html>
""")

        print(f"Created HTML index at {index_path}")


def main():
    parser = argparse.ArgumentParser(description="Create research-quality metric visualizations from MLflow data")
    parser.add_argument("--experiment-name", required=True, help="Name of the MLflow experiment")
    parser.add_argument("--tracking-uri", default="file:./mlruns", help="MLflow tracking URI")
    parser.add_argument("--output-dir", default=None, help="Output directory for visualizations")
    parser.add_argument("--metrics", nargs='+', help="Specific metrics to plot (default: all)")
    parser.add_argument("--loss-only", action="store_true", help="Only plot loss-related metrics")

    args = parser.parse_args()

    # Create visualizer
    visualizer = ResearchMetricVisualizer(
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        output_dir=args.output_dir
    )

    # Generate visualizations based on arguments
    visualizer.load_runs_data()
    visualizer.extract_metrics_data()

    if args.metrics:
        for metric in args.metrics:
            visualizer.plot_metric_by_time(metric)
    elif args.loss_only:
        visualizer.plot_loss_metrics()
    else:
        visualizer.create_full_research_report()


if __name__ == "__main__":
    main()
```


## download_all_annotations.py
```python
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
```

## evaluate.py
```python
# src/evaluate.py

import torch
from torchvision import transforms
from PIL import Image
from .models.generator import PConvUNet

def evaluate(image_path, mask_path, model_or_checkpoint_path, save_path):
    """
    Evaluate a model on a single image.

    Args:
        image_path: Path to the input image
        mask_path: Path to the mask
        model_or_checkpoint_path: Either a PConvUNet model instance or path to a checkpoint
        save_path: Path to save the inpainted image
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Adjust image size to match training
    img_size = (512, 512)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('L')
    mask = Image.open(mask_path).convert('L')

    image = transform(image).unsqueeze(0).to(device)
    mask = transform(mask).unsqueeze(0).to(device)
    mask = (mask > 0).float()  # Binarize the mask
    masked_img = image * mask

    # Handle either model instance or checkpoint path
    if isinstance(model_or_checkpoint_path, PConvUNet):
        generator = model_or_checkpoint_path
    else:
        generator = PConvUNet().to(device)
        # Update checkpoint loading with weights_only=True
        checkpoint = torch.load(model_or_checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['generator_state_dict'])
        else:
            generator.load_state_dict(checkpoint)

    generator.eval()

    with torch.no_grad():
        output = generator(masked_img, mask)

    # Convert tensors to images for saving
    output_img = output.cpu().squeeze().numpy()
    output_img = (output_img * 255).astype('uint8')
    output_pil = Image.fromarray(output_img, mode='L')

    # Resize to 500x500 if needed
    output_pil = output_pil.resize((500, 500), Image.BILINEAR)
    output_pil.save(save_path)
    print(f"Inpainted image saved to {save_path}")
```

## train.py
```python
# src/train.py

from typing import Dict, Optional
import torch
import time
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import numpy as np

from mvp_gan import ExperimentTracker
from .models.generator import PConvUNet
from .models.discriminator import Discriminator
from .utils.dataset import InpaintingDataset
from .utils.losses import InpaintingLoss, HumanGuidedLoss
from .evaluation.metrics import MaskEvaluator

logger = logging.getLogger(__name__)

def train(img_dir: Path,
         mask_dir: Path,
         generator: Optional[PConvUNet] = None,
         discriminator: Optional[Discriminator] = None,
         optimizer_G: Optional[torch.optim.Optimizer] = None,
         optimizer_D: Optional[torch.optim.Optimizer] = None,
         checkpoint_path: Optional[Path] = None,
         config: Optional[Dict] = None,
         experiment_tracker = None,
         val_img_dir: Optional[Path] = None,
         val_mask_dir: Optional[Path] = None):
    """
    Train the GAN model with validation-based model selection and experiment tracking.

    Args:
        img_dir: Directory containing training images
        mask_dir: Directory containing training masks
        generator: Optional pre-initialized generator model
        discriminator: Optional pre-initialized discriminator model
        optimizer_G: Optional pre-initialized generator optimizer
        optimizer_D: Optional pre-initialized discriminator optimizer
        checkpoint_path: Path to save/load model checkpoints
        config: Optional configuration dictionary
        experiment_tracker: Optional experiment tracker
        val_img_dir: Optional directory containing validation images
        val_mask_dir: Optional directory containing validation masks
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use default config if none provided
    if config is None:
        config = {
            'training': {
                'batch_size': 2,
                'learning_rate': 2e-4,
                'epochs': 10,
                'loss_weights': {
                    'perceptual': 0.1,
                    'tv': 0.1,
                }
            }
        }

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    # Setup training dataset
    try:
        train_dataset = InpaintingDataset(img_dir=img_dir, mask_dir=mask_dir, transform=transform)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training'].get('batch_size', 2),
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=True if device.type == 'cuda' else False
        )
    except Exception as e:
        logger.error(f"Failed to initialize training dataset: {str(e)}")
        raise

    # Setup validation dataset if provided
    val_loader = None
    if val_img_dir and val_mask_dir:
        try:
            val_dataset = InpaintingDataset(img_dir=val_img_dir, mask_dir=val_mask_dir, transform=transform)
            val_loader = DataLoader(
                val_dataset,
                batch_size=config['training'].get('batch_size', 2),
                shuffle=False,  # No need to shuffle validation
                num_workers=0,
                pin_memory=True if device.type == 'cuda' else False
            )
            logger.info("Validation dataset initialized")
        except Exception as e:
            logger.error(f"Failed to initialize validation dataset: {str(e)}")
            val_loader = None

    # Model setup - now accept pre-initialized models
    if generator is None:
        generator = PConvUNet().to(device)
    if discriminator is None:
        discriminator = Discriminator().to(device)

        # Initialize losses
    criterion = InpaintingLoss(
        perceptual_weight=config['training']['loss_weights']['perceptual'],
        tv_weight=config['training']['loss_weights']['tv'],
        device=device
    )
    adversarial_loss = torch.nn.BCEWithLogitsLoss()

    # Load existing checkpoint if available and models not provided.
    # This has been updated so loading from a checkpoint is ONLY if a model
    # is not already provided to train().
    if checkpoint_path and checkpoint_path.exists() and generator is None:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
                generator.load_state_dict(checkpoint['generator_state_dict'])
                if 'discriminator_state_dict' in checkpoint: # Handle case if D is saved
                    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                logger.info(f"Loaded checkpoint from {checkpoint_path}")
            else:
                generator.load_state_dict(checkpoint)
                logger.info(f"Loaded generator-only checkpoint from {checkpoint_path}")

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            raise


    # Optimizers - now accept pre-initialized optimizers
    if optimizer_G is None:
        optimizer_G = torch.optim.Adam(
            generator.parameters(),
            lr=config['training'].get('learning_rate', 2e-4)
        )
    if optimizer_D is None:
        optimizer_D = torch.optim.Adam(
            discriminator.parameters(),
            lr=config['training'].get('learning_rate', 2e-4)
        )

    # Log model architectures if tracking enabled
    if experiment_tracker is not None:
        experiment_tracker._log_model_architecture(generator)
        # experiment_tracker._log_model_architecture(discriminator) # Don't log D


    best_val_loss = float('inf')
    best_train_loss = float('inf')
    start_time = time.time()

    for epoch in range(config['training'].get('epochs', 10)):
        # Training phase
        generator.train()
        discriminator.train()
        epoch_metrics = {
            'g_loss': 0.0,
            'd_loss': 0.0,
            'real_loss': 0.0,
            'fake_loss': 0.0,
        }

        # Add this new key to track boundary loss
        epoch_metrics['boundary_loss'] = 0.0

        epoch_start = time.time()

        for batch_idx, data in enumerate(train_loader):
            try:
                real_imgs = data['image'].to(device)
                masks = data['mask'].to(device)
                masked_imgs = real_imgs * masks

                # Train Generator
                optimizer_G.zero_grad()
                gen_imgs = generator(masked_imgs, masks)

                # Calculate generator loss - MODIFY THIS PART
                g_loss = criterion(gen_imgs, real_imgs, masks)

                # Extract boundary loss for logging (if available)
                boundary_loss = 0.0
                if hasattr(criterion, 'boundary_loss') and criterion.boundary_weight > 0:
                    try:
                        boundary_loss = criterion.boundary_loss(gen_imgs, real_imgs, masks).item()
                        # Also add to epoch metrics to track average
                        epoch_metrics['boundary_loss'] += boundary_loss
                    except Exception as e:
                        logger.debug(f"Could not extract boundary loss: {e}")

                # Add adversarial loss component
                fake_validity = discriminator(gen_imgs)
                g_adv_loss = adversarial_loss(fake_validity, torch.ones_like(fake_validity, device=device))
                g_total_loss = g_loss + g_adv_loss

                g_total_loss.backward()
                optimizer_G.step()

                # Train Discriminator
                optimizer_D.zero_grad()
                real_validity = discriminator(real_imgs)
                fake_validity = discriminator(gen_imgs.detach())
                real_labels = torch.ones_like(real_validity, device=device)
                fake_labels = torch.zeros_like(fake_validity, device=device)
                real_loss = adversarial_loss(real_validity, real_labels)
                fake_loss = adversarial_loss(fake_validity, fake_labels)
                d_loss = 0.5 * (real_loss + fake_loss)
                d_loss.backward()
                optimizer_D.step()

                # Update metrics
                epoch_metrics['g_loss'] += g_total_loss.item()
                epoch_metrics['d_loss'] += d_loss.item()
                epoch_metrics['real_loss'] += real_loss.item()
                epoch_metrics['fake_loss'] += fake_loss.item()

                # Log batch metrics
                # Inside the training batch loop, add boundary metrics logging
                if experiment_tracker is not None and batch_idx % config['training'].get('log_interval', 10) == 0:
                    step = epoch * len(train_loader) + batch_idx

                    # Calculate boundary metrics if available
                    boundary_metrics = {}
                    try:
                        boundary_weight = config['training']['loss_weights'].get('boundary', 0.0)
                        if boundary_weight > 0:
                            from .evaluation.metrics import calculate_boundary_quality
                            boundary_metrics = calculate_boundary_quality(
                                gen_imgs.detach(),
                                real_imgs.detach(),
                                masks.detach()
                            )
                    except Exception as e:
                        logger.warning(f"Could not calculate boundary metrics: {e}")

                    batch_metrics = {
                        'g_loss': float(g_total_loss.item()),
                        'd_loss': float(d_loss.item()),
                        'real_loss': float(real_loss.item()),
                        'fake_loss': float(fake_loss.item()),
                        **boundary_metrics  # Add boundary metrics if available
                    }

                    experiment_tracker.log_training_batch(
                        pred=gen_imgs.detach(),
                        target=real_imgs.detach(),
                        model=generator,
                        optimizer=optimizer_G,
                        batch_metrics=batch_metrics,
                        step=step
                    )

            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                continue

        # Calculate epoch metrics
        num_batches = len(train_loader)
        epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        epoch_metrics['epoch_time'] = time.time() - epoch_start

        # Validation phase
        if val_loader is not None:
            generator.eval()
            val_g_loss = 0.0
            val_d_loss = 0.0

            with torch.no_grad():
                for val_batch in val_loader:
                    val_real_imgs = val_batch['image'].to(device)
                    val_masks = val_batch['mask'].to(device)
                    val_masked_imgs = val_real_imgs * val_masks

                    # Generate images
                    val_gen_imgs = generator(val_masked_imgs, val_masks)

                    # Calculate generator loss
                    val_g_total = criterion(val_gen_imgs, val_real_imgs, val_masks)
                    val_g_loss += val_g_total.item()

                    # Calculate discriminator loss for monitoring
                    val_real_validity = discriminator(val_real_imgs)
                    val_fake_validity = discriminator(val_gen_imgs)
                    val_d_real = adversarial_loss(val_real_validity, torch.ones_like(val_real_validity, device=device))
                    val_d_fake = adversarial_loss(val_fake_validity, torch.zeros_like(val_fake_validity, device=device))
                    val_d_loss += 0.5 * (val_d_real.item() + val_d_fake.item())

            val_g_loss /= len(val_loader)
            val_d_loss /= len(val_loader)

            # Log validation metrics
            if experiment_tracker is not None:
                val_metrics = {
                    'validation.g_loss': float(val_g_loss),
                    'validation.d_loss': float(val_d_loss)
                }
                experiment_tracker.log_metrics(val_metrics, step=epoch)

            # Save best model based on validation loss
            if val_g_loss < best_val_loss:
                best_val_loss = val_g_loss
                try:
                    checkpoint = {
                        'epoch': epoch,
                        'generator_state_dict': generator.state_dict(),
                        'discriminator_state_dict': discriminator.state_dict(), # Keep for now
                        'optimizer_G_state_dict': optimizer_G.state_dict(),
                        'optimizer_D_state_dict': optimizer_D.state_dict(),
                        'g_loss': float(epoch_metrics['g_loss']),
                        'd_loss': float(epoch_metrics['d_loss']),
                        'val_g_loss': float(val_g_loss),
                        'val_d_loss': float(val_d_loss),
                        'config': config
                    }
                    torch.save(checkpoint, checkpoint_path)

                    if experiment_tracker is not None:
                        scalar_metrics = {
                            'g_loss': float(epoch_metrics['g_loss']),
                            'd_loss': float(epoch_metrics['d_loss']),
                            'val_g_loss': float(val_g_loss),
                            'val_d_loss': float(val_d_loss),
                            'best_val_loss': float(best_val_loss),
                            'epoch': int(epoch)
                        }
                        generator.cpu()
                        if experiment_tracker is not None:
                            try:
                                # Move to CPU for logging
                                generator.cpu()

                                # Create a proper input example (single ndarray, not tuple)
                                input_example = np.zeros((1, 1, 512, 512), dtype=np.float32)

                                experiment_tracker.log_model(
                                    generator,
                                    "best_model_validation",
                                    metrics=scalar_metrics,
                                    input_example=input_example  # Pass this explicitly
                                )
                                # Move back to device
                                generator.to(device)
                            except Exception as e:
                                logger.error(f"Failed to log best model: {str(e)}")
                                generator.to(device)  # Ensure model returns to device

                    logger.info(f"Saved new best model with validation loss: {val_g_loss:.4f}")
                except Exception as e:
                    logger.error(f"Failed to save best model: {str(e)}")
        else:
            # If no validation set, fall back to training loss for model selection
            if epoch_metrics['g_loss'] < best_train_loss:
                best_train_loss = epoch_metrics['g_loss']
                try:
                    checkpoint = {
                        'epoch': epoch,
                        'generator_state_dict': generator.state_dict(),
                        'discriminator_state_dict': discriminator.state_dict(),  #Keep D
                        'optimizer_G_state_dict': optimizer_G.state_dict(),
                        'optimizer_D_state_dict': optimizer_D.state_dict(),
                        'g_loss': float(epoch_metrics['g_loss']),
                        'd_loss': float(epoch_metrics['d_loss']),
                        'config': config
                    }
                    torch.save(checkpoint, checkpoint_path)

                    if experiment_tracker is not None:
                        scalar_metrics = {
                            'g_loss': float(epoch_metrics['g_loss']),
                            'd_loss': float(epoch_metrics['d_loss']),
                            'best_loss': float(best_train_loss),
                            'epoch': int(epoch)
                        }
                        generator.cpu()
                        experiment_tracker.log_model(
                            generator,
                            "best_model_train",
                            metrics=scalar_metrics
                        )
                        generator.to(device)
                except Exception as e:
                    logger.error(f"Failed to save best model: {str(e)}")

        # Regular checkpoint save
        if epoch % config['training'].get('checkpoint_interval', 5) == 0:
            try:
                checkpoint_epoch_path = checkpoint_path.parent / f'checkpoint_epoch_{epoch}.pth'
                torch.save(checkpoint, checkpoint_epoch_path)
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {str(e)}")

        # Log epoch metrics
        if experiment_tracker is not None:
            clean_metrics = {
                'epoch.g_loss': float(epoch_metrics['g_loss']),
                'epoch.d_loss': float(epoch_metrics['d_loss']),
                'epoch.real_loss': float(epoch_metrics['real_loss']),
                'epoch.fake_loss': float(epoch_metrics['fake_loss']),
                'epoch.time': float(epoch_metrics['epoch_time'])
            }
            experiment_tracker.log_metrics(clean_metrics, step=epoch)

        # Log progress
        log_message = f"Epoch {epoch}: g_loss={epoch_metrics['g_loss']:.4f}, d_loss={epoch_metrics['d_loss']:.4f}"

        # Add boundary loss if it's being used
        if epoch_metrics['boundary_loss'] > 0:
            log_message += f", boundary_loss={epoch_metrics['boundary_loss']:.4f}"

        if val_loader is not None:
            log_message += f", val_g_loss={val_g_loss:.4f}, val_d_loss={val_d_loss:.4f}"

        log_message += f", time={epoch_metrics['epoch_time']:.2f}s"
        logger.info(log_message)

    # End training
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f}s")

    # Log final metrics
    if experiment_tracker is not None:
        final_metrics = {
            'training.total_time': float(total_time),
            'training.best_train_loss': float(best_train_loss)
        }
        if val_loader is not None:
            final_metrics['training.best_val_loss'] = float(best_val_loss)
            final_metrics['training.validation_improvement'] = float(best_val_loss - val_g_loss)
        experiment_tracker.log_metrics(final_metrics)
        # experiment_tracker.end_run() # No longer needed

    return {
        'best_train_loss': best_train_loss,
        'best_val_loss': best_val_loss if val_loader is not None else None,
        'total_time': total_time,
        'final_epoch': epoch
    }
```

## direct_match_dataset.py
```python
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

class DirectMatchDataset(Dataset):
    """Dataset that loads directly from matched file paths"""

    def __init__(self, matched_pairs, transform=None):
        self.matched_pairs = matched_pairs

        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        logger.info(f"Initialized DirectMatchDataset with {len(matched_pairs)} samples")

    def __len__(self):
        return len(self.matched_pairs)

    def __getitem__(self, idx):
        try:
            pair = self.matched_pairs[idx]

            # Load all images directly
            image = Image.open(pair['image_path']).convert('L')
            system_mask = Image.open(pair['system_mask_path']).convert('L')
            human_mask = Image.open(pair['human_mask_path']).convert('L')

            # CRITICAL: Ensure all images are resized to the expected format (512x512)
            image = image.resize((512, 512), Image.BILINEAR)
            system_mask = system_mask.resize((512, 512), Image.NEAREST)
            human_mask = human_mask.resize((512, 512), Image.NEAREST)

            # Apply transformations
            image_tensor = self.transform(image)
            system_mask_tensor = self.transform(system_mask)
            human_mask_tensor = self.transform(human_mask)

            # Ensure masks are binary (0 or 1)
            system_mask_tensor = (system_mask_tensor > 0.5).float()
            human_mask_tensor = (human_mask_tensor > 0.5).float()

            # Validate tensor dimensions (good practice for debugging)
            expected_shape = (1, 512, 512)
            if (image_tensor.shape != expected_shape or
                system_mask_tensor.shape != expected_shape or
                human_mask_tensor.shape != expected_shape):
                logger.warning(f"Shape mismatch: image={image_tensor.shape}, "
                            f"system_mask={system_mask_tensor.shape}, "
                            f"human_mask={human_mask_tensor.shape}")

            # Add validation for zero masks
            if human_mask_tensor.sum() == 0:
                logger.warning(f"Human mask for tile {pair['tile_name']} contains all zeros")

            return {
                'image': image_tensor,
                'mask': system_mask_tensor,
                'human_mask': human_mask_tensor,
                'tile_name': pair['tile_name']
            }

        except Exception as e:
            logger.error(f"Error loading item at index {idx}: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            # Return fallback tensors
            empty_tensor = torch.zeros((1, 512, 512), dtype=torch.float32)
            return {
                'image': empty_tensor,
                'mask': empty_tensor,
                'human_mask': empty_tensor,
                'tile_name': "error"
            }
```

## __init__.py
```python
# Add this line to the existing imports
from .direct_match_dataset import DirectMatchDataset

# Make sure the class is exported
__all__ = [..., 'DirectMatchDataset']  # Add to existing __all__ list
```

## losses.py
```python
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
import logging

logger = logging.getLogger(__name__)

class InpaintingLoss(nn.Module):
    def __init__(self,
                 perceptual_weight: float = 0.1,
                 tv_weight: float = 0.1,
                 boundary_weight: float = 0.5,
                 device=None):
        super().__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.l1_loss = nn.L1Loss().to(self.device)
        self.perceptual_weight = perceptual_weight
        self.tv_weight = tv_weight
        self.boundary_weight = boundary_weight  # New parameter for boundary loss

        # Load pre-trained VGG16 for perceptual loss
        # Note: We're initializing it directly on the correct device
        logger.info(f"Initializing VGG model on device: {self.device}")
        try:
            vgg_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            self.vgg_layers = vgg_model.features[:16].eval().to(self.device)
            for param in self.vgg_layers.parameters():
                param.requires_grad = False
        except Exception as e:
            logger.error(f"Error initializing VGG model: {str(e)}")
            raise

        # Initialize boundary-aware loss
        self.boundary_loss = BoundaryAwareLoss(device=self.device)
        self.boundary_loss = self.boundary_loss.to(self.device)

        logger.info("InpaintingLoss initialized successfully")

    def to(self, device):
        """Override to() to ensure all internal components are moved to the device"""
        logger.info(f"Moving InpaintingLoss to device: {device}")
        self.device = device
        self.l1_loss = self.l1_loss.to(device)
        self.vgg_layers = self.vgg_layers.to(device)

        # Make sure boundary loss is moved to the device too
        if hasattr(self, 'boundary_loss'):
            self.boundary_loss = self.boundary_loss.to(device)

        return super().to(device)

    def forward(self, input, target, mask):
        # Ensure inputs are on correct device
        input = input.to(self.device)
        target = target.to(self.device)
        mask = mask.to(self.device)

        # Log device information for debugging
        logger.debug(f"InpaintingLoss running on device: {self.device}")
        logger.debug(f"Input tensor device: {input.device}")
        logger.debug(f"VGG layers device: {next(self.vgg_layers.parameters()).device}")

        # Initialize total loss
        total_loss = torch.tensor(0.0, device=self.device)

        # L1 loss
        l1_loss = self.l1_loss(input, target)
        total_loss = total_loss + l1_loss

        # Perceptual Loss (if used)
        if self.perceptual_weight > 0:
            try:
                input_vgg = input.repeat(1, 3, 1, 1)  # Convert to 3 channels
                target_vgg = target.repeat(1, 3, 1, 1)

                # Double-check devices
                input_vgg = input_vgg.to(self.device)
                target_vgg = target_vgg.to(self.device)

                perceptual_loss = self.l1_loss(
                    self.vgg_layers(input_vgg),
                    self.vgg_layers(target_vgg)
                )
                total_loss = total_loss + self.perceptual_weight * perceptual_loss
            except Exception as e:
                logger.error(f"Error in perceptual loss: {str(e)}")
                perceptual_loss = torch.tensor(0.0, device=self.device)

        # Total Variation Loss on inpainted regions
        if self.tv_weight > 0:
            try:
                hole_mask = 1 - mask
                tv_loss = self.total_variation_loss(input * hole_mask)
                total_loss = total_loss + self.tv_weight * tv_loss
            except Exception as e:
                logger.error(f"Error in TV loss: {str(e)}")
                tv_loss = torch.tensor(0.0, device=self.device)

        # Boundary-aware loss
        if self.boundary_weight > 0:
            try:
                b_loss = self.boundary_loss(input, target, mask)
                # Add to total loss with weight
                total_loss = total_loss + self.boundary_weight * b_loss
                logger.debug(f"Boundary loss component: {b_loss.item()}")
            except Exception as e:
                logger.error(f"Error in boundary loss calculation: {str(e)}")
                # Don't add boundary loss component if it fails

        return total_loss

    def total_variation_loss(self, x):
        """Calculate total variation loss for smoothness"""
        batch_size = x.size(0)
        h_x = x.size(2)
        w_x = x.size(3)
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.numel()

class HumanGuidedLoss(InpaintingLoss):
    def __init__(self, config, device=None, **kwargs):
        # Make sure device isn't passed twice
        if 'device' in kwargs:
            del kwargs['device']

        # Get boundary weight from config if available
        boundary_weight = config['training'].get('loss_weights', {}).get('boundary', 0.5)

        # Initialize parent class
        super().__init__(
            device=device,
            boundary_weight=boundary_weight,
            **kwargs
        )

        self.human_feedback_weight = config['training']['modes']['human_guided']['human_feedback_weight']
        self.base_loss_weight = config['training']['modes']['human_guided']['base_loss_weight']
        logger.info(f"HumanGuidedLoss initialized with weights: base={self.base_loss_weight}, human={self.human_feedback_weight}, boundary={boundary_weight}")

    def forward(self, input, target, mask, human_feedback=None):
        # Ensure all tensors are on the correct device
        input = input.to(self.device)
        target = target.to(self.device)
        mask = mask.to(self.device)

        # Calculate base loss using parent class (which includes boundary loss)
        base_loss = super().forward(input, target, mask)

        # Initialize human_loss
        human_loss = torch.tensor(0.0, device=self.device)

        # Calculate human feedback loss (if provided)
        if human_feedback is not None and 'mask' in human_feedback and human_feedback['mask'] is not None:
            try:
                human_mask = human_feedback['mask'].to(self.device)
                human_guided_regions = (human_mask > 0).float().to(self.device)

                # Make sure there are non-zero values in the mask
                if human_guided_regions.sum() > 0:
                    human_loss = self.l1_loss(
                        input * human_guided_regions,
                        target * human_guided_regions
                    )

                    # Also apply boundary loss to human-masked regions for additional consistency
                    if self.boundary_weight > 0:
                        try:
                            human_boundary_loss = self.boundary_loss(
                                input,
                                target,
                                human_guided_regions
                            )
                            human_loss = human_loss + self.boundary_weight * human_boundary_loss
                        except Exception as e:
                            logger.error(f"Error computing human boundary loss: {str(e)}")

                    logger.debug(f"Human loss: {human_loss.item()}")
                else:
                    logger.warning("Human mask contains all zeros")
            except Exception as e:
                logger.error(f"Error computing human feedback loss: {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())

        # Combine losses according to configured weights
        total_loss = (
            self.base_loss_weight * base_loss +
            self.human_feedback_weight * human_loss
        )

        logger.debug(f"Total loss: {total_loss.item()} (base: {base_loss.item()}, human: {human_loss.item()})")
        return total_loss

class BoundaryAwareLoss(nn.Module):
    """
    Loss function that focuses on boundary regions between original and inpainted areas.

    This loss ensures smooth transitions at boundaries by:
    1. Detecting boundary regions using mask dilation/erosion
    2. Computing gradient consistency across boundaries
    3. Applying multi-scale boundary evaluation for robust performance
    """
    def __init__(self, boundary_width: int = 10, epsilon: float = 1e-6, device=None):
        """
        Initialize boundary-aware loss.

        Args:
            boundary_width: Width of the boundary region in pixels
            epsilon: Small value to prevent division by zero
            device: Computation device
        """
        super().__init__()
        self.boundary_width = boundary_width
        self.epsilon = epsilon

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Define Sobel filters for gradient calculation
        self.sobel_x = torch.tensor([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0]
        ], dtype=torch.float32).view(1, 1, 3, 3).to(self.device)

        self.sobel_y = torch.tensor([
            [-1.0, -2.0, -1.0],
            [0.0,  0.0,  0.0],
            [1.0,  2.0,  1.0]
        ], dtype=torch.float32).view(1, 1, 3, 3).to(self.device)

        logger.info(f"BoundaryAwareLoss initialized with boundary width: {boundary_width}, device: {self.device}")

    def to(self, device):
        """Override to() to ensure all tensors move to the specified device"""
        self.device = device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)
        return super().to(device)

    def extract_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Extract boundary region between masked and unmasked areas.

        Args:
            mask: Binary mask tensor where 1 indicates preserved regions
                 and 0 indicates regions to inpaint

        Returns:
            Boundary mask where 1 indicates boundary pixels
        """
        try:
            # Ensure mask is on the correct device
            mask = mask.to(self.device)

            # Dilate the mask
            dilated = F.max_pool2d(
                mask,
                kernel_size=self.boundary_width,
                stride=1,
                padding=self.boundary_width//2
            )

            # Erode the mask
            eroded = -F.max_pool2d(
                -mask,
                kernel_size=self.boundary_width,
                stride=1,
                padding=self.boundary_width//2
            )

            # Boundary is the difference between dilated and eroded
            boundary = dilated - eroded

            # Ensure boundary values are between 0 and 1
            boundary = torch.clamp(boundary, 0.0, 1.0)

            # Check if boundary contains numerical issues
            if torch.isnan(boundary).any() or torch.isinf(boundary).any():
                logger.warning("Numerical issues detected in boundary extraction")
                # Fallback to a safer version
                boundary = torch.zeros_like(mask)

                # Create a safe boundary using morphological operations
                with torch.no_grad():
                    # Simple dilation and erosion with smaller kernel
                    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
                    eroded = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
                    boundary = torch.clamp(dilated - eroded, 0.0, 1.0)

            return boundary

        except Exception as e:
            logger.error(f"Error in extract_boundary: {str(e)}")
            # Return empty boundary on error
            return torch.zeros_like(mask)

    def compute_gradients(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute image gradients using finite differences instead of convolution.

        Args:
            img: Input tensor image

        Returns:
            Tuple of (gradient_x, gradient_y) tensors
        """
        try:
            # Ensure image is on the correct device
            img = img.to(self.device)

            # Make sure image has the right shape
            if img.dim() == 3:  # Add batch dimension if missing
                img = img.unsqueeze(0)

            # Simple finite differences for gradients
            # Pad with zeros to maintain dimensions
            padded = F.pad(img, (1, 1, 1, 1), mode='replicate')

            # Horizontal gradient (central difference)
            grad_x = padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]
            grad_x = grad_x / 2.0  # Normalize

            # Vertical gradient (central difference)
            grad_y = padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]
            grad_y = grad_y / 2.0  # Normalize

            return grad_x, grad_y

        except Exception as e:
            logger.error(f"Error in compute_gradients: {str(e)}")
            # Return zero gradients on error
            return torch.zeros_like(img), torch.zeros_like(img)

    def gradient_consistency_loss(self, pred: torch.Tensor, target: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient consistency loss at boundary regions.

        Args:
            pred: Predicted (inpainted) image
            target: Target (ground truth) image
            boundary: Boundary region mask

        Returns:
            Gradient consistency loss value
        """
        try:
            # Ensure all inputs are the same shape
            if pred.shape != target.shape or pred.shape != boundary.shape:
                logger.warning("Shape mismatch in gradient_consistency_loss")
                # Simple reshaping to match
                target = F.interpolate(target, size=pred.shape[2:], mode='bilinear')
                boundary = F.interpolate(boundary, size=pred.shape[2:], mode='bilinear')

            # Simple intensity difference at boundary
            # This is a robust fallback that doesn't use gradients
            boundary_diff = torch.abs(pred - target) * boundary
            loss = boundary_diff.sum() / (boundary.sum() + self.epsilon)

            # Check for numerical issues
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("NaN or Inf detected in gradient_consistency_loss")
                return torch.tensor(0.01, device=self.device)

            return loss

        except Exception as e:
            logger.error(f"Error in gradient_consistency_loss: {str(e)}")
            # Return a small constant loss value on error
            return torch.tensor(0.01, device=self.device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the complete boundary-aware loss with robust fallback mechanisms.

        Args:
            pred: Predicted (inpainted) image
            target: Target (ground truth) image
            mask: Binary mask where 1 indicates preserved regions

        Returns:
            Boundary loss value
        """
        # Ensure all inputs are on the correct device
        pred = pred.to(self.device)
        target = target.to(self.device)
        mask = mask.to(self.device)

        try:
            # Simple fallback approach that will always work
            # Define a boundary region using simple dilation/erosion
            dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
            eroded = 1 - F.max_pool2d(1 - mask, kernel_size=3, stride=1, padding=1)
            boundary = torch.clamp(dilated - eroded, 0.0, 1.0)

            # If boundary is empty, return zero loss
            if torch.sum(boundary) < 1.0:
                return torch.tensor(0.0, device=self.device)

            # Simple L1 loss at boundary region (this always works)
            boundary_loss = torch.abs(pred - target) * boundary
            loss = boundary_loss.sum() / (boundary.sum() + self.epsilon)

            # Check for numerical issues
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("NaN or Inf detected in boundary loss")
                return torch.tensor(0.0, device=self.device)

            return loss

        except Exception as e:
            logger.error(f"Error in boundary loss calculation: {str(e)}")
            # Return zero loss on error to avoid breaking training
            return torch.tensor(0.0, device=self.device)
```

## metrics.py
```python
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np
from torchvision.transforms.functional import to_tensor
import psutil
import GPUtil
from datetime import datetime

class PerformanceMetrics:
    @staticmethod
    def calculate_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.item()

    @staticmethod
    def calculate_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
        """Calculate Structural Similarity Index"""
        C1 = (0.01 * 1.0) ** 2
        C2 = (0.03 * 1.0) ** 2

        mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
        mu2 = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean().item()

    @staticmethod
    def calculate_l1_l2(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
        """Calculate L1 and L2 distances"""
        l1_dist = F.l1_loss(pred, target).item()
        l2_dist = F.mse_loss(pred, target, reduction='mean').sqrt().item()
        return l1_dist, l2_dist

class TrainingMetrics:
    @staticmethod
    def calculate_gradient_norm(model: torch.nn.Module) -> Dict[str, float]:
        """Calculate gradient norms for model parameters"""
        total_norm = 0
        param_norms = {}

        for name, p in model.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                param_norms[f"grad_norm_{name}"] = param_norm.item()
                total_norm += param_norm.item() ** 2

        total_norm = total_norm ** 0.5
        param_norms['total_grad_norm'] = total_norm

        return param_norms

    @staticmethod
    def get_learning_rates(optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Get current learning rates"""
        return {f"lr_group_{i}": group['lr'] for i, group in enumerate(optimizer.param_groups)}

class ResourceMetrics:
    @staticmethod
    def get_gpu_metrics() -> Optional[Dict[str, float]]:
        """Get GPU utilization metrics"""
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return None

            gpu_metrics = {}
            for i, gpu in enumerate(gpus):
                gpu_metrics.update({
                    f"gpu_{i}_memory_used": gpu.memoryUsed,
                    f"gpu_{i}_memory_total": gpu.memoryTotal,
                    f"gpu_{i}_utilization": gpu.load * 100
                })
            return gpu_metrics
        except Exception:
            return None

    @staticmethod
    def get_cpu_metrics() -> Dict[str, float]:
        """Get CPU utilization metrics"""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent
        }

    @staticmethod
    def get_batch_timing(start_time: float) -> float:
        """Calculate batch processing time"""
        return datetime.now().timestamp() - start_time

class MetricsLogger:
    def __init__(self):
        self.performance_metrics = PerformanceMetrics()
        self.training_metrics = TrainingMetrics()
        self.resource_metrics = ResourceMetrics()

    def log_batch_metrics(self,
                         pred: torch.Tensor,
                         target: torch.Tensor,
                         model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer,
                         batch_start_time: float) -> Dict[str, float]:
        """Log comprehensive batch metrics"""
        metrics = {}

        # Performance metrics
        metrics['psnr'] = self.performance_metrics.calculate_psnr(pred, target)
        metrics['ssim'] = self.performance_metrics.calculate_ssim(pred, target)
        l1_dist, l2_dist = self.performance_metrics.calculate_l1_l2(pred, target)
        metrics['l1_distance'] = l1_dist
        metrics['l2_distance'] = l2_dist

        # Training metrics
        metrics.update(self.training_metrics.calculate_gradient_norm(model))
        metrics.update(self.training_metrics.get_learning_rates(optimizer))

        # Resource metrics
        gpu_metrics = self.resource_metrics.get_gpu_metrics()
        if gpu_metrics:
            metrics.update(gpu_metrics)
        metrics.update(self.resource_metrics.get_cpu_metrics())
        metrics['batch_time'] = self.resource_metrics.get_batch_timing(batch_start_time)

        return metrics

    def log_validation_metrics(self,
                             model: torch.nn.Module,
                             val_loader: torch.utils.data.DataLoader,
                             device: torch.device) -> Dict[str, float]:
        """Calculate validation metrics"""
        model.eval()
        val_metrics = {
            'val_psnr': 0,
            'val_ssim': 0,
            'val_l1': 0,
            'val_l2': 0
        }

        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch['image'].to(device))
                target = batch['target'].to(device)

                val_metrics['val_psnr'] += self.performance_metrics.calculate_psnr(pred, target)
                val_metrics['val_ssim'] += self.performance_metrics.calculate_ssim(pred, target)
                l1, l2 = self.performance_metrics.calculate_l1_l2(pred, target)
                val_metrics['val_l1'] += l1
                val_metrics['val_l2'] += l2

        # Average metrics
        num_batches = len(val_loader)
        return {k: v/num_batches for k, v in val_metrics.items()}
```

## dataset.py
```python
# src/utils/dataset.py

import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class InpaintingDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Get sorted list of filenames to match images and masks
        self.img_filenames = sorted([f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))])
        self.mask_filenames = sorted([f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))])

        assert len(self.img_filenames) == len(self.mask_filenames), "Number of images and masks do not match."

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        # Load grayscale image
        img_path = os.path.join(self.img_dir, self.img_filenames[idx])
        img = Image.open(img_path).convert('L')

        # Load mask
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])
        mask = Image.open(mask_path).convert('L')

        # Apply transformations
        if self.transform:
            # Ensure both are resized to the same dimensions
            img = self.transform(img)
            mask = self.transform(mask)
            mask = (mask > 0).float()  # Binarize the mask

        # Ensure mask has shape (1, H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        return {'image': img, 'mask': mask}
```

## discriminator.py
```python
# src/models/discriminator.py

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_channels, out_channels, normalization=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
```

## generator.py
```python
# # src/models/generator.py

import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from .pconv import PConv2d

class PConvUNet(nn.Module):
    def __init__(self):
        super(PConvUNet, self).__init__()

        # Encoder layers
        self.enc1 = PConv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.enc2 = PConv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.enc3 = PConv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.enc4 = PConv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.enc5 = PConv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.enc6 = PConv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.enc7 = PConv2d(512, 512, kernel_size=3, stride=2, padding=1)

        # Decoder layers
        self.dec7 = PConv2d(512 + 512, 512, kernel_size=3, padding=1)
        self.dec6 = PConv2d(512 + 512, 512, kernel_size=3, padding=1)
        self.dec5 = PConv2d(512 + 512, 512, kernel_size=3, padding=1)
        self.dec4 = PConv2d(512 + 256, 256, kernel_size=3, padding=1)
        self.dec3 = PConv2d(256 + 128, 128, kernel_size=3, padding=1)
        self.dec2 = PConv2d(128 + 64, 64, kernel_size=3, padding=1)
        self.dec1 = PConv2d(64, 64, kernel_size=3, padding=1)
        self.final = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x, mask):
        # Encoder
        e1, m1 = self.enc1(x, mask)
        e2, m2 = self.enc2(e1, m1)
        e3, m3 = self.enc3(e2, m2)
        e4, m4 = self.enc4(e3, m3)
        e5, m5 = self.enc5(e4, m4)
        e6, m6 = self.enc6(e5, m5)
        e7, m7 = self.enc7(e6, m6)

        # Decoder
        d6, dm6 = self.decode_step(e7, m7, e6, m6, self.dec7)
        d5, dm5 = self.decode_step(d6, dm6, e5, m5, self.dec6)
        d4, dm4 = self.decode_step(d5, dm5, e4, m4, self.dec5)
        d3, dm3 = self.decode_step(d4, dm4, e3, m3, self.dec4)
        d2, dm2 = self.decode_step(d3, dm3, e2, m2, self.dec3)
        d1, dm1 = self.decode_step(d2, dm2, e1, m1, self.dec2)

        # Final decoding without skip connection
        d0_up = interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)
        dm0_up = interpolate(dm1, scale_factor=2, mode='nearest')
        d0_up = self._pad_to_match(d0_up, x)
        dm0_up = self._pad_to_match(dm0_up, mask)
        m_combined = torch.max(dm0_up, mask)
        d0, _ = self.dec1(d0_up, m_combined)
        output = self.final(d0)
        output = torch.sigmoid(output)

        # Ensure that unmasked regions are copied from the input
        valid_mask = mask
        hole_mask = 1 - mask
        output = output * hole_mask + x * valid_mask

        return output

    def decode_step(self, up_feature, up_mask, skip_feature, skip_mask, decoder_layer):
        up_feature = interpolate(up_feature, scale_factor=2, mode='bilinear', align_corners=False)
        up_mask = interpolate(up_mask, scale_factor=2, mode='nearest')

        up_feature = self._pad_to_match(up_feature, skip_feature)
        up_mask = self._pad_to_match(up_mask, skip_mask)

        merged_feature = torch.cat([up_feature, skip_feature], dim=1)
        merged_mask = torch.max(up_mask, skip_mask)
        out_feature, out_mask = decoder_layer(merged_feature, merged_mask)
        return out_feature, out_mask

    def _pad_to_match(self, x, target):
        """Pads tensor x to match the size of target tensor along spatial dimensions."""
        diffY = target.size(2) - x.size(2)
        diffX = target.size(3) - x.size(3)
        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2])
        return x
```

## pconv.py
```python
# src/models/pconv.py

import torch
import torch.nn as nn

class PConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, batch_norm=True):
        super(PConv2d, self).__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.slide_winsize = self.input_conv.weight.data.shape[2] * self.input_conv.weight.data.shape[3]
        self.mask_conv = nn.Conv2d(1, 1, kernel_size, stride, padding, bias=False)

        # Initialize mask_conv weights to 1
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False

        # Optional batch normalization
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)

        self.activation = nn.ReLU()

    def forward(self, input, mask):
        # Multiply input by mask (broadcasting mask if necessary)
        input_masked = input * mask  # mask shape: [B,1,H,W]; input shape: [B,C,H,W]

        # Apply convolution to masked input
        output = self.input_conv(input_masked)

        # Update mask
        with torch.no_grad():
            output_mask = self.mask_conv(mask)
            output_mask = (output_mask > 0).float()

        # Calculate mask ratio
        mask_sum = self.mask_conv(mask)
        mask_ratio = self.slide_winsize / (mask_sum + 1e-8)
        mask_ratio = mask_ratio * (mask_sum > 0).float()

        # Normalize output
        output = output * mask_ratio

        # Apply batch normalization and activation
        if self.batch_norm:
            output = self.bn(output)
        output = self.activation(output)

        return output, output_mask
```

## results.py
```python
import json
from pathlib import Path
from typing import Dict


class ResultsManager:
    def __init__(self, config: Dict):
        self.results_dir = Path(config['data']['evaluation_results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_evaluation_results(self, results: Dict, experiment_name: str):
        """Save evaluation results with metadata"""
        results_path = self.results_dir / f"{experiment_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

    def load_results(self, experiment_name: str) -> Dict:
        """Load previous evaluation results"""
        results_path = self.results_dir / f"{experiment_name}_results.json"
        with open(results_path, 'r') as f:
            return json.load(f)
```

## evaluator.py
```python
# Add at top of evaluator.py
from typing import Dict
from pathlib import Path
import numpy as np
import torch.nn as nn
from ..utils.human_guided_dataset import HumanGuidedDataset
from .metrics import MaskEvaluator

class MetricsAggregator:
    def __init__(self):
        self.metrics = []

    def add(self, metric):
        self.metrics.append(metric)

    def get_summary(self) -> Dict[str, float]:
        return {
            "mean_iou": np.mean([m.iou for m in self.metrics]),
            "mean_precision": np.mean([m.precision for m in self.metrics]),
            "mean_recall": np.mean([m.recall for m in self.metrics])
        }

class GANEvaluator:
    def __init__(self, config: Dict):
        self.config = config
        self.metrics = MaskEvaluator(config)
        self.results_dir = Path(config['data']['evaluation_results_dir'])

    def evaluate_model(self,
                      generator: nn.Module,
                      eval_dataset: HumanGuidedDataset) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        metrics_aggregator = MetricsAggregator()

        for batch in eval_dataset:
            # Generate inpainted image
            output = generator(batch['image'], batch['mask'])

            # Calculate metrics
            if batch.get('human_mask') is not None:
                metrics = self.metrics.calculate_metrics(
                    output.cpu().numpy(),
                    batch['human_mask'].numpy()
                )
                metrics_aggregator.add(metrics)

        return metrics_aggregator.get_summary()
```

## metrics.py
```python
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import logging
import cv2
from typing import Dict, List, Optional, Tuple
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

@dataclass
class MaskMetrics:
    iou: float
    precision: float
    recall: float
    total_area: int
    feature_count: int
    average_feature_size: float

class MaskEvaluator:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def _identify_features(self, mask: np.ndarray) -> List:
        """Identify distinct features in the mask using connected components."""
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def evaluate_batch(self, gan_masks: List[np.ndarray],
                      human_masks: List[np.ndarray]) -> List[MaskMetrics]:
        """Evaluate a batch of masks."""
        return [self.calculate_metrics(gan, human)
                for gan, human in zip(gan_masks, human_masks)]

    def save_results(self, metrics: MaskMetrics, save_path: Path):
        """Save evaluation results."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'a') as f:
            metrics_dict = {k: v for k, v in metrics.__dict__.items()}
            f.write(f"{metrics_dict}\n")

    def _calculate_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.item()

    def _calculate_ssim(self, pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
        """Calculate Structural Similarity Index"""
        try:
            C1 = (0.01 * 1.0) ** 2
            C2 = (0.03 * 1.0) ** 2

            mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
            mu2 = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)

            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2

            sigma1_sq = F.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size//2) - mu1_sq
            sigma2_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=window_size//2) - mu2_sq
            sigma12 = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean().item()
        except Exception as e:
            logger.warning(f"Error calculating SSIM: {str(e)}")
            return 0.0

def calculate_boundary_quality(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                        boundary_width: int = 10) -> Dict[str, float]:
    """
    Calculate metrics that measure quality at the boundary between inpainted and original regions.
    Uses simple intensity differences instead of gradients to avoid dimension issues.
    """
    device = pred.device

    try:
        # Extract boundary (with safe operations)
        dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        eroded = 1 - F.max_pool2d(1 - mask, kernel_size=3, stride=1, padding=1)
        boundary = torch.clamp(dilated - eroded, 0.0, 1.0)

        # Handle empty boundary
        if torch.sum(boundary) < 1e-6:
            return {
                'boundary_mse': 0.0,
                'boundary_psnr': 0.0,
                'boundary_gradient_diff': 0.0
            }

        # MSE at boundary
        boundary_mse = torch.mean(((pred - target) * boundary) ** 2)

        # PSNR at boundary
        epsilon = 1e-6
        max_val = 1.0
        boundary_psnr = 10 * torch.log10(max_val**2 / (boundary_mse + epsilon))

        # Simple intensity variation as a proxy for gradient difference
        # This completely avoids the problematic convolution operations
        pred_diff = torch.abs(
            pred[:,:,1:,:] - pred[:,:,:-1,:]).mean() + torch.abs(
            pred[:,:,:,1:] - pred[:,:,:,:-1]).mean()

        target_diff = torch.abs(
            target[:,:,1:,:] - target[:,:,:-1,:]).mean() + torch.abs(
            target[:,:,:,1:] - target[:,:,:,:-1]).mean()

        boundary_gradient_diff = torch.abs(pred_diff - target_diff)

        return {
            'boundary_mse': boundary_mse.item(),
            'boundary_psnr': boundary_psnr.item(),
            'boundary_gradient_diff': boundary_gradient_diff.item()
        }

    except Exception as e:
        logger.error(f"Error calculating boundary quality: {str(e)}")
        return {
            'boundary_mse': 0.0,
            'boundary_psnr': 0.0,
            'boundary_gradient_diff': 0.0
        }

def calculate_metrics(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    """Calculate comprehensive metrics for evaluation"""
    metrics = {}

    # Standard metrics
    metrics['mse'] = F.mse_loss(pred, target).item()
    metrics['psnr'] = self._calculate_psnr(pred, target)
    metrics['ssim'] = self._calculate_ssim(pred, target)

    # Add boundary quality metrics
    boundary_width = self.config['evaluation']['metrics'].get('boundary_size', 10)
    boundary_metrics = calculate_boundary_quality(pred, target, mask, boundary_width)
    metrics.update(boundary_metrics)

    return metrics
```

## visualization.py
```python
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class ResultVisualizer:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def visualize_comparison(self,
                           original: np.ndarray,
                           inpainted: np.ndarray,
                           gan_mask: np.ndarray,
                           human_mask: np.ndarray,
                           save_path: Path):
        """Create visual comparison of results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        axes[0,0].imshow(original, cmap='gray')
        axes[0,0].set_title('Original')

        axes[0,1].imshow(inpainted, cmap='gray')
        axes[0,1].set_title('Inpainted')

        axes[1,0].imshow(gan_mask, cmap='gray')
        axes[1,0].set_title('GAN Mask')

        axes[1,1].imshow(human_mask, cmap='gray')
        axes[1,1].set_title('Human Annotation')

        plt.savefig(save_path)
        plt.close()
```

## __init__.py
```python
"""
Training module for DSM inpainting models.
"""

from .human_guided_trainer import HumanGuidedTrainer

__all__ = ['HumanGuidedTrainer']
```

## human_guided_trainer.py
```python
"""
Human-guided training module for DSM inpainting models.

This module contains the HumanGuidedTrainer class which implements fine-tuning
of pre-trained models using human-annotated masks.
"""

import time
import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict
from torch.utils.data import DataLoader

from ..evaluation.metrics import MaskEvaluator
from ..utils.losses import HumanGuidedLoss

logger = logging.getLogger(__name__)

class HumanGuidedTrainer:
    """
    Trainer class for human-guided fine-tuning of DSM inpainting models.

    This class handles the training process using human annotations to fine-tune
    generative models for terrain inpainting.
    """

    def __init__(self, config: Dict, experiment_tracker=None):
        """
        Initialize the human-guided trainer.

        Args:
            config: Configuration dictionary containing training parameters
            experiment_tracker: Optional experiment tracking instance for metrics logging
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics_evaluator = MaskEvaluator(config)
        self.experiment_tracker = experiment_tracker

        logger.info(f"Initialized HumanGuidedTrainer with device: {self.device}")

    def train(self, generator: nn.Module, train_dataset, num_epochs: int, checkpoint_dir: Path):
        """
        Train the model with human guidance.

        Args:
            generator: Generator model to train
            train_dataset: Dataset with human annotations
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
        """
        try:
            # if self.experiment_tracker is not None:
            #     if not hasattr(self.experiment_tracker, 'run') or self.experiment_tracker.run is None:
            #         logger.warning("No active MLflow run found. Training metrics may not be logged.")

            # Ensure model is on the correct device
            generator = generator.to(self.device)
            logger.info(f"Model moved to device: {self.device}")

            # Initialize loss function and optimizer - EXPLICITLY with device
            criterion = HumanGuidedLoss(self.config, device=self.device)
            logger.info(f"Criterion initialized on device: {self.device}")
            logger.info(f"VGG model device: {next(criterion.vgg_layers.parameters()).device}")

            optimizer = torch.optim.Adam(
                generator.parameters(),
                lr=self.config['training']['modes']['human_guided']['learning_rate']
            )

            # Create data loader with appropriate settings
            pin_memory = self.device.type == 'cuda'  # Only pin memory if using CUDA
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['training']['modes']['human_guided']['batch_size'],
                shuffle=True,
                num_workers=0,
                pin_memory=False  # Always set to False to avoid pin memory errors
            )

            logger.info(f"Created DataLoader with {len(train_dataset)} samples")

            best_loss = float('inf')
            start_time = time.time()

            for epoch in range(num_epochs):
                generator.train()
                epoch_loss = 0.0
                epoch_start = time.time()
                batch_count = 0
                success_count = 0

                # In mvp_gan/src/training/human_guided_trainer.py

                # This would be inside the train method of the HumanGuidedTrainer class
                for batch_idx, batch in enumerate(train_loader):
                    try:
                        # Move input tensors to device
                        images = batch['image'].to(self.device)
                        masks = batch['mask'].to(self.device)
                        human_masks = batch.get('human_mask')

                        logger.debug(f"Batch {batch_idx} - Images device: {images.device}, Masks device: {masks.device}")

                        if human_masks is not None:
                            human_masks = human_masks.to(self.device)
                            logger.debug(f"Human masks device: {human_masks.device}")

                        # Apply mask to input images
                        masked_images = images * masks

                        # Generate output
                        generated = generator(masked_images, masks)
                        logger.debug(f"Generated output device: {generated.device}")

                        # Calculate loss
                        # Insert the new code block here (starts with the line below)
                        human_feedback = {'mask': human_masks} if human_masks is not None else None
                        loss = criterion(generated, images, masks, human_feedback)

                        # Log boundary components if applicable
                        if self.experiment_tracker is not None and batch_idx % 10 == 0:
                            try:
                                boundary_weight = self.config['training']['loss_weights'].get('boundary', 0.0)
                                if boundary_weight > 0:
                                    from ..evaluation.metrics import calculate_boundary_quality
                                    boundary_metrics = calculate_boundary_quality(
                                        generated.detach(),
                                        images.detach(),
                                        masks.detach()
                                    )

                                # Add these to the logged metrics
                                self.experiment_tracker.log_metrics({
                                    f'batch.boundary_{k}': v for k, v in boundary_metrics.items()
                                }, step=epoch * len(train_loader) + batch_idx)
                            except Exception as e:
                                logger.warning(f"Could not log boundary metrics: {e}")
                        # End of new code block

                        logger.debug(f"Loss value: {loss.item()}, device: {loss.device}")

                        # Only add to epoch loss if valid
                        if not torch.isnan(loss) and not torch.isinf(loss):
                            epoch_loss += loss.item()
                            success_count += 1

                        # Update generator
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        # Log batch metrics
                        if self.experiment_tracker is not None and batch_idx % self.config['training'].get('log_interval', 10) == 0:
                            step = epoch * len(train_loader) + batch_idx
                            self.experiment_tracker.log_training_batch(
                                pred=generated.detach(),
                                target=images.detach(),
                                model=generator,
                                optimizer=optimizer,
                                batch_metrics={'loss': float(loss.item())},
                                step=step
                            )

                        logger.debug(f"Processed batch {batch_idx}: loss={loss.item():.6f}")
                        batch_count += 1

                    except Exception as e:
                        logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                        import traceback
                        logger.debug(traceback.format_exc())
                        continue

                # Calculate epoch metrics
                avg_epoch_loss = epoch_loss / max(1, success_count) if epoch_loss > 0 else 0.0
                epoch_time = time.time() - epoch_start

                # Log epoch metrics
                if self.experiment_tracker is not None:
                    self.experiment_tracker.log_metrics({
                        'epoch.loss': float(avg_epoch_loss),
                        'epoch.time': float(epoch_time),
                        'epoch.success_rate': float(success_count / max(1, batch_count))
                    }, step=epoch)

                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': generator.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': float(avg_epoch_loss),
                    'config': self.config
                }

                try:
                    # Save regular checkpoint
                    checkpoint_path = checkpoint_dir / f'generator_epoch_{epoch}.pth'
                    torch.save(checkpoint, checkpoint_path)
                    logger.debug(f"Saved checkpoint to {checkpoint_path}")

                    # Save best model if current loss is best
                    if avg_epoch_loss < best_loss and avg_epoch_loss > 0:
                        best_loss = avg_epoch_loss
                        best_model_path = checkpoint_dir / 'best_model.pth'
                        torch.save(checkpoint, best_model_path)
                        logger.info(f"New best model saved with loss: {best_loss:.6f}")

                        if self.experiment_tracker is not None:
                            try:
                                # Move to CPU for logging
                                generator.cpu()
                                self.experiment_tracker.log_model(
                                    generator,
                                    "best_human_guided_model",
                                    metrics={'loss': float(best_loss)}
                                )
                                # Move back to device
                                generator.to(self.device)
                            except Exception as e:
                                logger.error(f"Failed to log best model: {str(e)}")
                                generator.to(self.device)  # Ensure model returns to device

                except Exception as e:
                    logger.error(f"Failed to save checkpoint: {str(e)}")

                logger.info(
                    f"Epoch {epoch}: loss={avg_epoch_loss:.6f}, "
                    f"success_rate={success_count}/{batch_count}, "
                    f"time={epoch_time:.2f}s"
                )

            # End training
            total_time = time.time() - start_time
            logger.info(f"Human-guided training completed in {total_time:.2f}s")

            if self.experiment_tracker is not None:
                self.experiment_tracker.log_metrics({
                    'training.total_time': float(total_time),
                    'training.best_loss': float(best_loss)
                })

            return {
                'best_loss': best_loss,
                'total_time': total_time,
                'final_epoch': epoch,
                'success': True
            }

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            if self.experiment_tracker is not None:
                self.experiment_tracker.end_run()
            return {
                'best_loss': float('inf'),  # Or some other default value
                'total_time': 0,  # Or a calculated partial time
                'final_epoch': 0, # or the current epoch
                'success': False # Return False in case of failure
            }
```

## __init__.py
```python
from utils.experiment_tracking import ExperimentTracker

__all__ = ['ExperimentTracker']
```

## config.yaml
```yaml
training:
  loss_weights:
    perceptual: 0.1
    tv: 0.1
    boundary: 0       # Boundary loss weight - disabled for ablaition and baseline
    # boundary: 0.5       # Boundary loss weight
  modes:
    initial:
      epochs: 100
      batch_size: 32
    human_guided:
      epochs: 20
      batch_size: 5
      human_feedback_weight: 0.3
      base_loss_weight: 0.7
      learning_rate: 0.0001

evaluation:
  metrics:
    iou_threshold: 0.5
    precision_threshold: 0.7
    recall_threshold: 0.7
    boundary_size: 10    # Boundary size parameter for evaluation
  sampling:
    annotation_ratio: 0.01

  checkpoint_dir: "mvp_gan/checkpoints"
  checkpoint_file: "generator_epoch_49.pth"

mask_processing:
  roads:
    canny_low: 150        # Higher threshold to reduce noise
    canny_high: 300       # Higher threshold for strong edges only
    hough_threshold: 100  # More votes needed to detect a line
    hough_min_length: 100 # Longer minimum line length
    hough_max_gap: 20     # Slightly larger gap allowed for continuity
    line_thickness: 3     # Width of detected roads

  buildings: # TODO - likely via model similar to SEGMENT ANYTHING

  vegetation:
    min_area: 5000        # Minimum area for vegetation patches
    morph_kernel_size: 5  # Size of kernel for morphological operations

  fields:
    min_area: 10000       # Minimum area for field regions
    morph_kernel_size: 5  # Size of kernel for morphological operations

  visualization:
    enabled: true
    output_dir: "data/mask_visualization"

portal:
  base_url: "https://fkgsoftware.pythonanywhere.com"
  api_key: "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyIjoidGVzdCJ9.py3rCvl3ki2BLG2vS-WUnTIRsxK_46oJ_BVtd7gElag"
  # api_key: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiYWRtaW4ifQ.UZ7vyzSnem3Qk3Bl5tYTrbSXCErOqQYTB2xTXbUnyo4"

experiment_tracking:
  enabled: true
  tracking_uri: "file:./mlruns"  # Local filesystem by default
  experiment_name: "dsm_inpainting"
  tags:
    project: "terrain_generation"
    pipeline_version: "1.0"

geographical_split:
  enabled: true

data:
  raw_dir: "data/raw_data"
  processed_dir: "data/processed_data"
  output_dir: "data/output"
  input_zip_folder: "data/raw_data/input_zip_folder"
  extracted_dir: "data/raw_data/extracted"

  gan_images_dir: "mvp_gan/data/train/images"
  gan_masks_dir: "mvp_gan/data/train/masks"
  human_annotations_dir: "data/human_annotations"
  evaluation_results_dir: "data/evaluation_results"
  models_dir: "data/output/models"
  human_annotation_masks_dir: "human_annotation_masks"  # Directory name for human annotations within grid output

  parent_structure:
    processed:
      - "metadata"
      - "raw"
      - "train/images"
      - "train/masks"
      - "test/images"
      - "test/masks"
      - "val/images"
      - "val/masks"
    output:
      - "inpainted"
      - "colored"
      - "visualization"
      - "masks"
```

## mlflow_metrics_visualizer.py
```python
#!/usr/bin/env python3
"""
MLflow Metrics Aggregator and Visualizer

This script:
1. Scans all MLflow runs in a specified experiment
2. Groups them by run type (training, evaluation, etc.)
3. Collects metrics data from all runs
4. Normalizes timestamps
5. Generates plots for each metric showing values from all experiments
6. Adds trendlines to aid "at a glance understanding"

Usage:
    python mlflow_metrics_visualizer.py --experiment-name <name> --output-dir <dir>
"""

import os
import argparse
import sqlite3
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re
from scipy import stats

class MLflowMetricsVisualizer:
    def __init__(self, experiment_name='dsm_inpainting', mlruns_dir='./mlruns', output_dir=None):
        """
        Initialize the MLflow metrics visualizer.

        Args:
            experiment_name: Name of the MLflow experiment to analyze
            mlruns_dir: Directory containing MLflow runs data
            output_dir: Directory to save visualization outputs
        """
        self.experiment_name = experiment_name
        self.mlruns_dir = Path(mlruns_dir)
        self.output_dir = Path(output_dir or f"mlflow_metrics_viz_{experiment_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data structures
        self.runs_data = {}  # Dictionary of run_id -> run metadata
        self.metrics_data = defaultdict(lambda: defaultdict(list))  # metric_name -> run_id -> list of (time, value, step)
        self.run_types = defaultdict(list)  # run_type -> list of run_ids

        # Verify mlruns directory exists
        if not self.mlruns_dir.exists():
            raise FileNotFoundError(f"MLflow runs directory not found: {self.mlruns_dir}")

        # Find experiment directory
        self.experiment_dir = self._find_experiment_dir()
        if not self.experiment_dir:
            raise ValueError(f"Experiment '{experiment_name}' not found in {mlruns_dir}")

        print(f"Found experiment directory: {self.experiment_dir}")

    def _find_experiment_dir(self):
        """Find the experiment directory by name or ID."""
        # Check if experiment directory exists directly by name
        name_dir = self.mlruns_dir / self.experiment_name
        if name_dir.exists() and name_dir.is_dir():
            return name_dir

        # Check if there's a numeric directory containing the experiment
        for exp_dir in self.mlruns_dir.iterdir():
            if not exp_dir.is_dir() or exp_dir.name.startswith('.'):
                continue

            # Check meta.yaml if it exists
            meta_file = exp_dir / 'meta.yaml'
            if meta_file.exists():
                try:
                    with open(meta_file, 'r') as f:
                        content = f.read()
                        if f'name: {self.experiment_name}' in content:
                            return exp_dir
                except Exception:
                    continue

        # If still not found, look for experiment ID folders with runs
        # This is helpful when using SQLite or other backends where experiment name
        # might not be in directory name
        possible_dirs = []
        for exp_dir in self.mlruns_dir.iterdir():
            if not exp_dir.is_dir() or exp_dir.name.startswith('.'):
                continue

            # Check if any runs exist in this directory
            if list(exp_dir.glob('*/meta.yaml')):
                possible_dirs.append(exp_dir)

        # If there's exactly one directory with runs, use that
        if len(possible_dirs) == 1:
            return possible_dirs[0]

        # If we found multiple possible directories, log them for debugging
        if possible_dirs:
            print(f"Found multiple possible experiment directories: {[d.name for d in possible_dirs]}")
            print(f"Using the first one: {possible_dirs[0].name}")
            return possible_dirs[0]

        return None

    def _determine_run_type(self, run_id, run_data):
        """Determine the type of run based on tags or run name."""
        # Check for run name tag
        run_name = run_data.get('tags', {}).get('mlflow.runName', '')

        # Determine run type based on name patterns
        if 'train' in run_name.lower() and 'human' not in run_name.lower():
            return 'training_runs'
        elif 'eval' in run_name.lower():
            return 'evaluation_runs'
        elif 'human' in run_name.lower():
            return 'human_guided_runs'
        else:
            # Try to infer from metrics
            metrics_keys = set(run_data.get('metrics', {}).keys())
            if any('train' in m.lower() for m in metrics_keys):
                return 'training_runs'
            elif any('eval' in m.lower() for m in metrics_keys):
                return 'evaluation_runs'
            else:
                return 'other_runs'

    def _generate_human_readable_name(self, run_id, run_data):
        """Generate a human-readable name for the run if one isn't set."""
        # First check if there's a run name set
        if 'mlflow.runName' in run_data.get('tags', {}):
            return run_data['tags']['mlflow.runName']

        # If not, generate one based on the run type and a sequential number
        run_type = self._determine_run_type(run_id, run_data)
        base_name = {
            'training_runs': 'training_run',
            'evaluation_runs': 'evaluation_run',
            'human_guided_runs': 'human_guided_run',
            'other_runs': 'run'
        }.get(run_type, 'run')

        # Add a sequential number based on existing runs of this type
        existing_runs = len(self.run_types.get(run_type, []))
        return f"{base_name}_{existing_runs + 1:02d}"

    def scan_runs(self):
        """Scan all runs in the experiment and collect metadata."""
        # Find all run directories
        for run_dir in self.experiment_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name.startswith('.'):
                continue

            run_id = run_dir.name
            meta_file = run_dir / 'meta.yaml'

            if not meta_file.exists():
                continue

            # Load run metadata
            try:
                with open(meta_file, 'r') as f:
                    content = f.read()

                    # Extract basic info
                    run_data = {
                        'run_id': run_id,
                        'tags': {}
                    }

                    # Extract tags
                    tags_dir = run_dir / 'tags'
                    if tags_dir.exists():
                        for tag_file in tags_dir.iterdir():
                            if tag_file.is_file():
                                try:
                                    with open(tag_file, 'r') as tf:
                                        tag_value = tf.read().strip()
                                        run_data['tags'][tag_file.name] = tag_value
                                except Exception:
                                    pass

                    # Determine run type and store run data
                    run_type = self._determine_run_type(run_id, run_data)

                    # Generate a human-readable name if not set
                    if 'mlflow.runName' not in run_data['tags']:
                        run_data['tags']['mlflow.runName'] = self._generate_human_readable_name(run_id, run_data)

                    self.runs_data[run_id] = run_data
                    self.run_types[run_type].append(run_id)

            except Exception as e:
                print(f"Error reading metadata for run {run_id}: {e}")
                continue

        print(f"Found {len(self.runs_data)} runs in experiment {self.experiment_name}")
        for run_type, run_ids in self.run_types.items():
            print(f"  {run_type}: {len(run_ids)} runs")

        return self.runs_data

    def collect_metrics_data(self):
        """Collect metrics data from all runs."""
        # Process each run
        for run_id, run_data in self.runs_data.items():
            metrics_dir = self.experiment_dir / run_id / 'metrics'
            if not metrics_dir.exists():
                continue

            # Process each metric file
            for metric_file in metrics_dir.iterdir():
                if not metric_file.is_file():
                    continue

                metric_name = metric_file.name

                try:
                    # Read the metric data (timestamp, value, step)
                    metric_data = []
                    with open(metric_file, 'r') as f:
                        for line in f:
                            try:
                                # Format is: timestamp value step
                                parts = line.strip().split()
                                if len(parts) >= 3:
                                    timestamp = float(parts[0])
                                    value = float(parts[1])
                                    step = int(parts[2])
                                    metric_data.append((timestamp, value, step))
                            except Exception:
                                continue

                    # Store metric data if any valid entries found
                    if metric_data:
                        self.metrics_data[metric_name][run_id] = metric_data

                except Exception as e:
                    print(f"Error reading metric {metric_name} for run {run_id}: {e}")

        # Get metrics statistics
        total_metrics = len(self.metrics_data)
        total_data_points = sum(len(data) for metric_data in self.metrics_data.values()
                                for data in metric_data.values())

        print(f"Collected {total_data_points} data points for {total_metrics} metrics")
        return self.metrics_data

    def normalize_timestamps(self, metric_name, run_data, run_type="training_runs"):
        """
        Normalize timestamps for a metric across runs to create a continuous timeline.

        Args:
            metric_name: The name of the metric
            run_data: Dictionary of run_id -> list of (time, value, step)
            run_type: Type of run for grouping and sorting

        Returns:
            DataFrame with normalized timestamps for a continuous timeline
        """
        # Group runs by experiment or series
        # Now we identify runs in a sequential series based on run names
        # e.g., "training_run_01", "training_run_02", etc.
        run_series = defaultdict(list)

        # Extract sequence numbers from run names
        for run_id in run_data:
            if run_id not in self.runs_data:
                continue

            run_name = self.runs_data[run_id]['tags'].get('mlflow.runName', run_id[:8])

            # Try to extract a sequence identifier from the run name
            # Look for patterns like "training_run_01", "eval_02", etc.
            match = re.search(r'(?:^|_)(\d+)$', run_name)
            if match:
                sequence_num = int(match.group(1))
                # Get the run name prefix (everything before the number)
                prefix = run_name[:match.start()]
                run_series[prefix].append((run_id, sequence_num))
            else:
                # If no sequence number found, use the run name as-is
                run_series[run_name].append((run_id, 0))

        # Sort runs within each series by their sequence number
        for prefix, runs in run_series.items():
            run_series[prefix] = [run_id for run_id, _ in sorted(runs, key=lambda x: x[1])]

        # Prepare data for plotting with continuous timeline
        normalized_data = []

        # Process each series separately
        for series_name, run_ids in run_series.items():
            last_timestamp = 0  # Keeps track of the continuous timeline

            for run_id in run_ids:
                if not run_data.get(run_id):
                    continue

                # Get data points for this run, sorted by timestamp
                data_points = sorted(run_data[run_id], key=lambda x: x[0])
                if not data_points:
                    continue

                # Get relative times within this run
                run_start_time = data_points[0][0]

                # Get human-readable run name
                run_name = self.runs_data[run_id]['tags'].get('mlflow.runName', run_id[:8])

                # Add data points with continuous timeline
                for timestamp, value, step in data_points:
                    # Time within this run (in seconds)
                    relative_time = (timestamp - run_start_time) / 1000.0
                    # Continuous timeline
                    continuous_time = last_timestamp + relative_time

                    normalized_data.append({
                        'run_id': run_id,
                        'run_name': run_name,
                        'series': series_name,
                        'time': continuous_time,
                        'value': value,
                        'step': step
                    })

                # Update the last timestamp for the next run
                # Add the duration of this run to the continuous timeline
                if data_points:
                    run_duration = (data_points[-1][0] - run_start_time) / 1000.0
                    last_timestamp += run_duration + 10  # Add a small gap between runs

        return pd.DataFrame(normalized_data)

    def plot_metric(self, metric_name, run_type="training_runs"):
        """
        Generate a plot for a specific metric across all runs of a given type,
        with continuous timeline, color-coding by experiment, and added trendline.

        Args:
            metric_name: Name of the metric to plot
            run_type: Type of runs to include in the plot

        Returns:
            Path to the saved plot
        """
        # Get run IDs for the specified type
        run_ids = self.run_types.get(run_type, [])
        if not run_ids:
            print(f"No runs found of type: {run_type}")
            return None

        # Get metric data for these runs
        metric_data = {}
        for run_id in run_ids:
            if run_id in self.metrics_data[metric_name]:
                metric_data[run_id] = self.metrics_data[metric_name][run_id]

        if not metric_data:
            print(f"No data for metric '{metric_name}' in {run_type}")
            return None

        # Normalize timestamps to create continuous timeline
        df = self.normalize_timestamps(metric_name, metric_data, run_type)

        if df.empty:
            print(f"No valid data after normalization for metric '{metric_name}'")
            return None

        # Create figure
        plt.figure(figsize=(14, 7))

        # Create nice readable metric name
        display_name = metric_name.replace('_', ' ').replace('.', ' - ').title()

        # Plot by run_name for better readability
        sns.lineplot(data=df, x='time', y='value', hue='run_name',
                    marker='o', markersize=4, linestyle='-', alpha=0.7)

        # Add overall trendline
        if len(df) > 1:  # Need at least 2 points for a trendline
            # Calculate the linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(df['time'], df['value'])

            # Generate points for the trendline
            x_min, x_max = df['time'].min(), df['time'].max()
            x_trend = np.linspace(x_min, x_max, 100)
            y_trend = slope * x_trend + intercept

            # Plot the trendline with a thicker, darker line
            plt.plot(x_trend, y_trend, 'r--', linewidth=2, label=f'Trendline (slope: {slope:.4f})')

            # Add R-squared information
            plt.annotate(f'R² = {r_value**2:.4f}',
                        xy=(0.02, 0.95),
                        xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        plt.title(f"{display_name} - Continuous Timeline Across All Runs ({run_type.replace('_', ' ').title()})")
        plt.xlabel("Continuous Timeline (seconds)")
        plt.ylabel(display_name)
        plt.grid(True, alpha=0.3)

        # Add legend with better formatting
        plt.legend(title="Run Name", fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        # Create output directory
        output_path = self.output_dir / run_type / "metrics"
        output_path.mkdir(parents=True, exist_ok=True)

        # Save figure
        file_path = output_path / f"all_{metric_name}_plot.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved plot to {file_path}")
        return file_path

    def generate_all_metric_plots(self):
        """Generate plots for all metrics across all run types."""
        # Ensure data is loaded
        if not self.runs_data:
            self.scan_runs()

        if not self.metrics_data:
            self.collect_metrics_data()

        # Create plots for each run type and metric
        plots_created = 0

        for run_type, run_ids in self.run_types.items():
            if not run_ids:
                continue

            # Find all metrics for this run type
            run_metrics = set()
            for metric_name, run_data in self.metrics_data.items():
                if any(run_id in run_data for run_id in run_ids):
                    run_metrics.add(metric_name)

            print(f"Generating plots for {len(run_metrics)} metrics in {run_type}")

            # Create plots for each metric
            for metric_name in sorted(run_metrics):
                try:
                    self.plot_metric(metric_name, run_type)
                    plots_created += 1
                except Exception as e:
                    print(f"Error creating plot for {metric_name} in {run_type}: {e}")

        print(f"Created {plots_created} metric plots")
        return plots_created

    def generate_summary_report(self):
        """Generate a summary report of all runs and metrics."""
        # Ensure data is loaded
        if not self.runs_data:
            self.scan_runs()

        if not self.metrics_data:
            self.collect_metrics_data()

        # Create summary directory
        summary_dir = self.output_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)

        # Create run summary CSV
        runs_summary = []
        for run_id, run_data in self.runs_data.items():
            run_type = next((rt for rt, ids in self.run_types.items() if run_id in ids), "unknown")
            run_name = run_data.get('tags', {}).get('mlflow.runName', run_id[:8])

            # Count metrics for this run
            metric_count = sum(1 for metric_data in self.metrics_data.values()
                              if run_id in metric_data)

            # Get data point count
            data_point_count = sum(len(data) for metric_data in self.metrics_data.values()
                                   for rid, data in metric_data.items() if rid == run_id)

            runs_summary.append({
                'run_id': run_id,
                'run_name': run_name,
                'run_type': run_type,
                'metric_count': metric_count,
                'data_point_count': data_point_count
            })

        # Save runs summary
        runs_df = pd.DataFrame(runs_summary)
        runs_csv_path = summary_dir / "runs_summary.csv"
        runs_df.to_csv(runs_csv_path, index=False)

        # Create metrics summary CSV
        metrics_summary = []
        for metric_name, run_data in self.metrics_data.items():
            run_count = len(run_data)
            data_point_count = sum(len(data) for data in run_data.values())

            # Calculate min/max/avg values
            all_values = [value for data in run_data.values() for _, value, _ in data]

            metrics_summary.append({
                'metric_name': metric_name,
                'run_count': run_count,
                'data_point_count': data_point_count,
                'min_value': min(all_values) if all_values else None,
                'max_value': max(all_values) if all_values else None,
                'avg_value': sum(all_values) / len(all_values) if all_values else None
            })

        # Save metrics summary
        metrics_df = pd.DataFrame(metrics_summary)
        metrics_csv_path = summary_dir / "metrics_summary.csv"
        metrics_df.to_csv(metrics_csv_path, index=False)

        print(f"Saved run summary to {runs_csv_path}")
        print(f"Saved metrics summary to {metrics_csv_path}")

        return {
            'runs_summary': runs_csv_path,
            'metrics_summary': metrics_csv_path
        }

def main():
    parser = argparse.ArgumentParser(description="MLflow Metrics Aggregator and Visualizer")
    parser.add_argument("--experiment-name", default="dsm_inpainting",
                      help="Name of the MLflow experiment")
    parser.add_argument("--mlruns-dir", default="./mlruns",
                      help="Directory containing MLflow runs data")
    parser.add_argument("--output-dir", default=None,
                      help="Output directory for visualizations")
    parser.add_argument("--run-type", default=None,
                      help="Specific run type to analyze (training_runs, evaluation_runs, etc.)")
    parser.add_argument("--metric", default=None,
                      help="Specific metric to visualize")

    args = parser.parse_args()

    # Create visualizer
    visualizer = MLflowMetricsVisualizer(
        experiment_name=args.experiment_name,
        mlruns_dir=args.mlruns_dir,
        output_dir=args.output_dir
    )

    # Scan runs and collect metrics data
    visualizer.scan_runs()
    visualizer.collect_metrics_data()

    # If specific metric and run type provided, only plot that
    if args.metric and args.run_type:
        visualizer.plot_metric(args.metric, args.run_type)
    # If only run type provided, plot all metrics for that run type
    elif args.run_type:
        for metric_name in visualizer.metrics_data.keys():
            visualizer.plot_metric(metric_name, args.run_type)
    # If only metric provided, plot it for all run types
    elif args.metric:
        for run_type in visualizer.run_types.keys():
            visualizer.plot_metric(args.metric, run_type)
    # Otherwise, generate all plots and summary report
    else:
        visualizer.generate_all_metric_plots()
        visualizer.generate_summary_report()

if __name__ == "__main__":
    main()
```

## evaluationn_experiment.py
```python
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
```

## run_pipeline.sh
```bash
#!/bin/bash
#
# run_pipeline.sh - Improved script to run the main_pipeline.py with proper MLflow management
#
# This script provides a unified interface to run the pipeline in different modes while
# ensuring proper MLflow server management and error handling.

# Exit on error, unset variable, and pipe failure
set -e
set -u
set -o pipefail

# Configuration
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MLFLOW_CHECK_URL="http://localhost:5000"  # URL to check for MLflow

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"
chmod -R 755 "$LOG_DIR"

# Ensure mlruns directory exists with proper permissions
mkdir -p "./mlruns"
chmod -R 755 "./mlruns"

# Function to display usage information
show_usage() {
    echo "Usage: $0 [mode] [options]"
    echo "Modes:"
    echo "  train            - Run training mode"
    echo "  evaluate         - Run evaluation mode"
    echo "  human_guided     - Run human-guided training mode"
    echo "  all              - Run all modes sequentially"
    echo ""
    echo "Options:"
    echo "  --model PATH     - Specify input model path"
    echo "  --debug          - Enable debug logging"
    echo "  --no-mlflow      - Don't start/stop MLflow server (use existing)"
    echo "  --port PORT      - Specify the port for the MLflow UI (default: 5000)"
    echo ""
    echo "Example: $0 train --model models/best_model.pth"
}

# Function to run a specific mode
run_mode() {
    local mode=$1
    local log_file="${LOG_DIR}/pipeline_${mode}_${TIMESTAMP}.log"

    # Map human_guided to human_guided_train for main_pipeline.py
    local python_mode="$mode"
    if [ "$python_mode" = "human_guided" ]; then
        python_mode="human_guided_train"
    fi

    echo "Running pipeline in ${mode} mode..."
    echo "Logging to: ${log_file}"

    # Build command with any extra options
    CMD="python main_pipeline.py --mode $python_mode $EXTRA_ARGS"
    echo "Command: $CMD"

    # Set EXPERIMENT_MODE environment variable
    export EXPERIMENT_MODE=true

    # Run the pipeline and capture exit code
    $CMD 2>&1 | tee "${log_file}"
    exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        echo "✓ Successfully completed ${mode} mode"
        return 0
    else
        echo "✗ Failed in ${mode} mode (exit code: $exit_code)"
        return $exit_code
    fi
}

# Parse arguments
if [ $# -lt 1 ]; then
    show_usage
    exit 1
fi

MODE=$1
shift  # Remove mode from arguments

# Parse options
EXTRA_ARGS=""
NO_MLFLOW=false
MLFLOW_PORT="5000"  # Default port

while [ $# -gt 0 ]; do
    case "$1" in
        --model)
            EXTRA_ARGS="$EXTRA_ARGS --input_model $2"
            shift 2
            ;;
        --debug)
            EXTRA_ARGS="$EXTRA_ARGS --debug"
            shift
            ;;
        --no-mlflow)
            NO_MLFLOW=true
            shift
            ;;
        --port)
            MLFLOW_PORT="$2"
            EXTRA_ARGS="$EXTRA_ARGS --port $2" # Pass the port to main_pipeline if needed
            shift 2
            ;;
        --grid)
            EXTRA_ARGS="$EXTRA_ARGS --grid $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Update MLFLOW_CHECK_URL with the correct port
MLFLOW_CHECK_URL="http://localhost:${MLFLOW_PORT}"

# Trap to ensure cleanup on exit
trap 'echo "Exiting"; exit' EXIT

# Check if mlruns directory needs to be reset
if [ ! -d "./mlruns" ] || [ ! -d "./mlruns/dsm_inpainting" ]; then
    echo "MLflow directory structure not found or incomplete. Initializing..."
    ./reset_mlflow.sh
fi

# Process requested mode
case $MODE in
    "train"|"evaluate"|"human_guided")
        # Start MLflow for training modes using start_mlflow.sh
        if [ "$MODE" = "train" ] || [ "$MODE" = "human_guided" ]; then
            if [ "$NO_MLFLOW" = false ]; then  # Only start if --no-mlflow is not set
                 ./start_mlflow.sh "$MLFLOW_PORT"
                 # Wait a moment for MLflow to start
                 sleep 3
            fi
        fi

        # Run the requested mode
        run_mode "$MODE"
        ;;

    "all")
        echo "Running all modes sequentially..."

        # Start MLflow once for all modes using start_mlflow.sh
        if [ "$NO_MLFLOW" = false ]; then  # Only start if --no-mlflow is not set
            ./start_mlflow.sh "$MLFLOW_PORT"
            # Wait a moment for MLflow to start
            sleep 3
        fi

        # Run each mode
        FAILED=false
        for m in "train" "evaluate" "human_guided"; do
            echo ""
            echo "======== RUNNING $m MODE ========"
            if run_mode "$m"; then
                echo "✓ Completed $m mode"
            else
                echo "✗ Failed in $m mode"
                FAILED=true
                break
            fi
            echo "======== COMPLETED $m MODE ========"
            echo ""
            sleep 2  # Brief pause between modes
        done

        # Final status
        echo ""
        if [ "$FAILED" = false ]; then
            echo "✓ All modes completed successfully!"
            exit 0
        else
            echo "✗ Pipeline failed"
            exit 1
        fi
        ;;

    *)
        echo "Error: Invalid mode specified: $MODE"
        show_usage
        exit 1
        ;;
esac
```

## plot_run_phases.py
```python
#!/usr/bin/env python3
"""
MLflow Run Type Visualization Script

This script creates visualizations from MLflow experiment data, organizing runs by type
(training, evaluation, human-guided) and plotting metrics chronologically to show the
progression across different phases of model development.

Usage:
    python plot_run_phases.py --experiment-name <name> --tracking-uri <uri> --output-dir <dir>
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import yaml
from pathlib import Path
from datetime import datetime
import re

class RunPhaseVisualizer:
    def __init__(self, experiment_name, tracking_uri=None, output_dir=None):
        """
        Initialize the run phase visualizer.

        Args:
            experiment_name: Name of the MLflow experiment to visualize
            tracking_uri: MLflow tracking URI
            output_dir: Directory to save visualization outputs
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "file:./mlruns"
        self.output_dir = Path(output_dir or f"phase_viz_{experiment_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data structures
        self.runs_df = None
        self.runs_by_phase = None

        # Configure MLflow client
        mlflow.set_tracking_uri(self.tracking_uri)

        # Get experiment
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        if not self.experiment:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        print(f"Found experiment '{experiment_name}' with ID: {self.experiment.experiment_id}")

    def load_runs_data(self):
        """Load runs data for the experiment and categorize by phase."""
        # Get all runs for the experiment
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=["attribute.start_time ASC"]
        )

        if runs.empty:
            raise ValueError(f"No runs found for experiment '{self.experiment_name}'")

        # Clean up the DataFrame
        self.runs_df = runs.copy()

        # Parse timestamps
        self.runs_df['start_time_dt'] = pd.to_datetime(self.runs_df['start_time'], unit='ms')
        self.runs_df['end_time_dt'] = pd.to_datetime(self.runs_df['end_time'], unit='ms')

        # Calculate run duration
        self.runs_df['duration_minutes'] = (self.runs_df['end_time'] - self.runs_df['start_time']) / (1000 * 60)

        # Categorize runs by phase based on run name pattern
        self.runs_df['phase'] = 'unknown'

        # Use run name to categorize when available
        if 'tags.mlflow.runName' in self.runs_df.columns:
            # Training runs
            mask_training = self.runs_df['tags.mlflow.runName'].str.contains('train', case=False, na=False)
            self.runs_df.loc[mask_training, 'phase'] = 'training'

            # Evaluation runs
            mask_eval = self.runs_df['tags.mlflow.runName'].str.contains('eval', case=False, na=False)
            self.runs_df.loc[mask_eval, 'phase'] = 'evaluation'

            # Human-guided runs
            mask_human = self.runs_df['tags.mlflow.runName'].str.contains('human|guided', case=False, na=False)
            self.runs_df.loc[mask_human, 'phase'] = 'human_guided'

        # Group runs by phase
        self.runs_by_phase = {
            'training': self.runs_df[self.runs_df['phase'] == 'training'],
            'evaluation': self.runs_df[self.runs_df['phase'] == 'evaluation'],
            'human_guided': self.runs_df[self.runs_df['phase'] == 'human_guided'],
            'unknown': self.runs_df[self.runs_df['phase'] == 'unknown']
        }

        # Log run counts by phase
        for phase, df in self.runs_by_phase.items():
            print(f"Found {len(df)} {phase} runs")

        return self.runs_df

    def load_run_metadata(self):
        """
        Load run metadata directly from YAML files to get more detailed information.
        This is useful when MLflow API doesn't provide all needed details.
        """
        # Extract experiment path from tracking URI
        if self.tracking_uri.startswith("file:"):
            base_path = Path(self.tracking_uri.replace("file:", ""))
        else:
            print("Can only load metadata for file-based tracking URIs")
            return

        exp_path = base_path / self.experiment.experiment_id

        if not exp_path.exists():
            print(f"Experiment path not found: {exp_path}")
            return

        # Load metadata for each run
        run_metadata = []
        for run_dir in exp_path.iterdir():
            if run_dir.is_dir() and (run_dir / "meta.yaml").exists():
                try:
                    with open(run_dir / "meta.yaml", 'r') as f:
                        meta = yaml.safe_load(f)

                    # Extract run ID from directory name
                    run_id = run_dir.name

                    # Add to metadata collection
                    meta['run_id'] = run_id
                    run_metadata.append(meta)
                except Exception as e:
                    print(f"Error loading metadata for run {run_dir.name}: {e}")

        # Convert to DataFrame for easier manipulation
        meta_df = pd.DataFrame(run_metadata)

        # Merge with runs_df if possible
        if 'run_id' in meta_df.columns and self.runs_df is not None:
            # Keep only the columns not already in runs_df to avoid duplicates
            meta_cols = [c for c in meta_df.columns if c not in self.runs_df.columns or c == 'run_id']
            meta_df = meta_df[meta_cols]

            # Merge with runs_df
            self.runs_df = self.runs_df.merge(meta_df, on='run_id', how='left')

            print(f"Merged metadata for {len(meta_df)} runs")

        return meta_df

    def extract_metrics_by_phase(self):
        """Extract metrics and organize them by run phase."""
        if self.runs_df is None:
            self.load_runs_data()

        # Find all metrics columns
        metrics_cols = [col for col in self.runs_df.columns if col.startswith('metrics.')]

        # Create metrics dataframes by phase
        metrics_by_phase = {}

        for phase, phase_df in self.runs_by_phase.items():
            if len(phase_df) == 0:
                continue

            # Extract metrics for this phase
            phase_metrics = phase_df[['run_id', 'start_time_dt', 'phase'] + metrics_cols].copy()

            # Rename columns to remove 'metrics.' prefix
            phase_metrics.columns = [col.replace('metrics.', '') if col.startswith('metrics.') else col
                                     for col in phase_metrics.columns]

            metrics_by_phase[phase] = phase_metrics

        return metrics_by_phase

    def plot_metrics_across_phases(self, metrics=None, figsize=(15, 10)):
        """
        Plot metrics across all phases in chronological order.

        Args:
            metrics: List of specific metrics to plot (default: plot all)
            figsize: Figure size (width, height) in inches
        """
        if self.runs_df is None:
            self.load_runs_data()

        # Get metrics by phase
        metrics_by_phase = self.extract_metrics_by_phase()

        # Combine all phases for determining available metrics
        all_metrics = pd.concat(list(metrics_by_phase.values()), ignore_index=True)

        # Determine which metrics to plot
        available_metrics = [col for col in all_metrics.columns
                             if col not in ['run_id', 'start_time_dt', 'phase']]

        if metrics:
            plot_metrics = [m for m in metrics if m in available_metrics]
            if not plot_metrics:
                print(f"Warning: None of the specified metrics found. Available metrics: {available_metrics}")
                # Fall back to all metrics
                plot_metrics = available_metrics[:min(8, len(available_metrics))]
        else:
            # Prioritize common loss metrics
            loss_metrics = [m for m in available_metrics if 'loss' in m.lower()]
            if loss_metrics:
                plot_metrics = loss_metrics[:min(8, len(loss_metrics))]
            else:
                # Limit to reasonable number of metrics if no loss metrics found
                plot_metrics = available_metrics[:min(8, len(available_metrics))]

        print(f"Plotting metrics: {', '.join(plot_metrics)}")

        # Create figure
        fig, axes = plt.subplots(len(plot_metrics), 1, figsize=figsize, sharex=True)
        if len(plot_metrics) == 1:
            axes = [axes]

        # Phase colors and markers
        phase_styles = {
            'training': {'color': 'blue', 'marker': 'o', 'linestyle': '-', 'label': 'Training'},
            'evaluation': {'color': 'green', 'marker': 's', 'linestyle': '--', 'label': 'Evaluation'},
            'human_guided': {'color': 'red', 'marker': '^', 'linestyle': '-.', 'label': 'Human-Guided'},
            'unknown': {'color': 'gray', 'marker': 'x', 'linestyle': ':', 'label': 'Unknown'}
        }

        # Plot each metric
        for i, metric in enumerate(plot_metrics):
            ax = axes[i]

            # Plot each phase
            for phase, phase_df in metrics_by_phase.items():
                style = phase_styles[phase]

                if metric in phase_df.columns:
                    # Skip if no data for this metric in this phase
                    valid_data = phase_df[~phase_df[metric].isna()]

                    if not valid_data.empty:
                        # Sort by timestamp to ensure chronological order
                        valid_data = valid_data.sort_values('start_time_dt')

                        # Plot this phase
                        ax.plot(valid_data['start_time_dt'], valid_data[metric],
                                marker=style['marker'], color=style['color'],
                                linestyle=style['linestyle'], label=f"{style['label']} {metric}")

                        # Optionally add run annotations
                        for _, row in valid_data.iterrows():
                            ax.annotate(f"{row['run_id'][-6:]}",
                                        (row['start_time_dt'], row[metric]),
                                        textcoords="offset points",
                                        xytext=(0,10),
                                        ha='center',
                                        fontsize=7)

            # Customize plot
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')

            # Add vertical lines between phases
            phase_changes = self.runs_df.sort_values('start_time_dt')
            if len(phase_changes) > 1:
                for j in range(1, len(phase_changes)):
                    if phase_changes.iloc[j-1]['phase'] != phase_changes.iloc[j]['phase']:
                        ax.axvline(x=phase_changes.iloc[j]['start_time_dt'],
                                   color='black', linestyle='--', alpha=0.5)

        # Set common x-axis label
        axes[-1].set_xlabel('Run Start Time')

        # Add title
        plt.suptitle(f"Metrics Across Training Phases: {self.experiment_name}", fontsize=16)

        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / "metrics_across_phases.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved cross-phase metrics plot to {output_path}")

        return fig, axes

    def plot_loss_metrics(self, figsize=(15, 8)):
        """Plot specifically loss-related metrics across all phases."""
        if self.runs_df is None:
            self.load_runs_data()

        # Find all metrics columns
        metrics_cols = [col for col in self.runs_df.columns if col.startswith('metrics.')]

        # Filter for loss-related metrics
        loss_metrics = [col.replace('metrics.', '') for col in metrics_cols
                        if 'loss' in col.lower()]

        return self.plot_metrics_across_phases(metrics=loss_metrics, figsize=figsize)

    def create_phase_summary(self):
        """Create a summary table with key metrics by phase."""
        if self.runs_df is None:
            self.load_runs_data()

        # Extract metrics by phase
        metrics_by_phase = self.extract_metrics_by_phase()

        # Create summary dataframe
        summary_data = []

        for phase, phase_df in metrics_by_phase.items():
            if len(phase_df) == 0:
                continue

            # Get metrics columns
            metric_cols = [col for col in phase_df.columns
                          if col not in ['run_id', 'start_time_dt', 'phase']]

            # Calculate summary statistics for each metric
            for metric in metric_cols:
                valid_values = phase_df[~phase_df[metric].isna()][metric]

                if len(valid_values) > 0:
                    summary_data.append({
                        'Phase': phase,
                        'Metric': metric,
                        'Mean': valid_values.mean(),
                        'Min': valid_values.min(),
                        'Max': valid_values.max(),
                        'Count': len(valid_values)
                    })

        # Convert to dataframe
        summary_df = pd.DataFrame(summary_data)

        # Save to CSV
        output_path = self.output_dir / "phase_metrics_summary.csv"
        summary_df.to_csv(output_path, index=False)
        print(f"Saved phase metrics summary to {output_path}")

        return summary_df

    def create_full_report(self):
        """Generate a complete analysis with all visualizations and summaries."""
        # Load all data
        self.load_runs_data()
        self.load_run_metadata()

        # Plot combined metrics
        self.plot_metrics_across_phases()

        # Plot loss-specific metrics
        self.plot_loss_metrics()

        # Create summary tables
        self.create_phase_summary()

        print(f"Full report generated in {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize MLflow runs by phase")
    parser.add_argument("--experiment-name", required=True, help="Name of the MLflow experiment")
    parser.add_argument("--tracking-uri", default="file:./mlruns", help="MLflow tracking URI")
    parser.add_argument("--output-dir", default=None, help="Output directory for visualizations")
    parser.add_argument("--metrics", nargs='+', help="Specific metrics to plot")

    args = parser.parse_args()

    # Create visualizer
    visualizer = RunPhaseVisualizer(
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        output_dir=args.output_dir
    )

    # Generate visualizations
    if args.metrics:
        visualizer.load_runs_data()
        visualizer.load_run_metadata()
        visualizer.plot_metrics_across_phases(metrics=args.metrics)
    else:
        visualizer.create_full_report()


if __name__ == "__main__":
    main()
```

## mlflow_data_extractor.py
```python
#!/usr/bin/env python3
"""
MLflow Data Extractor

This script extracts data from MLflow experiment directories into clean, structured formats
that can be easily consumed by downstream statistical analysis.

Usage:
    python mlflow_data_extractor.py --experiment-dir EXP_DIR --output-dir OUTPUT_DIR

The script creates three output files:
1. experiment_metadata.json - Information about the experiment and its runs
2. metrics_data.json - All metrics data in a structured format
3. parameters_data.json - All parameter data in a structured format
"""

import argparse
import json
import logging
import re
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling NumPy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


class MLflowDataExtractor:
    def __init__(self, experiment_dir, output_dir):
        """
        Initialize the MLflow data extractor.

        Args:
            experiment_dir: Directory containing MLflow experiment data
            output_dir: Directory to save extracted data
        """
        self.experiment_dir = Path(experiment_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get experiment name from path
        self.experiment_name = self._get_experiment_name()

        # Initialize data structures
        self.experiment_metadata = {
            'id': str(self.experiment_dir),
            'name': self.experiment_name,
            'extraction_time': datetime.now().isoformat(),
            'runs': []
        }

        self.metrics_data = defaultdict(dict)  # metric_name -> {run_id -> [values]}
        self.parameters_data = defaultdict(dict)  # param_name -> {run_id -> value}
        self.run_groups = defaultdict(list)  # group_name -> [run_ids]

    def _get_experiment_name(self):
        """Extract a readable experiment name from the path."""
        # Get parent directory name as part of the experiment name for clarity
        parent_dir = self.experiment_dir.parent.parent.name
        exp_dir_name = self.experiment_dir.name

        # Try to read experiment name from meta.yaml if it exists
        meta_file = self.experiment_dir / 'meta.yaml'
        if meta_file.exists():
            try:
                with open(meta_file, 'r') as f:
                    content = f.read()
                    name_match = re.search(r'name: (.+)', content)
                    if name_match:
                        return f"{parent_dir}/{name_match.group(1).strip()}"
            except Exception as e:
                logger.warning(f"Could not read experiment name from meta.yaml: {e}")

        return f"{parent_dir}/{exp_dir_name}"

    def extract_data(self):
        """Extract all data from the experiment directory."""
        logger.info(f"Extracting data from experiment: {self.experiment_name}")

        # Find and process all run directories
        runs_found = self._process_runs()
        logger.info(f"Processed {runs_found} runs")

        # Group runs by parameters or tags
        self._group_runs()

        # Process metrics
        metrics_processed = self._process_metrics()
        logger.info(f"Processed {metrics_processed} metrics")

        # Calculate summary statistics
        self._calculate_summary_statistics()

        return {
            'experiment_metadata': self.experiment_metadata,
            'metrics_data': self.metrics_data,
            'parameters_data': self.parameters_data,
            'run_groups': self.run_groups
        }

    def _process_runs(self):
        """Process all run directories within the experiment."""
        runs_found = 0

        for run_dir in self.experiment_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name.startswith('.') or run_dir.name in ['artifacts', 'metrics', 'params', 'tags']:
                continue

            run_id = run_dir.name
            meta_file = run_dir / 'meta.yaml'

            if not meta_file.exists():
                logger.warning(f"Skipping run {run_id}: no meta.yaml found")
                continue

            try:
                # Extract run metadata
                run_data = {
                    'run_id': run_id,
                    'params': {},
                    'tags': {}
                }

                # Extract tags
                tags_dir = run_dir / 'tags'
                if tags_dir.exists():
                    for tag_file in tags_dir.iterdir():
                        if tag_file.is_file():
                            try:
                                with open(tag_file, 'r') as f:
                                    tag_value = f.read().strip()
                                    run_data['tags'][tag_file.name] = tag_value
                            except Exception as e:
                                logger.warning(f"Error reading tag {tag_file.name}: {e}")

                # Extract parameters
                params_dir = run_dir / 'params'
                if params_dir.exists():
                    for param_file in params_dir.iterdir():
                        if param_file.is_file():
                            try:
                                with open(param_file, 'r') as f:
                                    param_value = f.read().strip()
                                    run_data['params'][param_file.name] = param_value

                                    # Also store in parameters_data
                                    self.parameters_data[param_file.name][run_id] = param_value
                            except Exception as e:
                                logger.warning(f"Error reading parameter {param_file.name}: {e}")

                # Add run data to experiment metadata
                self.experiment_metadata['runs'].append(run_data)
                runs_found += 1

            except Exception as e:
                logger.error(f"Error processing run {run_id}: {e}")
                logger.error(traceback.format_exc())

        return runs_found

    def _process_metrics(self):
        """Process metrics for all runs."""
        metrics_processed = 0

        for run_data in self.experiment_metadata['runs']:
            run_id = run_data['run_id']
            metrics_dir = self.experiment_dir / run_id / 'metrics'

            if not metrics_dir.exists():
                logger.warning(f"No metrics directory found for run {run_id}")
                continue

            # Add metrics field to run data
            run_data['metrics'] = {}

            # Process each metric file
            for metric_file in metrics_dir.iterdir():
                if not metric_file.is_file():
                    continue

                metric_name = metric_file.name

                try:
                    # Read the metric data
                    values = []
                    timestamps = []
                    steps = []

                    with open(metric_file, 'r') as f:
                        for line in f:
                            try:
                                # Format is: timestamp value step
                                parts = line.strip().split()
                                if len(parts) >= 3:
                                    timestamp = float(parts[0])
                                    value = float(parts[1])
                                    step = int(parts[2])

                                    if np.isfinite(value):  # Skip NaN or Inf
                                        values.append(value)
                                        timestamps.append(timestamp)
                                        steps.append(step)
                            except Exception as e:
                                logger.warning(f"Error parsing metric line in {metric_file}: {e}")
                                continue

                    if values:
                        # Store metrics data
                        metric_data = {
                            'values': values,
                            'timestamps': timestamps,
                            'steps': steps
                        }

                        self.metrics_data[metric_name][run_id] = metric_data
                        metrics_processed += 1

                except Exception as e:
                    logger.error(f"Error processing metric {metric_name} for run {run_id}: {e}")
                    logger.error(traceback.format_exc())

        return metrics_processed

    def _group_runs(self):
        """Group runs by shared parameter values and other properties."""
        # Group by common parameter values
        param_groups = defaultdict(lambda: defaultdict(list))

        for run_data in self.experiment_metadata['runs']:
            run_id = run_data['run_id']

            # Group by each parameter value
            for param_name, param_value in run_data['params'].items():
                group_key = f"{param_name}={param_value}"
                param_groups[param_name][group_key].append(run_id)

        # Store param-based groups
        for param_name, groups in param_groups.items():
            for group_key, run_ids in groups.items():
                if len(run_ids) > 1:  # Only store groups with multiple runs
                    self.run_groups[f"param:{group_key}"] = run_ids

        # Group by user-defined run tags if available
        for run_data in self.experiment_metadata['runs']:
            run_id = run_data['run_id']

            # Look for tags that indicate groups
            for tag_name, tag_value in run_data['tags'].items():
                if 'group' in tag_name.lower() or 'set' in tag_name.lower():
                    group_key = f"{tag_name}={tag_value}"
                    self.run_groups[f"tag:{group_key}"].append(run_id)

    def _calculate_summary_statistics(self):
        """Calculate summary statistics for each metric across runs."""
        for metric_name, runs in self.metrics_data.items():
            # Add summary statistics to each run's metric data
            for run_id, data in runs.items():
                values = data['values']

                if values:
                    data['summary'] = {
                        'min': np.min(values),
                        'max': np.max(values),
                        'mean': np.mean(values),
                        'median': np.median(values),
                        'std': np.std(values),
                        'first': values[0],
                        'last': values[-1],
                        'count': len(values)
                    }

                    # Calculate improvement metrics
                    first_val = values[0]
                    last_val = values[-1]
                    data['summary']['improvement'] = last_val - first_val

                    if first_val != 0:
                        data['summary']['pct_improvement'] = ((last_val - first_val) / abs(first_val)) * 100
                    else:
                        data['summary']['pct_improvement'] = np.nan

                    # Add to run's summary in experiment metadata
                    for run_data in self.experiment_metadata['runs']:
                        if run_data['run_id'] == run_id:
                            if 'metrics_summary' not in run_data:
                                run_data['metrics_summary'] = {}
                            run_data['metrics_summary'][metric_name] = data['summary']
                            break

    def save_data(self):
        """Save all extracted data to output files."""
        # Save experiment metadata
        metadata_path = self.output_dir / 'experiment_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.experiment_metadata, f, indent=2, cls=NumpyEncoder)

        # Save metrics data
        metrics_path = self.output_dir / 'metrics_data.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_data, f, indent=2, cls=NumpyEncoder)

        # Save parameters data
        params_path = self.output_dir / 'parameters_data.json'
        with open(params_path, 'w') as f:
            json.dump(self.parameters_data, f, indent=2, cls=NumpyEncoder)

        # Save run groups
        groups_path = self.output_dir / 'run_groups.json'
        with open(groups_path, 'w') as f:
            json.dump(self.run_groups, f, indent=2, cls=NumpyEncoder)

        logger.info(f"Saved extracted data to {self.output_dir}")
        return {
            'metadata_path': str(metadata_path),
            'metrics_path': str(metrics_path),
            'params_path': str(params_path),
            'groups_path': str(groups_path)
        }


def main():
    parser = argparse.ArgumentParser(description="MLflow Data Extractor")
    parser.add_argument("--experiment-dir", required=True,
                      help="Directory containing MLflow experiment data")
    parser.add_argument("--output-dir", required=True,
                      help="Directory to save extracted data")
    args = parser.parse_args()

    try:
        extractor = MLflowDataExtractor(
            experiment_dir=args.experiment_dir,
            output_dir=args.output_dir
        )

        extractor.extract_data()
        output_files = extractor.save_data()

        # Print output file locations
        logger.info("Extraction complete. Output files:")
        for file_desc, file_path in output_files.items():
            logger.info(f"  - {file_desc}: {file_path}")

    except Exception as e:
        logger.error(f"Error during data extraction: {e}")
        logger.error(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## mlflow_statistical_analyzer.py
```python
#!/usr/bin/env python3
"""
MLflow Statistical Significance Tester - Streamlined Version

This script:
1. Loads metrics from multiple MLflow experiment directories
2. Performs statistical tests to compare experiments
3. Identifies significant differences between runs/experiments
4. Generates statistical reports

Usage:
    python mlflow_statistical_analyzer.py --experiment-dirs exp1,exp2,exp3 --output-dir results
"""

import os
import sys
import argparse
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
from scipy import stats
import itertools
from statsmodels.stats.multitest import multipletests
import json
import yaml
from tqdm import tqdm
import warnings
from datetime import datetime

# Suppress specific warnings
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")
warnings.filterwarnings("ignore", category=FutureWarning)

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):  # Correct handling for NumPy bool type
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Set up basic logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MLflowStatisticalAnalyzer:
    def __init__(self, experiment_dirs=None, output_dir=None, alpha=0.05,
                 exclude_metrics=None, correction_method='fdr_bh'):
        """
        Initialize the MLflow statistical analyzer.

        Args:
            experiment_dirs: List of directories containing MLflow experiment data
            output_dir: Directory to save analysis outputs
            alpha: Significance level for statistical tests (default: 0.05)
            exclude_metrics: List of metrics to exclude from analysis
            correction_method: Method for p-value correction for multiple testing
                              (options: 'bonferroni', 'fdr_bh', 'holm', etc.)
        """
        logger.info("Initializing MLflowStatisticalAnalyzer")

        self.experiment_dirs = [Path(d) for d in experiment_dirs or []]
        logger.info(f"Experiment directories: {self.experiment_dirs}")

        self.output_dir = Path(output_dir or "mlflow_statistical_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")

        self.alpha = alpha
        self.correction_method = correction_method
        logger.info(f"Statistical parameters: alpha={alpha}, correction={correction_method}")

        self.exclude_metrics = exclude_metrics or []
        if exclude_metrics:
            logger.info(f"Excluding metrics: {exclude_metrics}")

        # Initialize data structures
        self.experiments = {}  # Experiment data indexed by experiment ID
        self.runs_data = {}  # Run data indexed by run ID
        self.metrics_data = defaultdict(lambda: defaultdict(list))  # metric_name -> run_id -> list of (timestamp, value, step)

        # Statistical test results
        self.test_results = defaultdict(list)  # test_type -> list of test results

    def scan_experiments(self):
        """Scan all specified experiment directories and collect metadata."""
        logger.info(f"Scanning {len(self.experiment_dirs)} experiment directories...")

        for exp_dir in self.experiment_dirs:
            exp_name = exp_dir.name
            # Get the parent directory (will be EXPERIMENT_00_BASELINE, etc.)
            parent_dir = exp_dir.parent.parent.name
            logger.info(f"Processing experiment: {exp_name} from {parent_dir}")

            if not exp_dir.exists():
                logger.warning(f"Experiment directory not found: {exp_dir}")
                continue

            # Extract experiment metadata if available
            meta_file = exp_dir / 'meta.yaml'
            # Use the parent directory name to make the experiment ID unique
            experiment_id = f"{parent_dir}/{exp_name}"
            experiment_name = f"{parent_dir}/{exp_name}"

            if meta_file.exists():
                try:
                    with open(meta_file, 'r') as f:
                        content = f.read()
                        name_match = re.search(r'name: (.+)', content)
                        if name_match:
                            # Keep the parent directory in the name for clarity
                            experiment_name = f"{parent_dir}/{name_match.group(1).strip()}"
                            logger.info(f"Found experiment name: {experiment_name}")
                except Exception as e:
                    logger.error(f"Error reading experiment metadata: {e}")

            # Store experiment data
            self.experiments[experiment_id] = {
                'id': experiment_id,
                'name': experiment_name,
                'path': str(exp_dir),
                'runs': []
            }

            # Find all run directories
            runs_found = 0
            for run_dir in exp_dir.iterdir():
                if not run_dir.is_dir() or run_dir.name.startswith('.') or run_dir.name in ['artifacts', 'metrics', 'params', 'tags']:
                    continue

                run_id = run_dir.name
                meta_file = run_dir / 'meta.yaml'

                if not meta_file.exists():
                    logger.warning(f"Skipping run {run_id}: no meta.yaml found")
                    continue

                # Extract run metadata
                try:
                    run_data = {
                        'run_id': run_id,
                        'experiment_id': experiment_id,
                        'tags': {},
                        'params': {}
                    }

                    # Extract tags
                    tags_dir = run_dir / 'tags'
                    if tags_dir.exists():
                        for tag_file in tags_dir.iterdir():
                            if tag_file.is_file():
                                try:
                                    with open(tag_file, 'r') as tf:
                                        tag_value = tf.read().strip()
                                        run_data['tags'][tag_file.name] = tag_value
                                except Exception as e:
                                    logger.warning(f"Error reading tag {tag_file.name}: {e}")
                                    pass

                    # Extract parameters
                    params_dir = run_dir / 'params'
                    if params_dir.exists():
                        for param_file in params_dir.iterdir():
                            if param_file.is_file():
                                try:
                                    with open(param_file, 'r') as pf:
                                        param_value = pf.read().strip()
                                        run_data['params'][param_file.name] = param_value
                                except Exception as e:
                                    logger.warning(f"Error reading parameter {param_file.name}: {e}")
                                    pass

                    # Get run name or generate one
                    run_name = run_data['tags'].get('mlflow.runName', run_id[:8])
                    run_data['run_name'] = run_name

                    # Store run data
                    self.runs_data[run_id] = run_data
                    self.experiments[experiment_id]['runs'].append(run_id)
                    runs_found += 1

                except Exception as e:
                    logger.error(f"Error reading metadata for run {run_id}: {e}")
                    logger.error(traceback.format_exc())

            logger.info(f"Found {runs_found} runs in experiment {experiment_name}")

        # Print summary
        total_runs = sum(len(exp['runs']) for exp in self.experiments.values())
        logger.info(f"Found {len(self.experiments)} experiments with a total of {total_runs} runs")

        for exp_id, exp_data in self.experiments.items():
            logger.info(f"  {exp_data['name']}: {len(exp_data['runs'])} runs")

        return self.experiments

    def collect_metrics_data(self):
        """Collect metrics data from all runs across all experiments."""
        logger.info("Collecting metrics data...")

        for exp_id, exp_data in self.experiments.items():
            exp_dir = Path(exp_data['path'])
            logger.info(f"Processing metrics for experiment: {exp_data['name']}")

            # Process each run
            for run_id in tqdm(exp_data['runs'], desc=f"Processing runs in {exp_data['name']}"):
                run_metrics_dir = exp_dir / run_id / 'metrics'
                if not run_metrics_dir.exists():
                    logger.warning(f"No metrics directory found for run {run_id}")
                    continue

                metrics_found = 0
                # Process each metric file
                for metric_file in run_metrics_dir.iterdir():
                    if not metric_file.is_file():
                        continue

                    metric_name = metric_file.name

                    # Skip excluded metrics
                    if metric_name in self.exclude_metrics:
                        continue

                    try:
                        # Read the metric data (timestamp, value, step)
                        metric_data = []
                        with open(metric_file, 'r') as f:
                            for line in f:
                                try:
                                    # Format is: timestamp value step
                                    parts = line.strip().split()
                                    if len(parts) >= 3:
                                        timestamp = float(parts[0])
                                        value = float(parts[1])
                                        step = int(parts[2])
                                        metric_data.append((timestamp, value, step))
                                except Exception as e:
                                    logger.warning(f"Error parsing metric line in {metric_file}: {e}")
                                    continue

                        # Store metric data if any valid entries found
                        if metric_data:
                            self.metrics_data[metric_name][run_id] = metric_data
                            metrics_found += 1

                    except Exception as e:
                        logger.error(f"Error reading metric {metric_name} for run {run_id}: {e}")
                        logger.error(traceback.format_exc())

                logger.debug(f"Found {metrics_found} metrics for run {run_id}")

        # Get metrics statistics
        total_metrics = len(self.metrics_data)
        total_data_points = sum(len(data) for metric_data in self.metrics_data.values()
                               for data in metric_data.values())

        logger.info(f"Collected {total_data_points} data points for {total_metrics} metrics")

        # List the most common metrics
        metrics_counts = {m: sum(1 for _ in runs.keys()) for m, runs in self.metrics_data.items()}
        top_metrics = sorted(metrics_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        logger.info("Most common metrics:")
        for metric, count in top_metrics:
            logger.info(f"  {metric}: found in {count} runs")

        return self.metrics_data

    def prepare_metric_dataframe(self, metric_name):
        """
        Prepare a DataFrame for a specific metric with summary statistics.
        """
        if metric_name not in self.metrics_data:
            logger.warning(f"Metric {metric_name} not found in collected data")
            return pd.DataFrame()

        logger.debug(f"Preparing DataFrame for metric: {metric_name}")
        # Build dataframe rows
        rows = []

        for run_id, metric_points in self.metrics_data[metric_name].items():
            if run_id not in self.runs_data:
                logger.warning(f"Run {run_id} found in metrics but not in run data")
                continue

            run_data = self.runs_data[run_id]
            exp_id = run_data['experiment_id']

            if not metric_points:
                logger.warning(f"No metric points found for {metric_name} in run {run_id}")
                continue

            # Extract values
            values = [value for _, value, _ in metric_points]

            # Check for invalid values before calculations
            has_nan = any(np.isnan(v) for v in values)
            has_inf = any(np.isinf(v) for v in values)

            if has_nan or has_inf:
                logger.warning(f"Found {'NaN' if has_nan else ''} {'and' if has_nan and has_inf else ''} {'Inf' if has_inf else ''} values in {metric_name} for run {run_id}")
                # Filter out bad values for calculations
                valid_values = [v for v in values if np.isfinite(v)]
                if not valid_values:
                    logger.warning(f"No valid values for {metric_name} in run {run_id} after filtering - skipping")
                    continue
                values = valid_values

            # Calculate summary statistics
            try:
                # Safely calculate improvement percentage
                first_val = values[0]
                last_val = values[-1]
                improvement = last_val - first_val

                if first_val != 0:
                    pct_improvement = (improvement / abs(first_val)) * 100
                else:
                    # Avoid division by zero
                    pct_improvement = np.nan if improvement != 0 else 0
                    logger.debug(f"First value is zero for {metric_name} in run {run_id}, setting pct_improvement to {pct_improvement}")

                row = {
                    'experiment_id': exp_id,
                    'experiment_name': self.experiments[exp_id]['name'],
                    'run_id': run_id,
                    'run_name': run_data['run_name'],
                    'n_points': len(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'first': first_val,
                    'last': last_val,
                    'improvement': improvement,
                    'pct_improvement': pct_improvement
                }

                # Extract relevant parameters
                for param_key, param_value in run_data.get('params', {}).items():
                    # Add selected parameters that might be useful for analysis
                    if any(key_part in param_key for key_part in ['learning_rate', 'batch_size', 'epochs', 'loss_weight']):
                        try:
                            # Try to convert to float if numeric
                            row[f'param_{param_key}'] = float(param_value)
                        except ValueError:
                            row[f'param_{param_key}'] = param_value

                rows.append(row)
            except Exception as e:
                logger.error(f"Error calculating statistics for {metric_name} in run {run_id}: {e}")
                logger.error(traceback.format_exc())

        df = pd.DataFrame(rows)

        # Final validation of the dataframe
        if not df.empty:
            # Log warnings about any remaining NaN values
            na_counts = df.isna().sum()
            if na_counts.any():
                problematic_cols = na_counts[na_counts > 0]
                logger.warning(f"DataFrame for {metric_name} contains NaN values in columns: {problematic_cols.to_dict()}")

        logger.debug(f"Created DataFrame with {len(df)} rows for metric {metric_name}")
        return df

    def clean_metrics_data(self):
        """Clean metrics data by removing or replacing invalid values."""
        logger.info("Cleaning metrics data...")
        cleaned_count = 0
        metrics_removed = 0

        for metric_name, runs in list(self.metrics_data.items()):
            # Track runs with valid data for this metric
            valid_runs = 0

            for run_id, data_points in list(runs.items()):
                # Filter out NaN and Inf values
                valid_points = []
                for point in data_points:
                    timestamp, value, step = point
                    if np.isfinite(value):  # Checks for NaN and Inf
                        valid_points.append(point)
                    else:
                        cleaned_count += 1

                # Update with only valid points, or remove run if no valid points
                if valid_points:
                    self.metrics_data[metric_name][run_id] = valid_points
                    valid_runs += 1
                else:
                    del self.metrics_data[metric_name][run_id]

            # Remove metrics with insufficient data
            if valid_runs < 2:
                logger.warning(f"Metric '{metric_name}' has insufficient valid data points after cleaning")
                del self.metrics_data[metric_name]
                metrics_removed += 1

        logger.info(f"Cleaned {cleaned_count} invalid data points")
        logger.info(f"Removed {metrics_removed} metrics with insufficient data")
        logger.info(f"Remaining metrics after cleaning: {len(self.metrics_data)}")

        return self.metrics_data

    def run_statistical_tests(self):
        """
        Perform statistical tests on metrics data.

        This includes:
        1. Comparing metrics across experiments (ANOVA/Kruskal-Wallis)
        2. Pairwise comparisons with appropriate post-hoc tests
        3. Correlation analysis between metrics and parameters
        4. Multiple testing correction
        """
        logger.info("Running statistical tests...")

        # Process each metric
        for metric_name in tqdm(self.metrics_data.keys(), desc="Testing metrics"):
            logger.debug(f"Testing metric: {metric_name}")
            # Skip if not enough data
            if len(self.metrics_data[metric_name]) < 2:
                logger.debug(f"Skipping {metric_name}: not enough runs with this metric")
                continue

            # Prepare dataframe with summary statistics
            df = self.prepare_metric_dataframe(metric_name)

            if df.empty or len(df) < 2:
                logger.debug(f"Skipping {metric_name}: empty DataFrame or too few rows")
                continue

            # 1. Experiment comparison tests (are there differences across experiments?)
            self._test_experiment_differences(metric_name, df)

            # 2. Pairwise experiment comparison tests
            self._test_pairwise_experiments(metric_name, df)

            # 3. Parameter correlation tests
            self._test_parameter_correlations(metric_name, df)

        # 4. Apply multiple testing correction across all tests
        self._apply_multiple_testing_correction()

        # Log test result counts
        for test_type, results in self.test_results.items():
            logger.info(f"Completed {len(results)} {test_type}")

        return self.test_results

    def _test_experiment_differences(self, metric_name, df):
        """Test for differences in a metric across experiments."""
        # Skip if there's only one experiment
        if df['experiment_id'].nunique() < 2:
            logger.debug(f"Skipping experiment difference test for {metric_name}: only one experiment")
            return

        # Group by experiment and convert to list of numpy arrays for statistical tests
        groups = []
        for exp_id, group_df in df.groupby('experiment_id'):
            if len(group_df) >= 2:  # Ensure at least 2 samples per group
                groups.append(group_df['mean'].values)

        # Skip if not enough groups have sufficient samples
        if len(groups) < 2:
            logger.debug(f"Skipping experiment difference test for {metric_name}: insufficient groups")
            return

        # Try ANOVA first (parametric test)
        try:
            f_stat, p_value = stats.f_oneway(*groups)

            self.test_results['anova_tests'].append({
                'metric_name': metric_name,
                'test_type': 'ANOVA',
                'statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'experiments': list(df['experiment_id'].unique()),
                'sample_sizes': [len(group) for group in groups]
            })

            logger.debug(f"ANOVA for {metric_name}: F={f_stat:.4f}, p={p_value:.4e}")

        except Exception as e:
            logger.error(f"Error in ANOVA test for {metric_name}: {e}")
            logger.error(traceback.format_exc())

        # Try Kruskal-Wallis test (non-parametric)
        try:
            # Check if all groups are identical
            all_identical = True
            first_value = groups[0][0] if len(groups) > 0 and len(groups[0]) > 0 else None

            for group in groups:
                for value in group:
                    if value != first_value:
                        all_identical = False
                        break
                if not all_identical:
                    break

            if all_identical:
                logger.debug(f"Skipping Kruskal-Wallis test for {metric_name}: all values are identical")
                self.test_results['kruskal_tests'].append({
                    'metric_name': metric_name,
                    'test_type': 'Kruskal-Wallis',
                    'statistic': 0.0,
                    'p_value': 1.0,  # No difference = high p-value
                    'significant': False,
                    'experiments': list(df['experiment_id'].unique()),
                    'sample_sizes': [len(group) for group in groups],
                    'note': 'All values identical'
                })
            else:
                h_stat, p_value = stats.kruskal(*groups)
                self.test_results['kruskal_tests'].append({
                    'metric_name': metric_name,
                    'test_type': 'Kruskal-Wallis',
                    'statistic': h_stat,
                    'p_value': p_value,
                    'significant': p_value < self.alpha,
                    'experiments': list(df['experiment_id'].unique()),
                    'sample_sizes': [len(group) for group in groups]
                })
                logger.debug(f"Kruskal-Wallis for {metric_name}: H={h_stat:.4f}, p={p_value:.4e}")
        except Exception as e:
            logger.error(f"Error in Kruskal-Wallis test for {metric_name}: {e}")
            logger.error(traceback.format_exc())

    def _test_pairwise_experiments(self, metric_name, df):
        """Perform pairwise comparison tests between experiments."""
        # Get unique experiment IDs
        exp_ids = df['experiment_id'].unique()

        # Skip if there's only one experiment
        if len(exp_ids) < 2:
            logger.debug(f"Skipping pairwise tests for {metric_name}: only one experiment")
            return

        # Perform pairwise t-tests and Mann-Whitney U tests
        for exp1, exp2 in itertools.combinations(exp_ids, 2):
            # Get data for each experiment
            values1 = df[df['experiment_id'] == exp1]['mean'].values
            values2 = df[df['experiment_id'] == exp2]['mean'].values

            # Skip if either group has too few samples
            if len(values1) < 2 or len(values2) < 2:
                logger.debug(f"Skipping pairwise test between {exp1} and {exp2}: insufficient samples")
                continue

            # Calculate effect size (Cohen's d)
            mean1, mean2 = np.mean(values1), np.mean(values2)
            std1, std2 = np.std(values1), np.std(values2)
            n1, n2 = len(values1), len(values2)

            # Pooled standard deviation
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

            # Cohen's d
            cohens_d = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else np.nan

            # Perform t-test (parametric)
            try:
                t_stat, p_value = stats.ttest_ind(values1, values2, equal_var=False)

                self.test_results['t_tests'].append({
                    'metric_name': metric_name,
                    'test_type': 'Welch t-test',
                    'exp1': exp1,
                    'exp2': exp2,
                    'exp1_name': self.experiments[exp1]['name'],
                    'exp2_name': self.experiments[exp2]['name'],
                    'statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'effect_size_magnitude': self._interpret_cohens_d(cohens_d),
                    'significant': p_value < self.alpha,
                    'sample_sizes': [len(values1), len(values2)]
                })

                logger.debug(f"t-test for {metric_name} between {exp1} and {exp2}: t={t_stat:.4f}, p={p_value:.4e}, d={cohens_d:.2f}")

            except Exception as e:
                logger.error(f"Error in t-test for {metric_name} between {exp1} and {exp2}: {e}")
                logger.error(traceback.format_exc())

            # Perform Mann-Whitney U test (non-parametric)
            try:
                u_stat, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')

                self.test_results['mann_whitney_tests'].append({
                    'metric_name': metric_name,
                    'test_type': 'Mann-Whitney U',
                    'exp1': exp1,
                    'exp2': exp2,
                    'exp1_name': self.experiments[exp1]['name'],
                    'exp2_name': self.experiments[exp2]['name'],
                    'statistic': u_stat,
                    'p_value': p_value,
                    'significant': p_value < self.alpha,
                    'sample_sizes': [len(values1), len(values2)]
                })

                logger.debug(f"Mann-Whitney U for {metric_name} between {exp1} and {exp2}: U={u_stat:.4f}, p={p_value:.4e}")

            except Exception as e:
                logger.error(f"Error in Mann-Whitney U test for {metric_name} between {exp1} and {exp2}: {e}")
                logger.error(traceback.format_exc())

    def _test_parameter_correlations(self, metric_name, df):
        """Test for correlations between parameters and metric values."""
        # Find parameter columns
        param_cols = [col for col in df.columns if col.startswith('param_')]

        if not param_cols:
            logger.debug(f"No parameter columns found for {metric_name}")
            return

        # Test correlation for each parameter
        for param_col in param_cols:
            # Convert to numeric if possible, otherwise skip
            if df[param_col].dtype == object:
                try:
                    df[param_col] = pd.to_numeric(df[param_col])
                except (ValueError, TypeError):
                    logger.debug(f"Skipping non-numeric parameter {param_col}")
                    continue

            # Skip if not numeric
            if not pd.api.types.is_numeric_dtype(df[param_col]):
                logger.debug(f"Skipping non-numeric parameter {param_col}")
                continue

            # Skip parameters with no variation
            if df[param_col].nunique() <= 1:
                logger.debug(f"Skipping parameter {param_col} with no variation")
                continue

            # Calculate Pearson correlation
            try:
                pearson_r, p_value = stats.pearsonr(df[param_col], df['mean'])

                # Only record if reasonably strong correlation
                if abs(pearson_r) >= 0.3:  # Threshold for "moderate" correlation
                    self.test_results['parameter_correlations'].append({
                        'metric_name': metric_name,
                        'parameter': param_col.replace('param_', ''),
                        'test_type': 'Pearson Correlation',
                        'statistic': pearson_r,
                        'p_value': p_value,
                        'significant': p_value < self.alpha,
                        'correlation_strength': self._interpret_correlation(pearson_r),
                        'sample_size': len(df)
                    })

                    logger.debug(f"Pearson correlation for {metric_name} and {param_col}: r={pearson_r:.4f}, p={p_value:.4e}")

            except Exception as e:
                logger.error(f"Error in correlation test for {metric_name} and {param_col}: {e}")
                logger.error(traceback.format_exc())

            # Calculate Spearman rank correlation (non-parametric)
            try:
                spearman_r, p_value = stats.spearmanr(df[param_col], df['mean'])

                # Only record if reasonably strong correlation
                if abs(spearman_r) >= 0.3:  # Threshold for "moderate" correlation
                    self.test_results['parameter_correlations'].append({
                        'metric_name': metric_name,
                        'parameter': param_col.replace('param_', ''),
                        'test_type': 'Spearman Rank Correlation',
                        'statistic': spearman_r,
                        'p_value': p_value,
                        'significant': p_value < self.alpha,
                        'correlation_strength': self._interpret_correlation(spearman_r),
                        'sample_size': len(df)
                    })

                    logger.debug(f"Spearman correlation for {metric_name} and {param_col}: r={spearman_r:.4f}, p={p_value:.4e}")

            except Exception as e:
                logger.error(f"Error in Spearman correlation test for {metric_name} and {param_col}: {e}")
                logger.error(traceback.format_exc())

    def _interpret_cohens_d(self, d):
        """Interpret Cohen's d effect size."""
        if pd.isna(d):
            return "Unknown"
        if d < 0.2:
            return "Negligible"
        elif d < 0.5:
            return "Small"
        elif d < 0.8:
            return "Medium"
        else:
            return "Large"

    def _interpret_correlation(self, r):
        """Interpret correlation coefficient."""
        r_abs = abs(r)
        if r_abs < 0.3:
            return "Weak"
        elif r_abs < 0.5:
            return "Moderate"
        elif r_abs < 0.7:
            return "Strong"
        else:
            return "Very Strong"

    def _apply_multiple_testing_correction(self):
        """Apply correction for multiple testing to p-values."""
        logger.info("Applying multiple testing correction...")

        # Process each test type separately
        for test_type, results in self.test_results.items():
            if not results:
                logger.debug(f"No {test_type} to correct")
                continue

            # Extract p-values
            p_values = [result['p_value'] for result in results]

            try:
                # Apply correction
                reject, p_corrected, _, _ = multipletests(p_values, alpha=self.alpha, method=self.correction_method)

                # Update results with corrected values
                for i, result in enumerate(results):
                    result['p_value_corrected'] = p_corrected[i]
                    result['significant_corrected'] = reject[i]

                logger.info(f"Applied {self.correction_method} correction to {len(p_values)} {test_type}")

                # Count significant results after correction
                sig_count = sum(1 for r in results if r['significant_corrected'])
                logger.info(f"  {sig_count} significant results after correction ({sig_count/len(results)*100:.1f}%)")

            except Exception as e:
                logger.error(f"Error applying multiple testing correction to {test_type}: {e}")
                logger.error(traceback.format_exc())

    def save_results_to_files(self):
        """Save analysis results to files in the output directory."""
        logger.info(f"Saving analysis results to {self.output_dir}")

        # Make sure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save test results as JSON files
        results_dir = self.output_dir / "test_results"
        results_dir.mkdir(exist_ok=True)

        try:
            # Save each test type to a separate file
            for test_type, results in self.test_results.items():
                if not results:
                    continue

                filename = f"{test_type}.json"
                filepath = results_dir / filename

                with open(filepath, 'w') as f:
                    json.dump(results, f, indent=2, cls=NumpyEncoder)

                logger.info(f"Saved {len(results)} {test_type} results to {filepath}")

            # Save experiment and run metadata
            metadata = {
                "analysis_time": datetime.now().isoformat(),
                "alpha": self.alpha,
                "correction_method": self.correction_method,
                "experiment_count": len(self.experiments),
                "run_count": len(self.runs_data),
                "metrics_count": len(self.metrics_data),
                "experiments": self.experiments,
            }

            with open(self.output_dir / "analysis_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2, cls=NumpyEncoder)

            # Save metrics summary
            metrics_summary = {}
            for metric_name, runs_data in self.metrics_data.items():
                metrics_summary[metric_name] = {
                    "run_count": len(runs_data),
                    "total_points": sum(len(points) for points in runs_data.values())
                }

            with open(self.output_dir / "metrics_summary.json", 'w') as f:
                json.dump(metrics_summary, f, indent=2, cls=NumpyEncoder)

            # Generate a summary report of significant findings
            self._generate_summary_report()

        except Exception as e:
            logger.error(f"Error saving results to files: {e}")
            logger.error(traceback.format_exc())

            logger.info(f"Analysis results saved to {self.output_dir}")
        return self.output_dir

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MLflow Statistical Significance Tester")
    parser.add_argument("--experiment-dirs", required=True,
                        help="Comma-separated list of MLflow experiment directories")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save analysis outputs")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level for statistical tests (default: 0.05)")
    parser.add_argument("--correction-method", default="fdr_bh",
                        help="Method for multiple testing correction (default: fdr_bh)")
    args = parser.parse_args()

    # Split the comma-separated experiment directories
    experiment_dirs = args.experiment_dirs.split(',')

    # Create analyzer instance
    analyzer = MLflowStatisticalAnalyzer(
        experiment_dirs=experiment_dirs,
        output_dir=args.output_dir,
        alpha=args.alpha,
        correction_method=args.correction_method
    )

    # Run the analysis pipeline
    logger.info("Starting analysis pipeline")
    analyzer.scan_experiments()
    analyzer.collect_metrics_data()
    analyzer.clean_metrics_data()
    analyzer.run_statistical_tests()
    analyzer.save_results_to_files()
    logger.info("Analysis complete")

if __name__ == "__main__":
    main()

    def _generate_summary_report(self):
        """Generate a summary report of significant findings."""
        report_path = self.output_dir / "significant_findings_summary.txt"

        try:
            with open(report_path, 'w') as f:
                f.write("# MLflow Statistical Analysis: Significant Findings Summary\n\n")
                f.write(f"Analysis performed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Significance level (alpha): {self.alpha}\n")
                f.write(f"Multiple testing correction: {self.correction_method}\n\n")

                # Summary counts
                f.write("## Overview\n\n")
                f.write(f"- Analyzed {len(self.experiments)} experiments with a total of {len(self.runs_data)} runs\n")
                f.write(f"- Examined {len(self.metrics_data)} metrics\n\n")

                # Significant metrics from ANOVA/Kruskal-Wallis tests
                sig_metrics = set()
                for test_type in ['anova_tests', 'kruskal_tests']:
                    for result in self.test_results.get(test_type, []):
                        if result.get('significant_corrected', False):
                            sig_metrics.add(result['metric_name'])

                if sig_metrics:
                    f.write("## Metrics with Significant Differences Between Experiments\n\n")
                    for metric in sorted(sig_metrics):
                        f.write(f"- {metric}\n")
                    f.write("\n")

                # Significant pairwise comparisons
                f.write("## Significant Pairwise Differences\n\n")
                sig_pairs = []
                for result in self.test_results.get('t_tests', []):
                    if result.get('significant_corrected', False):
                        sig_pairs.append((
                            result['metric_name'],
                            result['exp1_name'],
                            result['exp2_name'],
                            result['p_value_corrected'],
                            result.get('cohens_d', 'N/A'),
                            result.get('effect_size_magnitude', 'N/A')
                        ))

                if sig_pairs:
                    for metric, exp1, exp2, p_val, cohen_d, effect in sorted(sig_pairs, key=lambda x: (x[0], x[3])):
                        f.write(f"- {metric}: {exp1} vs {exp2}\n")
                        f.write(f"  p={p_val:.6f}, Cohen's d={cohen_d}, effect size: {effect}\n")
                else:
                    f.write("No significant pairwise differences found after correction.\n")
                f.write("\n")

                # Significant parameter correlations
                f.write("## Significant Parameter Correlations\n\n")
                sig_correlations = []
                for result in self.test_results.get('parameter_correlations', []):
                    if result.get('significant_corrected', False):
                        sig_correlations.append((
                            result['metric_name'],
                            result['parameter'],
                            result['statistic'],
                            result['p_value_corrected'],
                            result['correlation_strength']
                        ))

                if sig_correlations:
                    for metric, param, stat, p_val, strength in sorted(sig_correlations, key=lambda x: (x[0], x[3])):
                        f.write(f"- {metric} correlates with {param}\n")
                        f.write(f"  r={stat:.4f}, p={p_val:.6f}, strength: {strength}\n")
                else:
                    f.write("No significant parameter correlations found after correction.\n")

            logger.info(f"Generated summary report at {report_path}")

        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            logger.error(traceback.format_exc())
```


## experiment_data.py
```python
#!/usr/bin/env python3
"""
MLflow Experiment Data Class

This module defines a class for loading and manipulating extracted MLflow experiment data.
It provides a clean interface for working with experiment data that can be used by
statistical analysis scripts.

Usage:
    from experiment_data import ExperimentData

    # Load a single experiment
    exp = ExperimentData.from_directory('/path/to/extracted/data')

    # Load multiple experiments
    exps = ExperimentData.load_multiple(['/path/to/exp1', '/path/to/exp2'])

    # Access metrics and parameters
    metric_values = exp.get_metric_values('loss')
    param_values = exp.get_parameter_values('learning_rate')
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentData:
    """Class for loading and working with extracted MLflow experiment data."""

    def __init__(self,
                 metadata: Dict[str, Any],
                 metrics_data: Dict[str, Dict[str, Any]],
                 parameters_data: Dict[str, Dict[str, str]],
                 run_groups: Dict[str, List[str]],
                 experiment_dir: str = None):
        """
        Initialize with experiment data.

        Args:
            metadata: Experiment metadata including run information
            metrics_data: Metrics data organized by metric name and run ID
            parameters_data: Parameter data organized by parameter name and run ID
            run_groups: Groups of runs organized by category
            experiment_dir: Directory containing the experiment data (optional)
        """
        self.metadata = metadata
        self.metrics_data = metrics_data
        self.parameters_data = parameters_data
        self.run_groups = run_groups
        self.experiment_dir = experiment_dir
        self.name = metadata.get('name', 'Unknown Experiment')
        self.id = metadata.get('id', 'Unknown ID')

        # Cache for calculated dataframes
        self._metrics_df_cache = {}
        self._summary_df_cache = {}

    @classmethod
    def from_directory(cls, directory_path: Union[str, Path]) -> 'ExperimentData':
        """
        Load experiment data from a directory containing the extracted JSON files.

        Args:
            directory_path: Path to the directory containing the extracted data

        Returns:
            ExperimentData instance with loaded data
        """
        directory = Path(directory_path)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        try:
            # Load metadata
            metadata_path = directory / 'experiment_metadata.json'
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Load metrics data
            metrics_path = directory / 'metrics_data.json'
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)

            # Load parameters data
            params_path = directory / 'parameters_data.json'
            with open(params_path, 'r') as f:
                parameters_data = json.load(f)

            # Load run groups
            groups_path = directory / 'run_groups.json'
            if groups_path.exists():
                with open(groups_path, 'r') as f:
                    run_groups = json.load(f)
            else:
                run_groups = {}

            return cls(
                metadata=metadata,
                metrics_data=metrics_data,
                parameters_data=parameters_data,
                run_groups=run_groups,
                experiment_dir=str(directory)
            )

        except Exception as e:
            logger.error(f"Error loading experiment data from {directory}: {e}")
            raise

    @classmethod
    def load_multiple(cls, directories: List[Union[str, Path]]) -> List['ExperimentData']:
        """
        Load multiple experiment data objects from directories.

        Args:
            directories: List of paths to directories containing extracted data

        Returns:
            List of ExperimentData instances
        """
        experiments = []

        for directory in directories:
            try:
                exp = cls.from_directory(directory)
                experiments.append(exp)
                logger.info(f"Loaded experiment: {exp.name}")
            except Exception as e:
                logger.error(f"Error loading experiment from {directory}: {e}")

        return experiments

    def get_run_ids(self, group: str = None) -> List[str]:
        """
        Get run IDs for this experiment, optionally filtered by group.

        Args:
            group: Optional group name to filter runs

        Returns:
            List of run IDs
        """
        if group and group in self.run_groups:
            return self.run_groups[group]

        return [run['run_id'] for run in self.metadata['runs']]

    def get_run_data(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get data for a specific run.

        Args:
            run_id: ID of the run to retrieve

        Returns:
            Dictionary of run data or None if not found
        """
        for run in self.metadata['runs']:
            if run['run_id'] == run_id:
                return run
        return None

    def get_metric_values(self, metric_name: str,
                          run_ids: List[str] = None,
                          summary_stat: str = None) -> Dict[str, Union[List[float], float]]:
        """
        Get values for a specific metric across runs.

        Args:
            metric_name: Name of the metric to retrieve
            run_ids: Optional list of run IDs to filter by
            summary_stat: Optional summary statistic to retrieve (e.g., 'mean', 'last')

        Returns:
            Dictionary with run IDs as keys and metric values or summary stats as values
        """
        if metric_name not in self.metrics_data:
            logger.warning(f"Metric '{metric_name}' not found in experiment {self.name}")
            return {}

        if run_ids is None:
            run_ids = self.get_run_ids()

        result = {}

        for run_id in run_ids:
            if run_id in self.metrics_data[metric_name]:
                run_data = self.metrics_data[metric_name][run_id]

                if summary_stat and 'summary' in run_data and summary_stat in run_data['summary']:
                    result[run_id] = run_data['summary'][summary_stat]
                elif summary_stat and summary_stat == 'all_summary':
                    result[run_id] = run_data.get('summary', {})
                else:
                    result[run_id] = run_data['values']

        return result

    def get_parameter_values(self, param_name: str, run_ids: List[str] = None) -> Dict[str, str]:
        """
        Get values for a specific parameter across runs.

        Args:
            param_name: Name of the parameter to retrieve
            run_ids: Optional list of run IDs to filter by

        Returns:
            Dictionary with run IDs as keys and parameter values as values
        """
        if param_name not in self.parameters_data:
            logger.warning(f"Parameter '{param_name}' not found in experiment {self.name}")
            return {}

        if run_ids is None:
            run_ids = self.get_run_ids()

        result = {}

        for run_id in run_ids:
            if run_id in self.parameters_data[param_name]:
                result[run_id] = self.parameters_data[param_name][run_id]

        return result

    def get_metric_summary_df(self, metric_names: List[str] = None) -> pd.DataFrame:
        """
        Get a DataFrame with summary statistics for metrics across runs.

        Args:
            metric_names: Optional list of metric names to include

        Returns:
            DataFrame with runs as rows and metric summaries as columns
        """
        # Use cached dataframe if available and no specific metrics requested
        if metric_names is None and 'all_metrics' in self._summary_df_cache:
            return self._summary_df_cache['all_metrics']

        cache_key = '_'.join(metric_names) if metric_names else 'all_metrics'
        if cache_key in self._summary_df_cache:
            return self._summary_df_cache[cache_key]

        # Get all metric names if not specified
        if metric_names is None:
            metric_names = list(self.metrics_data.keys())

        # Build rows for DataFrame
        rows = []

        for run in self.metadata['runs']:
            run_id = run['run_id']
            row = {
                'run_id': run_id,
                'experiment_name': self.name,
            }

            # Add parameters
            for param_name, param_values in self.parameters_data.items():
                if run_id in param_values:
                    # Try to convert parameters to numeric if possible
                    try:
                        row[f"param_{param_name}"] = float(param_values[run_id])
                    except (ValueError, TypeError):
                        row[f"param_{param_name}"] = param_values[run_id]

            # Add metric summaries
            for metric_name in metric_names:
                if metric_name in self.metrics_data and run_id in self.metrics_data[metric_name]:
                    metric_data = self.metrics_data[metric_name][run_id]
                    if 'summary' in metric_data:
                        for stat_name, stat_value in metric_data['summary'].items():
                            row[f"{metric_name}_{stat_name}"] = stat_value

            rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Cache the result
        self._summary_df_cache[cache_key] = df

        return df

    def get_metric_values_df(self,
                        metric_name: str,
                        run_ids: List[str] = None,
                        align_steps: bool = False) -> pd.DataFrame:
        """
        Get a DataFrame with all values for a specific metric across runs.

        Args:
            metric_name: Name of the metric to retrieve
            run_ids: Optional list of run IDs to filter by
            align_steps: Whether to align values by step number (useful for comparing epochs)

        Returns:
            DataFrame with steps as rows and runs as columns, or a long-format DataFrame
            if align_steps is False
        """
        if metric_name not in self.metrics_data:
            logger.warning(f"Metric '{metric_name}' not found in experiment {self.name}")
            return pd.DataFrame()

        # Use cached dataframe if available
        cache_key = f"{metric_name}_{align_steps}"
        if cache_key in self._metrics_df_cache:
            df = self._metrics_df_cache[cache_key]
            if run_ids is not None:
                # Filter by run_ids if provided
                if align_steps:
                    cols_to_keep = ['step'] + [c for c in df.columns if c in run_ids]
                    return df[cols_to_keep]
                else:
                    return df[df['run_id'].isin(run_ids)]
            return df

        if run_ids is None:
            run_ids = []
            for run_id in self.metrics_data[metric_name]:
                # Only include runs with valid data
                if 'values' in self.metrics_data[metric_name][run_id] and self.metrics_data[metric_name][run_id]['values']:
                    run_ids.append(run_id)

        # For step-aligned format (wide format)
        if align_steps:
            # Create a dictionary of step -> {run_id: value}
            step_dict = {}

            for run_id in run_ids:
                if run_id in self.metrics_data[metric_name]:
                    run_data = self.metrics_data[metric_name][run_id]
                    for step, value in zip(run_data['steps'], run_data['values']):
                        if step not in step_dict:
                            step_dict[step] = {}
                        step_dict[step][run_id] = value

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(step_dict, orient='index')
            df.index.name = 'step'
            df.reset_index(inplace=True)

        # For long format
        else:
            rows = []

            for run_id in run_ids:
                if run_id in self.metrics_data[metric_name]:
                    run_data = self.metrics_data[metric_name][run_id]

                    for step, value, timestamp in zip(run_data['steps'], run_data['values'], run_data['timestamps']):
                        rows.append({
                            'run_id': run_id,
                            'step': step,
                            'value': value,
                            'timestamp': timestamp
                        })

            df = pd.DataFrame(rows)

        # Cache the result
        self._metrics_df_cache[cache_key] = df

        return df

    def get_metric_comparison_df(self, metric_name: str, group_by: str = None) -> pd.DataFrame:
        """
        Get a DataFrame comparing a metric across different parameter values or groups.

        Args:
            metric_name: Name of the metric to compare
            group_by: Parameter or tag to group by (optional)

        Returns:
            DataFrame with metric statistics grouped by parameter values
        """
        if metric_name not in self.metrics_data:
            logger.warning(f"Metric '{metric_name}' not found in experiment {self.name}")
            return pd.DataFrame()

        # Get summary DataFrame
        df = self.get_metric_summary_df([metric_name])

        # If no grouping parameter specified, return the summary DataFrame
        if group_by is None:
            return df

        # Group by parameter
        group_col = f"param_{group_by}" if not group_by.startswith("param_") else group_by

        if group_col not in df.columns:
            logger.warning(f"Grouping column '{group_col}' not found in DataFrame")
            return df

        # Calculate statistics for each group
        grouped_stats = df.groupby(group_col).agg({
            f"{metric_name}_mean": ['mean', 'std', 'min', 'max', 'count'],
            f"{metric_name}_last": ['mean', 'std', 'min', 'max']
        }).reset_index()

        # Flatten multi-level columns
        grouped_stats.columns = [
            f"{col[0]}_{col[1]}" if col[1] else col[0]
            for col in grouped_stats.columns
        ]

        return grouped_stats

    def compare_with(self, other: 'ExperimentData', metric_name: str) -> pd.DataFrame:
        """
        Compare this experiment with another one for a specific metric.

        Args:
            other: Another ExperimentData instance to compare with
            metric_name: Name of the metric to compare

        Returns:
            DataFrame with comparison statistics
        """
        if metric_name not in self.metrics_data:
            logger.warning(f"Metric '{metric_name}' not found in experiment {self.name}")
            return pd.DataFrame()

        if metric_name not in other.metrics_data:
            logger.warning(f"Metric '{metric_name}' not found in experiment {other.name}")
            return pd.DataFrame()

        # Get summary DataFrames for both experiments
        df1 = self.get_metric_summary_df([metric_name])
        df2 = other.get_metric_summary_df([metric_name])

        # Add experiment name column
        df1['experiment'] = self.name
        df2['experiment'] = other.name

        # Combine
        combined = pd.concat([df1, df2])

        # Compute comparison statistics
        stats = combined.groupby('experiment').agg({
            f"{metric_name}_mean": ['mean', 'std', 'min', 'max', 'count'],
            f"{metric_name}_last": ['mean', 'std', 'min', 'max'],
            f"{metric_name}_improvement": ['mean', 'std']
        }).reset_index()

        # Flatten multi-level columns
        stats.columns = [
            f"{col[0]}_{col[1]}" if col[1] else col[0]
            for col in stats.columns
        ]

        return stats

    def export_to_csv(self, output_dir: Union[str, Path]) -> Dict[str, str]:
        """
        Export experiment data to CSV files for analysis in other tools.

        Args:
            output_dir: Directory to save CSV files

        Returns:
            Dictionary of file descriptions and paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs = {}

        # Export summary
        summary_df = self.get_metric_summary_df()
        summary_path = output_dir / f"{self.name.replace('/', '_')}_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        outputs['summary'] = str(summary_path)

        # Export each metric
        for metric_name in self.metrics_data:
            try:
                # Get the value DataFrame (long format)
                metric_df = self.get_metric_values_df(metric_name, align_steps=False)
                if not metric_df.empty:
                    metric_path = output_dir / f"{self.name.replace('/', '_')}_{metric_name}.csv"
                    metric_df.to_csv(metric_path, index=False)
                    outputs[f"metric_{metric_name}"] = str(metric_path)
            except Exception as e:
                logger.error(f"Error exporting metric {metric_name}: {e}")

        return outputs
```

## compare_experiments.py
```python
#!/usr/bin/env python3
"""
MLflow Experiment Comparison Script

This script compares metrics and performance across multiple MLflow experiments.
It creates visualizations to highlight differences between experiments and integrates
terrain generation evaluation metrics.

Usage:
    python compare_experiments.py --experiments <exp1> <exp2> ... [--output-dir <dir>]
                                   [--metrics <m1> <m2> ...] [--eval-data <dir>]
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union


class ExperimentComparer:
    def __init__(self, experiment_names, tracking_uri=None, output_dir=None):
        """
        Initialize the experiment comparer.

        Args:
            experiment_names: List of MLflow experiment names to compare
            tracking_uri: MLflow tracking URI (default: file:./mlruns)
            output_dir: Directory to save comparison outputs
        """
        self.experiment_names = experiment_names
        self.tracking_uri = tracking_uri or "file:./mlruns"
        self.output_dir = Path(output_dir or f"experiment_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data structures to hold experiment data
        self.experiments = {}  # {experiment_name: experiment_object}
        self.runs_data = {}    # {experiment_name: runs_df}

        # Connect to MLflow
        mlflow.set_tracking_uri(self.tracking_uri)

        # Load experiment data
        self._load_experiments()

        # Terrain evaluation metrics from external sources
        self.terrain_metrics = {}  # {experiment_name: metrics_dict}

    def _load_experiments(self):
        """Load experiment objects from MLflow."""
        for exp_name in self.experiment_names:
            experiment = mlflow.get_experiment_by_name(exp_name)
            if experiment:
                self.experiments[exp_name] = experiment
                print(f"Loaded experiment '{exp_name}' with ID: {experiment.experiment_id}")
            else:
                print(f"Warning: Experiment '{exp_name}' not found")

        if not self.experiments:
            raise ValueError("No valid experiments found")

    def load_runs_data(self):
        """Load runs data for all experiments."""
        for exp_name, experiment in self.experiments.items():
            # Get all runs for the experiment
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["attribute.start_time ASC"]
            )

            if runs.empty:
                print(f"Warning: No runs found for experiment '{exp_name}'")
                continue

            # Clean up the DataFrame
            runs_df = runs.copy()

            # Parse timestamps
            runs_df['start_time_dt'] = pd.to_datetime(runs_df['start_time'], unit='ms')
            runs_df['end_time_dt'] = pd.to_datetime(runs_df['end_time'], unit='ms')

            # Calculate run duration
            runs_df['duration_minutes'] = (runs_df['end_time'] - runs_df['start_time']) / (1000 * 60)

            # Add experiment name for identification
            runs_df['experiment_name'] = exp_name

            # Store in runs_data dictionary
            self.runs_data[exp_name] = runs_df

            print(f"Loaded data for {len(runs_df)} runs from experiment '{exp_name}'")

        # Return combined DataFrame for convenience
        return self.get_combined_runs()

    def get_combined_runs(self):
        """Get a combined DataFrame of all runs across experiments."""
        if not self.runs_data:
            self.load_runs_data()

        if not self.runs_data:
            return pd.DataFrame()  # Return empty DataFrame if no data

        return pd.concat(list(self.runs_data.values()), axis=0, ignore_index=True)

    def load_terrain_metrics(self, metrics_files=None):
        """
        Load terrain evaluation metrics from JSON files.

        Args:
            metrics_files: Dictionary mapping experiment names to metrics file paths,
                          or a directory containing metrics files named after experiments
        """
        if not metrics_files:
            return

        if isinstance(metrics_files, (str, Path)):
            # Assume it's a directory with files named after experiments
            metrics_dir = Path(metrics_files)
            for exp_name in self.experiment_names:
                metrics_file = metrics_dir / f"{exp_name}_evaluation.json"
                if metrics_file.exists():
                    try:
                        with open(metrics_file, 'r') as f:
                            self.terrain_metrics[exp_name] = json.load(f)
                        print(f"Loaded terrain metrics for '{exp_name}' from {metrics_file}")
                    except Exception as e:
                        print(f"Error loading terrain metrics for '{exp_name}': {e}")
        else:
            # Assume it's a dictionary mapping experiment names to file paths
            for exp_name, file_path in metrics_files.items():
                if exp_name in self.experiment_names and Path(file_path).exists():
                    try:
                        with open(file_path, 'r') as f:
                            self.terrain_metrics[exp_name] = json.load(f)
                        print(f"Loaded terrain metrics for '{exp_name}' from {file_path}")
                    except Exception as e:
                        print(f"Error loading terrain metrics for '{exp_name}': {e}")

    def compare_metrics(self, metrics=None, figsize=(14, 8)):
        """
        Compare specific metrics across experiments.

        Args:
            metrics: List of metrics to compare (default: all common metrics)
            figsize: Figure size (width, height) in inches
        """
        if not self.runs_data:
            self.load_runs_data()

        # Extract metrics data from all experiments
        all_metrics = {}
        for exp_name, runs_df in self.runs_data.items():
            # Find metrics columns
            metric_cols = [col for col in runs_df.columns if col.startswith('metrics.')]

            # Extract last values for each metric (representing final/best values)
            exp_metrics = {}
            for col in metric_cols:
                # Skip if all NaN
                if runs_df[col].isna().all():
                    continue

                # Get latest non-NaN value for each metric
                latest_idx = runs_df[~runs_df[col].isna()].index[-1]
                exp_metrics[col.replace('metrics.', '')] = runs_df.loc[latest_idx, col]

            all_metrics[exp_name] = exp_metrics

        # Find common metrics across all experiments
        common_metrics = set.intersection(*[set(m.keys()) for m in all_metrics.values()])

        if not common_metrics:
            print("No common metrics found across experiments")
            return None

        # Filter to requested metrics if specified
        if metrics:
            plot_metrics = [m for m in metrics if m in common_metrics]
            if not plot_metrics:
                print(f"None of the specified metrics found. Common metrics: {common_metrics}")
                return None
        else:
            # Limit to reasonable number of metrics if not specified
            plot_metrics = list(common_metrics)[:min(8, len(common_metrics))]

        # Create DataFrame for plotting
        plot_data = []
        for exp_name, exp_metrics in all_metrics.items():
            for metric in plot_metrics:
                plot_data.append({
                    'Experiment': exp_name,
                    'Metric': metric,
                    'Value': exp_metrics[metric]
                })

        plot_df = pd.DataFrame(plot_data)

        # Create barplot
        plt.figure(figsize=figsize)

        # Create subplots for each metric
        n_metrics = len(plot_metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)
        if n_metrics == 1:
            axes = [axes]

        # For consistent color scheme
        palette = sns.color_palette("tab10", len(self.experiment_names))
        exp_colors = {name: palette[i] for i, name in enumerate(self.experiment_names)}

        for i, metric in enumerate(plot_metrics):
            # Filter data for this metric
            metric_data = plot_df[plot_df['Metric'] == metric]

            # Create bar plot
            ax = axes[i]
            sns.barplot(x='Experiment', y='Value', data=metric_data, ax=ax, palette=exp_colors)

            # Customize plot
            ax.set_title(f"Comparison of {metric}")
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)

            # Add value labels on top of bars
            for j, p in enumerate(ax.patches):
                height = p.get_height()
                ax.text(p.get_x() + p.get_width()/2., height + height*0.01,
                        f'{height:.4f}', ha="center", va='bottom')

        # Adjust layout
        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / "metric_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved metric comparison to {output_path}")

        return fig, axes

    def compare_run_durations(self, figsize=(12, 6)):
        """Compare run durations across experiments."""
        if not self.runs_data:
            self.load_runs_data()

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Calculate average duration for each experiment
        avg_durations = []
        for exp_name, runs_df in self.runs_data.items():
            avg_duration = runs_df['duration_minutes'].mean()
            avg_durations.append({
                'Experiment': exp_name,
                'Average Duration (min)': avg_duration
            })

        avg_df = pd.DataFrame(avg_durations)

        # Create bar plot
        sns.barplot(x='Experiment', y='Average Duration (min)', data=avg_df, ax=ax)

        ax.set_title("Average Run Duration by Experiment")
        ax.set_xlabel("Experiment")
        ax.set_ylabel("Average Duration (minutes)")
        ax.grid(True, alpha=0.3)

        # Add value labels on top of bars
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., height + height*0.01,
                    f'{height:.2f}', ha="center", va='bottom')

        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / "run_duration_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved run duration comparison to {output_path}")

        return fig, ax

    def compare_metrics_over_time(self, metric, figsize=(12, 6)):
        """
        Compare how a specific metric evolved over time across experiments.

        Args:
            metric: The metric to compare
            figsize: Figure size (width, height) in inches
        """
        if not self.runs_data:
            self.load_runs_data()

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Full metric name
        metric_col = f"metrics.{metric}" if not metric.startswith("metrics.") else metric

        # Plot metric values over time for each experiment
        for exp_name, runs_df in self.runs_data.items():
            if metric_col in runs_df.columns and not runs_df[metric_col].isna().all():
                # Get run names if available
                if 'tags.mlflow.runName' in runs_df.columns:
                    run_names = runs_df['tags.mlflow.runName'].fillna(runs_df['run_id'].str[-8:])
                else:
                    run_names = runs_df['run_id'].str[-8:]

                # Plot only runs with this metric
                valid_runs = runs_df[~runs_df[metric_col].isna()]

                ax.plot(valid_runs.index, valid_runs[metric_col], 'o-', label=exp_name)

        ax.set_title(f"Comparison of {metric} Over Time")
        ax.set_xlabel("Run Index")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / f"metric_over_time_{metric}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved metric over time comparison to {output_path}")

        return fig, ax

    def compare_terrain_metrics(self, figsize=(14, 10)):
        """Compare terrain evaluation metrics across experiments."""
        if not self.terrain_metrics:
            print("No terrain metrics data loaded. Use load_terrain_metrics() first.")
            return None

        # Extract aggregate metrics from each experiment
        metrics_data = []
        for exp_name, metrics in self.terrain_metrics.items():
            if 'aggregate' in metrics:
                agg = metrics['aggregate']
                metrics_data.append({
                    'Experiment': exp_name,
                    'IoU': agg.get('mean_iou', 0),
                    'Precision': agg.get('mean_precision', 0),
                    'Recall': agg.get('mean_recall', 0),
                    'F1 Score': agg.get('mean_f1', 0),
                    'Largest Plausible Area (km²)': agg.get('max_plausible_area_sq_km', 0),
                    'Plausible Percentage': agg.get('mean_plausible_percentage', 0)
                })

        if not metrics_data:
            print("No valid aggregate metrics found in terrain metrics data")
            return None

        metrics_df = pd.DataFrame(metrics_data)

        # Set up plot
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        axes = axes.flatten()

        # Plot each metric
        metrics_to_plot = [
            'IoU', 'Precision', 'Recall', 'F1 Score',
            'Largest Plausible Area (km²)', 'Plausible Percentage'
        ]

        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            sns.barplot(x='Experiment', y=metric, data=metrics_df, ax=ax)

            ax.set_title(f"Comparison of {metric}")
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)

            # Rotate x-axis labels if needed
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            # Add value labels
            for j, p in enumerate(ax.patches):
                height = p.get_height()
                format_str = '{:.4f}' if metric in ['IoU', 'Precision', 'Recall', 'F1 Score'] else '{:.2f}'
                ax.text(p.get_x() + p.get_width()/2., height + height*0.01,
                        format_str.format(height), ha="center", va='bottom')

        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / "terrain_metrics_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved terrain metrics comparison to {output_path}")

        # Also create a summary table
        summary_path = self.output_dir / "terrain_metrics_summary.csv"
        metrics_df.to_csv(summary_path, index=False)
        print(f"Saved terrain metrics summary to {summary_path}")

        return fig, axes

    def create_radar_plot(self, figsize=(10, 10)):
        """Create a radar plot comparing key metrics across experiments."""
        if not self.terrain_metrics:
            print("No terrain metrics data loaded. Use load_terrain_metrics() first.")
            return None

        # Extract key metrics for radar plot
        radar_data = {}
        metrics = ['mean_iou', 'mean_precision', 'mean_recall', 'mean_f1',
                  'mean_plausible_percentage']

        display_names = {
            'mean_iou': 'IoU',
            'mean_precision': 'Precision',
            'mean_recall': 'Recall',
            'mean_f1': 'F1 Score',
            'mean_plausible_percentage': 'Plausible %'
        }

        # Get metrics for each experiment
        for exp_name, metric_data in self.terrain_metrics.items():
            if 'aggregate' in metric_data:
                radar_data[exp_name] = {
                    metric: metric_data['aggregate'].get(metric, 0)
                    for metric in metrics
                }

        if not radar_data:
            print("No valid metrics found for radar plot")
            return None

        # Create the radar plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, polar=True)

        # Set up the angles for each metric
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop

        # Set up the plot
        ax.set_theta_offset(np.pi / 2)  # Start from top
        ax.set_theta_direction(-1)  # Go clockwise

        # Set labels
        plt.xticks(angles[:-1], [display_names[m] for m in metrics])

        # Plot each experiment
        for i, (exp_name, values) in enumerate(radar_data.items()):
            # Extract values in the correct order and normalize to 0-1 range
            values_list = [values[m] for m in metrics]

            # Add the first value again to close the loop
            values_list += values_list[:1]

            # Plot the experiment
            ax.plot(angles, values_list, 'o-', linewidth=2, label=exp_name)
            ax.fill(angles, values_list, alpha=0.1)

        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        # Set y-axis limits
        ax.set_ylim(0, max([max(list(exp.values())) for exp in radar_data.values()]) * 1.1)

        # Set title
        plt.title('Experiment Comparison - Key Metrics', size=15, y=1.1)

        # Save figure
        output_path = self.output_dir / "radar_plot_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved radar plot to {output_path}")

        return fig, ax

    def generate_all_comparisons(self):
        """Generate all comparison visualizations."""
        if not self.runs_data:
            self.load_runs_data()

        # Create output subdirectory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.output_dir / f"comparison_report_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)

        # Save a copy of the combined runs data
        combined_runs = self.get_combined_runs()
        runs_csv_path = report_dir / "all_runs_data.csv"
        combined_runs.to_csv(runs_csv_path, index=False)

        # Extract common metrics across all experiments
        common_metrics = self._get_common_metrics()

        # Generate comparisons
        print(f"Generating comparisons for experiments: {', '.join(self.experiment_names)}")

        # 1. Compare metrics
        self.compare_metrics()
        plt.close()

        # 2. Compare run durations
        self.compare_run_durations()
        plt.close()

        # 3. Compare metrics over time for key metrics
        for metric in common_metrics[:min(3, len(common_metrics))]:
            try:
                self.compare_metrics_over_time(metric)
                plt.close()
            except Exception as e:
                print(f"Error generating metric over time comparison for {metric}: {e}")

        # 4. Compare terrain metrics if available
        if self.terrain_metrics:
            self.compare_terrain_metrics()
            plt.close()

            self.create_radar_plot()
            plt.close()

        # 5. Create comprehensive summary table
        self._create_summary_table(report_dir / "comprehensive_summary.csv")

        print(f"All comparisons generated and saved to {self.output_dir}")
        return report_dir

    def _get_common_metrics(self):
        """Get metrics that are common across all experiments."""
        common_metrics = None

        for exp_name, runs_df in self.runs_data.items():
            # Find metrics columns
            metric_cols = [col.replace('metrics.', '') for col in runs_df.columns
                        if col.startswith('metrics.')]

            # Remove metrics that are all NaN
            metric_cols = [col for col in metric_cols
                         if not runs_df[f'metrics.{col}'].isna().all()]

            # Update common metrics
            if common_metrics is None:
                common_metrics = set(metric_cols)
            else:
                common_metrics &= set(metric_cols)

        return list(common_metrics) if common_metrics else []

    def _create_summary_table(self, output_path):
        """Create a comprehensive summary table with MLflow and terrain metrics."""
        # Collect MLflow metrics
        mlflow_metrics = {}
        for exp_name, runs_df in self.runs_data.items():
            # Find metrics columns
            metric_cols = [col for col in runs_df.columns if col.startswith('metrics.')]

            # Get latest non-NaN value for each metric
            exp_metrics = {}
            for col in metric_cols:
                # Skip if all NaN
                if runs_df[col].isna().all():
                    continue

                # Get latest non-NaN value
                latest_idx = runs_df[~runs_df[col].isna()].index[-1]
                metric_name = col.replace('metrics.', '')
                exp_metrics[metric_name] = runs_df.loc[latest_idx, col]

            mlflow_metrics[exp_name] = exp_metrics

        # Collect terrain metrics
        terrain_agg = {}
        for exp_name, metrics in self.terrain_metrics.items():
            if 'aggregate' in metrics:
                terrain_agg[exp_name] = metrics['aggregate']

        # Combine into a single DataFrame
        all_data = []
        for exp_name in self.experiment_names:
            exp_data = {'Experiment': exp_name}

            # Add MLflow metrics
            if exp_name in mlflow_metrics:
                for metric, value in mlflow_metrics[exp_name].items():
                    exp_data[f'MLflow_{metric}'] = value

            # Add terrain metrics
            if exp_name in terrain_agg:
                for metric, value in terrain_agg[exp_name].items():
                    # Skip non-numeric metrics
                    if isinstance(value, (int, float)):
                        exp_data[f'Terrain_{metric}'] = value

            all_data.append(exp_data)

        # Create DataFrame and save
        summary_df = pd.DataFrame(all_data)
        summary_df.to_csv(output_path, index=False)
        print(f"Saved comprehensive summary to {output_path}")
        return summary_df

    def plot_metric_improvement(self, from_exp, to_exp, figsize=(12, 6)):
        """Plot the improvement in metrics from one experiment to another."""
        if not set([from_exp, to_exp]).issubset(set(self.experiment_names)):
            print(f"Both experiments must be in the loaded experiments: {self.experiment_names}")
            return None

        if not self.runs_data:
            self.load_runs_data()

        # Check if both experiments exist in runs data
        if from_exp not in self.runs_data or to_exp not in self.runs_data:
            print(f"Cannot find both experiments in runs data")
            return None

        # Get best metrics for each experiment
        from_metrics = {}
        to_metrics = {}

        for exp_name, exp_df in [(from_exp, self.runs_data[from_exp]),
                                (to_exp, self.runs_data[to_exp])]:
            # Find metrics columns
            metric_cols = [col for col in exp_df.columns if col.startswith('metrics.')]

            # For each metric, get the best value (assume higher is better)
            exp_metrics = {}
            for col in metric_cols:
                # Skip if all NaN
                if exp_df[col].isna().all():
                    continue

                # Get best value
                best_value = exp_df[col].max()  # Assume higher is better
                metric_name = col.replace('metrics.', '')

                if exp_name == from_exp:
                    from_metrics[metric_name] = best_value
                else:
                    to_metrics[metric_name] = best_value

        # Find common metrics
        common_metrics = set(from_metrics.keys()) & set(to_metrics.keys())

        if not common_metrics:
            print("No common metrics found between the two experiments")
            return None

        # Calculate improvement
        improvements = []
        for metric in common_metrics:
            rel_improvement = ((to_metrics[metric] - from_metrics[metric]) /
                              max(abs(from_metrics[metric]), 1e-10)) * 100

            improvements.append({
                'Metric': metric,
                'From': from_metrics[metric],
                'To': to_metrics[metric],
                'Absolute Diff': to_metrics[metric] - from_metrics[metric],
                'Relative Improvement (%)': rel_improvement
            })

        # Sort by absolute improvement
        improvements.sort(key=lambda x: abs(x['Absolute Diff']), reverse=True)

        # Create DataFrame
        imp_df = pd.DataFrame(improvements)

        # Create visualization
        fig, ax = plt.subplots(figsize=figsize)

        # Plot relative improvement as bar chart
        sns.barplot(x='Metric', y='Relative Improvement (%)', data=imp_df, ax=ax)

        ax.set_title(f"Metric Improvement: {from_exp} → {to_exp}")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Relative Improvement (%)")
        ax.grid(True, alpha=0.3)

        # Add value labels
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            # Use different position if negative
            if height < 0:
                ax.text(p.get_x() + p.get_width()/2., height - height*0.05,
                        f'{height:.2f}%', ha="center", va='top')
            else:
                ax.text(p.get_x() + p.get_width()/2., height + height*0.05,
                        f'{height:.2f}%', ha="center", va='bottom')

        # Rotate x-axis labels if many metrics
        if len(common_metrics) > 5:
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / f"improvement_{from_exp}_to_{to_exp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved improvement plot to {output_path}")

        # Save data
        data_path = self.output_dir / f"improvement_{from_exp}_to_{to_exp}.csv"
        imp_df.to_csv(data_path, index=False)

        return fig, ax


def main():
    parser = argparse.ArgumentParser(description="Compare multiple MLflow experiments")
    parser.add_argument("--experiments", nargs='+', required=True,
                      help="Names of MLflow experiments to compare")
    parser.add_argument("--tracking-uri", default="file:./mlruns",
                      help="MLflow tracking URI")
    parser.add_argument("--output-dir", default=None,
                      help="Output directory for comparison results")
    parser.add_argument("--metrics", nargs='+',
                      help="Specific metrics to compare")
    parser.add_argument("--eval-data",
                      help="Directory containing terrain evaluation data")
    parser.add_argument("--from-exp",
                      help="Source experiment for improvement comparison")
    parser.add_argument("--to-exp",
                      help="Target experiment for improvement comparison")

    args = parser.parse_args()

    # Create comparer
    comparer = ExperimentComparer(
        experiment_names=args.experiments,
        tracking_uri=args.tracking_uri,
        output_dir=args.output_dir
    )

    # Load runs data
    comparer.load_runs_data()

    # Load terrain evaluation data if provided
    if args.eval_data:
        comparer.load_terrain_metrics(args.eval_data)

    # If specific metrics provided, compare them
    if args.metrics:
        comparer.compare_metrics(metrics=args.metrics)
    else:
        # Otherwise, generate all comparisons
        comparer.generate_all_comparisons()

    # Generate improvement comparison if requested
    if args.from_exp and args.to_exp:
        comparer.plot_metric_improvement(args.from_exp, args.to_exp)


if __name__ == "__main__":
    main()
```

## main_pipeline.py
```python
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
```

## cleanup_pythonanywhere.sh
```bash
#!/bin/bash
#
# cleanup_pythonanywhere.sh - Script to clean up PythonAnywhere files
#
# This script provides a convenient way to clean up annotation and image files
# on your PythonAnywhere server.

# Default options
DRY_RUN=false
ANNOTATIONS=false
IMAGES=false
GRID=""
FORCE=false

# Display help message
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -a, --annotations       Delete files in annotations directory"
    echo "  -i, --images            Delete files in images directory"
    echo "  -g, --grid GRID         Filter by grid square prefix (e.g., NH70)"
    echo "  -d, --dry-run           List files that would be deleted without actually deleting"
    echo "  -f, --force             Skip confirmation prompt"
    echo "  -h, --help              Show this help message"
    echo
    echo "Examples:"
    echo "  $0 --annotations --grid NH70       Delete all NH70 annotations"
    echo "  $0 --images --dry-run              Show all images that would be deleted"
    echo "  $0 --annotations --images --force  Delete all annotations and images without confirmation"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -a|--annotations)
            ANNOTATIONS=true
            shift
            ;;
        -i|--images)
            IMAGES=true
            shift
            ;;
        -g|--grid)
            GRID="$2"
            shift
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Build the command
CMD="python utils/api/pythonanywhere_cleanup.py"

if [ "$ANNOTATIONS" = true ]; then
    CMD+=" --annotations"
fi

if [ "$IMAGES" = true ]; then
    CMD+=" --images"
fi

if [ -n "$GRID" ]; then
    CMD+=" --grid $GRID"
fi

if [ "$DRY_RUN" = true ]; then
    CMD+=" --dry-run"
fi

if [ "$FORCE" = true ]; then
    CMD+=" --force"
fi

# If no actions were specified, show help
if [ "$ANNOTATIONS" = false ] && [ "$IMAGES" = false ]; then
    echo "Error: You must specify at least one action (--annotations or --images)"
    show_help
    exit 1
fi

# Show the command that will be executed
echo "Executing: $CMD"

# Execute the command
eval $CMD

# Report completion
if [ $? -eq 0 ]; then
    echo "Cleanup completed successfully"
else
    echo "Cleanup encountered errors, check the logs for details"
    exit 1
fi
```

## plot_mlflow_experiment.py
```python
#!/usr/bin/env python3
"""
MLflow Experiment Visualization Script

This script creates visualizations from MLflow experiment data for a single experiment.
It generates plots for metrics over time, metric distributions, and summary statistics.

Usage:
    python plot_mlflow_experiment.py --experiment-name <name> [--output-dir <dir>]
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from pathlib import Path
from datetime import datetime
import re


class MLflowExperimentVisualizer:
    def __init__(self, experiment_name, tracking_uri=None, output_dir=None):
        """
        Initialize the MLflow experiment visualizer.

        Args:
            experiment_name: Name of the MLflow experiment to visualize
            tracking_uri: MLflow tracking URI (default: file:./mlruns)
            output_dir: Directory to save visualization outputs
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "file:./mlruns"
        self.output_dir = Path(output_dir or f"mlflow_viz_{experiment_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runs_df = None

        # Configure MLflow client
        mlflow.set_tracking_uri(self.tracking_uri)

        # Get experiment
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        if not self.experiment:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        print(f"Found experiment '{experiment_name}' with ID: {self.experiment.experiment_id}")

    def load_runs_data(self):
        """Load runs data for the experiment and transform it for visualization."""
        # Get all runs for the experiment
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=["attribute.start_time ASC"]
        )

        if runs.empty:
            raise ValueError(f"No runs found for experiment '{self.experiment_name}'")

        # Clean up the DataFrame
        self.runs_df = runs.copy()

        # Parse timestamps
        self.runs_df['start_time_dt'] = pd.to_datetime(self.runs_df['start_time'], unit='ms')
        self.runs_df['end_time_dt'] = pd.to_datetime(self.runs_df['end_time'], unit='ms')

        # Calculate run duration
        self.runs_df['duration_minutes'] = (self.runs_df['end_time'] - self.runs_df['start_time']) / (1000 * 60)

        # Generate human-readable run names if missing
        self._generate_readable_run_names()

        print(f"Loaded data for {len(self.runs_df)} runs")
        return self.runs_df

    def _generate_readable_run_names(self):
        """Generate human-readable run names for runs that don't have them."""
        # First, check if 'tags.mlflow.runName' column exists
        if 'tags.mlflow.runName' not in self.runs_df.columns:
            self.runs_df['tags.mlflow.runName'] = None

        # Group runs by type
        training_runs = []
        evaluation_runs = []
        human_guided_runs = []
        other_runs = []

        # Identify run types and assign to appropriate group
        for idx, row in self.runs_df.iterrows():
            # Use existing run name if available
            run_name = row.get('tags.mlflow.runName')

            # Determine run type if no name or generate new name if requested
            if pd.isna(run_name) or not run_name:
                # Try to determine type from parameters
                params = {k.replace('params.', ''): v for k, v in row.items()
                         if k.startswith('params.') and not pd.isna(v)}

                mode = params.get('mode', '').lower()

                if 'train' in mode and 'human' not in mode:
                    training_runs.append(idx)
                elif 'eval' in mode:
                    evaluation_runs.append(idx)
                elif 'human' in mode or 'human_guided' in mode:
                    human_guided_runs.append(idx)
                else:
                    # Check metrics as fallback
                    metrics = [k for k in row.index if k.startswith('metrics.') and not pd.isna(row[k])]
                    if any('train' in m.lower() for m in metrics):
                        training_runs.append(idx)
                    elif any('eval' in m.lower() for m in metrics):
                        evaluation_runs.append(idx)
                    else:
                        other_runs.append(idx)

        # Generate sequential run names
        for i, idx in enumerate(training_runs):
            if pd.isna(self.runs_df.loc[idx, 'tags.mlflow.runName']) or not self.runs_df.loc[idx, 'tags.mlflow.runName']:
                self.runs_df.loc[idx, 'tags.mlflow.runName'] = f"training_run_{i+1:02d}"

        for i, idx in enumerate(evaluation_runs):
            if pd.isna(self.runs_df.loc[idx, 'tags.mlflow.runName']) or not self.runs_df.loc[idx, 'tags.mlflow.runName']:
                self.runs_df.loc[idx, 'tags.mlflow.runName'] = f"evaluation_run_{i+1:02d}"

        for i, idx in enumerate(human_guided_runs):
            if pd.isna(self.runs_df.loc[idx, 'tags.mlflow.runName']) or not self.runs_df.loc[idx, 'tags.mlflow.runName']:
                self.runs_df.loc[idx, 'tags.mlflow.runName'] = f"human_guided_run_{i+1:02d}"

        for i, idx in enumerate(other_runs):
            if pd.isna(self.runs_df.loc[idx, 'tags.mlflow.runName']) or not self.runs_df.loc[idx, 'tags.mlflow.runName']:
                self.runs_df.loc[idx, 'tags.mlflow.runName'] = f"run_{i+1:02d}"

        # Fill in run_id for any remaining missing values as a fallback
        mask = pd.isna(self.runs_df['tags.mlflow.runName']) | (self.runs_df['tags.mlflow.runName'] == '')
        self.runs_df.loc[mask, 'tags.mlflow.runName'] = self.runs_df.loc[mask, 'run_id'].apply(lambda x: f"run_{x[-8:]}")

        print(f"Generated {mask.sum()} missing run names")
```

## reset_mlflow.sh
```bash
#!/bin/bash
#
# reset_mlflow.sh - Improved script to reset MLflow environment
#
# This script properly cleans up the MLflow environment, removing
# empty files and preserving important metadata.

set -e  # Exit on error

# Configuration
MLFLOW_DIR="./mlruns"
EXPERIMENT_NAME="dsm_inpainting"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="${MLFLOW_DIR}_backup_${TIMESTAMP}"

# Log messages with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Make backup of current MLflow data
backup_mlflow() {
    if [ -d "$MLFLOW_DIR" ]; then
        log "Creating backup at $BACKUP_DIR"
        mkdir -p "$(dirname "$BACKUP_DIR")"
        cp -r "$MLFLOW_DIR" "$BACKUP_DIR"
    else
        log "No MLflow directory to backup"
    fi
}

# Clean up existing MLflow environment
clean_mlflow() {
    log "Stopping any running MLflow processes"
    pkill -f "mlflow server" || true

    log "Removing MLflow data directory"
    rm -rf "$MLFLOW_DIR"
    mkdir -p "$MLFLOW_DIR"

    # Create subdirectories
    mkdir -p "$MLFLOW_DIR/models"
    mkdir -p "$MLFLOW_DIR/.trash"

    # Ensure correct permissions
    chmod -R 755 "$MLFLOW_DIR"
}

# Create base MLflow experiment structure
create_experiment() {
    log "Creating base experiment structure"

    mkdir -p "$MLFLOW_DIR/$EXPERIMENT_NAME"
    mkdir -p "$MLFLOW_DIR/$EXPERIMENT_NAME/artifacts"
    mkdir -p "$MLFLOW_DIR/$EXPERIMENT_NAME/metrics"
    mkdir -p "$MLFLOW_DIR/$EXPERIMENT_NAME/params"
    mkdir -p "$MLFLOW_DIR/$EXPERIMENT_NAME/tags"

    # Create meta.yaml with correct timestamp and experiment info
    CURRENT_TIME=$(date +%s%3N)
    cat > "$MLFLOW_DIR/$EXPERIMENT_NAME/meta.yaml" << EOL
artifact_location: file:./$MLFLOW_DIR/$EXPERIMENT_NAME
creation_time: $CURRENT_TIME
experiment_id: $EXPERIMENT_NAME
last_update_time: $CURRENT_TIME
lifecycle_stage: active
name: $EXPERIMENT_NAME
EOL

    # Ensure proper permissions
    chmod -R 755 "$MLFLOW_DIR"

    log "Created MLflow experiment structure for $EXPERIMENT_NAME"
}

# Main process
main() {
    log "Starting MLflow environment reset"

    # Backup existing data
    backup_mlflow

    # Clean MLflow environment
    clean_mlflow

    # Create new experiment structure
    create_experiment

    log "MLflow environment reset complete"
    log "You can start MLflow server with: ./start_mlflow.sh"
}

# Run the script
main
```

## evaluate_terrain.py
```python
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
```


## annotations_uploader.py
```python
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
```

## log_run.sh
```bash
#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Get current timestamp for unique log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_run_${TIMESTAMP}.log"

# Start MLflow in background
./start_mlflow.sh > "logs/mlflow_${TIMESTAMP}.log" 2>&1 &

# Wait a moment for MLflow to start
sleep 2

# Run the pipeline and tee output to both terminal and log file
python main_pipeline.py --mode train 2>&1 | tee "${LOG_FILE}"

# Optional: Kill MLflow server when done
# pkill -f "mlflow ui"
`

## requirements.txt
```plaintext
mlflow>=2.8.0
psutil>=5.9.0
gitpython>=3.1.40
GPUtil==1.4.0
torch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1
numpy==2.1.2
opencv-python-headless==4.10.0.84
PyYAML==6.0.2
requests==2.32.3
tqdm==4.66.5
Pillow==10.4.0
scikit-image==0.24.0
matplotlib==3.9.2
pandas==2.2.3
```
