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
