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
