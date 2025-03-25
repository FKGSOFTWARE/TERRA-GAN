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
