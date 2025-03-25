#!/bin/bash
#
# cleanup_empty_folders.sh
#
# This script removes empty directories across the codebase
# to clean up after processing the evaluation grid.

set -e  # Exit on error

# Timestamp for logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/cleanup_${TIMESTAMP}.log"

# Ensure logs directory exists
mkdir -p logs

# Log function
log() {
    local message="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $message" | tee -a "$LOG_FILE"
}

# Define directories that should be preserved even if empty
PRESERVE_DIRS=(
    ".git"
    "__pycache__"
    "mlruns"
    "mlruns_backup"
    ".trash"
)

# Function to check if a directory should be preserved
should_preserve() {
    local dir="$1"

    # Check if the directory matches any in the preserve list
    for preserve in "${PRESERVE_DIRS[@]}"; do
        if [[ "$dir" =~ /$preserve(/|$) ]]; then
            return 0  # Return true (0) if should be preserved
        fi
    done

    return 1  # Return false (1) if should not be preserved
}

# Function to find and remove empty directories with multiple passes
cleanup_empty_dirs() {
    local base_dir="$1"
    local total_removed=0
    local removed=1

    log "Cleaning empty directories in: $base_dir"

    # Keep removing directories until no more can be removed
    while [ "$removed" -gt 0 ]; do
        removed=0

        # Use find to identify empty directories, but filter out specific directories
        while IFS= read -r dir; do
            # Skip the base directory itself
            if [ "$dir" = "$base_dir" ]; then
                continue
            fi

            # Skip directories we want to preserve
            if should_preserve "$dir"; then
                continue
            fi

            # Remove the empty directory
            log "Removing empty directory: $dir"
            rmdir "$dir" 2>/dev/null && ((removed++)) || true
        done < <(find "$base_dir" -type d -empty 2>/dev/null | sort -r)

        total_removed=$((total_removed + removed))

        if [ "$removed" -gt 0 ]; then
            log "Removed $removed directories in this pass, continuing..."
        fi
    done

    log "Total: Removed $total_removed empty directories from $base_dir"
    return $total_removed
}

# Main cleanup process
log "Starting empty directory cleanup..."

# Look for all empty directories recursively
ROOT_DIR="$(pwd)"
log "Project root directory: $ROOT_DIR"

# These are the main directories we want to clean
CLEAN_DIRS=(
    "./data/processed_data"
    "./data/output"
    "./mvp_gan/data"
    "./TEST_01/final_results"
)

# Clean each directory
for dir in "${CLEAN_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        log "Processing directory: $dir"
        cleanup_empty_dirs "$dir"
    else
        log "Directory not found, skipping: $dir"
    fi
done

log "Cleanup completed successfully!"
