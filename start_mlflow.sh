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
