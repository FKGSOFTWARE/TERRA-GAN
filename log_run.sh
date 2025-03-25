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
