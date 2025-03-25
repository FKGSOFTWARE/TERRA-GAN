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
