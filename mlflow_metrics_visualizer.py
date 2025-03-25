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
