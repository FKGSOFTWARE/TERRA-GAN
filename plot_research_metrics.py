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
