import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ExperimentComparison:
    """Utility class for comparing multiple experiment runs"""

    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        self.tracking_uri = tracking_uri or "file:./mlruns"
        mlflow.set_tracking_uri(self.tracking_uri)
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        if not self.experiment:
            raise ValueError(f"Experiment {experiment_name} not found")

    def load_runs(self, metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """Load runs data into a DataFrame"""
        runs = mlflow.search_runs([self.experiment.experiment_id])
        if metrics:
            metric_cols = [f"metrics.{m}" for m in metrics]
            return runs[['run_id', 'start_time', *metric_cols]]
        return runs

    def compare_metrics(self,
                       metrics: List[str],
                       output_dir: Path,
                       run_ids: Optional[List[str]] = None):
        """Generate comparison plots for specified metrics"""
        runs_df = self.load_runs(metrics)
        if run_ids:
            runs_df = runs_df[runs_df['run_id'].isin(run_ids)]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create comparison plots
        for metric in metrics:
            metric_col = f"metrics.{metric}"

            # Line plot for metric over time
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=runs_df, x='start_time', y=metric_col, hue='run_id')
            plt.title(f'{metric} Over Time')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / f'{metric}_time_comparison.png')
            plt.close()

            # Box plot for metric distribution
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=runs_df, y=metric_col)
            plt.title(f'{metric} Distribution Across Runs')
            plt.tight_layout()
            plt.savefig(output_dir / f'{metric}_distribution.png')
            plt.close()

        # Generate summary statistics
        summary = runs_df.groupby('run_id')[
            [f"metrics.{m}" for m in metrics]
        ].agg(['mean', 'std', 'min', 'max'])

        summary.to_csv(output_dir / 'metrics_summary.csv')
        return summary

    def find_best_run(self,
                      metric: str,
                      higher_is_better: bool = False) -> Dict:
        """Find the best run based on a specific metric"""
        runs_df = self.load_runs([metric])
        metric_col = f"metrics.{metric}"

        best_idx = runs_df[metric_col].idxmax() if higher_is_better else runs_df[metric_col].idxmin()
        best_run = runs_df.loc[best_idx]

        return {
            'run_id': best_run['run_id'],
            'metric_value': best_run[metric_col],
            'start_time': best_run['start_time']
        }

    def compare_params(self,
                      params: List[str],
                      metric: str,
                      output_dir: Path):
        """Analyze parameter impact on specified metric"""
        runs = mlflow.search_runs([self.experiment.experiment_id])
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metric_col = f"metrics.{metric}"
        param_cols = [f"params.{p}" for p in params]

        # Create parameter impact plots
        for param in params:
            param_col = f"params.{param}"

            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=runs, x=param_col, y=metric_col)
            plt.title(f'Impact of {param} on {metric}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / f'{param}_impact.png')
            plt.close()

        # Calculate correlations
        correlations = runs[param_cols + [metric_col]].corr()[metric_col].sort_values()
        correlations.to_csv(output_dir / 'parameter_correlations.csv')

        return correlations
