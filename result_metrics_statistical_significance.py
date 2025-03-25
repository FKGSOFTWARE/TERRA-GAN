#!/usr/bin/env python3
"""
Statistical Significance Testing for Terrain Generation Experiments

This script performs statistical significance testing between experiment results:

1. Loads metrics from multiple experiment JSON files
2. Performs statistical hypothesis testing (t-tests, Mann-Whitney U, etc.)
3. Calculates effect sizes to quantify the magnitude of differences
4. Outputs comprehensive results in JSON format

Usage:
    python statistical_significance.py --experiments <exp1.json> <exp2.json> ...
                                      [--output <output.json>]
                                      [--significance-level 0.05]
                                      [--paired]
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
from scipy import stats
import warnings

# Configure warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


class StatisticalTester:
    """Statistical significance tester for comparing terrain generation experiments"""

    def __init__(self, experiments: List[Dict], experiment_names: List[str] = None,
                 output_file: str = None, alpha: float = 0.05, paired: bool = False):
        """
        Initialize the tester with experiment data

        Args:
            experiments: List of experiment data dictionaries (loaded from JSON)
            experiment_names: Optional names for experiments (defaults to exp1, exp2, etc.)
            output_file: Path to save JSON results
            alpha: Significance level (default: 0.05)
            paired: Whether to use paired tests (default: False)
        """
        self.experiments = experiments
        self.experiment_names = experiment_names or [f"exp{i+1}" for i in range(len(experiments))]

        if len(self.experiment_names) != len(self.experiments):
            raise ValueError("Number of experiment names must match number of experiments")

        self.output_file = output_file or f"stat_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        self.alpha = alpha
        self.paired = paired

        # Extract per-image metrics for statistical testing
        self.per_image_data = {}
        self.extract_per_image_data()

        # Results storage
        self.test_results = {}
        self.effect_sizes = {}
        self.descriptive_stats = None

    def extract_per_image_data(self):
        """Extract per-image metrics from each experiment for statistical testing"""
        for i, exp in enumerate(self.experiments):
            exp_name = self.experiment_names[i]

            # Check if experiment has per-image data
            if 'per_image' not in exp:
                print(f"Warning: Experiment {exp_name} does not have per-image data. Statistical tests may be limited.")
                continue

            # Extract metrics for each image
            per_image = exp['per_image']
            metrics = {}

            # Find all metrics in the first image
            if per_image:
                first_image = next(iter(per_image.values()))
                metric_names = list(first_image.keys())

                # Initialize metric lists
                for metric in metric_names:
                    metrics[metric] = []

                # Collect metric values for all images
                for img_id, img_metrics in per_image.items():
                    for metric in metric_names:
                        if metric in img_metrics:
                            # Ensure the value is numeric
                            try:
                                value = float(img_metrics[metric])
                                metrics[metric].append(value)
                            except (ValueError, TypeError):
                                continue

            self.per_image_data[exp_name] = metrics

    def calculate_descriptive_statistics(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Calculate descriptive statistics for all metrics across experiments"""
        stats_data = {}

        for exp_name, metrics in self.per_image_data.items():
            stats_data[exp_name] = {}

            for metric_name, values in metrics.items():
                if not values:
                    continue

                values_array = np.array(values)
                stats_data[exp_name][metric_name] = {
                    'mean': float(np.mean(values_array)),
                    'median': float(np.median(values_array)),
                    'std_dev': float(np.std(values_array)),
                    'min': float(np.min(values_array)),
                    'max': float(np.max(values_array)),
                    'count': int(len(values_array))
                }

        self.descriptive_stats = stats_data
        return stats_data

    def run_statistical_tests(self):
        """Run statistical tests for all metrics between all experiment pairs"""
        if len(self.experiments) < 2:
            print("Need at least 2 experiments to run statistical tests")
            return

        # Get all experiment pairs
        exp_pairs = []
        for i in range(len(self.experiments)):
            for j in range(i+1, len(self.experiments)):
                exp_pairs.append((self.experiment_names[i], self.experiment_names[j]))

        # Run tests for all metrics between all pairs
        for exp1_name, exp2_name in exp_pairs:
            if exp1_name not in self.per_image_data or exp2_name not in self.per_image_data:
                continue

            exp1_metrics = self.per_image_data[exp1_name]
            exp2_metrics = self.per_image_data[exp2_name]

            # Find common metrics
            common_metrics = set(exp1_metrics.keys()) & set(exp2_metrics.keys())

            # Initialize results for this pair
            pair_key = f"{exp1_name}_vs_{exp2_name}"
            self.test_results[pair_key] = {}
            self.effect_sizes[pair_key] = {}

            # Test each metric
            for metric in common_metrics:
                values1 = exp1_metrics[metric]
                values2 = exp2_metrics[metric]

                # Skip if insufficient data
                if len(values1) < 2 or len(values2) < 2:
                    continue

                # Check if this can be a paired test
                paired = self.paired
                if paired and len(values1) != len(values2):
                    print(f"Warning: Cannot use paired test for {metric} - unequal sample sizes")
                    paired = False

                # Run t-test
                try:
                    t_stat, p_value = self._run_t_test(values1, values2, paired)
                    self.test_results[pair_key][f"{metric}_t_test"] = {
                        'statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < self.alpha
                    }
                except Exception as e:
                    print(f"Warning: t-test failed for {metric}: {e}")

                # Run Mann-Whitney U test (non-parametric alternative)
                try:
                    u_stat, p_value = self._run_mann_whitney(values1, values2)
                    self.test_results[pair_key][f"{metric}_mann_whitney"] = {
                        'statistic': float(u_stat),
                        'p_value': float(p_value),
                        'significant': p_value < self.alpha
                    }
                except Exception as e:
                    print(f"Warning: Mann-Whitney test failed for {metric}: {e}")

                # Calculate effect size (Cohen's d)
                try:
                    effect_size = self._calculate_cohens_d(values1, values2)
                    self.effect_sizes[pair_key][metric] = float(effect_size)
                except Exception as e:
                    print(f"Warning: Effect size calculation failed for {metric}: {e}")

    def _run_t_test(self, values1, values2, paired=False):
        """Run t-test (paired or unpaired) between two sets of values"""
        if paired:
            return stats.ttest_rel(values1, values2)
        else:
            return stats.ttest_ind(values1, values2, equal_var=False)  # Welch's t-test

    def _run_mann_whitney(self, values1, values2):
        """Run Mann-Whitney U test between two sets of values"""
        return stats.mannwhitneyu(values1, values2, alternative='two-sided')

    def _calculate_cohens_d(self, values1, values2):
        """Calculate Cohen's d effect size"""
        n1, n2 = len(values1), len(values2)
        mean1, mean2 = np.mean(values1), np.mean(values2)
        var1, var2 = np.var(values1, ddof=1), np.var(values2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        # Cohen's d
        if pooled_std == 0:
            return 0  # No effect if no variation
        else:
            return (mean1 - mean2) / pooled_std

    def interpret_effect_size(self, d):
        """Interpret Cohen's d effect size"""
        d = abs(d)  # Use absolute value for interpretation
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    def create_summary_report(self):
        """Create a comprehensive summary of all statistical tests"""
        if not self.test_results:
            self.run_statistical_tests()

        # Calculate descriptive statistics if not already done
        if self.descriptive_stats is None:
            self.calculate_descriptive_statistics()

        # Create a summary of test results
        test_summary = {}
        for pair_key, tests in self.test_results.items():
            exp1, exp2 = pair_key.split('_vs_')

            if pair_key not in test_summary:
                test_summary[pair_key] = {}

            for test_key, result in tests.items():
                # Extract metric name and test type
                parts = test_key.rsplit('_', 2)
                if len(parts) >= 2:
                    metric = parts[0]
                    test_type = '_'.join(parts[1:])
                else:
                    metric = test_key
                    test_type = "unknown"

                # Get effect size
                effect_size = self.effect_sizes.get(pair_key, {}).get(metric, float('nan'))
                effect_interp = self.interpret_effect_size(effect_size) if not np.isnan(effect_size) else "unknown"

                # Get means for each experiment
                mean1 = np.mean(self.per_image_data[exp1].get(metric, [0]))
                mean2 = np.mean(self.per_image_data[exp2].get(metric, [0]))

                # Create metric entry if it doesn't exist
                if metric not in test_summary[pair_key]:
                    test_summary[pair_key][metric] = {}

                # Add test results
                test_summary[pair_key][metric][test_type] = {
                    'p_value': float(result['p_value']),
                    'significant': bool(result['significant']),
                    'statistic': float(result['statistic']),
                    'mean_1': float(mean1),
                    'mean_2': float(mean2),
                    'difference': float(mean1 - mean2),
                    'effect_size': float(effect_size) if not np.isnan(effect_size) else None,
                    'effect_interpretation': effect_interp
                }

        # Compile final results
        final_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'significance_level': self.alpha,
                'paired_tests': self.paired,
                'experiments': self.experiment_names
            },
            'descriptive_statistics': self.descriptive_stats,
            'test_results': test_summary
        }

        return final_results

    def save_results(self):
        """Save the results to a JSON file"""
        results = self.create_summary_report()

        with open(self.output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {self.output_file}")
        return self.output_file


def load_experiment_file(file_path):
    """Load an experiment results file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading experiment file {file_path}: {e}")
        return None


def extract_experiment_name(file_path):
    """Extract experiment name from file path"""
    path = Path(file_path)
    stem = path.stem

    # Remove common suffixes
    for suffix in ['_terrain_metrics', '_results', '_evaluation']:
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]

    return stem

def main():
    parser = argparse.ArgumentParser(description="Statistical significance testing for terrain generation experiments")
    parser.add_argument("--experiments", nargs='+', required=True, help="Paths to experiment JSON files")
    parser.add_argument("--names", nargs='+', help="Names for experiments (default: derived from filenames)")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    parser.add_argument("--significance-level", type=float, default=0.05, help="Significance level (default: 0.05)")
    parser.add_argument("--paired", action='store_true', help="Use paired tests where possible")

    args = parser.parse_args()

    # Validate arguments
    if args.names and len(args.names) != len(args.experiments):
        parser.error("Number of names must match number of experiments")

    # Load experiment data
    experiments = []
    experiment_names = []

    for i, exp_file in enumerate(args.experiments):
        exp_data = load_experiment_file(exp_file)
        if exp_data:
            experiments.append(exp_data)

            # Use provided name or extract from filename
            if args.names and i < len(args.names):
                name = args.names[i]
            else:
                name = extract_experiment_name(exp_file)

            experiment_names.append(name)
            print(f"Loaded experiment '{name}' from {exp_file}")

    if len(experiments) < 2:
        parser.error("Need at least 2 valid experiment files for comparison")

    # Create tester and run analysis
    tester = StatisticalTester(
        experiments=experiments,
        experiment_names=experiment_names,
        output_file=args.output,
        alpha=args.significance_level,
        paired=args.paired
    )

    tester.run_statistical_tests()
    tester.save_results()


if __name__ == "__main__":
    main()
