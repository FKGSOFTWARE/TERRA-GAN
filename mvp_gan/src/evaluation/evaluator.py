# Add at top of evaluator.py
from typing import Dict
from pathlib import Path
import numpy as np
import torch.nn as nn
from ..utils.human_guided_dataset import HumanGuidedDataset
from .metrics import MaskEvaluator

class MetricsAggregator:
    def __init__(self):
        self.metrics = []

    def add(self, metric):
        self.metrics.append(metric)

    def get_summary(self) -> Dict[str, float]:
        return {
            "mean_iou": np.mean([m.iou for m in self.metrics]),
            "mean_precision": np.mean([m.precision for m in self.metrics]),
            "mean_recall": np.mean([m.recall for m in self.metrics])
        }

class GANEvaluator:
    def __init__(self, config: Dict):
        self.config = config
        self.metrics = MaskEvaluator(config)
        self.results_dir = Path(config['data']['evaluation_results_dir'])

    def evaluate_model(self,
                      generator: nn.Module,
                      eval_dataset: HumanGuidedDataset) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        metrics_aggregator = MetricsAggregator()

        for batch in eval_dataset:
            # Generate inpainted image
            output = generator(batch['image'], batch['mask'])

            # Calculate metrics
            if batch.get('human_mask') is not None:
                metrics = self.metrics.calculate_metrics(
                    output.cpu().numpy(),
                    batch['human_mask'].numpy()
                )
                metrics_aggregator.add(metrics)

        return metrics_aggregator.get_summary()
