import json
from pathlib import Path
from typing import Dict


class ResultsManager:
    def __init__(self, config: Dict):
        self.results_dir = Path(config['data']['evaluation_results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_evaluation_results(self, results: Dict, experiment_name: str):
        """Save evaluation results with metadata"""
        results_path = self.results_dir / f"{experiment_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

    def load_results(self, experiment_name: str) -> Dict:
        """Load previous evaluation results"""
        results_path = self.results_dir / f"{experiment_name}_results.json"
        with open(results_path, 'r') as f:
            return json.load(f)
