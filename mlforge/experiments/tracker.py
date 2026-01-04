import json
import os
import time
from pathlib import Path
from typing import Any, Dict
import pickle


class ExperimentTracker:
    """Simple experiment tracking with local storage."""

    def __init__(self, experiment_name: str, base_dir: str = "./mlforge_experiments"):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)

        # Create experiment directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.exp_dir = self.base_dir / experiment_name / timestamp
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.params = {}
        self.metrics = {}
        self.artifacts = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()

    def log_params(self, **params):
        """Log hyperparameters."""
        self.params.update(params)

    def log_metrics(self, **metrics):
        """Log metrics."""
        self.metrics.update(metrics)

    def log_artifact(self, filepath: str):
        """Log artifact file."""
        import shutil
        dest = self.exp_dir / "artifacts" / Path(filepath).name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy(filepath, dest)
        self.artifacts.append(str(dest))

    def log_model(self, model, name: str):
        """Save model."""
        model_path = self.exp_dir / f"{name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        self.artifacts.append(str(model_path))

    def save(self):
        """Save experiment metadata."""
        metadata = {
            'experiment_name': self.experiment_name,
            'timestamp': time.time(),
            'params': self.params,
            'metrics': self.metrics,
            'artifacts': self.artifacts
        }

        with open(self.exp_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Experiment saved to: {self.exp_dir}")
