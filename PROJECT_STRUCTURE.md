# MLForge Project Structure

## 📁 Recommended Directory Layout

```
mlforge/
├── README.md                    # Main documentation (can use agents.md content)
├── setup.py                     # Package installation
├── requirements.txt             # Dependencies
├── LICENSE                      # MIT License
├── .gitignore                   # Git ignore file
│
├── mlforge/                     # Main package
│   ├── __init__.py
│   │
│   ├── optimization/            # Hyperparameter optimization
│   │   ├── __init__.py
│   │   ├── hyperband.py        # Hyperband implementation
│   │   ├── bayesian.py         # Optuna wrapper (optional)
│   │   └── base.py             # Base optimizer class
│   │
│   ├── experiments/             # Experiment tracking
│   │   ├── __init__.py
│   │   ├── tracker.py          # Main tracking class
│   │   ├── storage.py          # Backend storage (local/S3)
│   │   └── comparison.py       # Experiment comparison tools
│   │
│   ├── monitoring/              # Model monitoring
│   │   ├── __init__.py
│   │   ├── drift_detector.py   # Data drift detection
│   │   ├── performance.py      # Performance monitoring
│   │   ├── alerts.py           # Alert system
│   │   └── visualizations.py   # Monitoring dashboards
│   │
│   ├── features/                # Feature engineering
│   │   ├── __init__.py
│   │   ├── pipeline.py         # Feature pipeline
│   │   ├── transformers.py     # Custom transformers
│   │   ├── validation.py       # Data validation
│   │   └── config_parser.py    # YAML config parser
│   │
│   ├── registry/                # Model registry
│   │   ├── __init__.py
│   │   ├── model_store.py      # Model storage
│   │   ├── versioning.py       # Version management
│   │   └── deployment.py       # Deployment helpers
│   │
│   ├── utils/                   # Utilities
│   │   ├── __init__.py
│   │   ├── logging.py          # Logging configuration
│   │   ├── config.py           # Config management
│   │   └── metrics.py          # Metric calculations
│   │
│   └── cli/                     # Command-line interface
│       ├── __init__.py
│       └── main.py             # CLI commands
│
├── tests/                       # Unit tests
│   ├── __init__.py
│   ├── test_optimization/
│   ├── test_monitoring/
│   ├── test_features/
│   └── test_registry/
│
├── examples/                    # Usage examples
│   ├── quickstart.ipynb
│   ├── image_classification/
│   ├── nlp_sentiment/
│   └── tabular_prediction/
│
├── docs/                        # Documentation
│   ├── getting_started.md
│   ├── api_reference.md
│   ├── tutorials/
│   └── design_decisions.md
│
└── configs/                     # Example configs
    ├── features.yaml
    ├── monitoring.yaml
    └── optimization.yaml
```

## 🎯 Implementation Priority

### Phase 1: Core Foundation (Week 1-2)
**Priority: HIGH** - These are essential for basic functionality

1. **Basic Project Setup**
   ```bash
   mkdir mlforge
   cd mlforge
   git init
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   ```

2. **Package Structure**
   - Create `setup.py` with basic dependencies
   - Create `mlforge/__init__.py`
   - Create `requirements.txt`

3. **Hyperband Optimizer** (`mlforge/optimization/hyperband.py`)
   - Core algorithm implementation
   - Integration with PyTorch/TensorFlow
   - Simple API

### Phase 2: Experiment Tracking (Week 2-3)
**Priority: HIGH** - Critical for portfolio demonstration

1. **Experiment Tracker** (`mlforge/experiments/tracker.py`)
   - Log parameters, metrics, artifacts
   - Local storage backend (SQLite + file system)
   - Simple comparison functions

2. **Reproducibility Features**
   - Random seed management
   - Environment snapshot (pip freeze)
   - Configuration versioning

### Phase 3: Monitoring & Validation (Week 3-4)
**Priority: MEDIUM** - Shows production awareness

1. **Data Drift Detection** (`mlforge/monitoring/drift_detector.py`)
   - PSI (Population Stability Index)
   - KS test (Kolmogorov-Smirnov)
   - Simple alerting

2. **Model Performance Monitoring** (`mlforge/monitoring/performance.py`)
   - Metric tracking
   - Threshold-based alerts
   - Basic visualizations

### Phase 4: Feature Engineering (Week 4-5)
**Priority: MEDIUM** - Demonstrates pipeline thinking

1. **Feature Pipeline** (`mlforge/features/pipeline.py`)
   - YAML configuration support
   - Standard transformers (scaling, encoding)
   - sklearn compatibility

2. **Data Validation** (`mlforge/features/validation.py`)
   - Schema validation
   - Statistical profiling
   - Anomaly detection

### Phase 5: Polish & Documentation (Week 5-6)
**Priority: HIGH** - Critical for portfolio

1. **Examples** (`examples/`)
   - Image classification tutorial
   - NLP sentiment analysis
   - Tabular data prediction

2. **Documentation**
   - Comprehensive README
   - API documentation (Sphinx)
   - Usage tutorials

3. **Tests**
   - Unit tests for core components
   - Integration tests
   - CI/CD setup (GitHub Actions)

## 🛠️ Quick Start Implementation

### Step 1: Create `setup.py`

```python
from setuptools import setup, find_packages

setup(
    name="mlforge",
    version="0.1.0",
    description="Production-ready ML infrastructure framework",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "torch>=1.10.0",
        "pyyaml>=5.4.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "monitoring": [
            "plotly>=5.0.0",
            "dash>=2.0.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
```

### Step 2: Minimal Hyperband Implementation

```python
# mlforge/optimization/hyperband.py

import numpy as np
from typing import Callable, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class HyperbandOptimizer:
    """
    Hyperband hyperparameter optimization algorithm.
    
    Based on: "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization"
    Li et al., 2018
    
    Args:
        max_iter: Maximum iterations per configuration
        eta: Reduction factor for successive halving
        resource_attr: Name of resource attribute (e.g., 'epoch', 'num_samples')
    """
    
    def __init__(self, max_iter: int = 81, eta: int = 3, resource_attr: str = 'epoch'):
        self.max_iter = max_iter
        self.eta = eta
        self.resource_attr = resource_attr
        
        # Calculate number of brackets
        self.s_max = int(np.floor(np.log(max_iter) / np.log(eta)))
        self.B = (self.s_max + 1) * max_iter
        
        logger.info(f"Initialized Hyperband: max_iter={max_iter}, eta={eta}, s_max={self.s_max}")
    
    def optimize(
        self,
        model_fn: Callable,
        search_space: Dict[str, List[Any]],
        metric: str = 'accuracy',
        mode: str = 'max',
        num_samples: int = None
    ) -> Dict[str, Any]:
        """
        Run Hyperband optimization.
        
        Args:
            model_fn: Training function that takes (config, num_iters) and returns metric
            search_space: Dictionary of hyperparameter options
            metric: Metric to optimize
            mode: 'max' or 'min'
            num_samples: Total number of configurations to try
        
        Returns:
            Best configuration found
        """
        all_results = []
        
        for s in reversed(range(self.s_max + 1)):
            n = int(np.ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))
            r = self.max_iter * self.eta ** (-s)
            
            logger.info(f"Bracket s={s}: n={n} configs, r={r} initial iterations")
            
            # Generate random configurations
            configs = [self._sample_config(search_space) for _ in range(n)]
            
            # Successive halving
            for i in range(s + 1):
                n_i = int(n * self.eta ** (-i))
                r_i = int(r * self.eta ** i)
                
                logger.info(f"  Round {i}: {n_i} configs, {r_i} iterations each")
                
                # Evaluate configurations
                results = []
                for config in configs[:n_i]:
                    score = model_fn(config, num_iters=r_i)
                    results.append({
                        'config': config,
                        'score': score,
                        'iterations': r_i
                    })
                    all_results.append(results[-1])
                
                # Sort and keep top performers
                results = sorted(
                    results,
                    key=lambda x: x['score'],
                    reverse=(mode == 'max')
                )
                configs = [r['config'] for r in results]
        
        # Return best overall configuration
        best = max(all_results, key=lambda x: x['score']) if mode == 'max' \
               else min(all_results, key=lambda x: x['score'])
        
        logger.info(f"Best score: {best['score']:.4f}")
        return best['config']
    
    def _sample_config(self, search_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Randomly sample a configuration from search space."""
        return {
            key: np.random.choice(values)
            for key, values in search_space.items()
        }
```

### Step 3: Simple Experiment Tracker

```python
# mlforge/experiments/tracker.py

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
```

## 📝 Usage Example

```python
# example_usage.py

from mlforge.optimization import HyperbandOptimizer
from mlforge.experiments import ExperimentTracker
import torch
import torch.nn as nn

# Define model training function
def train_model(config, num_iters):
    # Your training code here
    model = nn.Linear(10, 2)
    # ... training loop for num_iters epochs ...
    return validation_accuracy

# Hyperparameter search
search_space = {
    'lr': [1e-4, 1e-3, 1e-2],
    'batch_size': [32, 64, 128],
    'hidden_size': [64, 128, 256]
}

optimizer = HyperbandOptimizer(max_iter=27, eta=3)

# Track experiment
with ExperimentTracker('my-experiment') as exp:
    best_config = optimizer.optimize(
        model_fn=train_model,
        search_space=search_space,
        metric='accuracy',
        mode='max'
    )
    
    exp.log_params(**best_config)
    exp.log_metrics(best_accuracy=0.95)
    exp.log_model(final_model, 'model_v1')
```

## 🎓 Learning Resources

### AWS SageMaker Concepts to Incorporate

1. **Processing Jobs** → Feature engineering pipelines
2. **Training Jobs** → Orchestrated training with HPO
3. **Model Registry** → Version management
4. **Model Monitor** → Drift detection
5. **Experiments** → Tracking and comparison
6. **Clarify** → Bias detection (future enhancement)

### Chip Huyen's Key Principles

1. **Iterative Development**: Start simple, add complexity
2. **Monitoring First**: Built-in observability
3. **Reproducibility**: Version everything
4. **Data-Centric**: Focus on data quality
5. **Fail-Safe**: Graceful degradation

## 📊 Portfolio Impact

This project demonstrates:

✅ **Production ML Skills**: Monitoring, HPO, experiment tracking  
✅ **Software Engineering**: Clean architecture, modular design  
✅ **AWS Knowledge**: SageMaker patterns, cloud-native thinking  
✅ **ML Systems Design**: Chip Huyen's principles applied  
✅ **Open Source**: Community contribution, documentation

**Perfect for**: Google, Meta, Microsoft, Amazon ML Engineer roles

## 🚀 Next Steps

1. **Start with Phase 1**: Get basic structure working
2. **Add one component at a time**: Don't try to build everything at once
3. **Write tests early**: Makes refactoring easier
4. **Document as you go**: Future you will thank present you
5. **Create examples**: Best way to validate your API design
6. **Get feedback**: Share early versions with peers/mentors

Good luck with your framework! 🎉
