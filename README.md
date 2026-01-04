# MLForge

Production-ready ML infrastructure framework for hyperparameter optimization, experiment tracking, and model monitoring.

## Overview

MLForge is a lightweight framework designed to bring production ML best practices to your machine learning projects. It provides essential tools for:

- **Hyperparameter Optimization**: Efficient search using Hyperband algorithm
- **Experiment Tracking**: Track parameters, metrics, and artifacts
- **Model Monitoring**: Data drift detection and performance monitoring (coming soon)
- **Feature Engineering**: Configurable pipelines and transformers (coming soon)

## Features

### Current Features

- **Hyperband Optimizer**: State-of-the-art hyperparameter optimization using successive halving
- **Experiment Tracker**: Simple, local-first experiment tracking with metadata storage
- **Modular Architecture**: Clean, extensible design for easy customization

### Roadmap

- [ ] Data drift detection
- [ ] Model performance monitoring
- [ ] Feature engineering pipelines
- [ ] Model registry and versioning
- [ ] CLI tools
- [ ] Integration with cloud storage (S3, GCS)

## Installation

### From Source

```bash
git clone https://github.com/Mituvinci/Forging_robust_ML_systems.git
cd Forging_robust_ML_systems
pip install -e .
```

### Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from mlforge.optimization import HyperbandOptimizer
from mlforge.experiments import ExperimentTracker

# Define your training function
def train_model(config, num_iters):
    # Your training code here
    # config contains hyperparameters
    # num_iters is the number of training epochs
    return validation_accuracy

# Define search space
search_space = {
    'lr': [1e-4, 1e-3, 1e-2],
    'batch_size': [32, 64, 128],
    'hidden_size': [64, 128, 256]
}

# Initialize optimizer
optimizer = HyperbandOptimizer(max_iter=27, eta=3)

# Track experiment
with ExperimentTracker('my-experiment') as exp:
    # Run optimization
    best_config = optimizer.optimize(
        model_fn=train_model,
        search_space=search_space,
        metric='accuracy',
        mode='max'
    )

    # Log results
    exp.log_params(**best_config)
    exp.log_metrics(best_accuracy=0.95)
```

## Examples

Check out the `examples/` directory for complete working examples:

- `quickstart.py`: Basic usage with a simple neural network

Run the quickstart example:

```bash
python examples/quickstart.py
```

## Architecture

```
mlforge/
├── optimization/       # Hyperparameter optimization
│   └── hyperband.py   # Hyperband implementation
├── experiments/       # Experiment tracking
│   └── tracker.py     # Experiment tracker
├── monitoring/        # Model monitoring (coming soon)
├── features/          # Feature engineering (coming soon)
└── registry/          # Model registry (coming soon)
```

## Documentation

### Hyperband Optimizer

The `HyperbandOptimizer` implements the Hyperband algorithm for efficient hyperparameter search.

**Parameters:**
- `max_iter` (int): Maximum iterations per configuration (default: 81)
- `eta` (int): Reduction factor for successive halving (default: 3)
- `resource_attr` (str): Name of resource attribute (default: 'epoch')

**Methods:**
- `optimize(model_fn, search_space, metric, mode)`: Run optimization

### Experiment Tracker

The `ExperimentTracker` provides simple experiment tracking with local storage.

**Parameters:**
- `experiment_name` (str): Name of the experiment
- `base_dir` (str): Base directory for storing experiments (default: './mlforge_experiments')

**Methods:**
- `log_params(**params)`: Log hyperparameters
- `log_metrics(**metrics)`: Log metrics
- `log_artifact(filepath)`: Log artifact file
- `log_model(model, name)`: Save model
- `save()`: Save experiment metadata

## Design Principles

MLForge is built on production ML best practices:

1. **Iterative Development**: Start simple, add complexity as needed
2. **Monitoring First**: Built-in observability
3. **Reproducibility**: Version everything
4. **Data-Centric**: Focus on data quality
5. **Modular Design**: Easy to extend and customize

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Acknowledgments

This project is inspired by:
- "Designing Machine Learning Systems" by Chip Huyen
- AWS SageMaker architecture patterns
- MLflow and other experiment tracking tools

## Author

Halima Akhter
