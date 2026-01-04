"""
Quickstart example for MLForge

This example demonstrates how to use MLForge for hyperparameter optimization
and experiment tracking with a simple neural network.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from mlforge.optimization import HyperbandOptimizer
from mlforge.experiments import ExperimentTracker
import numpy as np


# Simple neural network model
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Generate synthetic dataset
def generate_data(n_samples=1000, input_size=10, output_size=2):
    X = torch.randn(n_samples, input_size)
    y = torch.randint(0, output_size, (n_samples,))
    return TensorDataset(X, y)


# Training function for Hyperband
def train_model(config, num_iters):
    """
    Train a model with the given configuration for num_iters epochs.

    Args:
        config: Dictionary with hyperparameters (lr, batch_size, hidden_size)
        num_iters: Number of training epochs

    Returns:
        Validation accuracy
    """
    # Extract hyperparameters
    lr = config['lr']
    batch_size = config['batch_size']
    hidden_size = config['hidden_size']

    # Create datasets
    train_dataset = generate_data(n_samples=800)
    val_dataset = generate_data(n_samples=200)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model
    model = SimpleNet(input_size=10, hidden_size=hidden_size, output_size=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_iters):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    accuracy = correct / total
    return accuracy


def main():
    print("MLForge Quickstart Example")
    print("=" * 50)

    # Define search space
    search_space = {
        'lr': [1e-4, 1e-3, 1e-2],
        'batch_size': [32, 64, 128],
        'hidden_size': [64, 128, 256]
    }

    print("\nSearch space:")
    for key, values in search_space.items():
        print(f"  {key}: {values}")

    # Initialize Hyperband optimizer
    print("\nInitializing Hyperband optimizer...")
    optimizer = HyperbandOptimizer(max_iter=27, eta=3)

    # Run optimization with experiment tracking
    print("\nRunning hyperparameter optimization...")
    with ExperimentTracker('quickstart-example') as exp:
        best_config = optimizer.optimize(
            model_fn=train_model,
            search_space=search_space,
            metric='accuracy',
            mode='max'
        )

        print("\nBest configuration found:")
        for key, value in best_config.items():
            print(f"  {key}: {value}")

        # Train final model with best config
        print("\nTraining final model with best configuration...")
        final_accuracy = train_model(best_config, num_iters=27)

        # Log to experiment tracker
        exp.log_params(**best_config)
        exp.log_metrics(best_accuracy=final_accuracy)

        print(f"\nFinal model accuracy: {final_accuracy:.4f}")
        print(f"Experiment saved to: {exp.exp_dir}")


if __name__ == "__main__":
    main()
