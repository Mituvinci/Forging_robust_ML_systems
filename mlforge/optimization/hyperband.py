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
