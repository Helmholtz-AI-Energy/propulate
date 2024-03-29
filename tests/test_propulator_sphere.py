import random
import tempfile
from typing import Dict
import logging

import numpy as np

from propulate import Propulator
from propulate.utils import get_default_propagator, set_logger_config


def sphere(params: Dict[str, float]) -> float:
    """
    Sphere function: continuous, convex, separable, differentiable, unimodal

    Input domain: -5.12 <= x, y <= 5.12
    Global minimum 0 at (x, y) = (0, 0)

    Parameters
    ----------
    params : Dict[str, float]
        The function parameters.

    Returns
    -------
    float
        The function value.
    """
    return np.sum(np.array(list(params.values())) ** 2).item()


def test_propulator():
    """Test single worker using Propulator to optimize sphere."""
    rng = random.Random(42)  # Separate random number generator for optimization.
    limits = {
        "a": (-5.12, 5.12),
        "b": (-5.12, 5.12),
    }
    with tempfile.TemporaryDirectory() as checkpoint_path:
        set_logger_config(
            level=logging.INFO,
            log_file=checkpoint_path + "/propulate.log",
            log_to_stdout=True,
            log_rank=False,
            colors=True,
        )
        # Set up evolutionary operator.
        propagator = get_default_propagator(
            pop_size=4,
            limits=limits,
            crossover_prob=0.7,
            mutation_prob=9.0,
            random_init_prob=0.1,
            rng=rng,
        )

        # Set up propulator performing actual optimization.
        propulator = Propulator(
            loss_fn=sphere,
            propagator=propagator,
            rng=rng,
            generations=100,
            checkpoint_path=checkpoint_path,
        )

        # Run optimization and print summary of results.
        propulator.propulate()
        best = propulator.summarize()

        assert best[0][0].loss < 0.8
