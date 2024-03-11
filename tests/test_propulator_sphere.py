import random
import tempfile
from typing import Dict
from operator import attrgetter
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
    params: dict[str, float]
            function parameters
    Returns
    -------
    float
        function value
    """
    return np.sum(np.array(list(params.values())) ** 2)


def test_Propulator():
    """
    Test single worker using Propulator to optimize sphere.
    """
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
        propagator = get_default_propagator(  # Get default evolutionary operator.
            pop_size=4,  # Breeding pool size
            limits=limits,  # Search-space limits
            mate_prob=0.7,  # Crossover probability
            mut_prob=9.0,  # Mutation probability
            random_prob=0.1,  # Random-initialization probability
            rng=rng,  # Random number generator
        )

        # Set up propulator performing actual optimization.
        propulator = Propulator(
            loss_fn=sphere,
            propagator=propagator,
            rng=rng,
            generations=10,
            checkpoint_path=checkpoint_path,
        )

        # Run optimization and print summary of results.
        propulator.propulate()
        propulator.summarize()
        best = min(propulator.population, key=attrgetter("loss"))

        assert best.loss < 0.8
