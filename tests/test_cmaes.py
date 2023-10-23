import random
import tempfile
from typing import Dict
from operator import attrgetter

import numpy as np

from propulate import Propulator
from propulate.propagators import CMAPropagator, BasicCMA


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


def test_PSO():
    """
    Test single worker using Propulator to optimize sphere using a PSO propagator.
    """
    return
    rng = random.Random(42)  # Separate random number generator for optimization.
    limits = {
        "a": (-5.12, 5.12),
        "b": (-5.12, 5.12),
    }
    with tempfile.TemporaryDirectory() as checkpoint_path:
        # Set up evolutionary operator.

        adapter = BasicCMA()
        propagator = CMAPropagator(adapter, limits, rng=rng)

        # Set up propulator performing actual optimization.
        propulator = Propulator(
            loss_fn=sphere,
            propagator=propagator,
            generations=10,
            checkpoint_path=checkpoint_path,
            rng=rng,
        )

        # Run optimization and print summary of results.
        propulator.propulate()
        propulator.summarize()
        best = min(propulator.population, key=attrgetter("loss"))

        assert best.loss < 10.0
