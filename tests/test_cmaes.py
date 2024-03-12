import random
import tempfile
from typing import Dict

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
    params : Dict[str, float]
        The function parameters.

    Returns
    -------
    float
        The function value.
    """
    return np.sum(np.array(list(params.values())) ** 2).item()


def test_cmaes():
    """Test single worker using Propulator to optimize sphere using a CMA-ES propagator."""
    rng = random.Random(42)  # Separate random number generator for optimization.
    limits = {
        "a": (-5.12, 5.12),
        "b": (-5.12, 5.12),
    }
    with tempfile.TemporaryDirectory() as checkpoint_path:
        # Set up evolutionary operator.
        adapter = BasicCMA()
        propagator = CMAPropagator(adapter, limits, rng=rng)

        # Set up Propulator performing actual optimization.
        propulator = Propulator(
            loss_fn=sphere,
            propagator=propagator,
            rng=rng,
            generations=100,
            checkpoint_path=checkpoint_path,
        )
        # Run optimization and print summary of results.
        propulator.propulate()
        best = propulator.summarize(top_n=1, debug=2)
        assert best[0][0].loss < 1
