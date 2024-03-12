import random
import tempfile
from typing import Dict

from mpi4py import MPI
import numpy as np

from propulate import Propulator
from propulate.propagators import Conditional
from propulate.propagators.pso import BasicPSO, InitUniformPSO


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


def test_pso():
    """Test single worker using Propulator to optimize sphere using a PSO propagator."""
    rng = random.Random(42)  # Separate random number generator for optimization.
    limits = {
        "a": (-5.12, 5.12),
        "b": (-5.12, 5.12),
    }
    with tempfile.TemporaryDirectory() as checkpoint_path:
        # Set up evolutionary operator.
        pso_propagator = BasicPSO(
            inertia=0.729,
            c_cognitive=1.49334,
            c_social=1.49445,
            rank=MPI.COMM_WORLD.rank,  # MPI rank TODO fix when implemented proper MPI parallel tests
            limits=limits,
            rng=rng,
        )
        init = InitUniformPSO(limits, rng=rng, rank=MPI.COMM_WORLD.rank)
        propagator = Conditional(1, pso_propagator, init)  # TODO MPIify

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
        best = propulator.summarize(top_n=1, debug=2)
        assert best[0][0].loss < 30.0
