import random
import tempfile

import pytest
from mpi4py import MPI

from propulate import Propulator
from propulate.propagators import Conditional
from propulate.propagators.pso import BasicPSO, InitUniformPSO
from propulate.utils.benchmark_functions import sphere


@pytest.mark.mpi
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
