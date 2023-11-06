import random
import tempfile
from operator import attrgetter

import pytest

from propulate import Propulator
from propulate.propagators import CMAPropagator, BasicCMA
from propulate.utils import sphere


@pytest.mark.mpi_skip
def test_PSO():
    """
    Test single worker using Propulator to optimize sphere using a PSO propagator.
    """
    rng = random.Random(42)  # Separate random number generator for optimization.
    limits = {
        "a": (-5.12, 5.12),
        "b": (-5.12, 5.12),
    }
    with tempfile.TemporaryDirectory() as checkpoint_directory:
        # Set up evolutionary operator.

        adapter = BasicCMA()
        propagator = CMAPropagator(adapter, limits, rng=rng)

        # Set up propulator performing actual optimization.
        propulator = Propulator(
            loss_fn=sphere,
            propagator=propagator,
            generations=10,
            checkpoint_directory=checkpoint_directory,
            rng=rng,
        )

        # Run optimization and print summary of results.
        propulator.propulate()
        propulator.summarize()
        best = min(propulator.population, key=attrgetter("loss"))

        assert best.loss < 10.0
