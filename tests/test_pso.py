import pathlib
import random

import pytest
from mpi4py import MPI

from propulate import Propulator
from propulate.propagators import Conditional, Propagator
from propulate.propagators.pso import (
    BasicPSO,
    CanonicalPSO,
    ConstrictionPSO,
    InitUniformPSO,
    VelocityClampingPSO,
)
from propulate.utils.benchmark_functions import get_function_search_space, sphere

limits = get_function_search_space("sphere")[1]
rank = MPI.COMM_WORLD.rank
rng = random.Random(42 + rank)


@pytest.fixture(
    params=[
        BasicPSO(
            inertia=0.729,
            c_cognitive=1.49445,
            c_social=1.49445,
            rank=rank,
            limits=limits,
            rng=rng,
        ),
        VelocityClampingPSO(
            inertia=0.729,
            c_cognitive=1.49445,
            c_social=1.49445,
            rank=rank,
            limits=limits,
            rng=rng,
            v_limits=0.6,
        ),
        ConstrictionPSO(
            c_cognitive=2.05,
            c_social=2.05,
            rank=rank,
            limits=limits,
            rng=rng,
        ),
        CanonicalPSO(
            c_cognitive=2.05, c_social=2.05, rank=rank, limits=limits, rng=rng
        ),
    ]
)
def pso_propagator(request: pytest.FixtureRequest) -> Propagator:
    """Iterate over PSO propagator variants."""
    return request.param


@pytest.mark.mpi
def test_pso(pso_propagator: Propagator, mpi_tmp_path: pathlib.Path) -> None:
    """
    Test single worker using Propulator to optimize a benchmark function using the default genetic propagator.

    Parameters
    ----------
    pso_propagator : BasicPSO
        The PSO propagator variant to test.
    mpi_tmp_path : pathlib.Path
        The temporary checkpoint directory.
    """
    # Set up evolutionary operator.
    init = InitUniformPSO(limits, rng=rng, rank=rank)
    propagator = Conditional(1, pso_propagator, init)

    # Set up propulator performing actual optimization.
    propulator = Propulator(
        loss_fn=sphere,
        propagator=propagator,
        rng=rng,
        generations=100,
        checkpoint_path=mpi_tmp_path,
    )

    # Run optimization and print summary of results.
    propulator.propulate()
