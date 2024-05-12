import copy
import logging
import pathlib
import random

import deepdiff
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
from propulate.utils import set_logger_config
from propulate.utils.benchmark_functions import get_function_search_space, sphere

log = logging.getLogger("propulate")  # Get logger instance.
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
        CanonicalPSO(c_cognitive=2.05, c_social=2.05, rank=rank, limits=limits, rng=rng),
    ]
)
def pso_propagator(request: pytest.FixtureRequest) -> Propagator:
    """Iterate over PSO propagator variants."""
    return request.param


@pytest.mark.mpi
def test_pso(pso_propagator: Propagator, mpi_tmp_path: pathlib.Path) -> None:
    """
    Test a pso propagator.

    Parameters
    ----------
    pso_propagator : BasicPSO
        The PSO propagator variant to test.
    mpi_tmp_path : pathlib.Path
        The temporary checkpoint directory.
    """
    set_logger_config()
    # Set up pso propagator.
    init = InitUniformPSO(limits, rng=rng, rank=rank)
    propagator = Conditional(limits, 1, pso_propagator, init)

    # Set up propulator performing actual optimization.
    propulator = Propulator(
        loss_fn=sphere,
        propagator=propagator,
        rng=rng,
        generations=10,
        checkpoint_path=mpi_tmp_path,
    )

    # Run optimization and print summary of results.
    propulator.propulate()
    log.handlers.clear()


@pytest.mark.mpi
def test_pso_checkpointing(pso_propagator, mpi_tmp_path: pathlib.Path):
    """
    Test velocity checkpointing when using a PSO propagator.

    Parameters
    ----------
    pso_propagator : BasicPSO
        The PSO propagator variant to test.
    mpi_tmp_path : pathlib.Path
        The temporary checkpoint directory.
    """
    set_logger_config()
    # Set up pso propagator.
    init = InitUniformPSO(limits, rng=rng, rank=rank)
    propagator = Conditional(limits, 1, pso_propagator, init)

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

    old_population = copy.deepcopy(
        propulator.population
    )  # Save population list from the last run.
    del propulator  # Delete propulator object.
    MPI.COMM_WORLD.barrier()  # Synchronize all processes.

    propulator = Propulator(
        loss_fn=sphere,
        propagator=propagator,
        generations=20,
        checkpoint_path=mpi_tmp_path,
        rng=rng,
    )  # Set up new propulator starting from checkpoint.

    # As the number of requested generations is smaller than the number of generations from the run before,
    # no new evaluations are performed. Thus, the length of both Propulators' populations must be equal.
    assert (
        len(deepdiff.DeepDiff(old_population, propulator.population, ignore_order=True))
        == 0
    )
    log.handlers.clear()


# TODO test resuming pso run from a non-pso checkpoint
