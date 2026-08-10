import copy
import pathlib
import random

import deepdiff
import numpy as np
import pytest
from mpi4py import MPI

from propulate import Propulator
from propulate.population import Individual
from propulate.propagators.base import Propagator
from propulate.utils import get_default_propagator, set_logger_config
from propulate.utils.benchmark_functions import get_function_search_space


@pytest.fixture(
    params=[
        "rosenbrock",
        "step",
        "quartic",
        "rastrigin",
        "griewank",
        "schwefel",
        "bisphere",
        "birastrigin",
        "bukin",
        "eggcrate",
        "himmelblau",
        "keane",
        "leon",
        "sphere",
    ]
)
def function_name(request: pytest.FixtureRequest) -> str:
    """Define benchmark function parameter sets as used in tests."""
    return request.param


def test_propulator(function_name: str, mpi_tmp_path: pathlib.Path) -> None:
    """
    Test standard Propulator to optimize the benchmark functions using the default genetic propagator.

    This test is run both sequentially and in parallel.

    Parameters
    ----------
    function_name : str
        The function name.
    mpi_tmp_path : pathlib.Path
        The temporary checkpoint directory.
    """
    rng = random.Random(42 + MPI.COMM_WORLD.rank)  # Random number generator for optimization
    benchmark_function, limits = get_function_search_space(function_name)
    set_logger_config(log_file=mpi_tmp_path / "log.log")
    propagator = get_default_propagator(
        pop_size=4,
        limits=limits,
        rng=rng,
    )  # Set up evolutionary operator.
    propulator = Propulator(
        loss_fn=benchmark_function,
        propagator=propagator,
        rng=rng,
        generations=10,
        checkpoint_path=mpi_tmp_path,
    )  # Set up propulator performing actual optimization.
    propulator.propulate()  # Run optimization and print summary of results.


def test_propulator_checkpointing(mpi_tmp_path: pathlib.Path) -> None:
    """
    Test standard Propulator checkpointing for the sphere benchmark function.

    This test is run both sequentially and in parallel.

    Parameters
    ----------
    mpi_tmp_path : pathlib.Path
        The temporary checkpoint directory.
    """
    rng = random.Random(42 + MPI.COMM_WORLD.rank)  # Separate random number generator for optimization
    benchmark_function, limits = get_function_search_space("sphere")

    propagator = get_default_propagator(  # Get default evolutionary operator.
        pop_size=4,  # Breeding pool size
        limits=limits,  # Search-space limits
        rng=rng,  # Random number generator
    )
    propulator = Propulator(
        loss_fn=benchmark_function,
        propagator=propagator,
        generations=10,
        checkpoint_path=mpi_tmp_path,
        rng=rng,
    )  # Set up propulator performing actual optimization.

    propulator.propulate()  # Run optimization and print summary of results.

    old_population = copy.deepcopy(propulator.population)  # Save population list from the last run.
    del propulator  # Delete propulator object.
    MPI.COMM_WORLD.barrier()  # Synchronize all processes.

    propulator = Propulator(
        loss_fn=benchmark_function,
        propagator=propagator,
        generations=5,
        checkpoint_path=mpi_tmp_path,
        rng=rng,
    )  # Set up new propulator starting from checkpoint.

    # As the number of requested generations is smaller than the number of generations from the run before,
    # no new evaluations are performed. Thus, the length of both Propulators' populations must be equal.
    assert len(deepdiff.DeepDiff(old_population, propulator.population, ignore_order=True)) == 0


class _RankTrackingPropagator(Propagator):
    """Minimal propagator that records rank context set by Propulator before breeding."""

    def __init__(self, limits):
        super().__init__(parents=-1, offspring=1, rng=random.Random(0))
        self.limits = limits
        self.rank = -999
        self.world_size = -999
        self.seen_rank = None
        self.seen_world_size = None
        self.context_calls = 0

    def set_worker_context(self, worker_rank: int, worker_size: int) -> None:
        self.context_calls += 1
        self.rank = worker_rank
        self.world_size = worker_size

    def __call__(self, inds):  # type: ignore[override]
        self.seen_rank = self.rank
        self.seen_world_size = self.world_size
        return Individual(np.array([0.0, 0.0], dtype=float), self.limits, generation=0, rank=self.rank)


def test_propulator_sets_propagator_worker_context(mpi_tmp_path: pathlib.Path) -> None:
    """Propulator should set worker context via propagator API during initialization."""
    _, limits = get_function_search_space("sphere")
    propagator = _RankTrackingPropagator(limits)
    rng = random.Random(42 + MPI.COMM_WORLD.rank)
    set_logger_config(log_file=mpi_tmp_path / "log_rank_context.log")

    propulator = Propulator(
        loss_fn=lambda p: float(np.sum(np.array(list(p.values())) ** 2)),
        propagator=propagator,
        rng=rng,
        generations=1,
        checkpoint_path=mpi_tmp_path,
    )

    assert propagator.context_calls == 1
    assert propagator.rank == propulator.island_comm.rank
    assert propagator.world_size == propulator.island_comm.size

    _ = propulator._breed()
    assert propagator.seen_rank == propulator.island_comm.rank
    assert propagator.seen_world_size == propulator.island_comm.size
