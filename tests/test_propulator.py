import copy
import logging
import pathlib
import random

import deepdiff
import pytest
from mpi4py import MPI

from propulate import Propulator
from propulate.utils import get_default_propagator, set_logger_config
from propulate.utils.benchmark_functions import get_function_search_space

log = logging.getLogger("propulate")  # Get logger instance.


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
    set_logger_config()
    benchmark_function, limits = get_function_search_space(function_name)
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
    log.handlers.clear()


def test_propulator_checkpointing(mpi_tmp_path: pathlib.Path) -> None:
    """
    Test standard Propulator checkpointing for the sphere benchmark function.

    This test is run both sequentially and in parallel.

    Parameters
    ----------
    mpi_tmp_path : pathlib.Path
        The temporary checkpoint directory.
    """
    first_generations = 20
    second_generations = 40
    set_logger_config(level=logging.DEBUG)
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
        generations=first_generations,
        checkpoint_path=mpi_tmp_path,
        rng=rng,
    )  # Set up propulator performing actual optimization.

    propulator.propulate()  # Run optimization and print summary of results.
    assert len(propulator.population) == first_generations * propulator.propulate_comm.Get_size()

    old_population = copy.deepcopy(propulator.population)  # Save population list from the last run.
    del propulator  # Delete propulator object.
    MPI.COMM_WORLD.barrier()  # Synchronize all processes.

    propulator = Propulator(
        loss_fn=benchmark_function,
        propagator=propagator,
        generations=second_generations,
        checkpoint_path=mpi_tmp_path,
        rng=rng,
    )  # Set up new propulator starting from checkpoint.

    # As the number of requested generations is smaller than the number of generations from the run before,
    # no new evaluations are performed. Thus, the length of both Propulators' populations must be equal.
    assert len(deepdiff.DeepDiff(old_population, propulator.population, ignore_order=True)) == 0
    propulator.propulate()
    # NOTE make sure nothing was overwritten
    seniors = [ind for ind in propulator.population if ind.generation < first_generations]
    assert len(deepdiff.DeepDiff(old_population, seniors, ignore_order=True)) == 0
    assert len(propulator.population) == second_generations * propulator.propulate_comm.Get_size()

    log.handlers.clear()


# TODO test loading a checkpoint with an unevaluated individual
