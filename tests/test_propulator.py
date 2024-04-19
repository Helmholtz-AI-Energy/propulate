import copy
import logging
import random

import deepdiff
import pytest
from mpi4py import MPI

from propulate import Propulator
from propulate.utils import get_default_propagator, set_logger_config
from propulate.utils.benchmark_functions import get_function_search_space


@pytest.fixture(
    params=[
        ("rosenbrock", 0.0, 0.1),
        ("step", -25.0, 2.0),
        ("quartic", 0.0, 1000.0),
        ("rastrigin", 0.0, 1000.0),
        ("griewank", 0.0, 10000.0),
        ("schwefel", 0.0, 10000.0),
        ("bisphere", 0.0, 1000.0),
        ("birastrigin", 0.0, 1000.0),
        ("bukin", 0.0, 100.0),
        ("eggcrate", -1.0, 10.0),
        ("himmelblau", 0.0, 1.0),
        ("keane", 0.6736675, 1.0),
        ("leon", 0.0, 10.0),
        ("sphere", 0.0, 0.01),  # (fname, expected, abs)
    ]
)
def function_parameters(request):
    """Define benchmark function parameter sets as used in tests."""
    return request.param


def test_propulator(function_parameters, mpi_tmp_path) -> None:
    """
    Test single worker using Propulator to optimize a benchmark function using the default genetic propagator.

    Parameters
    ----------
    function_parameters : tuple
        The tuple containing (fname, expected, abs).
    """
    fname, expected, abs_tolerance = function_parameters
    rng = random.Random(42)  # Separate random number generator for optimization
    function, limits = get_function_search_space(fname)
    set_logger_config(
        level=logging.INFO,
        log_file=mpi_tmp_path / "propulate.log",
        log_to_stdout=True,
        log_rank=False,
        colors=True,
    )
    # Set up evolutionary operator.
    propagator = get_default_propagator(
        pop_size=4,
        limits=limits,
        crossover_prob=0.7,
        mutation_prob=0.9,
        random_init_prob=0.1,
        rng=rng,
    )

    # Set up propulator performing actual optimization.
    propulator = Propulator(
        loss_fn=function,
        propagator=propagator,
        rng=rng,
        generations=100,
        checkpoint_path=mpi_tmp_path,
    )

    # Run optimization and print summary of results.
    propulator.propulate()
    assert propulator.summarize(top_n=1, debug=2)[0][0].loss == pytest.approx(
        expected=expected, abs=abs_tolerance
    )


def test_propulator_checkpointing(mpi_tmp_path) -> None:
    """Test single worker Propulator checkpointing."""
    rng = random.Random(42)  # Separate random number generator for optimization
    function, limits = get_function_search_space("sphere")

    propagator = get_default_propagator(  # Get default evolutionary operator.
        pop_size=4,  # Breeding pool size
        limits=limits,  # Search-space limits
        crossover_prob=0.7,  # Crossover probability
        mutation_prob=0.9,  # Mutation probability
        random_init_prob=0.1,  # Random-initialization probability
        rng=rng,  # Random number generator
    )
    propulator = Propulator(
        loss_fn=function,
        propagator=propagator,
        generations=1000,
        checkpoint_path=mpi_tmp_path,
        rng=rng,
    )

    propulator.propulate()

    old_population = copy.deepcopy(propulator.population)
    del propulator
    MPI.COMM_WORLD.barrier()

    propulator = Propulator(
        loss_fn=function,
        propagator=propagator,
        generations=20,
        checkpoint_path=mpi_tmp_path,
        rng=rng,
    )

    # print(old_population)
    # print(propulator.population)
    print(deepdiff.DeepDiff(old_population, propulator.population, ignore_order=True))
    print(len(old_population), len(propulator.population))

    assert (
        len(deepdiff.DeepDiff(old_population, propulator.population, ignore_order=True))
        == 0
    )
