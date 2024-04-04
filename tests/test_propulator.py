import copy
import logging
import random
import tempfile

import deepdiff
import pytest

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


@pytest.mark.mpi_skip
def test_propulator(function_parameters) -> None:
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
    with tempfile.TemporaryDirectory() as checkpoint_path:
        set_logger_config(
            level=logging.INFO,
            log_file=checkpoint_path + "/propulate.log",
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
            checkpoint_path=checkpoint_path,
        )

        # Run optimization and print summary of results.
        propulator.propulate()
        assert propulator.summarize(top_n=1, debug=2)[0][0].loss == pytest.approx(
            expected=expected, abs=abs_tolerance
        )


@pytest.mark.mpi_skip
def test_propulator_checkpointing() -> None:
    """Test single worker Propulator checkpointing."""
    rng = random.Random(42)  # Separate random number generator for optimization
    function, limits = get_function_search_space("sphere")

    with tempfile.TemporaryDirectory() as checkpoint_directory:
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
            checkpoint_path=checkpoint_directory,
            rng=rng,
        )

        propulator.propulate()

        old_population = copy.deepcopy(propulator.population)
        del propulator

        propulator = Propulator(
            loss_fn=function,
            propagator=propagator,
            generations=20,
            checkpoint_path=checkpoint_directory,
            rng=rng,
        )

        assert (
            len(
                deepdiff.DeepDiff(
                    old_population, propulator.population, ignore_order=True
                )
            )
            == 0
        )
