import copy
import logging
import random
import tempfile

import deepdiff
import pytest

from propulate import Propulator
from propulate.utils import get_default_propagator, set_logger_config
from propulate.utils.benchmark_functions import get_function_search_space


@pytest.mark.mpi_skip
def test_propulator() -> None:
    """Test single worker using Propulator to optimize sphere."""
    rng = random.Random(42)  # Separate random number generator for optimization
    function, limits = get_function_search_space("sphere")
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

        assert propulator.summarize()[0][0].loss < 0.8


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
            generations=10,
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
