import random
import tempfile
from operator import attrgetter
import logging
from copy import deepcopy

from propulate import Propulator
from propulate.utils import get_default_propagator, set_logger_config, sphere


def test_Propulator():
    """
    Test single worker using Propulator to optimize sphere.
    """
    rng = random.Random(42)  # Separate random number generator for optimization.
    limits = {
        "a": (-5.12, 5.12),
        "b": (-5.12, 5.12),
    }
    with tempfile.TemporaryDirectory() as checkpoint_directory:
        set_logger_config(
            level=logging.INFO,
            log_file=checkpoint_directory + "/propulate.log",
            log_to_stdout=True,
            log_rank=False,
            colors=True,
        )
        # Set up evolutionary operator.
        propagator = get_default_propagator(  # Get default evolutionary operator.
            pop_size=4,  # Breeding pool size
            limits=limits,  # Search-space limits
            mate_prob=0.7,  # Crossover probability
            mut_prob=9.0,  # Mutation probability
            random_prob=0.1,  # Random-initialization probability
            rng=rng,  # Random number generator
        )

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

        assert best.loss < 0.8


def test_checkpointing_propulator():
    rng = random.Random(42)  # Separate random number generator for optimization.
    limits = {
        "a": (-5.12, 5.12),
        "b": (-5.12, 5.12),
    }

    with tempfile.TemporaryDirectory() as checkpoint_directory:
        propagator = get_default_propagator(limits=limits, rng=rng)
        propulator = Propulator(
            loss_fn=sphere,
            propagator=propagator,
            generations=10,
            checkpoint_directory=checkpoint_directory,
            rng=rng,
        )

        propulator.propulate()

        population = deepcopy(propulator.population)

        del propulator

        propulator = Propulator(
            loss_fn=sphere,
            propagator=propagator,
            generations=20,
            checkpoint_directory=checkpoint_directory,
            rng=rng,
        )

        # TODO properly check the loaded from checkpoint  population is the same as the one retained from the first run
        assert population == propulator.population
