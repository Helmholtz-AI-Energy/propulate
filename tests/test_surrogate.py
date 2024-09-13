import logging
import random
from pathlib import Path
from typing import Dict, Generator, Tuple, Union

import numpy as np
import pytest
from mpi4py import MPI

from propulate import Islands, Propulator, surrogate
from propulate.utils import get_default_propagator, set_logger_config

pytestmark = pytest.mark.filterwarnings(
    "ignore::DeprecationWarning",
    match="Assigning the 'data' attribute is an inherently unsafe operation and will be removed in the future.",
)

log = logging.getLogger(__name__)  # Get logger instance.
set_logger_config(level=logging.DEBUG)


def ind_loss(params: Dict[str, Union[int, float, str]]) -> Generator[float, None, None]:
    """
    Toy iterative loss function for evolutionary optimization with ``Propulate``.

    A decay with some noise and some bumps and an eventual increase.

    Parameters
    ----------
    params : Dict[str, int | float | str]
        The parameters to be optimized.

    Returns
    -------
    Generator[float, None, None]
        Yields the current loss.
    """
    rng = np.random.default_rng(seed=MPI.COMM_WORLD.rank)
    num_iterations = 300
    for i in range(num_iterations):
        yield (10 * params["start"] * np.exp(-i / 10) + rng.standard_normal() / 100 + params["limit"] + 1 / 10000 * i**2)


def test_static(mpi_tmp_path: Path) -> None:
    """Test static surrogate using a dummy function."""
    pop_size = 2 * MPI.COMM_WORLD.size  # Breeding population size
    limits: Dict[str, Union[Tuple[int, int], Tuple[float, float], Tuple[str, ...]]] = {
        "start": (0.1, 7.0),
        "limit": (-1.0, 1.0),
    }  # Define search space.
    rng = random.Random(MPI.COMM_WORLD.rank + 100)  # Set up separate random number generator for evolutionary optimizer.
    num_generations = 4

    propagator = get_default_propagator(  # Get default evolutionary operator.
        pop_size=pop_size,  # Breeding population size
        limits=limits,  # Search space
        crossover_prob=0.7,  # Crossover probability
        mutation_prob=0.4,  # Mutation probability
        random_init_prob=0.1,  # Random-initialization probability
        rng=rng,  # Random number generator for evolutionary optimizer
    )
    propulator = Propulator(
        loss_fn=ind_loss,
        propagator=propagator,
        generations=num_generations,
        checkpoint_path=mpi_tmp_path,
        rng=rng,
        surrogate_factory=lambda: surrogate.StaticSurrogate(),
    )  # Set up propulator performing actual optimization.

    propulator.propulate(debug=1)  # Run optimization and print summary of results.
    MPI.COMM_WORLD.barrier()


@pytest.mark.mpi(min_size=8)
def test_static_island(mpi_tmp_path: Path) -> None:
    """Test static surrogate using a dummy function."""
    pop_size = 2 * MPI.COMM_WORLD.size  # Breeding population size
    limits: Dict[str, Union[Tuple[int, int], Tuple[float, float], Tuple[str, ...]]] = {
        "start": (0.1, 7.0),
        "limit": (-1.0, 1.0),
    }  # Define search space.
    rng = random.Random(MPI.COMM_WORLD.rank + 100)  # Set up separate random number generator for evolutionary optimizer.
    num_generations = 4

    propagator = get_default_propagator(  # Get default evolutionary operator.
        pop_size=pop_size,  # Breeding population size
        limits=limits,  # Search space
        crossover_prob=0.7,  # Crossover probability
        mutation_prob=0.4,  # Mutation probability
        random_init_prob=0.1,  # Random-initialization probability
        rng=rng,  # Random number generator for evolutionary optimizer
    )
    islands = Islands(  # Set up island model.
        loss_fn=ind_loss,  # Loss function to optimize
        propagator=propagator,  # Evolutionary operator
        rng=rng,  # Random number generator
        generations=num_generations,  # Number of generations per worker
        num_islands=2,  # Number of islands
        checkpoint_path=mpi_tmp_path,
        surrogate_factory=lambda: surrogate.StaticSurrogate(),
    )
    islands.propulate(  # Run evolutionary optimization.
        logging_interval=1,  # Logging interval
        debug=2,  # Verbosity level
    )
    islands.summarize(top_n=1, debug=2)
    MPI.COMM_WORLD.barrier()


def test_dynamic(mpi_tmp_path: Path) -> None:
    """Test static surrogate using a dummy function."""
    pop_size = 2 * MPI.COMM_WORLD.size  # Breeding population size
    limits: Dict[str, Union[Tuple[int, int], Tuple[float, float], Tuple[str, ...]]] = {
        "start": (0.1, 7.0),
        "limit": (-1.0, 1.0),
    }  # Define search space.
    rng = random.Random(MPI.COMM_WORLD.rank + 100)  # Set up separate random number generator for evolutionary optimizer.
    num_generations = 4

    propagator = get_default_propagator(  # Get default evolutionary operator.
        pop_size=pop_size,  # Breeding population size
        limits=limits,  # Search space
        crossover_prob=0.7,  # Crossover probability
        mutation_prob=0.4,  # Mutation probability
        random_init_prob=0.1,  # Random-initialization probability
        rng=rng,  # Random number generator for evolutionary optimizer
    )
    propulator = Propulator(
        loss_fn=ind_loss,
        propagator=propagator,
        generations=num_generations,
        checkpoint_path=mpi_tmp_path,
        rng=rng,
        surrogate_factory=lambda: surrogate.DynamicSurrogate(limits),
    )  # Set up propulator performing actual optimization.

    propulator.propulate(logging_interval=1)  # Run optimization and print summary of results.
    MPI.COMM_WORLD.barrier()


@pytest.mark.mpi(min_size=8)
@pytest.mark.filterwarnings(
    "ignore::DeprecationWarning",
    match="Assigning the 'data' attribute is an inherently unsafe operation and will be removed in the future.",
)
def test_dynamic_island(mpi_tmp_path: Path) -> None:
    """Test dynamic surrogate using a dummy function."""
    num_generations = 4  # Number of generations
    pop_size = 2 * MPI.COMM_WORLD.size  # Breeding population size
    limits: Dict[str, Union[Tuple[int, int], Tuple[float, float], Tuple[str, ...]]] = {
        "start": (0.1, 7.0),
        "limit": (-1.0, 1.0),
        "num_iterations": (100, 1000),
    }  # Define search space.
    rng = random.Random(MPI.COMM_WORLD.rank + 100)  # Set up separate random number generator for evolutionary optimizer.
    propagator = get_default_propagator(  # Get default evolutionary operator.
        pop_size=pop_size,  # Breeding population size
        limits=limits,  # Search space
        rng=rng,  # Random number generator for evolutionary optimizer
    )
    islands = Islands(  # Set up island model.
        loss_fn=ind_loss,  # Loss function to optimize
        propagator=propagator,  # Evolutionary operator
        rng=rng,  # Random number generator
        generations=num_generations,  # Number of generations per worker
        num_islands=2,  # Number of islands
        checkpoint_path=mpi_tmp_path,
        surrogate_factory=lambda: surrogate.DynamicSurrogate(limits),
    )
    islands.propulate(  # Run evolutionary optimization.
        logging_interval=1,  # Logging interval
        debug=2,  # Verbosity level
    )
    islands.summarize(top_n=1, debug=2)
