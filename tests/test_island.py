import copy
import pathlib
import random
from typing import Tuple

import deepdiff
import numpy as np
import pytest
from mpi4py import MPI

from propulate import Islands
from propulate.propagators import SelectMax, SelectMin
from propulate.utils import get_default_propagator, set_logger_config
from propulate.utils.benchmark_functions import get_function_search_space


@pytest.fixture(
    params=[
        ("rosenbrock", 0.0),
        ("step", -25.0),
        ("quartic", 0.0),
        ("rastrigin", 0.0),
        ("griewank", 0.0),
        ("schwefel", 0.0),
        ("bisphere", 0.0),
        ("birastrigin", 0.0),
        ("bukin", 0.0),
        ("eggcrate", -1.0),
        ("himmelblau", 0.0),
        ("keane", 0.6736675),
        ("leon", 0.0),
        ("sphere", 0.0),  # (fname, expected)
    ]
)
def function_parameters(request):
    """Define benchmark function parameter sets as used in tests."""
    return request.param


@pytest.mark.mpi(min_size=4)
def test_islands(
    function_parameters: Tuple[str, float], mpi_tmp_path: pathlib.Path
) -> None:
    """
    Test basic island functionality (only run in parallel with at least four processes).

    Parameters
    ----------
    function_parameters : Tuple
        The tuple containing each function name along with its global minimum.
    mpi_tmp_path : pathlib.Path
        The temporary checkpoint directory.
    """
    rng = random.Random(
        42 + MPI.COMM_WORLD.rank
    )  # Separate random number generator for optimization
    function, limits = get_function_search_space(function_parameters[0])
    set_logger_config(log_file=mpi_tmp_path / "log.log")

    # Set up evolutionary operator.
    propagator = get_default_propagator(
        pop_size=4,
        limits=limits,
        rng=rng,
    )

    # Set up island model.
    islands = Islands(
        loss_fn=function,
        propagator=propagator,
        rng=rng,
        generations=100,
        num_islands=2,
        migration_probability=0.9,
        pollination=False,
        checkpoint_path=mpi_tmp_path,
    )

    # Run actual optimization.
    islands.evolve(
        top_n=1,
        logging_interval=10,
        debug=2,
    )


@pytest.mark.mpi(min_size=4)
def test_checkpointing_isolated(
    function_parameters: Tuple[str, float], mpi_tmp_path: pathlib.Path
) -> None:
    """
    Test isolated island checkpointing without migration (only run in parallel with at least four processes).

    Parameters
    ----------
    function_parameters : Tuple
        The tuple containing each function name along with its global minimum.
    mpi_tmp_path : pathlib.Path
        The temporary checkpoint directory.
    """
    rng = random.Random(
        42 + MPI.COMM_WORLD.rank
    )  # Separate random number generator for optimization
    function, limits = get_function_search_space(function_parameters[0])
    set_logger_config(log_file=mpi_tmp_path / "log.log")

    # Set up evolutionary operator.
    propagator = get_default_propagator(
        pop_size=4,
        limits=limits,
        rng=rng,
    )

    # Set up island model.
    islands = Islands(
        loss_fn=function,
        propagator=propagator,
        rng=rng,
        generations=100,
        num_islands=2,
        migration_probability=0.0,
        checkpoint_path=mpi_tmp_path,
    )

    # Run actual optimization.
    islands.evolve(
        top_n=1,
        logging_interval=10,
        debug=2,
    )

    old_population = copy.deepcopy(islands.propulator.population)
    del islands

    islands = Islands(
        loss_fn=function,
        propagator=propagator,
        rng=rng,
        generations=100,
        num_islands=2,
        migration_probability=0.0,
        checkpoint_path=mpi_tmp_path,
    )

    assert (
        len(
            deepdiff.DeepDiff(
                old_population, islands.propulator.population, ignore_order=True
            )
        )
        == 0
    )


@pytest.mark.mpi(min_size=4)
def test_checkpointing_migration(
    function_parameters: Tuple[str, float], mpi_tmp_path: pathlib.Path
) -> None:
    """
    Test island checkpointing with migration (only run in parallel with at least four processes).

    Parameters
    ----------
    function_parameters : Tuple
        The tuple containing each function name along with its global minimum.
    mpi_tmp_path : pathlib.Path
        The temporary checkpoint directory.
    """
    rng = random.Random(
        42 + MPI.COMM_WORLD.rank
    )  # Separate random number generator for optimization
    function, limits = get_function_search_space(function_parameters[0])
    set_logger_config(log_file=mpi_tmp_path / "log.log")

    # Set up evolutionary operator.
    propagator = get_default_propagator(
        pop_size=4,
        limits=limits,
        rng=rng,
    )

    # Set up island model.
    islands = Islands(
        loss_fn=function,
        propagator=propagator,
        rng=rng,
        generations=100,
        num_islands=2,
        migration_probability=0.9,
        pollination=False,  # TODO fixtureize
        checkpoint_path=mpi_tmp_path,
    )

    # Run actual optimization.
    islands.evolve(
        top_n=1,
        logging_interval=10,
        debug=2,
    )

    old_population = copy.deepcopy(islands.propulator.population)
    del islands

    islands = Islands(
        loss_fn=function,
        propagator=propagator,
        rng=rng,
        generations=100,
        num_islands=2,
        migration_probability=0.9,
        emigration_propagator=SelectMin,
        immigration_propagator=SelectMax,
        pollination=False,  # TODO fixtureize
        checkpoint_path=mpi_tmp_path,
    )

    assert (
        len(
            deepdiff.DeepDiff(
                old_population, islands.propulator.population, ignore_order=True
            )
        )
        == 0
    )


@pytest.mark.mpi(min_size=4)
def test_checkpointing_pollination(
    function_parameters: Tuple[str, float], mpi_tmp_path: pathlib.Path
) -> None:
    """
    Test island checkpointing with pollination (only run in parallel with at least four processes).

    Parameters
    ----------
    function_parameters : Tuple
        The tuple containing each function name along with its global minimum.
    mpi_tmp_path : pathlib.Path
        The temporary checkpoint directory.
    """
    rng = random.Random(
        42 + MPI.COMM_WORLD.rank
    )  # Separate random number generator for optimization
    function, limits = get_function_search_space(function_parameters[0])
    set_logger_config(log_file=mpi_tmp_path / "propulate.log")

    # Set up evolutionary operator.
    propagator = get_default_propagator(
        pop_size=4,
        limits=limits,
        rng=rng,
    )

    # Set up island model.
    islands = Islands(
        loss_fn=function,
        propagator=propagator,
        rng=rng,
        generations=100,
        num_islands=2,
        migration_probability=0.9,
        pollination=False,  # TODO fixtureize
        checkpoint_path=mpi_tmp_path,
    )

    # Run actual optimization.
    islands.evolve(
        top_n=1,
        logging_interval=10,
        debug=2,
    )

    old_population = copy.deepcopy(islands.propulator.population)
    del islands

    islands = Islands(
        loss_fn=function,
        propagator=propagator,
        rng=rng,
        generations=100,
        num_islands=2,
        migration_probability=0.9,
        pollination=True,  # TODO fixtureize
        checkpoint_path=mpi_tmp_path,
    )

    assert (
        len(
            deepdiff.DeepDiff(
                old_population, islands.propulator.population, ignore_order=True
            )
        )
        == 0
    )


@pytest.mark.mpi(min_size=8)
def test_checkpointing_unequal_populations(
    function_parameters: Tuple[str, float], mpi_tmp_path: pathlib.Path
) -> None:
    """
    Test island checkpointing for inhomogeneous island sizes (only run in parallel with at least eight processes).

    Parameters
    ----------
    function_parameters : Tuple
        The tuple containing each function name along with its global minimum.
    mpi_tmp_path : pathlib.Path
        The temporary checkpoint directory.
    """
    rng = random.Random(
        42 + MPI.COMM_WORLD.rank
    )  # Separate random number generator for optimization
    function, limits = get_function_search_space(function_parameters[0])
    set_logger_config(log_file=mpi_tmp_path / "log.log")

    # Set up evolutionary operator.
    propagator = get_default_propagator(
        pop_size=4,
        limits=limits,
        rng=rng,
    )

    # Set up island model.
    islands = Islands(
        loss_fn=function,
        propagator=propagator,
        rng=rng,
        generations=100,
        num_islands=2,
        island_sizes=np.array([1, 3]),
        migration_probability=0.9,
        pollination=False,  # TODO fixtureize
        checkpoint_path=mpi_tmp_path,
    )

    # Run actual optimization.
    islands.evolve(
        top_n=1,
        logging_interval=10,
        debug=2,
    )

    old_population = copy.deepcopy(islands.propulator.population)
    del islands

    islands = Islands(
        loss_fn=function,
        propagator=propagator,
        rng=rng,
        generations=100,
        num_islands=2,
        island_sizes=np.array([3, 5]),
        migration_probability=0.9,
        pollination=True,  # TODO fixtureize
        checkpoint_path=mpi_tmp_path,
    )

    assert (
        len(
            deepdiff.DeepDiff(
                old_population, islands.propulator.population, ignore_order=True
            )
        )
        == 0
    )
