import copy
import logging
import random

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


@pytest.mark.mpi(min_size=4)
def test_island(function_parameters, mpi_tmp_path) -> None:
    """Test basic island functionality."""
    fname, expected, abs_tolerance = function_parameters
    rng = random.Random(
        42 + MPI.COMM_WORLD.rank
    )  # Separate random number generator for optimization
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

    # Set up migration topology.
    migration_topology = 1 * np.ones(  # Set up fully connected migration topology.
        (2, 2), dtype=int
    )
    np.fill_diagonal(
        migration_topology, 0
    )  # An island does not send migrants to itself.

    # Set up island model.
    islands = Islands(
        loss_fn=function,
        propagator=propagator,
        rng=rng,
        generations=100,
        num_islands=2,
        migration_topology=migration_topology,
        migration_probability=0.9,
        emigration_propagator=SelectMin,
        immigration_propagator=SelectMax,
        pollination=False,  # TODO fixtureize
        checkpoint_path=mpi_tmp_path,
    )

    # Run actual optimization.
    islands.evolve(
        top_n=1,
        logging_interval=10,
        debug=2,
    )


@pytest.mark.mpi
def test_checkpointing_isolated(function_parameters, mpi_tmp_path):
    """Test island checkpointing without migration."""
    fname, expected, abs_tolerance = function_parameters
    rng = random.Random(
        42 + MPI.COMM_WORLD.rank
    )  # Separate random number generator for optimization
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

    # Set up migration topology.
    migration_topology = 1 * np.ones(  # Set up fully connected migration topology.
        (2, 2), dtype=int
    )
    np.fill_diagonal(
        migration_topology, 0
    )  # An island does not send migrants to itself.

    # Set up island model.
    islands = Islands(
        loss_fn=function,
        propagator=propagator,
        rng=rng,
        generations=100,
        num_islands=2,
        migration_topology=migration_topology,
        migration_probability=0.0,
        emigration_propagator=SelectMin,
        immigration_propagator=SelectMax,
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
        migration_topology=migration_topology,
        migration_probability=0.0,
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


@pytest.mark.mpi
def test_checkpointing_migration(function_parameters, mpi_tmp_path):
    """Test island checkpointing without migration."""
    fname, expected, abs_tolerance = function_parameters
    rng = random.Random(
        42 + MPI.COMM_WORLD.rank
    )  # Separate random number generator for optimization
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

    # Set up migration topology.
    migration_topology = 1 * np.ones(  # Set up fully connected migration topology.
        (2, 2), dtype=int
    )
    np.fill_diagonal(
        migration_topology, 0
    )  # An island does not send migrants to itself.

    # Set up island model.
    islands = Islands(
        loss_fn=function,
        propagator=propagator,
        rng=rng,
        generations=100,
        num_islands=2,
        migration_topology=migration_topology,
        migration_probability=0.0,
        emigration_propagator=SelectMin,
        immigration_propagator=SelectMax,
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
        migration_topology=migration_topology,
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


def test_checkpointing_pollination(function_parameters, mpi_tmp_path):
    """Test island checkpointing without migration."""
    fname, expected, abs_tolerance = function_parameters
    rng = random.Random(
        42 + MPI.COMM_WORLD.rank
    )  # Separate random number generator for optimization
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

    # Set up migration topology.
    migration_topology = 1 * np.ones(  # Set up fully connected migration topology.
        (2, 2), dtype=int
    )
    np.fill_diagonal(
        migration_topology, 0
    )  # An island does not send migrants to itself.

    # Set up island model.
    islands = Islands(
        loss_fn=function,
        propagator=propagator,
        rng=rng,
        generations=100,
        num_islands=2,
        migration_topology=migration_topology,
        migration_probability=0.0,
        emigration_propagator=SelectMin,
        immigration_propagator=SelectMax,
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
        migration_topology=migration_topology,
        migration_probability=0.9,
        emigration_propagator=SelectMin,
        immigration_propagator=SelectMax,
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


# @pytest.mark.mpi
# def test_checkpointing_unequal_populations():
#     # TODO test loading unequal islands
#     raise
#
#
# @pytest.mark.mpi
# def test_checkpointing_setup_downgrade():
#     # TODO write checkpoint with large setup and read it with a small one
#     raise
#
#
# @pytest.mark.mpi
# def test_checkpointing_setup_upgrade():
#     # TOOD write checkpoint with small setup and read it with a large one
#     raise
