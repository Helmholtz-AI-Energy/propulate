import copy
import pathlib
import random
from typing import Callable, Dict, Tuple

import deepdiff
import numpy as np
import pytest
from mpi4py import MPI

from propulate import Islands
from propulate.propagators import Propagator
from propulate.utils import get_default_propagator, set_logger_config
from propulate.utils.benchmark_functions import get_function_search_space


@pytest.fixture(scope="module")
def global_variables():
    """Get global variables used by most of the tests in this module."""
    rng = random.Random(
        42 + MPI.COMM_WORLD.rank
    )  # Set up separate random number generator for optimization.
    function, limits = get_function_search_space(
        "sphere"
    )  # Get function and search space to optimize.
    propagator = get_default_propagator(
        pop_size=4,
        limits=limits,
        rng=rng,
    )  # Set up evolutionary operator.
    yield rng, function, limits, propagator


@pytest.fixture(
    params=[
        True,
        False,
    ]
)
def pollination(request):
    """Iterate through pollination parameter."""
    return request.param


@pytest.mark.mpi(min_size=4)
def test_islands(
    global_variables: Tuple[
        random.Random, Callable, Dict[str, Tuple[float, float]], Propagator
    ],
    pollination: bool,
    mpi_tmp_path: pathlib.Path,
) -> None:
    """
    Test basic island functionality (only run in parallel with at least four processes).

    Parameters
    ----------
    global_variables : Tuple[random.Random, Callable, Dict[str, Tuple[float, float]], propulate.Propagator]
        Global variables used by most of the tests in this module.
    pollination : bool
        Whether pollination or real migration should be used.
    mpi_tmp_path : pathlib.Path
        The temporary checkpoint directory.
    """
    rng, function, limits, propagator = global_variables
    set_logger_config(log_file=mpi_tmp_path / "log.log")

    # Set up island model.
    islands = Islands(
        loss_fn=function,
        propagator=propagator,
        rng=rng,
        generations=100,
        num_islands=2,
        migration_probability=0.9,
        pollination=pollination,
        checkpoint_path=mpi_tmp_path,
    )

    # Run actual optimization.
    islands.evolve(
        debug=2,
    )


@pytest.mark.mpi(min_size=4)
def test_checkpointing_isolated(
    global_variables: Tuple[
        random.Random, Callable, Dict[str, Tuple[float, float]], Propagator
    ],
    mpi_tmp_path: pathlib.Path,
) -> None:
    """
    Test isolated island checkpointing without migration (only run in parallel with at least four processes).

    Parameters
    ----------
    global_variables : Tuple[random.Random, Callable, Dict[str, Tuple[float, float]], propulate.Propagator]
        Global variables used by most of the tests in this module.
    mpi_tmp_path : pathlib.Path
        The temporary checkpoint directory.
    """
    rng, function, limits, propagator = global_variables
    set_logger_config(log_file=mpi_tmp_path / "log.log")

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
    islands.evolve(top_n=1, debug=2)

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
def test_checkpointing(
    global_variables: Tuple[
        random.Random, Callable, Dict[str, Tuple[float, float]], Propagator
    ],
    pollination: bool,
    mpi_tmp_path: pathlib.Path,
) -> None:
    """
    Test island checkpointing with migration and pollination (only run in parallel with at least four processes).

    Parameters
    ----------
    global_variables : Tuple[random.Random, Callable, Dict[str, Tuple[float, float]], propulate.Propagator]
        Global variables used by most of the tests in this module.
    pollination : bool
        Whether pollination or real migration should be used.
    mpi_tmp_path : pathlib.Path
        The temporary checkpoint directory.
    """
    rng, function, limits, propagator = global_variables
    set_logger_config(log_file=mpi_tmp_path / "log.log")

    # Set up island model.
    islands = Islands(
        loss_fn=function,
        propagator=propagator,
        rng=rng,
        generations=100,
        num_islands=2,
        migration_probability=0.9,
        pollination=pollination,
        checkpoint_path=mpi_tmp_path,
    )

    # Run actual optimization.
    islands.evolve(
        top_n=1,
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
        pollination=pollination,
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
    global_variables: Tuple[
        random.Random, Callable, Dict[str, Tuple[float, float]], Propagator
    ],
    pollination: bool,
    mpi_tmp_path: pathlib.Path,
) -> None:
    """
    Test island checkpointing for inhomogeneous island sizes (only run in parallel with at least eight processes).

    Parameters
    ----------
    global_variables : Tuple[random.Random, Callable, Dict[str, Tuple[float, float]], propulate.Propagator]
        Global variables used by most of the tests in this module.
    pollination : bool
        Whether pollination or real migration should be used.
    mpi_tmp_path : pathlib.Path
        The temporary checkpoint directory.
    """
    rng, function, limits, propagator = global_variables
    set_logger_config(log_file=mpi_tmp_path / "log.log")

    # Set up island model.
    islands = Islands(
        loss_fn=function,
        propagator=propagator,
        rng=rng,
        generations=100,
        num_islands=2,
        island_sizes=np.array([3, 5]),
        migration_probability=0.9,
        pollination=pollination,
        checkpoint_path=mpi_tmp_path,
    )

    # Run actual optimization.
    islands.evolve(
        top_n=1,
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
        pollination=pollination,
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
