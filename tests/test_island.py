import copy
import logging
import pathlib
import random
from typing import Callable, Dict, Tuple

import deepdiff
import h5py
import numpy as np
import pytest
from mpi4py import MPI

from propulate import Islands
from propulate.propagators import Propagator
from propulate.utils import get_default_propagator, set_logger_config
from propulate.utils.benchmark_functions import get_function_search_space


@pytest.fixture(scope="module")
def global_variables() -> Tuple[random.Random, Callable, Dict, Propagator]:
    """Get global variables used by most of the tests in this module."""
    rng = random.Random(42 + MPI.COMM_WORLD.rank)  # Set up separate random number generator for optimization.
    benchmark_function, limits = get_function_search_space("sphere")  # Get function and search space to optimize.
    propagator = get_default_propagator(
        pop_size=4,
        limits=limits,
        rng=rng,
    )  # Set up evolutionary operator.
    return rng, benchmark_function, limits, propagator


@pytest.fixture(
    params=[
        True,
        False,
    ]
)
def pollination(request: pytest.FixtureRequest) -> bool:
    """Iterate through pollination parameter."""
    return request.param


@pytest.mark.mpi(min_size=4)
def test_islands(
    global_variables: Tuple[random.Random, Callable, Dict[str, Tuple[float, float]], Propagator],
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
    log = logging.getLogger("propulate")  # Get logger instance.
    set_logger_config(level=logging.DEBUG)
    rng, benchmark_function, limits, propagator = global_variables

    # Set up island model.
    islands = Islands(
        loss_fn=benchmark_function,
        propagator=propagator,
        rng=rng,
        generations=10,
        num_islands=2,
        migration_probability=0.9,
        pollination=pollination,
        checkpoint_path=mpi_tmp_path,
    )

    # Run actual optimization.
    islands.propulate(
        debug=2,
    )
    log.handlers.clear()


@pytest.mark.mpi(min_size=4)
def test_islands_checkpointing_isolated(
    global_variables: Tuple[random.Random, Callable, Dict[str, Tuple[float, float]], Propagator],
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
    first_generations = 20
    second_generations = 40
    log = logging.getLogger("propulate")  # Get logger instance.
    set_logger_config(level=logging.DEBUG)
    rng, benchmark_function, limits, propagator = global_variables

    # Set up island model.
    islands = Islands(
        loss_fn=benchmark_function,
        propagator=propagator,
        rng=rng,
        generations=first_generations,
        num_islands=2,
        migration_probability=0.0,
        checkpoint_path=mpi_tmp_path,
    )

    # Run actual optimization.
    islands.propulate(debug=2)
    assert len(islands.propulator.population) == first_generations * islands.propulator.island_comm.Get_size()

    old_population = copy.deepcopy(islands.propulator.population)
    del islands
    MPI.COMM_WORLD.barrier()  # Synchronize all processes.

    islands = Islands(
        loss_fn=benchmark_function,
        propagator=propagator,
        rng=rng,
        generations=second_generations,
        num_islands=2,
        migration_probability=0.0,
        checkpoint_path=mpi_tmp_path,
    )
    assert len(old_population) == len(islands.propulator.population)

    assert len(deepdiff.DeepDiff(old_population, islands.propulator.population, ignore_order=True)) == 0
    log.handlers.clear()


# TODO test, that there are no clones in the population
@pytest.mark.mpi(min_size=4)
def test_islands_checkpointing(
    global_variables: Tuple[random.Random, Callable, Dict[str, Tuple[float, float]], Propagator],
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
    first_generations = 20
    second_generations = 40
    log = logging.getLogger("propulate")  # Get logger instance.
    set_logger_config()
    rng, benchmark_function, limits, propagator = global_variables

    # Set up island model.
    islands = Islands(
        loss_fn=benchmark_function,
        propagator=propagator,
        rng=rng,
        generations=first_generations,
        num_islands=2,
        migration_probability=0.9,
        pollination=pollination,
        checkpoint_path=mpi_tmp_path,
    )

    # Run actual optimization.
    islands.propulate(debug=2)

    old_population = copy.deepcopy(islands.propulator.population)
    del islands

    islands = Islands(
        loss_fn=benchmark_function,
        propagator=propagator,
        rng=rng,
        generations=second_generations,
        num_islands=2,
        migration_probability=0.9,
        pollination=pollination,
        checkpoint_path=mpi_tmp_path,
    )

    assert len(deepdiff.DeepDiff(old_population, islands.propulator.population, ignore_order=True)) == 0
    islands.propulate()
    log.handlers.clear()


@pytest.mark.mpi(min_size=8)
def test_islands_checkpointing_unequal_populations(
    global_variables: Tuple[random.Random, Callable, Dict[str, Tuple[float, float]], Propagator],
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
    first_generations = 20
    second_generations = 40
    log = logging.getLogger("propulate")  # Get logger instance.
    set_logger_config()
    rng, benchmark_function, limits, propagator = global_variables

    # Set up island model.
    islands = Islands(
        loss_fn=benchmark_function,
        propagator=propagator,
        rng=rng,
        generations=first_generations,
        num_islands=2,
        island_sizes=np.array([3, 5]),
        migration_probability=0.9,
        pollination=pollination,
        checkpoint_path=mpi_tmp_path,
    )

    # Run actual optimization.
    islands.propulate(debug=2)

    old_population = copy.deepcopy(islands.propulator.population)
    del islands

    islands = Islands(
        loss_fn=benchmark_function,
        propagator=propagator,
        rng=rng,
        generations=second_generations,
        num_islands=2,
        island_sizes=np.array([3, 5]),
        migration_probability=0.9,
        pollination=pollination,
        checkpoint_path=mpi_tmp_path,
    )

    assert len(deepdiff.DeepDiff(old_population, islands.propulator.population, ignore_order=True)) == 0
    # TODO compare active only
    log.handlers.clear()


@pytest.mark.mpi(min_size=8)
def test_islands_checkpointing_incomplete(
    global_variables: Tuple[random.Random, Callable, Dict[str, Tuple[float, float]], Propagator],
    pollination: bool,
    mpi_tmp_path: pathlib.Path,
) -> None:
    """
    Test island checkpointing where individuals in the checkpoint have not finished evaluating.

    Parameters
    ----------
    global_variables : Tuple[random.Random, Callable, Dict[str, Tuple[float, float]], propulate.Propagator]
        Global variables used by most of the tests in this module.
    pollination : bool
        Whether pollination or real migration should be used.
    mpi_tmp_path : pathlib.Path
        The temporary checkpoint directory.
    """
    first_generations = 20
    second_generations = 40
    log = logging.getLogger("propulate")  # Get logger instance.
    set_logger_config()
    rng, benchmark_function, limits, propagator = global_variables

    island_sizes = np.array([3, 5])
    # Set up island model.
    islands = Islands(
        loss_fn=benchmark_function,
        propagator=propagator,
        rng=rng,
        generations=first_generations,
        num_islands=2,
        island_sizes=island_sizes,
        migration_probability=0.9,
        pollination=pollination,
        checkpoint_path=mpi_tmp_path,
    )

    # Run actual optimization.
    islands.propulate(debug=2)
    MPI.COMM_WORLD.barrier()
    # NOTE manipulate checkpoint
    # NOTE this is the index of the last started generation
    started_first_generations = [
        2,
        first_generations // 2,
        first_generations // 4,
        1,
        first_generations - 1,
        first_generations // 3,
        3,
        first_generations // 2,
    ]
    assert started_first_generations == MPI.COMM_WORLD.size
    island_colors = np.concatenate([idx * np.ones(el, dtype=int) for idx, el in enumerate(island_sizes)]).ravel()

    if MPI.COMM_WORLD.rank == 0:
        with h5py.File(mpi_tmp_path / "ckpt.hdf5", "r+") as f:
            for i, g in enumerate(started_first_generations):
                f["generations"][i] = g
            for worker, g in enumerate(started_first_generations):
                island_idx = island_colors[worker]
                # Worker_idx is the island rank
                worker_idx = worker - np.concatenate(np.array([0]), island_sizes)[island_idx - 1]
                if worker == 2 or worker == 5:
                    # skip some workers who finished their last evaluation
                    continue
                f[f"{island_idx}"][f"{worker_idx}"]["loss"][g:] = np.nan

    old_population = copy.deepcopy(islands.propulator.population)
    del islands
    MPI.COMM_WORLD.barrier()

    # Set up island model.
    islands = Islands(
        loss_fn=benchmark_function,
        propagator=propagator,
        rng=rng,
        generations=second_generations,
        num_islands=2,
        island_sizes=island_sizes,
        migration_probability=0.9,
        pollination=pollination,
        checkpoint_path=mpi_tmp_path,
    )

    # NOTE check that only the correct number of individuals were read
    island_idx = islands.propulator.island_idx
    assert len(islands.propulator.population) == sum(started_first_generations[: island_sizes[island_idx]]) + len(
        started_first_generations[: island_sizes[island_idx]]
    )

    # NOTE check that in the loaded population on each island all ranks but one have an unevaluated individual
    island_root = islands.propulator.propulate_comm.rank - islands.propulator.island_comm.rank
    island_started_first_generations = started_first_generations[island_root : island_root + islands.propulator.island_comm.size]
    for rank in range(islands.propulator.island_comm.size):
        pop_key = (island_idx, rank, island_started_first_generations[rank])
        if islands.propulator.propulate_comm.rank not in {2, 5}:
            assert np.isnan(islands.propulator.population[pop_key].loss)

    islands.propulate(debug=2)
    # NOTE check that the total number is correct
    # NOTE because of migration it only has to be summed over all islands
    MPI.COMM_WORLD.barrier()
    final_pop_sizes = islands.propulator.propulate_comm.allgather(len(islands.propulator.population))
    assert final_pop_sizes[0] + final_pop_sizes[3] == second_generations * islands.propulator.propulate_comm.size
    # NOTE check there are no unevaluated individuals anymore
    assert [np.isnan(ind.loss) for ind in islands.propulator.population.values()].count(False) == len(islands.propulator.population)
    # NOTE check that loss for entire population is written to disk
    if MPI.COMM_WORLD.rank == 0:
        with h5py.File(mpi_tmp_path / "ckpt.hdf5", "r") as f:
            for island in range(island_sizes.size):
                for worker in range(island_sizes[island]):
                    assert not np.isnan(f[f"{island}"][f"{worker}"]["loss"][:].any())
    # NOTE check that no started individuals have been overwritten
    # NOTE old_population might be incomplete without final synch, so we only compare the ones we have.
    # NOTE over all ranks, each ind should be checked on at least once.
    for island in range(island_sizes.size):
        for rank in range(island_sizes[island]):
            finished_first_generations = started_first_generations[rank]
            if island == 0 and rank == 2:
                finished_first_generations += 1
            if island == 1 and rank == 2:
                finished_first_generations += 1
            for g in range(finished_first_generations + 1):
                pop_key = (island, rank, started_first_generations[rank])
                if pop_key in old_population:
                    assert old_population[pop_key].position == pytest.approx(islands.propulator.population[pop_key].position)

    log.handlers.clear()
