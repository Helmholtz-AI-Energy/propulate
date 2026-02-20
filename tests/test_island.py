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
from propulate.utils.consistency_checks import final_synch, population_consistency_check


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
def test_islands_simple(
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
        generations=50,
        num_islands=2,
        migration_probability=0.9,
        pollination=pollination,
        checkpoint_path=mpi_tmp_path,
    )

    # Run actual optimization.
    islands.propulate()
    MPI.COMM_WORLD.barrier()
    final_synch(islands.propulator)
    population_consistency_check(islands.propulator)
    log.handlers.clear()
    MPI.COMM_WORLD.barrier()
    print(f"Finished simple island test, pollination: {pollination}")


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
    islands.propulate()
    final_synch(islands.propulator)
    population_consistency_check(islands.propulator)
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

    MPI.COMM_WORLD.barrier()


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
    set_logger_config(level=logging.DEBUG)
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
    islands.propulate()
    final_synch(islands.propulator)
    population_consistency_check(islands.propulator)

    old_population = copy.deepcopy(islands.propulator._get_active_individuals())
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

    assert len(deepdiff.DeepDiff(old_population, list(islands.propulator.population.values()), ignore_order=True)) == 0
    islands.propulate()
    final_synch(islands.propulator)
    population_consistency_check(islands.propulator)
    log.handlers.clear()

    MPI.COMM_WORLD.barrier()


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
    set_logger_config(level=logging.DEBUG)
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
    islands.propulate()
    final_synch(islands.propulator)
    population_consistency_check(islands.propulator)

    old_population = copy.deepcopy(islands.propulator._get_active_individuals())
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

    assert len(deepdiff.DeepDiff(old_population, list(islands.propulator.population.values()), ignore_order=True)) == 0
    log.handlers.clear()

    MPI.COMM_WORLD.barrier()


@pytest.mark.mpi(min_size=8, max_size=8)
def test_islands_checkpointing_incomplete_isolated(
    global_variables: Tuple[random.Random, Callable, Dict[str, Tuple[float, float]], Propagator],
    pollination: bool,
    mpi_tmp_path: pathlib.Path,
) -> None:
    """
    Test island checkpointing where individuals in the checkpoint have not finished evaluating.

    For now this test is for islands without migration, since that breaks the count.

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
    set_logger_config(level=logging.DEBUG)
    rng, benchmark_function, limits, propagator = global_variables
    migration_probability = 0.0
    num_islands = 2

    island_sizes = np.array([3, 5])
    # Set up island model.
    islands = Islands(
        loss_fn=benchmark_function,
        propagator=propagator,
        rng=rng,
        generations=first_generations,
        num_islands=num_islands,
        island_sizes=island_sizes,
        migration_probability=migration_probability,
        pollination=pollination,
        checkpoint_path=mpi_tmp_path,
    )

    # Run actual optimization.
    islands.propulate()
    final_synch(islands.propulator)
    population_consistency_check(islands.propulator)
    MPI.COMM_WORLD.barrier()

    # NOTE manipulate checkpoint
    # NOTE for these workers their last evaluation in the checkpoint has finished
    workers_last_finished = {2, 5}
    workers_last_finished_island = {0: {2}, 1: {2}}
    # NOTE this is the index of the last started generation
    started_first_generations = np.array(
        [
            2,
            first_generations // 2,  # 10
            first_generations // 4,  # 5
            1,
            first_generations // 3,  # 6
            first_generations - 1,  # NOTE this one is completely finished
            3,
            first_generations - 1,  # NOTE this one has started the last generation, but not finished
        ]
    )
    assert len(started_first_generations) == MPI.COMM_WORLD.size
    island_colors = np.concatenate([idx * np.ones(el, dtype=int) for idx, el in enumerate(island_sizes)]).ravel()

    if MPI.COMM_WORLD.rank == 0:
        with h5py.File(mpi_tmp_path / "ckpt.hdf5", "r+") as f:
            for i, g in enumerate(started_first_generations):
                f["generations"][i] = g
            for worker, g in enumerate(started_first_generations):
                island_idx = island_colors[worker]
                # Worker_idx is the island rank
                # [0, 1, 2, 3, 4, 5, 6, 7] world
                # [0, 1, 2, 0, 1, 2, 3, 4] island rank
                # [0, 0, 0, 1, 1, 1, 1, 1] island colors
                island_worker_idx = worker - np.cumsum(np.concatenate((np.array([0]), island_sizes)))[island_idx]
                if worker in workers_last_finished:
                    # skip some workers who finished their last evaluation
                    f[f"{island_idx}"][f"{island_worker_idx}"]["loss"][g + 1 :] = np.nan
                else:
                    f[f"{island_idx}"][f"{island_worker_idx}"]["loss"][g:] = np.nan

    MPI.COMM_WORLD.barrier()
    with h5py.File(mpi_tmp_path / "ckpt.hdf5", "r") as f:
        island_worker_idx = islands.propulator.island_comm.rank
        island_idx = islands.propulator.island_idx

    old_population = copy.deepcopy(islands.propulator.population)
    del islands
    MPI.COMM_WORLD.barrier()

    # Set up island model.
    islands = Islands(
        loss_fn=benchmark_function,
        propagator=propagator,
        rng=rng,
        generations=second_generations,
        num_islands=num_islands,
        island_sizes=island_sizes,
        migration_probability=migration_probability,
        pollination=pollination,
        checkpoint_path=mpi_tmp_path,
    )
    MPI.COMM_WORLD.barrier()

    # NOTE check that only the correct number of individuals were read

    # NOTE
    island_idx = islands.propulator.island_idx
    finished_on_island = sum(started_first_generations[island_colors == island_idx])
    last_finished_mask = np.zeros(len(started_first_generations), dtype=bool)
    last_finished_mask[np.array([x for x in workers_last_finished])] = True
    finished_on_island += sum((island_colors == island_idx)[last_finished_mask])
    assert finished_on_island == [18, 49][island_idx]

    # NOTE check that the values read are the same:
    count = 0
    for island in range(island_sizes.size):
        for island_rank in range(island_sizes[island]):
            prop_rank = island_rank + np.cumsum(np.concatenate((np.array([0]), island_sizes)))[island]
            finished_first_generations = started_first_generations[prop_rank]
            if island == 0 and island_rank == 2:
                finished_first_generations += 1
            if island == 1 and island_rank == 2:
                finished_first_generations += 1
            for g in range(finished_first_generations):
                pop_key = (island, island_rank, g)
                if pop_key in old_population:
                    count += 1
                    assert old_population[pop_key].position == pytest.approx(islands.propulator.population[pop_key].position)
    assert count > 0

    # NOTE check only evaluated individuals are in active breeding population
    assert len(islands.propulator._get_active_individuals()) == len(
        [ind for ind in islands.propulator._get_active_individuals() if not np.isnan(ind.loss)]
    )
    # NOTE check all evaluated individuals are in active breeding population
    assert len(islands.propulator._get_active_individuals()) == finished_on_island

    # NOTE check that the correct number of individuals in the population are evaluated
    MPI.COMM_WORLD.barrier()
    assert len(
        [ind for ind in islands.propulator.population.values() if np.isnan(ind.loss)]
    ) == islands.propulator.island_comm.size - len(workers_last_finished_island[islands.propulator.island_idx])

    # NOTE after loading the checkpoint, all workers except those in workers_last_finished should have one not evaluated individual
    island_root = islands.propulator.propulate_comm.rank - islands.propulator.island_comm.rank
    island_started_first_generations = started_first_generations[island_root : island_root + islands.propulator.island_comm.size]

    for island_rank in range(islands.propulator.island_comm.size):
        pop_key = (island_idx, island_rank, island_started_first_generations[island_rank])
        if island_rank not in workers_last_finished_island[island_idx]:
            assert np.isnan(islands.propulator.population[pop_key].loss)

    MPI.COMM_WORLD.barrier()

    count = 0
    for island in range(island_sizes.size):
        for island_rank in range(island_sizes[island]):
            prop_rank = island_rank + np.cumsum(np.concatenate((np.array([0]), island_sizes)))[island]
            finished_first_generations = started_first_generations[prop_rank]
            if island == 0 and island_rank == 2:
                finished_first_generations += 1
            if island == 1 and island_rank == 2:
                finished_first_generations += 1
            for g in range(finished_first_generations):
                pop_key = (island, island_rank, g)
                if pop_key in old_population:
                    assert old_population[pop_key].position == pytest.approx(islands.propulator.population[pop_key].position)
                    count += 1
    assert count > 0

    MPI.COMM_WORLD.barrier()
    islands.propulate()
    MPI.COMM_WORLD.barrier()
    count = 0
    for island in range(island_sizes.size):
        for island_rank in range(island_sizes[island]):
            prop_rank = island_rank + np.cumsum(np.concatenate((np.array([0]), island_sizes)))[island]
            finished_first_generations = started_first_generations[prop_rank]
            if island == 0 and island_rank == 2:
                finished_first_generations += 1
            if island == 1 and island_rank == 2:
                finished_first_generations += 1
            for g in range(finished_first_generations):
                pop_key = (island, island_rank, g)
                if pop_key in old_population:
                    assert old_population[pop_key].position == pytest.approx(islands.propulator.population[pop_key].position)
                    count += 1
    assert count > 0

    final_synch(islands.propulator)
    population_consistency_check(islands.propulator)
    # NOTE check that the total number is correct
    # NOTE because of migration it only has to be summed over all islands
    MPI.COMM_WORLD.barrier()
    final_pop_sizes = islands.propulator.propulate_comm.allgather(len(islands.propulator.population))
    assert final_pop_sizes[0] + final_pop_sizes[3] == second_generations * islands.propulator.propulate_comm.size
    assert len(islands.propulator._get_active_individuals()) == second_generations * island_sizes[island_idx]
    # NOTE check there are no unevaluated individuals anymore
    assert [np.isnan(ind.loss) for ind in islands.propulator.population.values()].count(False) == len(islands.propulator.population)
    # NOTE check that loss for entire population is written to disk
    if MPI.COMM_WORLD.rank == 0:
        with h5py.File(mpi_tmp_path / "ckpt.hdf5", "r") as f:
            for island in range(island_sizes.size):
                for worker in range(island_sizes[island]):
                    assert not np.isnan(f[f"{island}"][f"{worker}"]["loss"][:].any())

    # NOTE check that no started individuals have been overwritten
    count = 0
    for island in range(island_sizes.size):
        for island_rank in range(island_sizes[island]):
            prop_rank = island_rank + np.cumsum(np.concatenate((np.array([0]), island_sizes)))[island]
            finished_first_generations = started_first_generations[prop_rank]
            if island == 0 and island_rank == 2:
                finished_first_generations += 1
            if island == 1 and island_rank == 2:
                finished_first_generations += 1
            for g in range(finished_first_generations):
                pop_key = (island, island_rank, g)
                if pop_key in old_population:
                    assert old_population[pop_key].position == pytest.approx(islands.propulator.population[pop_key].position)
                    count += 1
    assert count > 0

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
    # TODO
    pass
