import copy
import logging
import pathlib
import random

import deepdiff
import h5py
import numpy as np
import pytest
from mpi4py import MPI

from propulate import Propulator
from propulate.utils import get_default_propagator, set_logger_config
from propulate.utils.benchmark_functions import get_function_search_space
from propulate.utils.consistency_checks import final_synch, population_consistency_check


@pytest.fixture(
    params=[
        "rosenbrock",
        "step",
        "quartic",
        "rastrigin",
        "griewank",
        "schwefel",
        "bisphere",
        "birastrigin",
        "bukin",
        "eggcrate",
        "himmelblau",
        "keane",
        "leon",
        "sphere",
    ]
)
def function_name(request: pytest.FixtureRequest) -> str:
    """Define benchmark function parameter sets as used in tests."""
    return request.param


def test_propulator_simple(function_name: str, mpi_tmp_path: pathlib.Path) -> None:
    """
    Test standard Propulator to optimize the benchmark functions using the default genetic propagator.

    This test is run both sequentially and in parallel.

    Parameters
    ----------
    function_name : str
        The function name.
    mpi_tmp_path : pathlib.Path
        The temporary checkpoint directory.
    """
    rng = random.Random(42 + MPI.COMM_WORLD.rank)  # Random number generator for optimization

    log = logging.getLogger("propulate")  # Get logger instance.
    set_logger_config()
    benchmark_function, limits = get_function_search_space(function_name)
    propagator = get_default_propagator(
        pop_size=4,
        limits=limits,
        rng=rng,
    )  # Set up evolutionary operator.
    propulator = Propulator(
        loss_fn=benchmark_function,
        propagator=propagator,
        rng=rng,
        generations=10,
        checkpoint_path=mpi_tmp_path,
    )  # Set up propulator performing actual optimization.
    propulator.propulate()  # Run optimization and print summary of results.
    final_synch(propulator)
    population_consistency_check(propulator)

    log.handlers.clear()


def test_propulator_checkpointing(mpi_tmp_path: pathlib.Path) -> None:
    """
    Test standard Propulator checkpointing for the sphere benchmark function.

    This test is run both sequentially and in parallel.

    Parameters
    ----------
    mpi_tmp_path : pathlib.Path
        The temporary checkpoint directory.
    """
    first_generations = 20
    second_generations = 40
    log = logging.getLogger("propulate")  # Get logger instance.
    set_logger_config(level=logging.DEBUG)
    rng = random.Random(42 + MPI.COMM_WORLD.rank)  # Separate random number generator for optimization
    benchmark_function, limits = get_function_search_space("sphere")

    propagator = get_default_propagator(  # Get default evolutionary operator.
        pop_size=4,  # Breeding pool size
        limits=limits,  # Search-space limits
        rng=rng,  # Random number generator
    )
    propulator = Propulator(
        loss_fn=benchmark_function,
        propagator=propagator,
        generations=first_generations,
        checkpoint_path=mpi_tmp_path,
        rng=rng,
    )  # Set up propulator performing actual optimization.

    propulator.propulate()  # Run optimization and print summary of results.
    final_synch(propulator)
    population_consistency_check(propulator)
    assert len(propulator.population) == first_generations * propulator.propulate_comm.Get_size()

    old_population = copy.deepcopy(propulator.population)  # Save population list from the last run.
    del propulator  # Delete propulator object.
    MPI.COMM_WORLD.barrier()  # Synchronize all processes.

    propulator = Propulator(
        loss_fn=benchmark_function,
        propagator=propagator,
        generations=second_generations,
        checkpoint_path=mpi_tmp_path,
        rng=rng,
    )  # Set up new propulator starting from checkpoint.

    # As the number of requested generations is smaller than the number of generations from the run before,
    # no new evaluations are performed. Thus, the length of both Propulators' populations must be equal.
    assert len(deepdiff.DeepDiff(old_population, propulator.population, ignore_order=True)) == 0
    propulator.propulate()
    final_synch(propulator)
    population_consistency_check(propulator)
    # NOTE make sure nothing was overwritten
    seniors = {k: v for (k, v) in propulator.population.items() if v.generation < first_generations}
    assert len(deepdiff.DeepDiff(old_population, seniors, ignore_order=True)) == 0
    assert len(propulator.population) == second_generations * propulator.propulate_comm.Get_size()

    log.handlers.clear()


def test_propulator_checkpointing_incomplete(mpi_tmp_path: pathlib.Path) -> None:
    """
    Test Propulator checkpointing where the last in the checkpoint evaluation has not finished.

    Parameters
    ----------
    mpi_tmp_path : pathlib.Path
        The temporary checkpoint directory.
    """
    first_generations = 20
    second_generations = 40
    log = logging.getLogger("propulate")  # Get logger instance.
    set_logger_config(level=logging.DEBUG)
    rng = random.Random(42 + MPI.COMM_WORLD.rank)  # Separate random number generator for optimization
    benchmark_function, limits = get_function_search_space("sphere")

    propagator = get_default_propagator(  # Get default evolutionary operator.
        pop_size=4,  # Breeding pool size
        limits=limits,  # Search-space limits
        rng=rng,  # Random number generator
    )
    propulator = Propulator(
        loss_fn=benchmark_function,
        propagator=propagator,
        generations=first_generations,
        checkpoint_path=mpi_tmp_path,
        rng=rng,
    )  # Set up propulator performing actual optimization.

    propulator.propulate()  # Run optimization and
    final_synch(propulator)
    population_consistency_check(propulator)
    assert len(propulator.population) == first_generations * propulator.propulate_comm.Get_size()
    MPI.COMM_WORLD.barrier()  # Synchronize all processes.
    # NOTE manipulate the written checkpoint, delete last result for some
    # NOTE this is an index, so e.g. [0, 1, 2] are present not just [0, 1]
    started_first_generations = [
        2,
        first_generations - 1,
        first_generations // 2,
        first_generations // 3,
    ] + [first_generations - 1] * max(0, MPI.COMM_WORLD.size - 4)
    started_first_generations = started_first_generations[0 : MPI.COMM_WORLD.size]

    if MPI.COMM_WORLD.rank == 0:
        with h5py.File(mpi_tmp_path / "ckpt.hdf5", "r+") as f:
            for i, g in enumerate(started_first_generations):
                f["generations"][i] = g
            for worker, g in enumerate(started_first_generations[:-1]):  # NOTE last worker has fully evaluated last ind
                f["0"][f"{worker}"]["loss"][g:] = np.nan

    MPI.COMM_WORLD.barrier()  # Synchronize all processes.
    # NOTE without a final synch these might be incomplete on some ranks
    # NOTE that's why we only compare against the ones that are present
    old_population = copy.deepcopy(propulator.population)  # Save population list from the last run.
    del propulator  # Delete propulator object.
    MPI.COMM_WORLD.barrier()  # Synchronize all processes.

    propulator = Propulator(
        loss_fn=benchmark_function,
        propagator=propagator,
        generations=second_generations,
        checkpoint_path=mpi_tmp_path,
        rng=rng,
    )  # Set up new propulator starting from checkpoint.
    # NOTE check that only the correct number of individuals were read
    assert len(propulator.population) == sum(started_first_generations) + len(started_first_generations)
    # NOTE check that in the loaded population all ranks but one have an unevaluated individual
    for rank in range(MPI.COMM_WORLD.size - 1):
        pop_key = (0, rank, started_first_generations[rank])
        assert np.isnan(propulator.population[pop_key].loss)
    assert sum([np.isnan(ind.loss) for ind in propulator.population.values()]) == MPI.COMM_WORLD.size - 1

    propulator.propulate()
    # NOTE check that the total number is correct
    final_synch(propulator)
    population_consistency_check(propulator)

    assert len(propulator.population) == second_generations * propulator.propulate_comm.Get_size()
    # NOTE check there are no unevaluated individuals anymore
    assert [np.isnan(ind.loss) for ind in propulator.population.values()].count(False) == len(propulator.population)

    # NOTE check that loss for entire population is written to disk
    if MPI.COMM_WORLD.rank == 0:
        with h5py.File(mpi_tmp_path / "ckpt.hdf5", "r") as f:
            for worker in range(MPI.COMM_WORLD.size):
                assert not np.isnan(f["0"][f"{worker}"]["loss"][:]).any()

    # NOTE check that no started individuals have been overwritten
    # NOTE old_population might be incomplete without final synch, so we only compare the ones we have.
    # NOTE over all ranks, each ind should be checked on at least once.
    count = 0
    for rank in range(MPI.COMM_WORLD.size):
        for g in range(started_first_generations[rank] + 1):
            pop_key = (0, rank, started_first_generations[rank])
            if pop_key in old_population:
                count += 1
                assert old_population[pop_key].position == pytest.approx(propulator.population[pop_key].position)
    assert count > 0
    # TODO check the worker that got finished

    log.handlers.clear()
