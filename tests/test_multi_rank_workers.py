import logging
import pathlib
import random
from typing import Dict

import h5py
import numpy as np
import pytest
from mpi4py import MPI

from propulate import Islands
from propulate.utils import get_default_propagator, set_logger_config
from propulate.utils.consistency_checks import final_synch, population_consistency_check

log = logging.getLogger("propulate")  # Get logger instance.


@pytest.fixture(
    params=[
        True,
        False,
    ]
)
def pollination(request: pytest.FixtureRequest) -> bool:
    """Iterate through pollination parameter."""
    return request.param


def parallel_sphere(params: Dict[str, float], comm: MPI.Comm = MPI.COMM_SELF) -> float:
    """
    Parallel sphere function to showcase using multi-rank workers in Propulate.

    Sphere function: continuous, convex, separable, differentiable, unimodal

    Input domain: -5.12 <= x, y <= 5.12
    Global minimum 0 at (x, y) = (0, 0)

    Parameters
    ----------
    params : Dict[str, float]
        The function parameters.
    comm : MPI.Comm
        The communicator of the worker.

    Returns
    -------
    float
        The function value.
    """
    if comm != MPI.COMM_SELF:
        term = list(params.values())[comm.rank] ** 2  # Each rank squares one of the inputs.
        return comm.allreduce(term)  # Return the sum over all squared inputs.
    else:
        return np.sum(np.array(list(params.values())) ** 2).item()


@pytest.mark.mpi(min_size=8)
def test_multi_rank_workers(pollination: bool, mpi_tmp_path: pathlib.Path) -> None:
    """
    Test multi-rank workers. Two islands with two workers with two ranks each.

    Parameters
    ----------
    mpi_tmp_path : pathlib.Path
        The temporary checkpoint directory.
    """
    set_logger_config(level=logging.DEBUG)
    full_world_comm = MPI.COMM_WORLD  # Get full world communicator.

    rng = random.Random(42 + full_world_comm.rank)
    limits = {
        "a": (-5.12, 5.12),
        "b": (-5.12, 5.12),
    }
    propagator = get_default_propagator(  # Get default evolutionary operator.
        pop_size=2,  # Breeding pool size
        limits=limits,  # Search-space limits
        rng=rng,  # Separate random number generator for Propulate optimization
    )

    # Set up island model.
    islands = Islands(
        loss_fn=parallel_sphere,  # Loss function to be minimized
        propagator=propagator,  # Propagator, i.e., evolutionary operator to be used
        rng=rng,  # Separate random number generator for Propulate optimization
        generations=10,  # Overall number of generations
        num_islands=2,  # Number of islands
        migration_probability=0.9,  # Migration probability
        pollination=pollination,  # Whether to use pollination or migration
        checkpoint_path=mpi_tmp_path,  # Checkpoint path
        # ----- SPECIFIC FOR MULTI-RANK UCS ----
        ranks_per_worker=2,  # Number of ranks per (multi rank) worker
    )

    # Run actual optimization.
    log.debug("initialized islands")
    islands.propulate(
        logging_interval=10,  # Logging interval
    )
    log.handlers.clear()


@pytest.mark.mpi(min_size=8)
def test_multi_rank_workers_resume_unfinished(mpi_tmp_path: pathlib.Path) -> None:
    """
    Test resuming multi-rank workers from a checkpoint in which the last evaluations have not finished.

    All ranks of a worker have to take part in the evaluation of the resumed individual (the loss function is
    collective over the worker's ranks), so the worker's root rank must involve them before the ordinary loop.
    """
    set_logger_config(level=logging.DEBUG)
    full_world_comm = MPI.COMM_WORLD
    rng = random.Random(42 + full_world_comm.rank)
    limits = {"a": (-5.12, 5.12), "b": (-5.12, 5.12)}
    island_sizes = np.array([2, 2])  # 8 ranks: two islands with two workers with two ranks each
    first_generations = 10
    second_generations = 20

    def make(generations: int) -> Islands:
        return Islands(
            loss_fn=parallel_sphere,
            propagator=get_default_propagator(pop_size=2, limits=limits, rng=rng),
            rng=rng,
            generations=generations,
            num_islands=2,
            island_sizes=island_sizes,
            migration_probability=0.0,
            checkpoint_path=mpi_tmp_path,
            ranks_per_worker=2,
        )

    islands = make(first_generations)
    islands.propulate()
    del islands
    full_world_comm.barrier()

    # NOTE pretend that no worker finished evaluating its last individual
    if full_world_comm.rank == 0:
        with h5py.File(mpi_tmp_path / "ckpt.hdf5", "r+") as f:
            for island in range(len(island_sizes)):
                for worker in range(island_sizes[island]):
                    f[f"{island}"][f"{worker}"]["loss"][first_generations - 1] = np.nan
    full_world_comm.barrier()

    islands = make(second_generations)
    propulator = islands.propulator
    if propulator.propulate_comm is not None:
        assert propulator.generation == first_generations - 1
        assert np.isnan(propulator.population[(propulator.island_idx, propulator.island_comm.rank, first_generations - 1)].loss)
    islands.propulate()
    if propulator.propulate_comm is not None:
        final_synch(propulator)
        population_consistency_check(propulator)
        assert all(not np.isnan(ind.loss) for ind in propulator.population.values())
    full_world_comm.barrier()
    if full_world_comm.rank == 0:
        with h5py.File(mpi_tmp_path / "ckpt.hdf5", "r") as f:
            for island in range(len(island_sizes)):
                for worker in range(island_sizes[island]):
                    assert not np.isnan(f[f"{island}"][f"{worker}"]["loss"][:]).any()
    log.handlers.clear()


# TODO surrogate
