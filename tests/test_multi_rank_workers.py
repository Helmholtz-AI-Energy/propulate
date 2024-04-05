import random
from typing import Dict

import numpy as np
import pytest
from mpi4py import MPI

from propulate import Islands
from propulate.propagators import SelectMax, SelectMin
from propulate.utils import get_default_propagator, set_logger_config


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
        term = (
            list(params.values())[comm.rank] ** 2
        )  # Each rank squares one of the inputs.
        return comm.allreduce(term)  # Return the sum over all squared inputs.
    else:
        return np.sum(np.array(list(params.values())) ** 2).item()


@pytest.mark.mpi(min_size=8)
def test_multi_rank_workers(mpi_tmp_path):
    """Test multi rank workers. Two Islands with at least two workers with two ranks each."""
    full_world_comm = MPI.COMM_WORLD  # Get full world communicator.
    set_logger_config()

    rng = random.Random(42 + full_world_comm.rank)
    limits = {
        "a": (-5.12, 5.12),
        "b": (-5.12, 5.12),
    }
    function = parallel_sphere
    propagator = get_default_propagator(  # Get default evolutionary operator.
        pop_size=2,  # Breeding pool size
        limits=limits,  # Search-space limits
        rng=rng,  # Separate random number generator for Propulate optimization
    )

    # Set up migration topology.
    migration_topology = 1 * np.ones((2, 2), dtype=int)
    np.fill_diagonal(
        migration_topology, 0
    )  # An island does not send migrants to itself.

    # Set up island model.
    islands = Islands(
        loss_fn=function,  # Loss function to be minimized
        propagator=propagator,  # Propagator, i.e., evolutionary operator to be used
        rng=rng,  # Separate random number generator for Propulate optimization
        generations=1000,  # Overall number of generations
        num_islands=2,  # Number of islands
        migration_topology=migration_topology,  # Migration topology
        migration_probability=0.9,  # Migration probability
        emigration_propagator=SelectMin,  # Selection policy for emigrants
        immigration_propagator=SelectMax,  # Selection policy for immigrants
        pollination=False,  # Whether to use pollination or migration
        checkpoint_path=mpi_tmp_path,  # Checkpoint path
        # ----- SPECIFIC FOR MULTI-RANK UCS ----
        ranks_per_worker=2,  # Number of ranks per (multi rank) worker
    )

    # Run actual optimization.
    islands.evolve(
        top_n=1,  # Print top-n best individuals on each island in summary.
        logging_interval=10,  # Logging interval
        debug=1,  # Debug level
    )
