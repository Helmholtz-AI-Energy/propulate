"""
Multi-rank workers in Propulate using the example of a parallel sphere function.

Tested with 8 processes overall, 2 islands, and 2 ranks per worker, where each worker calculates one of the squared
terms in the (in this case) two-dimensional sphere function. In general, the parallel sphere function's dimension
should equal the number of ranks per worker.
"""
import pathlib
import random
from typing import Dict

import numpy as np
from mpi4py import MPI

from propulate import Islands
from propulate.propagators import SelectMax, SelectMin
from propulate.utils import get_default_propagator, set_logger_config
from propulate.utils.benchmark_functions import parse_arguments


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


if __name__ == "__main__":
    full_world_comm = MPI.COMM_WORLD  # Get full world communicator.

    config, _ = parse_arguments()

    # Set up separate logger for Propulate optimization.
    set_logger_config(
        level=config.logging_level,  # Logging level
        log_file=f"{config.checkpoint}/{pathlib.Path(__file__).stem}.log",  # Logging path
        log_to_stdout=True,  # Print log on stdout.
        log_rank=False,  # Do not prepend MPI rank to logging messages.
        colors=True,  # Use colors.
    )

    rng = random.Random(
        config.seed + full_world_comm.rank
    )  # Separate random number generator for optimization.
    # Set callable function + search-space limits.
    limits = {
        "a": (-5.12, 5.12),
        "b": (-5.12, 5.12),
    }
    function = parallel_sphere

    # Set up evolutionary operator.
    propagator = get_default_propagator(  # Get default evolutionary operator.
        pop_size=config.pop_size,  # Breeding pool size
        limits=limits,  # Search-space limits
        crossover_prob=config.crossover_probability,  # Crossover probability
        mutation_prob=config.mutation_probability,  # Mutation probability
        random_init_prob=config.random_init_probability,  # Random-initialization probability
        rng=rng,  # Separate random number generator for Propulate optimization
    )

    # Set up migration topology.
    migration_topology = (
        config.num_migrants
        * np.ones(  # Set up fully connected migration topology.
            (config.num_islands, config.num_islands), dtype=int
        )
    )
    np.fill_diagonal(
        migration_topology, 0
    )  # An island does not send migrants to itself.

    # Set up island model.
    islands = Islands(
        loss_fn=function,  # Loss function to be minimized
        propagator=propagator,  # Propagator, i.e., evolutionary operator to be used
        rng=rng,  # Separate random number generator for Propulate optimization
        generations=config.generations,  # Overall number of generations
        num_islands=config.num_islands,  # Number of islands
        migration_topology=migration_topology,  # Migration topology
        migration_probability=config.migration_probability,  # Migration probability
        emigration_propagator=SelectMin,  # Selection policy for emigrants
        immigration_propagator=SelectMax,  # Selection policy for immigrants
        pollination=config.pollination,  # Whether to use pollination or migration
        checkpoint_path=config.checkpoint,  # Checkpoint path
        # ----- SPECIFIC FOR MULTI-RANK UCS ----
        ranks_per_worker=config.ranks_per_worker,  # Number of ranks per (multi rank) worker
    )

    # Run actual optimization.
    islands.propulate(
        logging_interval=config.logging_interval,  # Logging interval
        debug=config.verbosity,  # Debug level
    )
    islands.summarize(top_n=config.top_n, debug=config.verbosity)
