"""
Multi-rank workers in Propulate using the example of a parallel sphere function.
Tested with 8 processes overall, 2 islands, and 2 ranks per worker, where each worker calculates one of the squared
terms in the (in this case) two-dimensional sphere function. In general, the parallel sphere function's dimension
should equal the number of ranks per worker.
"""
import argparse
import logging
import random
import pathlib
from typing import Dict

import numpy as np
from mpi4py import MPI

from propulate import Islands
from propulate.propagators import SelectMin, SelectMax
from propulate.utils import get_default_propagator, set_logger_config


def parallel_sphere(params: Dict[str, float], comm: MPI.Comm) -> float:
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
    term = list(params.values())[comm.rank] ** 2  # Each rank squares one of the inputs.
    return comm.allreduce(term)  # Return the sum over all squared inputs.


if __name__ == "__main__":
    full_world_comm = MPI.COMM_WORLD  # Get full world communicator.

    parser = argparse.ArgumentParser(
        prog="Simple Propulator example",
        description="Set up and run a basic Propulator optimization of mathematical functions.",
    )
    parser.add_argument(
        "--generations", type=int, default=1000
    )  # Number of generations
    parser.add_argument(
        "--seed", type=int, default=0
    )  # Seed for Propulate random number generator
    parser.add_argument("--verbosity", type=int, default=1)  # Verbosity level
    parser.add_argument(
        "--checkpoint", type=str, default="./"
    )  # Path for loading and writing checkpoints.
    parser.add_argument(
        "--pop_size", type=int, default=2 * full_world_comm.size
    )  # Breeding pool size
    parser.add_argument(
        "--crossover_probability", type=float, default=0.7
    )  # Crossover probability
    parser.add_argument(
        "--mutation_probability", type=float, default=0.4
    )  # Mutation probability
    parser.add_argument("--random_init_probability", type=float, default=0.1)
    parser.add_argument(
        "--num_islands", type=int, default=2
    )  # Number of separate evolutionary islands
    parser.add_argument(
        "--ranks_per_worker", type=int, default=2
    )  # number of sub ranks that each worker will use
    parser.add_argument(
        "--migration_probability", type=float, default=0.9
    )  # Migration probability
    parser.add_argument("--num_migrants", type=int, default=1)
    parser.add_argument("--pollination", action="store_true")
    parser.add_argument(
        "--top_n", type=int, default=1
    )  # Print top-n best individuals on each island in summary.
    parser.add_argument("--logging_int", type=int, default=10)  # Logging interval
    config = parser.parse_args()

    # Set up separate logger for Propulate optimization.
    set_logger_config(
        level=logging.INFO,  # Logging level
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
        mate_prob=config.crossover_probability,  # Crossover probability
        mut_prob=config.mutation_probability,  # Mutation probability
        random_prob=config.random_init_probability,  # Random-initialization probability
        rng=rng,  # Random number generator
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
        rng=rng,  # Separate random number generator
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
    islands.evolve(
        top_n=config.top_n,  # Print top-n best individuals on each island in summary.
        logging_interval=config.logging_int,  # Logging interval
        debug=config.verbosity,  # Verbosity level
    )
