#!/usr/bin/env python3
import argparse
import random
from mpi4py import MPI

from propulate import Islands
from propulate.propagators import SelectMin, SelectMax
from propulate.utils import get_default_propagator
from function_benchmark import *


if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    parser = argparse.ArgumentParser(
        prog="Simple Propulator example",
        description="Set up and run a basic Propulator optimization of mathematical functions."
    )
    parser.add_argument(  # Function to optimize
        "-f",
        "--function",
        type=str,
        choices=[
            "bukin", "eggcrate", "himmelblau", "keane", "leon", "rastrigin", "schwefel", "sphere", "step",
            "rosenbrock", "quartic", "bisphere", "birastrigin", "griewank"
        ],
        default="sphere"
    )
    parser.add_argument("-g", "--generations", type=int, default=1000)  # Number of generations
    parser.add_argument("-s", "--seed", type=int, default=0)  # Seed for Propulate random number generator
    parser.add_argument("-v", "--verbosity", type=int, default=1)  # Verbosity level
    parser.add_argument("-ckpt", "--checkpoint", type=str, default="./")  # Path for loading and writing checkpoints.
    parser.add_argument("-p", "--pop_size", type=int, default=2*comm.size)  # Breeding pool size
    parser.add_argument("-cp", "--crossover_probability", type=float, default=0.7)  # Crossover probability
    parser.add_argument("-mp", "--mutation_probability", type=float, default=0.4)  # Mutation probability
    parser.add_argument("-rp", "--random_init_probability", type=float, default=0.1)
    parser.add_argument("-i", "--num_islands", type=int, default=2)  # Number of separate evolutionary islands
    parser.add_argument("-migp", "--migration_probability", type=float, default=0.9)  # Migration probability
    parser.add_argument("-m", "--num_migrants", type=int, default=1)
    parser.add_argument("-pln", "--pollination", action="store_true")
    parser.add_argument("-t", "--top_n", type=int, default=1)  # Print top-n best individuals on each island in summary.
    parser.add_argument("-l", "--logging_int", type=int, default=10)  # Logging interval
    config = parser.parse_args()

    rng = random.Random(config.seed+comm.rank)  # Separate random number generator for optimization.
    function, limits = get_function_search_space(config.function)  # Get callable function + search-space limits.

    # Set up evolutionary operator.
    propagator = get_default_propagator(  # Get default evolutionary operator.
        pop_size=config.pop_size,  # Breeding pool size
        limits=limits,  # Search-space limits
        mate_prob=config.crossover_probability,  # Crossover probability
        mut_prob=config.mutation_probability,  # Mutation probability
        random_prob=config.random_init_probability,  # Random-initialization probability
        rng=rng  # Random number generator
    )

    # Set up migration topology.
    migration_topology = config.num_migrants * np.ones(        # Set up fully connected migration topology.
        (config.num_islands, config.num_islands),
        dtype=int
    )
    np.fill_diagonal(migration_topology, 0)             # An island does not send migrants to itself.

    # Set up island model.
    islands = Islands(
        loss_fn=function,
        propagator=propagator,
        rng=rng,
        generations=config.generations,
        num_islands=config.num_islands,
        migration_topology=migration_topology,
        migration_probability=config.migration_probability,
        emigration_propagator=SelectMin,
        immigration_propagator=SelectMax,
        pollination=config.pollination,
        checkpoint_path=config.checkpoint
    )

    # Run actual optimization.
    islands.evolve(top_n=config.top_n, logging_interval=config.logging_int, debug=config.verbosity)
