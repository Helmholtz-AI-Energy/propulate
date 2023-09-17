#!/usr/bin/env python3
import random
import argparse
import logging

from mpi4py import MPI

from propulate import Propulator
from propulate.utils import get_default_propagator, set_logger_config
from function_benchmark import *

if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        print(
            "#################################################\n"
            "# PROPULATE: Parallel Propagator of Populations #\n"
            "#################################################\n"
        )

    parser = argparse.ArgumentParser(
        prog="Simple Propulator example",
        description="Set up and run a basic Propulator optimization of mathematical functions.",
    )
    parser.add_argument(  # Function to optimize
        "-f",
        "--function",
        type=str,
        choices=[
            "bukin",
            "eggcrate",
            "himmelblau",
            "keane",
            "leon",
            "rastrigin",
            "schwefel",
            "sphere",
            "step",
            "rosenbrock",
            "quartic",
            "bisphere",
            "birastrigin",
            "griewank",
        ],
        default="sphere",
    )
    parser.add_argument(
        "-g", "--generations", type=int, default=1000
    )  # Number of generations
    parser.add_argument(
        "-s", "--seed", type=int, default=0
    )  # Seed for Propulate random number generator
    parser.add_argument("-v", "--verbosity", type=int, default=1)  # Verbosity level
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, default="./"
    )  # Path for loading and writing checkpoints.
    parser.add_argument(
        "-p", "--pop_size", type=int, default=2 * comm.size
    )  # Breeding pool size
    parser.add_argument(
        "-cp", "--crossover_probability", type=float, default=0.7
    )  # Crossover probability
    parser.add_argument(
        "-mp", "--mutation_probability", type=float, default=0.4
    )  # Mutation probability
    parser.add_argument("-rp", "--random_init_probability", type=float, default=0.1)
    parser.add_argument("-t", "--top_n", type=int, default=1)
    parser.add_argument("-l", "--logging_int", type=int, default=10)
    config = parser.parse_args()

    # Set up separate logger for Propulate optimization.
    set_logger_config(
        level=logging.INFO,  # logging level
        log_file=f"{config.checkpoint}/propulator.log",  # logging path
        log_to_stdout=True,  # Print log on stdout.
        log_rank=False,  # Do not prepend MPI rank to logging messages.
        colors=True,  # Use colors.
    )

    rng = random.Random(
        config.seed + comm.rank
    )  # Separate random number generator for optimization.
    function, limits = get_function_search_space(
        config.function
    )  # Get callable function + search-space limits.

    # Set up evolutionary operator.
    propagator = get_default_propagator(  # Get default evolutionary operator.
        pop_size=config.pop_size,  # Breeding pool size
        limits=limits,  # Search-space limits
        mate_prob=config.crossover_probability,  # Crossover probability
        mut_prob=config.mutation_probability,  # Mutation probability
        random_prob=config.random_init_probability,  # Random-initialization probability
        rng=rng,  # Random number generator
    )

    # Set up propulator performing actual optimization.
    propulator = Propulator(
        loss_fn=function,
        propagator=propagator,
        comm=comm,
        generations=config.generations,
        checkpoint_path=config.checkpoint,
        rng=rng,
    )

    # Run optimization and print summary of results.
    propulator.propulate(logging_interval=config.logging_int, debug=config.verbosity)
    propulator.summarize(top_n=config.top_n, debug=config.verbosity)
