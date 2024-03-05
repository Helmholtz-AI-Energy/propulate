"""Simple Propulator example script."""
import random
import argparse
import logging
import pathlib

from mpi4py import MPI

from propulate import Propulator
from propulate.utils import get_default_propagator, set_logger_config
from function_benchmark import get_function_search_space

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
        "--pop_size", type=int, default=2 * comm.size
    )  # Breeding pool size
    parser.add_argument(
        "--crossover_probability", type=float, default=0.7
    )  # Crossover probability
    parser.add_argument(
        "--mutation_probability", type=float, default=0.4
    )  # Mutation probability
    parser.add_argument("--random_init_probability", type=float, default=0.1)
    parser.add_argument("--top_n", type=int, default=1)
    parser.add_argument("--logging_int", type=int, default=10)
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
        loss_fn=function,  # Loss function to be minimized
        propagator=propagator,  # Propagator, i.e., evolutionary operator to be used
        island_comm=comm,  # Communicator to be used
        generations=config.generations,  # Number of generations
        checkpoint_path=config.checkpoint,  # Checkpoint path
        rng=rng,  # Separate random number generator for Propulate optimization
    )

    # Run optimization and print summary of results.
    propulator.propulate(
        logging_interval=config.logging_int,  # Logging interval
        debug=config.verbosity,  # Verbosity level
    )
    propulator.summarize(
        top_n=config.top_n,  # Print top-n best individuals on each island in summary.
        debug=config.verbosity,  # Verbosity level
    )
