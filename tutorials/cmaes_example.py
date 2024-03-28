"""Simple example script using CMA-ES."""
import pathlib
import random

from function_benchmark import get_function_search_space, parse_arguments
from mpi4py import MPI

from propulate import Propulator
from propulate.propagators import ActiveCMA, BasicCMA, CMAPropagator
from propulate.utils import set_logger_config

if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        print(
            "#################################################\n"
            "# PROPULATE: Parallel Propagator of Populations #\n"
            "#################################################\n"
        )

    config, _ = parse_arguments(comm)

    # Set up separate logger for Propulate optimization.
    set_logger_config(
        level=config.logging_level,  # Logging level
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
    if config.adapter == "basic":
        adapter = BasicCMA()
    elif config.adapter == "active":
        adapter = ActiveCMA()
    else:
        raise ValueError("Adapter can be either 'basic' or 'active'.")

    propagator = CMAPropagator(adapter, limits, rng=rng)

    # Set up propulator performing actual optimization.
    propulator = Propulator(
        loss_fn=function,
        propagator=propagator,
        rng=rng,
        island_comm=comm,
        generations=config.generations,
        checkpoint_path=config.checkpoint,
    )

    # Run optimization and print summary of results.
    propulator.propulate(
        logging_interval=config.logging_interval, debug=config.verbosity
    )
    propulator.summarize(top_n=config.top_n, debug=config.verbosity)
