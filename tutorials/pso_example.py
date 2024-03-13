"""
This files contains an example use case for the PSO propagators. Here, you can choose between benchmark functions and
optimize them. The example shows how to set up Propulate in order to use it with PSO.
"""
import argparse
import pathlib
import random
from typing import Dict

from mpi4py import MPI

from propulate import set_logger_config, Propulator
from propulate.propagators import Conditional, Propagator
from propulate.propagators.pso import (
    BasicPSO,
    VelocityClampingPSO,
    ConstrictionPSO,
    CanonicalPSO,
    InitUniformPSO,
)
from function_benchmark import get_function_search_space, parse_arguments

if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        print(
            "#################################################\n"
            "# PROPULATE: Parallel Propagator of Populations #\n"
            "#################################################\n"
        )

    config, hp_set = parse_arguments(comm)

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

    if config.variant in ("Constriction", "Canonical"):
        if not hp_set["cognitive"]:
            config.cognitive = 2.05
        if not hp_set["social"]:
            config.social = 2.05

    pso_propagator: Propagator
    if config.variant == "Basic":
        pso_propagator = BasicPSO(
            config.inertia,
            config.cognitive,
            config.social,
            MPI.COMM_WORLD.rank,
            limits,
            rng,
        )
    elif config.variant == "VelocityClamping":
        pso_propagator = VelocityClampingPSO(
            config.inertia,
            config.cognitive,
            config.social,
            MPI.COMM_WORLD.rank,
            limits,
            rng,
            config.clamping_factor,
        )
    elif config.variant == "Constriction":
        pso_propagator = ConstrictionPSO(
            config.cognitive, config.social, MPI.COMM_WORLD.rank, limits, rng
        )
    elif config.variant == "Canonical":
        pso_propagator = CanonicalPSO(
            config.cognitive, config.social, MPI.COMM_WORLD.rank, limits, rng
        )
    else:
        raise ValueError("Invalid PSO propagator name given.")
    init = InitUniformPSO(limits, rng=rng, rank=MPI.COMM_WORLD.rank)
    propagator = Conditional(config.pop_size, pso_propagator, init)

    propulator = Propulator(
        function,
        propagator,
        rng=rng,
        island_comm=comm,
        generations=config.generations,
        checkpoint_path=config.checkpoint,
    )
    propulator.propulate(
        logging_interval=config.logging_interval, debug=config.verbosity
    )
    propulator.summarize(top_n=config.top_n, debug=config.verbosity)
