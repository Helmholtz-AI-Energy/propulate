"""
This files contains an example use case for the PSO propagators. Here, you can choose between benchmark functions and
optimize them.

The example shows, how to set up Propulate in order to use it with PSO.
"""
import argparse
import random
from typing import Dict

from mpi4py import MPI

from propulate import set_logger_config, Propulator
from propulate.propagators import Conditional
from propulate.propagators.pso import (
    Basic,
    VelocityClamping,
    Constriction,
    Canonical,
    InitUniform,
)
from tutorials.function_benchmark import get_function_search_space

if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        print(
            "#################################################\n"
            "# PROPULATE: Parallel Propagator of Populations #\n"
            "#################################################\n"
        )

    parser = argparse.ArgumentParser(
        prog="Simple PSO example",
        description="Set up and run a basic particle swarm optimization of mathematical functions.",
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
    parser.add_argument(
        "-v", "--verbosity", type=int, default=1, choices=range(6)
    )  # Verbosity level
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, default="./"
    )  # Path for loading and writing checkpoints.
    parser.add_argument(
        "-p", "--pop_size", type=int, default=2 * comm.size
    )  # Breeding pool size
    parser.add_argument(
        "-var",
        "--variant",
        type=str,
        choices=["Basic", "VelocityClamping", "Constriction", "Canonical"],
        default="Basic",
    )  # PSO variant to run

    hp_set: Dict[str, bool] = {
        "inertia": False,
        "cognitive": False,
        "social": False,
    }

    class ParamSettingCatcher(argparse.Action):
        """
        This class extends argparse's Action class in order to allow for an action, that logs, if one of the PSO HP
        was actually set.
        """

        def __call__(self, parser, namespace, values, option_string=None):
            hp_set[self.dest] = True
            super().__call__(parser, namespace, values, option_string)

    parser.add_argument(
        "--inertia", type=float, default=0.729, action=ParamSettingCatcher
    )  # Inertia weight
    parser.add_argument(
        "--cognitive", type=float, default=1.49445, action=ParamSettingCatcher
    )  # Cognitive factor
    parser.add_argument(
        "--social", type=float, default=1.49445, action=ParamSettingCatcher
    )  # Social factor
    parser.add_argument(
        "--clamping_factor", type=float, default=0.6
    )  # Clamping factor for velocity clamping
    parser.add_argument("-t", "--top_n", type=int, default=1)
    parser.add_argument("-l", "--logging_int", type=int, default=10)
    config = parser.parse_args()

    # Set up separate logger for Propulate optimization.
    set_logger_config(
        level=10 * config.verbosity,  # logging level
        log_file=f"{config.checkpoint}/propulator.log",  # logging path
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
    pso_propagator = {
        "Basic": Basic(
            config.inertia,
            config.cognitive,
            config.social,
            MPI.COMM_WORLD.rank,
            limits,
            rng,
        ),
        "VelocityClamping": VelocityClamping(
            config.inertia,
            config.cognitive,
            config.social,
            MPI.COMM_WORLD.rank,
            limits,
            rng,
            config.clamping_factor,
        ),
        "Constriction": Constriction(
            config.cognitive, config.social, MPI.COMM_WORLD.rank, limits, rng
        ),
        "Canonical": Canonical(
            config.cognitive, config.social, MPI.COMM_WORLD.rank, limits, rng
        ),
    }[config.variant]
    init = InitUniform(limits, rng=rng, rank=MPI.COMM_WORLD.rank)
    propagator = Conditional(config.pop_size, pso_propagator, init)

    propulator = Propulator(
        function,
        propagator,
        comm=comm,
        generations=config.generations,
        checkpoint_path=config.checkpoint,
        rng=rng,
    )
    propulator.propulate(config.logging_int, config.verbosity)
    propulator.summarize(top_n=config.top_n, debug=config.verbosity)
