"""Simple example script using Bayesian Optimization."""

import pathlib
import random

import numpy as np
from mpi4py import MPI

from propulate import Propulator
from propulate.utils import set_logger_config
from propulate.utils.benchmark_functions import (
    get_function_search_space,
    parse_arguments,
)

# Import your Bayesian optimizer bits
from propulate.propagators.bayes_opt import (
    BayesianOptimizer,
    SingleCPUFitter,
    MultiCPUFitter,
)
from sklearn.gaussian_process.kernels import RBF


class RandomBoxSearch:
    """Simple random search to optimize the acquisition function."""
    def __init__(self, n_samples: int = 256) -> None:
        self.n_samples = n_samples

    def optimize(self, acq_func, bounds: np.ndarray, rng: random.Random) -> np.ndarray:
        lows, highs = bounds
        dim = lows.shape[0]
        best_x = None
        best_v = float("inf")
        for _ in range(self.n_samples):
            x = np.array([rng.uniform(l, u) for l, u in zip(lows, highs)], dtype=float)
            v = acq_func(x)
            if v < best_v:
                best_v = v
                best_x = x
        return best_x


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

    rng = random.Random(config.seed + comm.rank)  # Separate RNG for optimization.
    benchmark_function, limits = get_function_search_space(config.function)  # Get function + limits.

    # Pick a fitter based on MPI world size (multi-rank -> MultiCPUFitter).
    if comm.size > 1:
        fitter = MultiCPUFitter(comm=comm, n_restarts_per_rank=1, seed=config.seed)
    else:
        fitter = SingleCPUFitter()

    # Simple acquisition optimizer (random search within bounds).
    acq_optimizer = RandomBoxSearch(n_samples=256)

    # Kernel (RBF with per-dimension length scales).
    dim = len(limits)
    kernel = RBF(length_scale=np.ones(dim))

    # Set up Bayesian optimizer propagator.
    propagator = BayesianOptimizer(
        limits=limits,
        optimizer=acq_optimizer,
        rank=comm.rank,
        fitter=fitter,
        kernel=kernel,
        acquisition_type="EI",              # EI, PI, or UCB
        acquisition_params={"xi": 0.01},    # For UCB use {"kappa": ...}
        rank_stretch=True,                  # Diversify acquisition params across ranks
        factor_min=0.5,
        factor_max=2.0,
        sparse=True,                        # Keep surrogate fitting light
        sparse_params={"max_points": 500},
        rng=rng,
    )

    # Set up propulator performing actual optimization.
    propulator = Propulator(
        loss_fn=benchmark_function,
        propagator=propagator,
        rng=rng,
        island_comm=comm,
        generations=config.generations,
        checkpoint_path=config.checkpoint,
    )

    # Run optimization and print summary of results.
    propulator.propulate(logging_interval=config.logging_interval, debug=config.verbosity)
    propulator.summarize(top_n=config.top_n, debug=config.verbosity)
