import logging
import pathlib
import random
from typing import Tuple

import numpy as np
import pytest

from propulate import Propulator
from propulate.propagators import ParallelNelderMead
from propulate.utils import set_logger_config
from propulate.utils.benchmark_functions import get_function_search_space

log = logging.getLogger("propulate")  # Get logger instance.


@pytest.fixture(
    params=[
        ("rosenbrock", 0.0),
        ("step", -25.0),
        ("quartic", 0.0),
        ("rastrigin", 0.0),
        ("griewank", 0.0),
        ("schwefel", 0.0),
        ("bisphere", 0.0),
        ("birastrigin", 0.0),
        ("bukin", 0.0),
        ("eggcrate", -1.0),
        ("himmelblau", 0.0),
        ("keane", 0.6736675),
        ("leon", 0.0),
        ("sphere", 0.0),  # (fname, expected)
    ]
)
def function_parameters(request: pytest.FixtureRequest) -> Tuple[str, float]:
    """Define benchmark function parameter sets as used in tests."""
    return request.param


def test_cmaes(
    function_parameters: Tuple[str, float], mpi_tmp_path: pathlib.Path
) -> None:
    """
    Test Propulator to optimize a benchmark function using a Nelder-Mead propagator.

    This test is run both sequentially and in parallel.

    Parameters
    ----------
    function_parameters : Tuple[str, float]
        The tuple containing each function name along with its global minimum.
    mpi_tmp_path : pathlib.Path
        The temporary checkpoint directory.
    """
    set_logger_config()
    rng = random.Random(42)  # Separate random number generator for optimization.
    function, limits = get_function_search_space(function_parameters[0])
    # Set up evolutionary operator.
    low = np.array([v[0] for v in limits.values()])
    high = np.array([v[1] for v in limits.values()])
    start_point = np.random.default_rng(seed=235231).uniform(low=low, high=high)
    propagator = ParallelNelderMead(limits, rng=rng, start=start_point)

    # Set up Propulator performing actual optimization.
    propulator = Propulator(
        loss_fn=function,
        propagator=propagator,
        rng=rng,
        generations=100,
        checkpoint_path=mpi_tmp_path,
    )
    # Run optimization and print summary of results.
    propulator.propulate()
    propulator.summarize()
    log.handlers.clear()
