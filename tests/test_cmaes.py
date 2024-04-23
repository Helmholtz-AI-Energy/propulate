import pathlib
import random
from typing import Tuple

import pytest

from propulate import Propulator
from propulate.propagators import BasicCMA, CMAPropagator
from propulate.utils.benchmark_functions import get_function_search_space


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
def function_parameters(request):
    """Define benchmark function parameter sets as used in tests."""
    return request.param


def test_cmaes(
    function_parameters: Tuple[str, float], mpi_tmp_path: pathlib.Path
) -> None:
    """
    Test single worker using Propulator to optimize a benchmark function using a CMA-ES propagator.

    This test is run both sequentially and in parallel.

    Parameters
    ----------
    function_parameters : Tuple[str, float]
        The tuple containing each function name along with its global minimum.
    mpi_tmp_path : pathlib.Path
        The temporary checkpoint directory.
    """
    rng = random.Random(42)  # Separate random number generator for optimization.
    function, limits = get_function_search_space(function_parameters[0])
    # Set up evolutionary operator.
    adapter = BasicCMA()
    propagator = CMAPropagator(adapter, limits, rng=rng)

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
