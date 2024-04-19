import random

import pytest

from propulate import Propulator
from propulate.propagators import BasicCMA, CMAPropagator
from propulate.utils.benchmark_functions import get_function_search_space


@pytest.fixture(
    params=[
        ("rosenbrock", 0.0, 0.1),
        ("step", -25.0, 0.0),
        ("quartic", 0.0, 10.0),
        ("rastrigin", 0.0, 0.1),
        ("griewank", 0.0, 10.0),
        ("schwefel", 0.0, 10.0),
        ("bisphere", 0.0, 100.0),
        ("birastrigin", 0.0, 100.0),
        ("bukin", 0.0, 10.0),
        ("eggcrate", -1.0, 1.0),
        ("himmelblau", 0.0, 1.0),
        ("keane", 0.6736675, 0.1),
        ("leon", 0.0, 0.1),
        ("sphere", 0.0, 0.001),  # (fname, expected, abs)
    ]
)
def function_parameters(request):
    """Define benchmark function parameter sets as used in tests."""
    return request.param


def test_cmaes(function_parameters, mpi_tmp_path):
    """
    Test single worker using Propulator to optimize a benchmark function using a CMA-ES propagator.

    Parameters
    ----------
    function_parameters : tuple
        The tuple containing (fname, expected, abs).
    """
    fname, expected, abs_tolerance = function_parameters
    rng = random.Random(42)  # Separate random number generator for optimization.
    function, limits = get_function_search_space(fname)
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
    # assert propulator.summarize(top_n=1, debug=2)[0][0].loss == pytest.approx(
    #     expected=expected, abs=abs_tolerance
    # )


# @pytest.mark.mpi
# def test_cmaes_migration():
#     raise
#
#
# @pytest.mark.mpi
# def test_cmaes_categorical():
#     raise
