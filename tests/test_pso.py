import random
from typing import Tuple

import pytest
from mpi4py import MPI

from propulate import Propulator
from propulate.propagators import Conditional
from propulate.propagators.pso import BasicPSO, InitUniformPSO
from propulate.utils.benchmark_functions import get_function_search_space


@pytest.fixture(
    params=[
        ("rosenbrock", 0.0, 0.1),
        ("step", -25.0, 2.0),
        ("quartic", 0.0, 1000.0),
        ("rastrigin", 0.0, 1000.0),
        ("griewank", 0.0, 10000.0),
        ("schwefel", 0.0, 10000.0),
        ("bisphere", 0.0, 1000.0),
        ("birastrigin", 0.0, 1000.0),
        ("bukin", 0.0, 100.0),
        ("eggcrate", -1.0, 10.0),
        ("himmelblau", 0.0, 1.0),
        ("keane", 0.6736675, 1.0),
        ("leon", 0.0, 10.0),
        ("sphere", 0.0, 0.01),  # (fname, expected, abs)
    ]
)
def function_parameters(request):
    """Define benchmark function parameter sets as used in tests."""
    return request.param


@pytest.mark.mpi
def test_pso(function_parameters: Tuple[str, float, float], mpi_tmp_path):
    """
    Test single worker using Propulator to optimize a benchmark function using the default genetic propagator.

    Parameters
    ----------
    function_parameters : tuple
        The tuple containing (fname, expected, abs).
    """
    fname, expected, abs_tolerance = function_parameters
    rng = random.Random(42)  # Separate random number generator for optimization.
    function, limits = get_function_search_space("sphere")
    # Set up evolutionary operator.
    pso_propagator = BasicPSO(
        inertia=0.729,
        c_cognitive=1.49334,
        c_social=1.49445,
        rank=MPI.COMM_WORLD.rank,  # MPI rank
        limits=limits,
        rng=rng,
    )
    init = InitUniformPSO(limits, rng=rng, rank=MPI.COMM_WORLD.rank)
    propagator = Conditional(1, pso_propagator, init)

    # Set up propulator performing actual optimization.
    propulator = Propulator(
        loss_fn=function,
        propagator=propagator,
        rng=rng,
        generations=100,
        checkpoint_path=mpi_tmp_path,
    )

    # Run optimization and print summary of results.
    propulator.propulate()
    assert propulator.summarize(top_n=1, debug=2)[0][0].loss == pytest.approx(
        expected=expected, abs=abs_tolerance
    )
