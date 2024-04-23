import pathlib
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


@pytest.mark.mpi
def test_pso(function_parameters: Tuple[str, float, float], mpi_tmp_path: pathlib.Path):
    """
    Test single worker using Propulator to optimize a benchmark function using the default genetic propagator.

    Parameters
    ----------
    function_parameters : Tuple
        The tuple containing each function name along with its global minimum.
    mpi_tmp_path : pathlib.Path
        The temporary checkpoint directory.
    """
    rng = random.Random(42)  # Separate random number generator for optimization.
    function, limits = get_function_search_space(function_parameters[0])
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
