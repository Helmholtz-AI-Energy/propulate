import pathlib
import random

import pytest

from propulate import Propulator
from propulate.propagators import ActiveCMA, BasicCMA, CMAPropagator
from propulate.utils.benchmark_functions import get_function_search_space


@pytest.fixture(params=[BasicCMA(), ActiveCMA()])
def cma_adapter(request):
    """Iterate over CMA adapters (basic and active)."""
    return request.param


def test_cmaes_basic(cma_adapter, mpi_tmp_path: pathlib.Path) -> None:
    """
    Test Propulator to optimize a benchmark function using CMA-ES propagators.

    This test is run both sequentially and in parallel.

    Parameters
    ----------
    cma_adapter : CMAAdapter
        The CMA adapter used, either basic or active.
    mpi_tmp_path : pathlib.Path
        The temporary checkpoint directory.
    """
    rng = random.Random(42)  # Separate random number generator for optimization.
    benchmark_function, limits = get_function_search_space("sphere")
    # Set up evolutionary operator.
    adapter = cma_adapter
    propagator = CMAPropagator(adapter, limits, rng=rng)

    # Set up Propulator performing actual optimization.
    propulator = Propulator(
        loss_fn=benchmark_function,
        propagator=propagator,
        rng=rng,
        generations=100,
        checkpoint_path=mpi_tmp_path,
    )
    # Run optimization and print summary of results.
    propulator.propulate()
