import logging
import pathlib
import random

import pytest

from propulate import Propulator
from propulate.propagators import ActiveCMA, BasicCMA, CMAAdapter, CMAPropagator
from propulate.utils import set_logger_config
from propulate.utils.benchmark_functions import get_function_search_space

log = logging.getLogger("propulate")  # Get logger instance.


@pytest.fixture(params=[BasicCMA(), ActiveCMA()])
def cma_adapter(request: pytest.FixtureRequest) -> CMAAdapter:
    """Iterate over CMA adapters (basic and active)."""
    return request.param


def test_cmaes_basic(cma_adapter: CMAAdapter, mpi_tmp_path: pathlib.Path) -> None:
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
    set_logger_config()
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
        generations=10,
        checkpoint_path=mpi_tmp_path,
    )
    # Run optimization and print summary of results.
    propulator.propulate()
    log.handlers.clear()


# TODO test with pollination
