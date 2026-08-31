import logging
import pathlib
import random

import pytest

from propulate import Islands, Propulator
from propulate.propagators import ActiveCMA, BasicCMA, CMAAdapter, CMAPropagator
from propulate.utils import set_logger_config
from propulate.utils.benchmark_functions import get_function_search_space
from propulate.utils.consistency_checks import final_synch, population_consistency_check


@pytest.fixture(params=[BasicCMA(), ActiveCMA()])
def cma_adapter(request: pytest.FixtureRequest) -> CMAAdapter:
    """Iterate over CMA adapters (basic and active)."""
    return request.param


@pytest.fixture(
    params=[
        True,
        False,
    ]
)
def pollination(request: pytest.FixtureRequest) -> bool:
    """Iterate through pollination parameter."""
    return request.param


def test_cmaes_basic(cma_adapter: CMAAdapter, pollination: bool, mpi_tmp_path: pathlib.Path) -> None:
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
    log = logging.getLogger("propulate")  # Get logger instance.
    set_logger_config(level=logging.DEBUG)

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
    final_synch(propulator)
    population_consistency_check(propulator)
    log.handlers.clear()


@pytest.mark.mpi(min_size=8)
def test_cmaes_islands(cma_adapter: CMAAdapter, pollination: bool, mpi_tmp_path: pathlib.Path) -> None:
    """
    Test Propulator to optimize a benchmark function using CMA-ES propagators with the island model.

    This test is run in parallel.

    Parameters
    ----------
    cma_adapter : CMAAdapter
        The CMA adapter used, either basic or active.
    mpi_tmp_path : pathlib.Path
        The temporary checkpoint directory.
    """
    log = logging.getLogger("propulate")  # Get logger instance.
    set_logger_config(level=logging.DEBUG)

    rng = random.Random(42)  # Separate random number generator for optimization.
    benchmark_function, limits = get_function_search_space("sphere")
    # Set up evolutionary operator.
    adapter = cma_adapter
    propagator = CMAPropagator(adapter, limits, rng=rng)

    # Set up Propulator performing actual optimization.
    islands = Islands(
        loss_fn=benchmark_function,
        propagator=propagator,
        rng=rng,
        generations=10,
        num_islands=2,
        migration_probability=0.9,
        pollination=pollination,
        checkpoint_path=mpi_tmp_path,
    )
    # Run optimization and print summary of results.
    islands.propulate()
    final_synch(islands.propulator)
    population_consistency_check(islands.propulator)
    log.handlers.clear()
