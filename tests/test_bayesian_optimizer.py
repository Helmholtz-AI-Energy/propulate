# tests/test_bayes_acquisitions.py
import copy
import pathlib
import random
from typing import Callable, Dict, Tuple, Union

import deepdiff
import numpy as np
import pytest
from mpi4py import MPI
from sklearn.gaussian_process.kernels import RBF

from propulate import Propulator
from propulate.utils import set_logger_config
from propulate.utils.benchmark_functions import get_function_search_space
from propulate.propagators.bayesopt import (
    BayesianOptimizer,
    expected_improvement,
    create_acquisition,
)


@pytest.fixture(
    params=[
        "sphere",
        "rosenbrock",
    ]
)
def function_name(request: pytest.FixtureRequest) -> str:
    """Define benchmark function parameter sets as used in tests."""
    return request.param


def _make_bayes_propagator(limits, rng: random.Random) -> BayesianOptimizer:
    """Helper to build a BayesianOptimizer with simple acquisition optimizer.

    Note: Parallel fitters are currently disabled; the optimizer defaults to a
    single-CPU fitter internally.
    """
    dim = len(limits)
    return BayesianOptimizer(
        limits=limits,
        rank=MPI.COMM_WORLD.rank,
        world_size=MPI.COMM_WORLD.size,
        acquisition_type="EI",
        acquisition_params={"xi": 0.01},
        rank_stretch=True,          # diversify across ranks
        factor_min=0.5,
        factor_max=2.0,
        sparse=True,                # keep training sets light in tests
        sparse_params={"max_points": 200},
        rng=rng,
    )


def test_bayes_propagator(function_name: str, mpi_tmp_path: pathlib.Path) -> None:
    """
    Test Propulator to optimize the benchmark functions using the Bayesian optimizer propagator.

    This test is run both sequentially and in parallel.

    Parameters
    ----------
    function_name : str
        The function name.
    mpi_tmp_path : pathlib.Path
        The temporary checkpoint directory.
    """
    rng = random.Random(42 + MPI.COMM_WORLD.rank)  # Random number generator for optimization
    benchmark_function, limits = get_function_search_space(function_name)
    set_logger_config(log_file=mpi_tmp_path / "log.log")

    propagator = _make_bayes_propagator(limits, rng)

    propulator = Propulator(
        loss_fn=benchmark_function,
        propagator=propagator,
        rng=rng,
        generations=2,
        checkpoint_path=mpi_tmp_path,
    )  # Set up propulator performing actual optimization.

    propulator.propulate()  # Run optimization and print summary of results.


def test_bayes_propagator_checkpointing(mpi_tmp_path: pathlib.Path) -> None:
    """
    Test Propulator checkpointing for the sphere benchmark function with the Bayesian optimizer.

    This test is run both sequentially and in parallel.

    Parameters
    ----------
    mpi_tmp_path : pathlib.Path
        The temporary checkpoint directory.
    """
    rng = random.Random(42 + MPI.COMM_WORLD.rank)  # Separate random number generator for optimization
    benchmark_function, limits = get_function_search_space("sphere")

    propagator = _make_bayes_propagator(limits, rng)

    propulator = Propulator(
        loss_fn=benchmark_function,
        propagator=propagator,
        generations=10,
        checkpoint_path=mpi_tmp_path,
        rng=rng,
    )  # Set up propulator performing actual optimization.

    propulator.propulate()  # Run optimization and print summary of results.

    old_population = copy.deepcopy(propulator.population)  # Save population list from the last run.
    del propulator  # Delete propulator object.
    MPI.COMM_WORLD.barrier()  # Synchronize all processes.

    propulator = Propulator(
        loss_fn=benchmark_function,
        propagator=propagator,
        generations=5,
        checkpoint_path=mpi_tmp_path,
        rng=rng,
    )  # Set up new propulator starting from checkpoint.

    # As the number of requested generations is smaller than the number of generations from the run before,
    # no new evaluations are performed. Thus, the length of both Propulators' populations must be equal.
    assert len(deepdiff.DeepDiff(old_population, propulator.population, ignore_order=True)) == 0


class _DummyModel:
    """Minimal surrogate mock that returns fixed (mu, sigma)."""
    def __init__(self, mu: float, sigma: float):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.calls = 0

    def predict(self, X, return_std=True):
        self.calls += 1
        X = np.atleast_2d(X)
        n = X.shape[0]
        mu = np.full((n,), self.mu, dtype=float)
        std = np.full((n,), self.sigma, dtype=float)
        return mu, std


def test_expected_improvement_function_basic() -> None:
    """
    Unit test for the vectorized expected_improvement helper.
    """
    mu = np.array([0.0, 0.9, 2.0])
    sigma = np.array([0.1, 0.5, 1.0])
    f_best = 1.0
    xi = 0.01

    ei = expected_improvement(mu, sigma, f_best, minimize=True, xi=xi)

    # Sanity checks
    assert ei.shape == mu.shape
    assert np.all(ei >= 0.0)  # EI is nonnegative
    # If mu << f_best, EI should be relatively large (given sigma)
    assert ei[0] > ei[1]
    # If mu >> f_best, EI should be close to 0
    assert ei[2] < 0.5


def test_expected_improvement_sigma_zero_path() -> None:
    """
    EI should reduce to max(f_best - mu - xi, 0) when sigma == 0.
    """
    f_best = 1.0
    x = np.array([0.123, -0.4])

    # Case 1: improvement negative -> EI = 0
    m1 = _DummyModel(mu=1.2, sigma=0.0)
    acq = create_acquisition("EI", xi=0.0)
    v1 = acq.evaluate(x, m1, f_best)
    assert v1 == pytest.approx(0.0)

    # Case 2: improvement positive -> EI = improvement
    m2 = _DummyModel(mu=0.8, sigma=0.0)
    v2 = acq.evaluate(x, m2, f_best)
    assert v2 == pytest.approx(0.2, abs=1e-12)


def test_ei_matches_closed_form_against_helper() -> None:
    """
    EI acquisition.evaluate should numerically match expected_improvement()
    on a single-point query given (mu, sigma).
    """
    f_best = 0.75
    xi = 0.01
    mu, sigma = 0.5, 0.2
    model = _DummyModel(mu=mu, sigma=sigma)
    acq = create_acquisition("EI", xi=xi)

    x = np.array([0.3, -0.1, 0.7])
    v = acq.evaluate(x, model, f_best)

    ref = float(expected_improvement(np.array([mu]), np.array([sigma]), f_best, minimize=True, xi=xi)[0])
    assert v == pytest.approx(ref, rel=1e-12, abs=1e-12)
    assert model.calls == 1


def test_probability_improvement_behavior() -> None:
    """
    PI should return Phi((f_best - mu - xi)/sigma) for minimization.
    Also check the sigma -> 0 limits via cdf(±inf) -> {1, 0}.
    """
    x = np.array([0.1, 0.2])

    # Standard finite-sigma case
    f_best, xi = 1.0, 0.05
    mu, sigma = 0.8, 0.2
    model = _DummyModel(mu=mu, sigma=sigma)
    acq = create_acquisition("PI", xi=xi)
    v = acq.evaluate(x, model, f_best)

    from scipy.stats import norm
    z = (f_best - mu - xi) / sigma
    ref = float(norm.cdf(z))
    assert v == pytest.approx(ref, rel=1e-12, abs=1e-12)

    # Sigma == 0 positive improvement -> 1
    model2 = _DummyModel(mu=0.7, sigma=0.0)
    v2 = acq.evaluate(x, model2, f_best)
    assert v2 == pytest.approx(1.0)

    # Sigma == 0 negative improvement -> 0
    model3 = _DummyModel(mu=1.5, sigma=0.0)
    v3 = acq.evaluate(x, model3, f_best)
    assert v3 == pytest.approx(0.0)


def test_upper_confidence_bound_matches_formula() -> None:
    """
    UCB acquisition returns - (mu - kappa * sigma) for minimization.
    """
    x = np.array([0.0])
    kappa = 1.96
    mu, sigma = 0.5, 0.2
    model = _DummyModel(mu=mu, sigma=sigma)

    acq = create_acquisition("UCB", kappa=kappa)
    v = acq.evaluate(x, model, f_best=-123.0)  # f_best is unused for UCB

    expected = mu - kappa * sigma
    assert v == pytest.approx(expected, rel=1e-12, abs=1e-12)

    # Increasing kappa should increase the returned value (-mu + kappa*sigma)
    acq2 = create_acquisition("UCB", kappa=kappa * 2)
    v2 = acq2.evaluate(x, model, f_best=0.0)
    assert v2 < v


def test_rank_stretching_parameters_across_all_acquisitions() -> None:
    """
    Check that rank_stretch rescales xi (EI/PI) and kappa (UCB) from factor_min to factor_max.
    """
    size = 5
    r0 = 0
    rN = size - 1

    # EI / PI -> xi scaling
    acq_ei_0 = create_acquisition("EI", rank_stretch=True, rank=r0, size=size, factor_min=0.5, factor_max=2.0, xi=0.02)
    acq_ei_N = create_acquisition("EI", rank_stretch=True, rank=rN, size=size, factor_min=0.5, factor_max=2.0, xi=0.02)
    assert acq_ei_0.xi == pytest.approx(0.5 * 0.02)
    assert acq_ei_N.xi == pytest.approx(2.0 * 0.02)

    acq_pi_0 = create_acquisition("PI", rank_stretch=True, rank=r0, size=size, factor_min=0.5, factor_max=2.0, xi=0.01)
    acq_pi_N = create_acquisition("PI", rank_stretch=True, rank=rN, size=size, factor_min=0.5, factor_max=2.0, xi=0.01)
    assert acq_pi_0.xi == pytest.approx(0.5 * 0.01)
    assert acq_pi_N.xi == pytest.approx(2.0 * 0.01)

    # UCB -> kappa scaling
    acq_ucb_0 = create_acquisition("UCB", rank_stretch=True, rank=r0, size=size, factor_min=0.5, factor_max=2.0, kappa=1.0)
    acq_ucb_N = create_acquisition("UCB", rank_stretch=True, rank=rN, size=size, factor_min=0.5, factor_max=2.0, kappa=1.0)
    assert acq_ucb_0.kappa == pytest.approx(0.5 * 1.0)
    assert acq_ucb_N.kappa == pytest.approx(2.0 * 1.0)


def test_acquisition_interfaces_call_predict_once() -> None:
    """
    Smoke test that evaluate() calls model.predict() exactly once for each acquisition.
    """
    x = np.array([0.1, -0.2, 0.3])
    f_best = 0.4

    m1 = _DummyModel(mu=0.2, sigma=0.5)
    m2 = _DummyModel(mu=0.2, sigma=0.5)
    m3 = _DummyModel(mu=0.2, sigma=0.5)

    create_acquisition("EI", xi=0.01).evaluate(x, m1, f_best)
    create_acquisition("PI", xi=0.01).evaluate(x, m2, f_best)
    create_acquisition("UCB", kappa=1.5).evaluate(x, m3, f_best)

    assert m1.calls == 1
    assert m2.calls == 1
    assert m3.calls == 1


def test_bayesian_optimizer_mixed_example(mpi_tmp_path: pathlib.Path) -> None:
    """
    Test BayesianOptimizer with mixed-type search space.
    """
    rng = random.Random(42 + MPI.COMM_WORLD.rank)

    limits: Dict[str, Union[Tuple[float, float], Tuple[int, int], Tuple[str, ...]]] = {
        "x": (0.0, 5.0),
        "n": (1, 10),
        "method": ("fast", "slow"),
    }

    def mixed_function(params):
        """Simple test function with mixed types."""
        penalty = 1.0 if params["method"] == "slow" else 0.0
        return (params["x"] - 2.5) ** 2 + (params["n"] - 5) ** 2 + penalty

    propagator = BayesianOptimizer(
        limits=limits,
        rank=MPI.COMM_WORLD.rank,
        world_size=MPI.COMM_WORLD.size,
        n_initial=10,
        acquisition_type="EI",
        rng=rng,
    )

    propulator = Propulator(
        loss_fn=mixed_function,
        propagator=propagator,
        rng=rng,
        generations=3,
        checkpoint_path=mpi_tmp_path,
    )

    propulator.propulate()

    # Verify we can find reasonable solutions
    assert len(propulator.population) > 0

    # Verify types are preserved
    for ind in propulator.population:
        assert isinstance(ind["x"], float)
        assert isinstance(ind["n"], int)
        assert isinstance(ind["method"], str)
        assert ind["method"] in limits["method"]
