# tests/test_bayesian_optimizer.py
import copy
import pathlib
import random
from typing import Dict, Tuple, Union

import deepdiff
import numpy as np
import pytest
from mpi4py import MPI

import propulate.propagators.bayesopt as bayesopt_module
from propulate import Propulator
from propulate.population import Individual
from propulate.propagators.bayesopt import (
    BayesianOptimizer,
    MultiStartAcquisitionOptimizer,
    SingleCPUFitter,
    SurrogateFitter,
    _project_to_discrete,
    _sparse_select_indices,
    create_acquisition,
    create_fitter,
    expected_improvement,
)
from propulate.utils import set_logger_config
from propulate.utils.benchmark_functions import get_function_search_space


@pytest.fixture(
    params=[
        "sphere",
        "rosenbrock",
    ]
)
def function_name(request: pytest.FixtureRequest) -> str:
    """Define benchmark function parameter sets as used in tests."""
    return request.param


def _position_dim_from_limits(limits) -> int:
    """Compute BO position dimension including one-hot categorical expansion."""
    return sum(len(bounds) if isinstance(bounds[0], str) else 1 for bounds in limits.values())


def _fast_test_optimizer(limits) -> MultiStartAcquisitionOptimizer:
    """Construct a lightweight acquisition optimizer for tests."""
    position_dim = _position_dim_from_limits(limits)
    return MultiStartAcquisitionOptimizer(
        n_candidates=max(64, 16 * position_dim),
        n_restarts=2,
        minimize_options={"maxiter": 80, "maxfun": 800, "ftol": 1e-6, "gtol": 1e-5},
    )


def _make_bayes_propagator(limits, rng: random.Random) -> BayesianOptimizer:
    """Helper to build a BayesianOptimizer with simple acquisition optimizer.

    Note: Parallel fitters are currently disabled; the optimizer defaults to a
    single-CPU fitter internally.
    """
    return BayesianOptimizer(
        limits=limits,
        rank=MPI.COMM_WORLD.rank,
        world_size=MPI.COMM_WORLD.size,
        acquisition_type="EI",
        acquisition_params={"xi": 0.01},
        rank_stretch=True,  # diversify across ranks
        factor_min=0.5,
        factor_max=2.0,
        sparse=True,  # keep training sets light in tests
        sparse_params={"max_points": 200},
        optimizer=_fast_test_optimizer(limits),
        optimize_hyperparameters=False,
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


class _CountingModel:
    """Surrogate mock that records predict() call counts and batch sizes."""

    def __init__(self):
        self.calls = 0
        self.batch_sizes = []

    def predict(self, X, return_std=True):
        self.calls += 1
        X = np.atleast_2d(X)
        n = X.shape[0]
        self.batch_sizes.append(n)
        mu = np.sum(X, axis=1, dtype=float)
        std = np.full((n,), 0.1, dtype=float)
        return mu, std


class _DeterministicOptimizer:
    """Simple optimizer stub that probes acquisition once and returns the origin."""

    def optimize(self, acq_func, bounds, rng=None):
        dim = bounds.shape[1]
        x = np.zeros(dim, dtype=float)
        _ = acq_func(x)
        return x


def _record_acquisition_calls(monkeypatch: pytest.MonkeyPatch):
    calls = []
    real_create = bayesopt_module.create_acquisition

    def _wrapped_create(acq_type, **params):
        calls.append((acq_type, dict(params)))
        return real_create(acq_type, **params)

    monkeypatch.setattr(bayesopt_module, "create_acquisition", _wrapped_create)
    return calls


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


def test_ei_maximization_path_matches_helper() -> None:
    """
    EI acquisition.evaluate should match expected_improvement() for maximize mode.
    """
    f_best = 0.75
    xi = 0.02
    mu, sigma = 1.0, 0.3
    model = _DummyModel(mu=mu, sigma=sigma)
    acq = create_acquisition("EI", xi=xi, minimize=False)

    x = np.array([0.3, -0.1, 0.7])
    v = acq.evaluate(x, model, f_best)

    ref = float(expected_improvement(np.array([mu]), np.array([sigma]), f_best, minimize=False, xi=xi)[0])
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


def test_rank_stretch_false_keeps_parameters_identical_across_ranks() -> None:
    """When rank_stretch=False, per-rank acquisition parameters should stay unchanged."""
    size = 5
    acq_ei_0 = create_acquisition("EI", rank_stretch=False, rank=0, size=size, xi=0.02)
    acq_ei_N = create_acquisition("EI", rank_stretch=False, rank=size - 1, size=size, xi=0.02)
    assert acq_ei_0.xi == pytest.approx(0.02)
    assert acq_ei_N.xi == pytest.approx(0.02)

    acq_ucb_0 = create_acquisition("UCB", rank_stretch=False, rank=0, size=size, kappa=1.0)
    acq_ucb_N = create_acquisition("UCB", rank_stretch=False, rank=size - 1, size=size, kappa=1.0)
    assert acq_ucb_0.kappa == pytest.approx(1.0)
    assert acq_ucb_N.kappa == pytest.approx(1.0)


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


def test_multistart_optimizer_vectorizes_candidate_acquisition_evaluation() -> None:
    """Candidate scoring should use one batched model.predict call when acq_func supports it."""
    model = _CountingModel()
    acq = create_acquisition("UCB", kappa=1.0)
    optimizer = MultiStartAcquisitionOptimizer(
        n_candidates=64,
        n_restarts=0,
        polish=False,
    )

    def acq_func(x):
        arr = np.asarray(x)
        if arr.ndim == 1:
            return acq.evaluate(arr, model, f_best=0.0)
        return acq.evaluate_batch(arr, model, f_best=0.0)

    bounds = np.array([np.zeros(3), np.ones(3)], dtype=float)
    x_best = optimizer.optimize(acq_func, bounds=bounds, rng=random.Random(0))

    assert x_best.shape == (3,)
    assert model.calls == 1
    assert model.batch_sizes == [64]


def test_acquisition_switching_activates_second_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """_select_next should switch acquisition type and params at the configured generation."""
    calls = _record_acquisition_calls(monkeypatch)

    opt = BayesianOptimizer(
        limits={"x": (0.0, 1.0)},
        rank=0,
        acquisition_type="EI",
        acquisition_params={"xi": 0.2},
        second_acquisition_type="UCB",
        acq_switch_generation=3,
        second_acquisition_params={"kappa": 3.5},
        anneal_acquisition=False,
        rank_stretch=False,
        optimizer=_DeterministicOptimizer(),
        rng=random.Random(0),
    )
    model = _DummyModel(mu=0.0, sigma=1.0)

    opt._select_next(model, f_best=0.0, current_generation=1)  # t=2, still EI
    opt._select_next(model, f_best=0.0, current_generation=2)  # t=3, switch to UCB

    assert calls[0][0].upper() == "EI"
    assert calls[0][1]["xi"] == pytest.approx(0.2)
    assert "kappa" not in calls[0][1]
    assert calls[1][0].upper() == "UCB"
    assert calls[1][1]["kappa"] == pytest.approx(3.5)
    assert "xi" not in calls[1][1]


@pytest.mark.parametrize(
    ("acq_type", "param_name", "default_value"),
    [
        ("EI", "xi", 0.01),
        ("UCB", "kappa", 1.96),
    ],
)
def test_annealing_uses_default_parameter_when_not_explicitly_set(
    monkeypatch: pytest.MonkeyPatch,
    acq_type: str,
    param_name: str,
    default_value: float,
) -> None:
    """Annealing should apply even when acquisition_params is omitted."""
    calls = _record_acquisition_calls(monkeypatch)

    opt = BayesianOptimizer(
        limits={"x": (0.0, 1.0)},
        rank=0,
        acquisition_type=acq_type,
        acquisition_params=None,
        anneal_acquisition=True,
        rank_stretch=False,
        optimizer=_DeterministicOptimizer(),
        rng=random.Random(1),
    )
    model = _DummyModel(mu=0.0, sigma=1.0)

    opt._select_next(model, f_best=0.0, current_generation=0)  # t=1
    opt._select_next(model, f_best=0.0, current_generation=50)  # t=51

    first = float(calls[0][1][param_name])
    later = float(calls[1][1][param_name])
    expected_first = default_value / np.sqrt(1.0 + 0.05 * 1.0)
    expected_later = default_value / np.sqrt(1.0 + 0.05 * 51.0)

    assert first == pytest.approx(expected_first)
    assert later == pytest.approx(expected_later)
    assert later < first


@pytest.mark.parametrize(
    ("acq_type", "param_name", "param_value"),
    [
        ("EI", "xi", 0.2),
        ("UCB", "kappa", 3.0),
    ],
)
def test_annealing_disabled_keeps_acquisition_parameter_constant(
    monkeypatch: pytest.MonkeyPatch,
    acq_type: str,
    param_name: str,
    param_value: float,
) -> None:
    """When annealing is disabled, acquisition params should not decay over time."""
    calls = _record_acquisition_calls(monkeypatch)

    opt = BayesianOptimizer(
        limits={"x": (0.0, 1.0)},
        rank=0,
        acquisition_type=acq_type,
        acquisition_params={param_name: param_value},
        anneal_acquisition=False,
        rank_stretch=False,
        optimizer=_DeterministicOptimizer(),
        rng=random.Random(2),
    )
    model = _DummyModel(mu=0.0, sigma=1.0)

    opt._select_next(model, f_best=0.0, current_generation=0)
    opt._select_next(model, f_best=0.0, current_generation=50)

    assert float(calls[0][1][param_name]) == pytest.approx(param_value)
    assert float(calls[1][1][param_name]) == pytest.approx(param_value)


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
        optimizer=_fast_test_optimizer(limits),
        optimize_hyperparameters=False,
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


# --- Mixed-type and BO robustness tests ---
class _RecordingModel:
    def predict(self, X, return_std=True):
        X = np.atleast_2d(X)
        n = X.shape[0]
        return np.zeros(n, dtype=float), np.ones(n, dtype=float)


class _RecordingFitter(SurrogateFitter):
    def __init__(self):
        self.optimize_flags = []

    def fit(self, kernel, X, y, **kwargs):
        self.optimize_flags.append(bool(kwargs.get("optimize_hyperparameters", False)))
        return _RecordingModel()


def test_bayesian_optimizer_integers_only():
    """Test BayesianOptimizer with integer parameters only."""
    limits = {
        "a": (1, 10),
        "b": (5, 20),
    }
    rng = random.Random(42)
    opt = BayesianOptimizer(
        limits=limits,
        rank=0,
        n_initial=5,
        optimizer=_fast_test_optimizer(limits),
        optimize_hyperparameters=False,
        rng=rng,
    )

    # Check position dimension
    assert opt.position_dim == 2
    assert opt.dim == 2

    # Generate initial individuals
    inds = []
    for _ in range(10):
        ind = opt(inds)
        inds.append(ind)
        ind.loss = float(sum(ind[k] ** 2 for k in limits))

    # Check that all values are integers
    for ind in inds:
        assert isinstance(ind["a"], int)
        assert isinstance(ind["b"], int)
        assert limits["a"][0] <= ind["a"] <= limits["a"][1]
        assert limits["b"][0] <= ind["b"] <= limits["b"][1]


def test_bayesian_optimizer_categorical_only():
    """Test BayesianOptimizer with categorical parameters only."""
    limits = {
        "activation": ("relu", "tanh", "sigmoid"),
        "optimizer": ("adam", "sgd"),
    }
    rng = random.Random(42)
    opt = BayesianOptimizer(
        limits=limits,
        rank=0,
        n_initial=5,
        optimizer=_fast_test_optimizer(limits),
        optimize_hyperparameters=False,
        rng=rng,
    )

    # Check position dimension (one-hot encoding)
    assert opt.position_dim == 3 + 2  # 3 for activation, 2 for optimizer
    assert opt.dim == 2  # number of parameters

    # Generate individuals
    inds = []
    for _ in range(10):
        ind = opt(inds)
        inds.append(ind)
        ind.loss = 1.0  # dummy loss

    # Check that all values are valid categories
    for ind in inds:
        assert ind["activation"] in limits["activation"]
        assert ind["optimizer"] in limits["optimizer"]

    # Check one-hot encoding in position array
    ind = inds[0]
    assert np.sum(ind.position[0:3]) == pytest.approx(1.0)  # exactly one activation
    assert np.sum(ind.position[3:5]) == pytest.approx(1.0)  # exactly one optimizer


def test_bayesian_optimizer_mixed_types():
    """Test BayesianOptimizer with float, int, and categorical parameters."""
    limits = {
        "learning_rate": (0.001, 0.1),
        "num_layers": (1, 10),
        "activation": ("relu", "tanh", "sigmoid"),
    }
    rng = random.Random(42)
    opt = BayesianOptimizer(
        limits=limits,
        rank=0,
        n_initial=10,
        optimizer=_fast_test_optimizer(limits),
        optimize_hyperparameters=False,
        rng=rng,
    )

    # Check dimensions
    assert opt.position_dim == 1 + 1 + 3  # float + int + 3 categories
    assert opt.dim == 3

    # Generate individuals
    inds = []
    for _ in range(14):
        ind = opt(inds)
        inds.append(ind)
        # Simple loss function
        lr_penalty = (ind["learning_rate"] - 0.01) ** 2
        layer_penalty = (ind["num_layers"] - 5) ** 2
        act_bonus = 0.0 if ind["activation"] == "relu" else 0.5
        ind.loss = lr_penalty + layer_penalty + act_bonus

    # Check types
    for ind in inds:
        assert isinstance(ind["learning_rate"], float)
        assert isinstance(ind["num_layers"], int)
        assert isinstance(ind["activation"], str)
        assert 0.001 <= ind["learning_rate"] <= 0.1
        assert 1 <= ind["num_layers"] <= 10
        assert ind["activation"] in limits["activation"]


def test_project_to_discrete():
    """Test the projection helper function."""
    limits = {
        "float_param": (0.0, 10.0),
        "int_param": (1, 5),
        "cat_param": ("a", "b", "c"),
    }
    param_types = {key: type(limits[key][0]) for key in limits}

    # Test float passthrough and int rounding
    x = np.array([5.5, 3.7, 0.2, 0.6, 0.1])  # float, int, cat_one_hot
    x_proj = _project_to_discrete(x, limits, param_types)

    assert x_proj[0] == pytest.approx(5.5)  # float unchanged
    assert x_proj[1] == 4.0  # int rounded
    # Categorical: argmax of [0.2, 0.6, 0.1] = index 1 ("b")
    assert np.sum(x_proj[2:5]) == pytest.approx(1.0)
    assert np.argmax(x_proj[2:5]) == 1

    # Test boundary clipping for integers
    x2 = np.array([5.5, 6.8, 0.3, 0.3, 0.4])
    x2_proj = _project_to_discrete(x2, limits, param_types)
    assert x2_proj[1] == 5.0  # clipped to max


def test_project_to_discrete_edge_cases():
    """Test projection function edge cases."""
    limits = {
        "int_param": (0, 10),
        "cat_param": ("x", "y"),
    }
    param_types = {key: type(limits[key][0]) for key in limits}

    # Test negative one-hot values (should still project correctly)
    x = np.array([5.5, -1.0, -2.0])  # int, cat_one_hot (negative values)
    x_proj = _project_to_discrete(x, limits, param_types)

    assert x_proj[0] == 6.0  # rounded
    # Should still produce valid one-hot despite negative values
    assert np.sum(x_proj[1:3]) == pytest.approx(1.0)
    assert x_proj[1] == 1.0 or x_proj[2] == 1.0

    # Test integer boundary values
    x_low = np.array([-1.0, 0.5, 0.5])
    x_low_proj = _project_to_discrete(x_low, limits, param_types)
    assert x_low_proj[0] == 0.0  # clipped to min

    x_high = np.array([11.0, 0.5, 0.5])
    x_high_proj = _project_to_discrete(x_high, limits, param_types)
    assert x_high_proj[0] == 10.0  # clipped to max


def test_bayesian_optimizer_sparse_mixed_types():
    """Test sparse selection works with mixed-type position arrays."""
    limits = {
        "x": (0.0, 1.0),
        "y": (1, 10),
        "cat": ("a", "b"),
    }
    rng = random.Random(42)
    opt = BayesianOptimizer(
        limits=limits,
        rank=0,
        n_initial=5,
        sparse=True,
        sparse_params={"max_points": 20},
        optimizer=_fast_test_optimizer(limits),
        optimize_hyperparameters=False,
        rng=rng,
    )

    # Generate many individuals
    inds = []
    for i in range(45):
        ind = opt(inds)
        inds.append(ind)
        ind.loss = float(i)  # monotonic loss

    # Should trigger sparse selection
    # Verify types are preserved
    for ind in inds:
        assert isinstance(ind["x"], float)
        assert isinstance(ind["y"], int)
        assert isinstance(ind["cat"], str)


def test_bayesian_optimizer_invalid_limits():
    """Test that invalid limits raise appropriate errors."""
    rng = random.Random(42)

    # Empty limits
    with pytest.raises(ValueError, match="cannot be empty"):
        BayesianOptimizer(limits={}, rank=0, rng=rng)

    # Inverted bounds
    with pytest.raises(ValueError, match="lower bound must be < upper bound"):
        BayesianOptimizer(limits={"x": (10.0, 5.0)}, rank=0, rng=rng)

    # Duplicate categories
    with pytest.raises(ValueError, match="duplicate categories"):
        BayesianOptimizer(limits={"cat": ("a", "b", "a")}, rank=0, rng=rng)

    # Single category
    with pytest.raises(ValueError, match="at least 2"):
        BayesianOptimizer(limits={"cat": ("a",)}, rank=0, rng=rng)

    # Invalid hyperparameter schedule values
    with pytest.raises(ValueError, match="hp_opt_warmup_fits"):
        BayesianOptimizer(limits={"x": (0.0, 1.0)}, rank=0, hp_opt_warmup_fits=-1, rng=rng)

    with pytest.raises(ValueError, match="hp_opt_period"):
        BayesianOptimizer(limits={"x": (0.0, 1.0)}, rank=0, hp_opt_period=0, rng=rng)


def test_acquisition_optimization_mixed_types():
    """Test that acquisition function optimization works with mixed types."""
    limits = {
        "continuous": (0.0, 1.0),
        "discrete": (0, 10),
        "categorical": ("option_a", "option_b", "option_c"),
    }
    rng = random.Random(42)
    opt = BayesianOptimizer(
        limits=limits,
        rank=0,
        n_initial=10,
        acquisition_type="EI",
        optimizer=_fast_test_optimizer(limits),
        optimize_hyperparameters=False,
        rng=rng,
    )

    # Warm up with initial design
    inds = []
    for _ in range(10):
        ind = opt(inds)
        inds.append(ind)
        ind.loss = rng.random()

    # Next individual should use acquisition function
    new_ind = opt(inds)

    # Verify types
    assert isinstance(new_ind["continuous"], float)
    assert isinstance(new_ind["discrete"], int)
    assert isinstance(new_ind["categorical"], str)
    assert new_ind["categorical"] in limits["categorical"]


def test_hyperparameter_optimization_schedule_mixed():
    """Test that GP hyperparameter optimization works with mixed types."""
    limits = {
        "x": (0.0, 5.0),
        "n": (1, 20),
    }
    rng = random.Random(42)
    opt = BayesianOptimizer(
        limits=limits,
        rank=0,
        n_initial=10,
        fitter=SingleCPUFitter(n_restarts=0, random_state=0, alpha=1e-3),
        optimize_hyperparameters=True,
        hp_opt_warmup_fits=1,
        hp_opt_period=100,
        optimizer=_fast_test_optimizer(limits),
        p_explore_start=1.0,
        p_explore_end=1.0,
        rng=rng,
    )

    # Generate enough points to trigger hyperparameter optimization
    inds = []
    for _ in range(20):
        ind = opt(inds)
        inds.append(ind)
        # Use a non-degenerate objective with moderate stochasticity so GP
        # hyperparameter optimization remains well-conditioned in MPI runs.
        ind.loss = 0.3 * (ind["x"] - 2.5) ** 2 + 0.05 * (ind["n"] - 10) ** 2 + 0.8 * np.sin(4.0 * ind["x"]) + 0.5 * rng.random()

    # Should have fit multiple times
    assert opt._hp_fit_calls >= 3
    assert opt._hp_opt_calls >= 1


def test_hyperparameter_schedule_uses_optimized_fit_counter():
    """Warmup optimization should start once enough data exists, regardless of earlier non-optimizing fits."""
    limits = {f"x{i}": (0.0, 1.0) for i in range(20)}  # threshold = max(40, n_initial//2)
    fitter = _RecordingFitter()
    opt = BayesianOptimizer(
        limits=limits,
        rank=0,
        n_initial=5,
        optimize_hyperparameters=True,
        fitter=fitter,
        rng=random.Random(0),
    )

    flags = {}
    for generation in range(5, 47):
        X = np.zeros((generation, opt.position_dim), dtype=float)
        y = np.zeros((generation,), dtype=float)
        opt._fit_surrogate(X, y, current_generation=generation)
        flags[generation] = fitter.optimize_flags[-1]

    assert all(not flags[g] for g in range(5, 40))
    assert flags[40] is True
    assert flags[41] is True
    assert flags[42] is True
    assert flags[43] is False
    assert flags[44] is False
    assert flags[45] is True
    assert flags[46] is False
    assert opt._hp_opt_calls == 4


def test_hyperparameter_schedule_accepts_custom_warmup_and_period():
    """Custom schedule arguments should control warmup and periodic optimization cadence."""
    limits = {f"x{i}": (0.0, 1.0) for i in range(10)}  # threshold = max(20, n_initial//2)
    fitter = _RecordingFitter()
    opt = BayesianOptimizer(
        limits=limits,
        rank=0,
        n_initial=5,
        optimize_hyperparameters=True,
        fitter=fitter,
        hp_opt_warmup_fits=1,
        hp_opt_period=2,
        rng=random.Random(1),
    )

    flags = {}
    for generation in range(5, 26):
        X = np.zeros((generation, opt.position_dim), dtype=float)
        y = np.zeros((generation,), dtype=float)
        opt._fit_surrogate(X, y, current_generation=generation)
        flags[generation] = fitter.optimize_flags[-1]

    assert all(not flags[g] for g in range(5, 20))
    assert flags[20] is True
    assert flags[21] is False
    assert flags[22] is True
    assert flags[23] is False
    assert flags[24] is True
    assert flags[25] is False
    assert opt._hp_opt_calls == 3


def test_epsilon_greedy_exploration_mixed():
    """Test epsilon-greedy exploration maintains type constraints."""
    limits = {
        "a": (0.0, 1.0),
        "b": (1, 5),
        "c": ("x", "y", "z"),
    }
    rng = random.Random(12345)  # Seed to get random exploration
    opt = BayesianOptimizer(
        limits=limits,
        rank=0,
        n_initial=5,
        p_explore_start=1.0,  # Force exploration
        optimizer=_fast_test_optimizer(limits),
        optimize_hyperparameters=False,
        rng=rng,
    )

    inds = []
    for _ in range(12):
        ind = opt(inds)
        inds.append(ind)
        ind.loss = rng.random()

        # Verify types even during exploration
        assert isinstance(ind["a"], float)
        assert isinstance(ind["b"], int)
        assert isinstance(ind["c"], str)
        assert ind["c"] in limits["c"]


def test_backward_compatibility_float_only():
    """Test that float-only optimization still works (backward compatibility)."""
    limits = {"x": (0.0, 10.0), "y": (-5.0, 5.0)}
    rng = random.Random(42)
    opt = BayesianOptimizer(
        limits=limits,
        rank=0,
        n_initial=5,
        optimizer=_fast_test_optimizer(limits),
        optimize_hyperparameters=False,
        rng=rng,
    )

    # Should work exactly as before
    assert opt.position_dim == 2
    assert opt.dim == 2

    inds = []
    for _ in range(15):
        ind = opt(inds)
        inds.append(ind)
        ind.loss = ind["x"] ** 2 + ind["y"] ** 2

    # Find best
    best = min(inds, key=lambda i: i.loss)
    assert best.loss < 5.0  # Should find something near (0, 0)


def test_position_dim_warning():
    """Test that a warning is issued for high-dimensional position spaces."""
    # Create limits with many categorical variables to exceed threshold
    limits = {f"cat{i}": tuple(f"opt{j}" for j in range(10)) for i in range(12)}  # 12 * 10 = 120 dimensions

    with pytest.warns(UserWarning, match="Position dimension.*is large"):
        BayesianOptimizer(limits=limits, rank=0)


def test_mixed_types_with_different_acquisitions():
    """Test mixed types work with different acquisition functions."""
    limits = {
        "x": (0.0, 1.0),
        "n": (1, 5),
        "opt": ("a", "b"),
    }
    rng = random.Random(42)

    for acq_type in ["EI", "PI", "UCB"]:
        opt = BayesianOptimizer(
            limits=limits,
            rank=0,
            n_initial=10,
            acquisition_type=acq_type,
            optimizer=_fast_test_optimizer(limits),
            optimize_hyperparameters=False,
            rng=rng,
        )

        inds = []
        for _ in range(10):
            ind = opt(inds)
            inds.append(ind)
            ind.loss = rng.random()

        # Verify types for all acquisition functions
        for ind in inds:
            assert isinstance(ind["x"], float)
            assert isinstance(ind["n"], int)
            assert isinstance(ind["opt"], str)


def test_initial_designs_with_mixed_types():
    """Test different initial design methods with mixed types."""
    limits = {
        "x": (0.0, 1.0),
        "n": (1, 5),
        "cat": ("a", "b", "c"),
    }

    for design in ["sobol", "lhs", "random"]:
        rng = random.Random(42)
        opt = BayesianOptimizer(
            limits=limits,
            rank=0,
            n_initial=10,
            initial_design=design,
            optimizer=_fast_test_optimizer(limits),
            optimize_hyperparameters=False,
            rng=rng,
        )

        inds = []
        for _ in range(10):
            ind = opt(inds)
            inds.append(ind)
            ind.loss = 1.0

        # Verify types for all initial designs
        for ind in inds:
            assert isinstance(ind["x"], float)
            assert isinstance(ind["n"], int)
            assert isinstance(ind["cat"], str)


def test_initial_design_default_is_sobol_and_invalid_rejected():
    """Default initial design should be sobol and unknown values should error."""
    limits = {"x": (0.0, 1.0)}
    opt = BayesianOptimizer(limits=limits, rank=0, rng=random.Random(42))
    assert opt.initial_design == "sobol"

    with pytest.raises(ValueError, match="initial_design"):
        BayesianOptimizer(limits=limits, rank=0, initial_design="foobar", rng=random.Random(42))


def test_insufficient_data_fallback_handles_mixed_types():
    """Fallback random exploration must work when all observed losses are non-finite."""
    limits = {
        "x": (0.0, 1.0),
        "n": (1, 5),
        "cat": ("a", "b", "c"),
    }
    rng = random.Random(42)
    opt = BayesianOptimizer(
        limits=limits,
        rank=0,
        n_initial=1,
        rng=rng,
    )

    # One initial point to pass warm-start gate.
    seed_ind = opt([])
    seed_ind.loss = float("nan")

    # Triggers fallback branch after filtering non-finite losses.
    ind = opt([seed_ind])

    assert isinstance(ind["x"], float)
    assert isinstance(ind["n"], int)
    assert isinstance(ind["cat"], str)
    assert 0.0 <= ind["x"] <= 1.0
    assert 1 <= ind["n"] <= 5
    assert ind["cat"] in limits["cat"]


def test_sparse_subsample_all_nonfinite_is_safe():
    """Sparse subsampling should return empty training arrays when all losses are non-finite."""
    limits = {
        "x": (0.0, 1.0),
        "cat": ("a", "b"),
    }
    opt = BayesianOptimizer(
        limits=limits,
        rank=0,
        n_initial=1,
        sparse=True,
        sparse_params={"max_points": 5},
        rng=random.Random(42),
    )

    inds = []
    for i in range(6):
        position = np.array([i / 10.0, 1.0, 0.0], dtype=float)
        ind = Individual(position, limits, generation=i, rank=0)
        ind.loss = float("nan") if i % 2 == 0 else float("inf")
        inds.append(ind)

    X, y, sub_inds = opt._subsample(inds)
    assert X.shape == (0, opt.position_dim)
    assert y.shape == (0,)
    assert sub_inds == []

    # Should not crash and should return a valid fallback individual.
    next_ind = opt(inds)
    assert isinstance(next_ind["x"], float)
    assert isinstance(next_ind["cat"], str)
    assert next_ind["cat"] in limits["cat"]


def test_sparse_subsample_filters_nonfinite_and_keeps_finite():
    """Sparse subsampling should drop non-finite points and keep only finite training data."""
    limits = {"x": (0.0, 1.0)}
    opt = BayesianOptimizer(
        limits=limits,
        rank=0,
        n_initial=1,
        sparse=True,
        sparse_params={"max_points": 3, "top_m": 1},
        rng=random.Random(7),
    )

    losses = [float("nan"), 4.0, float("inf"), 3.0, 2.0, 1.0]
    inds = []
    for i, loss in enumerate(losses):
        ind = Individual(np.array([i / 10.0], dtype=float), limits, generation=i, rank=0)
        ind.loss = loss
        inds.append(ind)

    X, y, sub_inds = opt._subsample(inds)
    assert 1 <= X.shape[0] <= opt.max_points
    assert X.shape[1] == opt.position_dim
    assert len(sub_inds) == X.shape[0]
    assert np.all(np.isfinite(y))


def test_sparse_select_indices_preserves_top_m_elites() -> None:
    """Sparse selector must always include the top_m best objective values."""
    X = np.linspace(0.0, 1.0, 10).reshape(-1, 1)
    y = np.array([9.0, 8.0, 0.2, 7.0, 6.0, 5.0, 4.0, 0.1, 3.0, 2.0], dtype=float)
    lows = np.array([0.0], dtype=float)
    highs = np.array([1.0], dtype=float)

    idx = _sparse_select_indices(
        X,
        y,
        lows,
        highs,
        max_points=5,
        top_m=2,
    )
    assert len(idx) == 5
    elite = set(np.argsort(y)[:2].tolist())
    assert elite.issubset(set(idx.tolist()))


def test_generation_monotonic_when_sparse_drops_local_rank():
    """Generation must be derived from full local history, not sparse-selected subset."""
    limits = {"x": (0.0, 1.0)}
    opt = BayesianOptimizer(
        limits=limits,
        rank=0,
        n_initial=1,
        sparse=True,
        sparse_params={"max_points": 5, "top_m": 5},
        p_explore_start=0.0,
        p_explore_end=0.0,
        optimizer=_fast_test_optimizer(limits),
        optimize_hyperparameters=False,
        rng=random.Random(11),
    )

    inds = []
    # Local-rank history at generations 1..10 with bad losses.
    for gen in range(1, 11):
        ind = Individual(np.array([0.9], dtype=float), limits, generation=gen, rank=0)
        ind.loss = 100.0 + gen
        inds.append(ind)
    # Remote rank has better losses, so sparse top_m keeps only rank 1 points.
    for gen in range(1, 11):
        ind = Individual(np.array([0.1], dtype=float), limits, generation=gen, rank=1)
        ind.loss = float(gen)
        inds.append(ind)

    _, _, sub_inds = opt._subsample(inds)
    assert {ind.rank for ind in sub_inds} == {1}

    next_ind = opt(inds)
    assert next_ind.generation == 11


def test_bayesian_optimizer_repr_contains_key_fields() -> None:
    """Repr should include key BO configuration for debugging."""
    opt = BayesianOptimizer(
        limits={"x": (0.0, 1.0)},
        rank=2,
        world_size=4,
        acquisition_type="PI",
        sparse=True,
        n_initial=7,
        rng=random.Random(0),
    )
    text = repr(opt)
    assert "BayesianOptimizer(" in text
    assert "rank=2" in text
    assert "world_size=4" in text
    assert "acquisition_type='PI'" in text
    assert "sparse=True" in text
    assert "n_initial=7" in text


@pytest.mark.parametrize("fitter_type", ["multi_cpu", "single_gpu", "multi_gpu"])
def test_create_fitter_unsupported_backends_raise_not_implemented(fitter_type: str) -> None:
    """Factory should fail early for fitter backends that are not implemented."""
    with pytest.raises(NotImplementedError, match="not implemented"):
        create_fitter(fitter_type)


def test_bayesian_optimizer_ordinal_integers():
    """Test BayesianOptimizer with ordinal integer parameters (e.g., batch sizes)."""
    limits = {
        "x": (0.0, 1.0),
        "batch_size": (16, 32, 64, 128),
    }
    rng = random.Random(42)
    opt = BayesianOptimizer(
        limits=limits,
        rank=0,
        n_initial=10,
        optimizer=_fast_test_optimizer(limits),
        optimize_hyperparameters=False,
        rng=rng,
    )

    # Position dim: 1 (float) + 1 (ordinal int) = 2
    assert opt.position_dim == 2
    assert opt.dim == 2

    inds = []
    for _ in range(14):
        ind = opt(inds)
        inds.append(ind)
        ind.loss = (ind["x"] - 0.5) ** 2 + abs(ind["batch_size"] - 64)

    # Verify all batch_size values are from the allowed set
    allowed = {16, 32, 64, 128}
    for ind in inds:
        assert isinstance(ind["x"], float)
        assert isinstance(ind["batch_size"], int)
        assert ind["batch_size"] in allowed, f"batch_size={ind['batch_size']} not in {allowed}"
