"""Tests for BayesianOptimizer with mixed-type (integer and categorical) support."""
import random

import numpy as np
import pytest

from propulate.propagators.bayesopt import BayesianOptimizer, SurrogateFitter, _project_to_discrete
from propulate.population import Individual


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
        rng=rng,
    )

    # Check dimensions
    assert opt.position_dim == 1 + 1 + 3  # float + int + 3 categories
    assert opt.dim == 3

    # Generate individuals
    inds = []
    for _ in range(20):
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
    assert (x_proj[1] == 1.0 or x_proj[2] == 1.0)

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
        sparse_params={"max_points": 50},
        rng=rng,
    )

    # Generate many individuals
    inds = []
    for i in range(100):
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
        n_initial=15,
        acquisition_type="EI",
        rng=rng,
    )

    # Warm up with initial design
    inds = []
    for _ in range(15):
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
        optimize_hyperparameters=True,
        rng=rng,
    )

    # Generate enough points to trigger hyperparameter optimization
    inds = []
    for i in range(30):
        ind = opt(inds)
        inds.append(ind)
        ind.loss = (ind["x"] - 2.5) ** 2 + (ind["n"] - 10) ** 2

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
        rng=rng,
    )

    inds = []
    for _ in range(20):
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
    limits = {
        f"cat{i}": tuple(f"opt{j}" for j in range(10)) for i in range(12)
    }  # 12 * 10 = 120 dimensions

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
            rng=rng,
        )

        inds = []
        for _ in range(15):
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
