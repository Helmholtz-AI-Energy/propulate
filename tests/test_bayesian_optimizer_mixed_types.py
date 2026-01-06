"""Tests for BayesianOptimizer with mixed-type (integer and categorical) support."""
import random

import numpy as np
import pytest

from propulate.propagators.bayesopt import BayesianOptimizer, _project_to_discrete
from propulate.population import Individual


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
