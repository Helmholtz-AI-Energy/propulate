"""Focused tests for the optional GP backend, independent of checkpoint storage."""

import os
import subprocess
import sys
from typing import Any, Callable
from unittest.mock import Mock

import numpy as np
import pytest

from propulate.surrogate import DynamicSurrogate


def test_base_import_without_sklearn() -> None:
    """Base imports and static surrogates work when sklearn is unavailable."""
    script = """
import sys
sys.modules["sklearn"] = None
import propulate
from propulate.surrogate import StaticSurrogate, DynamicSurrogate
StaticSurrogate()
try:
    DynamicSurrogate({"x": (0.0, 1.0)})
except ImportError as exc:
    assert 'propulate[surrogate]' in str(exc)
else:
    raise AssertionError("missing optional dependency was not reported")
"""
    env = dict(os.environ)
    # Avoid initializing another MPI job from an MPI-launched test process.
    env["MPI4PY_RC_INITIALIZE"] = "0"
    subprocess.run([sys.executable, "-c", script], check=True, env=env, timeout=30)


def test_global_fit_and_merge() -> None:
    """Completed and received observations produce finite scalar GP predictions."""
    pytest.importorskip("sklearn")
    model = DynamicSurrogate({"x": (0.0, 1.0)})
    model.start_run(Mock(position=np.array([0.2])))
    assert model.cancel(4.0) is False
    model.update(3.0)
    model.merge((np.array([0.8]), 2.0))
    assert model.history_X.shape == (2, 1)
    assert model.global_gpr is not None
    assert np.isfinite(model.global_gpr.predict([[0.5]])).all()
    np.testing.assert_array_equal(model.data()[0], [0.8])


def test_local_mean_and_variance_rule() -> None:
    """Residual fitting restores the prior mean and retains variance-based pruning."""
    pytest.importorskip("sklearn")

    model = DynamicSurrogate({"x": (0.0, 1.0)})
    model.first_run = False
    model.history_Y = np.array([[1.0]])
    model.global_gpr = Mock()
    model.global_gpr.predict.return_value = np.array([10.0])
    model.start_run(Mock(position=np.array([0.5])))
    model.max_idx = 5
    gp = Mock(kernel_=model.local_kernel)
    gp.predict.return_value = (np.array([-5.0]), np.array([2.0]))
    from unittest.mock import patch

    with patch.object(model, "_fit_local", side_effect=lambda: setattr(model, "local_gpr", gp)):
        # Restored mean=5, variance=4: (5-4)*0.8 <= 1, so retain the run.
        # Subtracting std instead would incorrectly prune it.
        assert model.cancel(12.0) is False
        gp.predict.return_value = (np.array([-2.0]), np.array([2.0]))
        assert model.cancel(11.0) is True


def test_constructor_preserves_random_state() -> None:
    """Constructing a surrogate does not reset application random generators."""
    import random

    pytest.importorskip("sklearn")
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    DynamicSurrogate({"x": (0.0, 1.0)})
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert numpy_state[0] == after[0]
    np.testing.assert_array_equal(numpy_state[1], after[1])
    assert numpy_state[2:] == after[2:]


def test_existing_history_rules_unchanged() -> None:
    """Document existing zero-loss and run-completion behavior without fixing it."""
    pytest.importorskip("sklearn")
    model = DynamicSurrogate({"x": (0.0, 1.0)})
    model.start_run(Mock(position=np.array([0.2])))
    model.cancel(4.0)
    model.cancel(3.0)
    model.update(2.0)
    model.start_run(Mock(position=np.array([0.8])))
    model.current_run_data = np.array([[4.0], [3.0]])
    model.update(1.0)
    assert model.history_Y.shape == (1, 1)
    model.merge((np.array([0.5]), 0.0))
    assert model.history_Y.shape == (1, 1)


def test_trainable_local_mean() -> None:
    """Joint fitting moves the global initial mean and optimizes the kernel."""
    pytest.importorskip("sklearn")
    from unittest.mock import patch

    from scipy.linalg import cho_solve
    from scipy.optimize import minimize

    model = DynamicSurrogate({"x": (0.0, 1.0)})
    model.first_run = False
    model.global_gpr = Mock()
    model.global_gpr.predict.return_value = np.array([20.0])
    model.start_run(Mock(position=np.array([0.5])))
    model.current_run_data = np.array([[3.0], [2.0], [1.5], [1.2]])
    initial_theta = model.local_kernel.theta.copy()
    with patch("scipy.optimize.minimize", wraps=minimize) as optimizer:
        model._fit_local()
    initial = optimizer.call_args.args[1]
    assert initial[-1] == 20.0
    np.testing.assert_array_equal(initial[:-1], initial_theta)
    assert abs(model.mean_function - 20.0) > 1.0
    assert model.local_gpr is not None
    assert not np.allclose(model.local_gpr.kernel_.theta, initial_theta)
    # At the joint optimum the constant mean is the generalized least-squares mean.
    ones = np.ones(4)
    inverse_y = cho_solve((model.local_gpr.L_, True), model.current_run_data.ravel())
    inverse_ones = cho_solve((model.local_gpr.L_, True), ones)
    expected_mean = ones @ inverse_y / (ones @ inverse_ones)
    assert model.mean_function == pytest.approx(expected_mean, abs=1e-3)
    assert np.isfinite(model.local_gpr.predict([[10]], return_std=True)).all()
    fitted_theta = model.local_gpr.kernel_.theta.copy()
    fitted_mean = model.mean_function
    model.current_run_data = np.vstack([model.current_run_data, [1.1]])
    with patch("scipy.optimize.minimize", wraps=minimize) as optimizer:
        model._fit_local()
    np.testing.assert_allclose(optimizer.call_args.args[1][:-1], fitted_theta)
    assert optimizer.call_args.args[1][-1] == fitted_mean
    model.start_run(Mock(position=np.array([0.8])))
    assert model.mean_function == 20.0
    assert model.local_gpr is None


def test_joint_objective_gradient() -> None:
    """The joint mean/kernel gradient matches finite differences."""
    pytest.importorskip("sklearn")
    from unittest.mock import patch

    from scipy.optimize import OptimizeResult, check_grad

    model = DynamicSurrogate({"x": (0.0, 1.0)})
    model.current_run_data = np.array([[3.0], [2.0], [1.0]])
    model.mean_function = 4.0

    def inspect_objective(
        fun: Callable[[np.ndarray], tuple[float, np.ndarray]], initial: np.ndarray, **kwargs: Any
    ) -> OptimizeResult:
        assert check_grad(lambda p: fun(p)[0], lambda p: fun(p)[1], initial) < 1e-5
        return OptimizeResult(x=initial, fun=fun(initial)[0])

    with patch("scipy.optimize.minimize", side_effect=inspect_objective):
        model._fit_local()
