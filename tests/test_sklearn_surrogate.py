"""Focused tests for the optional GP backend, independent of checkpoint storage."""

import os
import subprocess
import sys
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

    with patch("sklearn.gaussian_process.GaussianProcessRegressor", return_value=gp):
        # Restored mean=5, variance=4: (5-4)*0.8 <= 1, so retain the run.
        # Subtracting std instead would incorrectly prune it.
        assert model.cancel(12.0) is False
        np.testing.assert_array_equal(gp.fit.call_args.args[1], [2.0])
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
