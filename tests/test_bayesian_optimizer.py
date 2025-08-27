# tests/test_bayes_acquisitions.py
import copy
import pathlib
import random

import numpy as np
import pytest
from mpi4py import MPI  # Kept for style consistency

from propulate.propagators.bayes_opt import (
    expected_improvement,
    create_acquisition,
)


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

    expected = -(mu - kappa * sigma)
    assert v == pytest.approx(expected, rel=1e-12, abs=1e-12)

    # Increasing kappa should increase the returned value (-mu + kappa*sigma)
    acq2 = create_acquisition("UCB", kappa=kappa * 2)
    v2 = acq2.evaluate(x, model, f_best=0.0)
    assert v2 > v


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
