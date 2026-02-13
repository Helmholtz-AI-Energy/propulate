from abc import ABC, abstractmethod
import logging
import random
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Protocol, cast

import numpy as np
from scipy.optimize import fmin_l_bfgs_b, minimize
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, RBF, ConstantKernel as C, WhiteKernel, Matern

from ..propagators import Propagator
from ..population import Individual, _normalize_param_type


def get_default_kernel_sklearn(
    dim: int,
    categorical_blocks: Optional[List[Tuple[int, int]]] = None,
) -> Kernel:
    """
    Get default Matern kernel for sklearn GaussianProcessRegressor.

    Parameters
    ----------
    dim : int
        Position array dimension (accounts for one-hot encoding of categoricals).
        This may be larger than the number of parameters due to categorical encoding.
    categorical_blocks : list of (offset, n_categories), optional
        One-hot dimension blocks. When provided, these dimensions get larger
        initial length scales (5.0) and a tighter lower bound (1.0) to reduce
        their influence relative to continuous dimensions.

    Returns
    -------
    Kernel
        Default kernel combining Constant, Matern, and WhiteKernel.
    """
    length_scales = np.ones(dim)
    ls_bounds = np.full((dim, 2), [1e-2, 1e3])
    if categorical_blocks:
        for cat_offset, n_categories in categorical_blocks:
            for j in range(n_categories):
                length_scales[cat_offset + j] = 5.0
                ls_bounds[cat_offset + j] = [1.0, 1e3]
    kernel = (
        C(1.0, (1e-3, 1e5)) *
        Matern(length_scale=length_scales, length_scale_bounds=ls_bounds)
        + WhiteKernel(noise_level=1e-1, noise_level_bounds=(1e-8, 1e1))
    )
    return kernel
    
_bo_log = logging.getLogger("propulate.bayesopt")


def _log_gp_diagnostics(
    model: GaussianProcessRegressor,
    y: np.ndarray,
    rank: int,
    generation: int,
    optimized_hyperparameters: bool,
) -> None:
    """Log GP kernel diagnostics after fitting.

    Reports training data statistics, fitted kernel parameters, and warns
    when length_scale values are at or near their bounds.
    """
    if not hasattr(model, "kernel_"):
        return

    _bo_log.info(
        "[BO rank=%d gen=%d] GP fit (hp_opt=%s): n_train=%d, "
        "y_min=%.4g, y_max=%.4g, y_mean=%.4g, y_std=%.4g",
        rank, generation, optimized_hyperparameters,
        len(y), y.min(), y.max(), y.mean(), y.std(),
    )
    _bo_log.info(
        "[BO rank=%d gen=%d] Fitted kernel: %s",
        rank, generation, model.kernel_,
    )
    _bo_log.info(
        "[BO rank=%d gen=%d] Log-marginal-likelihood: %.4f",
        rank, generation, model.log_marginal_likelihood_value_,
    )

    # Check whether Matern / RBF length_scale is near its bounds.
    # Kernel structure: (C * Matern) + WhiteKernel  →  k1 = C*Matern, k2 = White
    #                   k1.k1 = C, k1.k2 = Matern
    try:
        k_stationary = model.kernel_.k1.k2  # Matern or RBF
        ls = np.atleast_1d(k_stationary.length_scale)
        bounds = np.atleast_2d(k_stationary.length_scale_bounds)
        lb = bounds[:, 0]
        ub = bounds[:, 1]
        at_lb = int(np.sum(ls <= lb * 1.05))
        at_ub = int(np.sum(ls >= ub * 0.95))
        if at_lb > 0:
            _bo_log.warning(
                "[BO rank=%d gen=%d] %d/%d length_scale dims at LOWER bound "
                "(ls=%s, lb=%s)",
                rank, generation, at_lb, len(ls), ls, lb,
            )
        if at_ub > 0:
            _bo_log.warning(
                "[BO rank=%d gen=%d] %d/%d length_scale dims at UPPER bound "
                "(ls=%s, ub=%s)",
                rank, generation, at_ub, len(ls), ls, ub,
            )
    except (AttributeError, TypeError, IndexError):
        pass  # kernel structure may differ from expected layout

    # Report noise level from WhiteKernel if present.
    try:
        noise = model.kernel_.k2.noise_level
        _bo_log.info(
            "[BO rank=%d gen=%d] WhiteKernel noise_level: %.4g",
            rank, generation, noise,
        )
    except AttributeError:
        pass


# --- Sparse selection helpers -------------------------------------------------
def _sparse_select_indices(
    X: np.ndarray,
    y: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    max_points: int,
    top_m: int = 150,
    continuous_dims: Optional[List[int]] = None,
    categorical_blocks: Optional[List[Tuple[int, int]]] = None,
) -> np.ndarray:
    """
    Select up to ``max_points`` indices from (X, y) using a deterministic strategy:
    - Always keep the top-M best points by y (lower is better).
    - Fill the remaining budget with greedy farthest-point sampling in a
      whitened feature space for better geometric spread.

    When ``continuous_dims`` and ``categorical_blocks`` are provided, uses a
    weighted mixed distance: whitened Euclidean on continuous dimensions plus
    Hamming distance on categorical dimensions, balanced so each block
    contributes equally.  Otherwise falls back to pure Euclidean on all dims.

    Returns
    -------
    idx : np.ndarray (k,)
        Selected indices into X/y (k <= max_points).
    """
    n = X.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    # Cap parameters
    k_budget = min(max_points, n)
    m = max(0, min(top_m, k_budget))

    # Sort by objective (ascending, best first) and keep top-M
    order = np.argsort(y)
    selected = list(order[:m])
    if len(selected) >= k_budget:
        return np.array(selected[:k_budget], dtype=int)

    # Build distance function
    use_mixed = continuous_dims is not None and categorical_blocks is not None
    if use_mixed:
        assert continuous_dims is not None and categorical_blocks is not None
        n_cont = len(continuous_dims)
        n_cat_params = len(categorical_blocks)

        # Whiten continuous dims only
        if n_cont > 0:
            cont_idx = np.array(continuous_dims)
            X_cont = X[:, cont_idx]
            cont_lows = lows[cont_idx]
            cont_highs = highs[cont_idx]
            cont_scale = np.where((cont_highs - cont_lows) == 0.0, 1.0, cont_highs - cont_lows)
            X_cont_01 = (X_cont - cont_lows) / cont_scale
            cont_mean = X_cont_01.mean(axis=0)
            cont_std = X_cont_01.std(axis=0)
            cont_std[cont_std < 1e-12] = 1.0
            Xw_cont = (X_cont_01 - cont_mean) / cont_std
        else:
            Xw_cont = np.empty((n, 0))

        # Extract categorical indices (argmax of each one-hot block)
        if n_cat_params > 0:
            cat_indices = np.zeros((n, n_cat_params), dtype=int)
            for j, (off, n_c) in enumerate(categorical_blocks):  # type: ignore[arg-type]
                cat_indices[:, j] = np.argmax(X[:, off:off + n_c], axis=1)
        else:
            cat_indices = np.empty((n, 0), dtype=int)

        # Per-block weights so continuous and categorical blocks contribute equally
        w_cont = 1.0 / max(n_cont, 1) if n_cont > 0 else 0.0
        w_cat = 1.0 / max(n_cat_params, 1) if n_cat_params > 0 else 0.0

        def _dist_from(i: int, pool_idx: np.ndarray) -> np.ndarray:
            d = np.zeros(len(pool_idx))
            if n_cont > 0:
                diff = Xw_cont[pool_idx] - Xw_cont[i]
                d += w_cont * np.sum(diff ** 2, axis=1)
            if n_cat_params > 0:
                hamming = np.sum(cat_indices[pool_idx] != cat_indices[i], axis=1)
                d += w_cat * hamming.astype(float)
            return d
    else:
        # Legacy path: pure Euclidean on all dims
        scale = np.where((highs - lows) == 0.0, 1.0, (highs - lows))
        X01 = (X - lows) / scale
        xmean = X01.mean(axis=0)
        xstd = X01.std(axis=0)
        xstd[xstd < 1e-12] = 1.0
        Xw = (X01 - xmean) / xstd

        def _dist_from(i: int, pool_idx: np.ndarray) -> np.ndarray:
            return np.sum((Xw[pool_idx] - Xw[i]) ** 2, axis=1)

    # Pool are non-selected indices
    all_idx = np.arange(n)
    mask_sel = np.zeros(n, dtype=bool)
    mask_sel[selected] = True
    pool = all_idx[~mask_sel]

    if pool.size == 0:
        return np.array(selected, dtype=int)

    # Initialize min-distance-to-selected for all candidates in pool
    dist_min = np.full(pool.shape[0], np.inf, dtype=float)
    if selected:
        for s in selected:
            ds = _dist_from(s, pool)
            dist_min = np.minimum(dist_min, ds)
    else:
        # If nothing selected yet (m == 0), seed with the best point to anchor
        s0 = int(order[0])
        selected.append(s0)
        mask_sel[s0] = True
        pool = all_idx[~mask_sel]
        dist_min = _dist_from(s0, pool)

    # Greedy farthest-point until budget is met
    while len(selected) < k_budget and pool.size > 0:
        k = int(np.argmax(dist_min))
        j = int(pool[k])
        selected.append(j)

        # Remove chosen j from pool and its distance entry
        pool = np.delete(pool, k)
        if pool.size == 0:
            break
        dnew = _dist_from(j, pool)
        dist_min = np.delete(dist_min, k)
        dist_min = np.minimum(dist_min, dnew)

    return np.array(selected[:k_budget], dtype=int)
    
class SupportsPredict(Protocol):
    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, np.ndarray]: ...

def expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
    minimize: bool = True,
    xi: float = 0.01,
) -> np.ndarray:
    """
    Compute Expected Improvement for minimization or maximization.
    """
    if minimize:
        improvement = f_best - mu - xi
    else:
        improvement = mu - f_best - xi
    with np.errstate(divide="ignore", invalid="ignore"):
        eps = 1e-12
        Z = improvement / np.maximum(sigma, eps)
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0] = np.maximum(improvement[sigma == 0], 0)
    return ei


class AcquisitionFunction(ABC):
    """
    Base class for acquisition functions.
    """
    @abstractmethod
    def evaluate(
        self,
        x: np.ndarray,
        model: Union[GaussianProcessRegressor, SupportsPredict],
        f_best: float,
    ) -> float:
        """
        Evaluate acquisition at x given surrogate and current best.
        """
        ...

    def _predict(
        self,
        x: np.ndarray,
        model: Union[GaussianProcessRegressor, SupportsPredict],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and std at x using the surrogate model.
        """
        x = np.atleast_2d(x)
        mu, sigma = cast(Tuple[np.ndarray, np.ndarray], model.predict(x, return_std=True))
        return mu, sigma


class ExpectedImprovement(AcquisitionFunction):
    """
    Expected Improvement acquisition for minimization.
    """
    def __init__(self, xi: float = 0.01):
        self.xi = xi

    def evaluate(
        self,
        x: np.ndarray,
        model,
        f_best: float,
    ) -> float:
        mu, sigma = self._predict(x, model)
        ei = expected_improvement(mu, sigma, f_best, minimize=True, xi=self.xi)
        return float(ei[0])


class ProbabilityImprovement(AcquisitionFunction):
    """
    Probability of Improvement acquisition for minimization.
    """
    def __init__(self, xi: float = 0.01):
        self.xi = xi

    def evaluate(self, x: np.ndarray, model, f_best: float) -> float:
        mu, sigma = self._predict(x, model)
        imp = f_best - mu - self.xi

        # Safe Z with explicit sigma==0 handling
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = np.divide(imp, sigma, out=np.full_like(imp, np.inf), where=sigma > 0)
            pi = norm.cdf(Z)
            # If sigma == 0, PI is 1 when improvement>0 else 0
            pi = np.where(sigma == 0, (imp > 0).astype(float), pi)

        return float(pi[0])


class UpperConfidenceBound(AcquisitionFunction):
    """Lower Confidence Bound (named UCB here for legacy reasons) for minimization.

    We define ``ucb = mu - kappa * sigma`` so that *minimizing* this value trades off
    exploitation (low mean) and exploration (high uncertainty). This is effectively
    the Lower Confidence Bound (LCB) formulation used for minimization problems.
    """
    def __init__(self, kappa: float = 1.96):
        self.kappa = kappa

    def evaluate(
        self,
        x: np.ndarray,
        model,
        f_best: float,
    ) -> float:
        mu, sigma = self._predict(x, model)
        ucb = mu - self.kappa * sigma
        return float(ucb[0])


LowerConfidenceBound = UpperConfidenceBound


class AcquisitionType(Enum):
    EI = "EI"
    PI = "PI"
    UCB = "UCB"


def create_acquisition(
    acq_type: str,
    *,
    rank_stretch: bool = False,
    rank: Optional[int] = None,
    size: Optional[int] = None,
    factor_min: float = 0.5,
    factor_max: float = 2.0,
    **params
) -> AcquisitionFunction:
    """
    Factory to create an acquisition function by name.
    If rank_stretch=True and size>1, linearly rescales “xi” (for EI/PI)
    or “kappa” (for UCB) by factor_min→factor_max across ranks 0…size-1.
    """
    try:
        at = AcquisitionType(acq_type.upper())
    except ValueError:
        valid = [e.value for e in AcquisitionType]
        raise ValueError(f"Unknown acquisition type '{acq_type}'. Valid types: {valid}")

    # only do per-rank stretch if requested
    if rank_stretch and rank is not None and size and size > 1:
        rel = rank / (size - 1)  # goes 0.0 … 1.0
        stretch = factor_min + rel * (factor_max - factor_min)
        if at in (AcquisitionType.EI, AcquisitionType.PI):
            xi0 = params.get("xi", 0.01)
            params["xi"] = xi0 * stretch
        elif at == AcquisitionType.UCB:
            k0 = params.get("kappa", 1.96)
            params["kappa"] = k0 * stretch

    # instantiate the right class
    if at == AcquisitionType.EI:
        return ExpectedImprovement(**params)
    elif at == AcquisitionType.PI:
        return ProbabilityImprovement(**params)
    else:  # at == AcquisitionType.UCB
        return UpperConfidenceBound(**params)


class MultiStartAcquisitionOptimizer:
    """Multi-start L-BFGS-B optimizer for acquisition functions."""

    def __init__(
        self,
        n_candidates: int = 256,
        n_restarts: int = 5,
        polish: bool = True,
        minimize_options: Optional[Dict[str, Union[int, float]]] = None,
    ) -> None:
        if n_candidates <= 0:
            raise ValueError("n_candidates must be positive")
        if n_restarts < 0:
            raise ValueError("n_restarts must be non-negative")
        self.n_candidates = n_candidates
        self.n_restarts = n_restarts
        self.polish = polish
        self.minimize_options: Dict[str, Union[int, float]] = (
            {"maxiter": 200, "maxfun": 2000, "ftol": 1e-9, "gtol": 1e-6}
            if minimize_options is None
            else dict(minimize_options)
        )

    @staticmethod
    def _make_generator(rng: Optional[random.Random], dim: int) -> np.random.Generator:
        if isinstance(rng, random.Random):
            seed = rng.randrange(0, 2**32 - 1)
            return np.random.default_rng(seed)
        if isinstance(rng, np.random.Generator):  # type: ignore[arg-type]
            return rng
        # fall back to default generator seeded from entropy
        return np.random.default_rng()

    def _sample_candidates(
        self,
        lows: np.ndarray,
        highs: np.ndarray,
        rng: Optional[random.Random],
        n_points: int,
    ) -> np.ndarray:
        generator = self._make_generator(rng, len(lows))
        # Generator.uniform supports vector bounds via broadcasting
        samples = generator.uniform(lows, highs, size=(n_points, len(lows)))
        return samples

    def optimize(
        self,
        acq_func,
        bounds: np.ndarray,
        rng: Optional[random.Random] = None,
    ) -> np.ndarray:
        lows, highs = bounds
        dim = lows.shape[0]
        n_candidates = max(self.n_candidates, max(1, self.n_restarts))
        candidates = self._sample_candidates(lows, highs, rng, n_candidates)

        # Be robust to occasional numerical issues in acquisition evaluation
        vals: List[float] = []
        for x in candidates:
            try:
                v = float(acq_func(x))
                if not np.isfinite(v):
                    v = np.inf
            except Exception:
                # Treat any failure as an invalid candidate
                v = np.inf
            vals.append(v)
        values = np.array(vals, dtype=float)
        finite_mask = np.isfinite(values)
        if not np.any(finite_mask):
            # fall back to random point if everything failed
            return candidates[0]

        candidates = candidates[finite_mask]
        values = values[finite_mask]

        order = np.argsort(values)
        best_idx = order[0]
        best_x = candidates[best_idx]
        best_val = values[best_idx]

        if self.polish and self.n_restarts > 0:
            bounds_pairs = list(zip(lows, highs))
            for idx in order[: min(self.n_restarts, len(order))]:
                x0 = candidates[idx]
                try:
                    res = minimize(
                        acq_func,
                        x0,
                        method="L-BFGS-B",
                        bounds=bounds_pairs,
                        options=self.minimize_options,
                    )
                except Exception:
                    continue

                if not np.isfinite(res.fun):
                    continue

                if res.fun < best_val:
                    best_val = float(res.fun)
                    best_x = np.clip(res.x, lows, highs)

        return best_x


def _robust_lbfgs(obj_func, initial_theta, bounds):
    """
    Replacement for sklearn's default optimizer.
    Tries L-BFGS-B with friendlier options; if line search still fails,
    falls back to a bounded Powell step (gradient-free).
    Returns (theta_opt, f_min).
    """
    def f_and_g(theta):
        # sklearn passes obj_func(theta, eval_gradient=True) -> (f, g)
        return obj_func(theta, eval_gradient=True)

    # 1) Two attempts with progressively looser / longer line search
    attempts = [
        dict(maxiter=200, maxls=50,  factr=1e7, pgtol=1e-8),
        dict(maxiter=400, maxls=100, factr=1e9, pgtol=1e-6),
    ]
    best = (None, np.inf)
    for opts in attempts:
        # Cast to Any to silence strict type checks from stubs; runtime accepts these options.
        theta, fval, info = cast(Any, fmin_l_bfgs_b)(
            f_and_g, initial_theta, bounds=bounds, **opts
        )
        if fval < best[1]:
            best = (theta, fval)
        if info.get("warnflag", 1) == 0:  # converged
            return theta, fval  # success -> no warnings

    # 2) Fallback: gradient-free Powell within bounds (stable, slower)
    def f_only(theta):
        return obj_func(theta, eval_gradient=False)

    res = minimize(f_only, best[0] if best[0] is not None else initial_theta,
                   method="Powell", bounds=bounds,
                   options={"maxiter": 1000, "xtol": 1e-4, "ftol": 1e-4, "disp": False})
    return res.x, res.fun
    
    
class SurrogateFitter(ABC):
    """
    Base class for surrogate model fitters leveraging different compute backends.
    """
    @abstractmethod
    def fit(
        self,
        kernel: Kernel,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> Union[GaussianProcessRegressor, SupportsPredict]:
        """
        Fit and return a surrogate model trained on (X, y).
        """
        ...


class SingleCPUFitter(SurrogateFitter):
    def __init__(self, optimize_hyperparameters: bool = True,
                 n_restarts: int = 5, random_state: Optional[int] = None,
                 alpha: float = 1e-6):
        self.optimize_hyperparameters = optimize_hyperparameters
        self.n_restarts = n_restarts
        self.random_state = random_state
        self.alpha = alpha

    def fit(self, kernel, X, y, **kwargs):
        flag = kwargs.get("optimize_hyperparameters", self.optimize_hyperparameters)
        opt = _robust_lbfgs if flag else None
        nre = self.n_restarts if flag else 0
        gp = GaussianProcessRegressor(
            kernel=kernel, optimizer=opt, n_restarts_optimizer=nre,
            normalize_y=True, random_state=self.random_state, alpha=self.alpha
        )
        gp.fit(X, y)
        return gp

class MultiCPUFitter(SurrogateFitter):
    """Stub for a parallel CPU fitter (not yet implemented in this PR)."""

    def fit(self, kernel: Kernel, X: np.ndarray, y: np.ndarray, **kwargs):
        raise NotImplementedError(
            "MultiCPUFitter is not supported in this Bayesian Optimizer PR. "
            "Please use SingleCPUFitter for now."
        )


class SingleGPUFitter(SurrogateFitter):
    """Stub for a single-GPU fitter (not yet implemented in this PR)."""

    def fit(self, kernel: Kernel, X: np.ndarray, y: np.ndarray, **kwargs) -> SupportsPredict:
        raise NotImplementedError(
            "SingleGPUFitter is not supported in this Bayesian Optimizer PR. "
            "Please use SingleCPUFitter for now."
        )


class MultiGPUFitter(SurrogateFitter):
    """Stub for a multi-GPU fitter (not yet implemented in this PR)."""

    def fit(self, kernel: Kernel, X: np.ndarray, y: np.ndarray, **kwargs) -> SupportsPredict:
        raise NotImplementedError(
            "MultiGPUFitter is not supported in this Bayesian Optimizer PR. "
            "Please use SingleCPUFitter for now."
        )

class FitterType(Enum):
    SINGLE_CPU = "single_cpu"
    MULTI_CPU = "multi_cpu"
    SINGLE_GPU = "single_gpu"
    MULTI_GPU = "multi_gpu"


def create_fitter(
    fitter_type: str,
    **kwargs
) -> SurrogateFitter:
    """
    Factory to create a surrogate fitter by resource type.
    """
    try:
        ft = FitterType(fitter_type.lower())
    except ValueError:
        valid = [e.value for e in FitterType]
        raise ValueError(f"Unknown fitter type '{fitter_type}'. Valid types: {valid}")

    if ft == FitterType.SINGLE_CPU:
        return SingleCPUFitter(**kwargs)
    elif ft == FitterType.MULTI_CPU:
        return MultiCPUFitter(**kwargs)
    elif ft == FitterType.SINGLE_GPU:
        return SingleGPUFitter(**kwargs)
    elif ft == FitterType.MULTI_GPU:
        return MultiGPUFitter(**kwargs)
    else:
        raise ValueError(f"Unsupported fitter type '{fitter_type}'.")


def _project_to_discrete(
    x: np.ndarray,
    limits: Dict[str, Union[Tuple[float, float], Tuple[int, int], Tuple[str, ...]]],
    param_types: Dict[str, type],
) -> np.ndarray:
    """
    Project continuous position array to valid discrete/categorical values.

    This function handles:
    - Integer parameters: rounds to nearest integer and clips to bounds
    - Categorical parameters: projects one-hot vectors to valid one-hot encoding
    - Float parameters: passes through unchanged (with bounds clipping)

    Parameters
    ----------
    x : np.ndarray
        Continuous position array from acquisition optimization
    limits : Dict[str, Union[Tuple[float, float], Tuple[int, int], Tuple[str, ...]]]
        Search space limits
    param_types : Dict[str, type]
        Type of each parameter

    Returns
    -------
    np.ndarray
        Projected position array with valid discrete/categorical values
    """
    x_proj = x.copy()
    offset = 0

    for key in limits:
        if param_types[key] is str:
            # Categorical: project to valid one-hot encoding
            n_categories = len(limits[key])
            one_hot_vec = x[offset:offset + n_categories]

            # Use softmax to get probabilities, then argmax for winner-takes-all
            # This handles negative values and out-of-bounds gracefully
            exp_vals = np.exp(one_hot_vec - np.max(one_hot_vec))  # numerical stability
            probs = exp_vals / np.sum(exp_vals)
            winner_idx = np.argmax(probs)

            # Set to valid one-hot encoding
            x_proj[offset:offset + n_categories] = 0.0
            x_proj[offset + winner_idx] = 1.0

            offset += n_categories

        elif param_types[key] is int:
            if len(limits[key]) == 2:
                # Integer range: round and clip to bounds
                low, high = limits[key]
                rounded_val = np.rint(x[offset])
                x_proj[offset] = float(np.clip(rounded_val, low, high))
            else:
                # Ordinal integer: snap to nearest valid value
                values = np.array(limits[key], dtype=float)
                idx = int(np.argmin(np.abs(values - x[offset])))
                x_proj[offset] = float(values[idx])
            offset += 1

        else:  # float
            # Float: just clip to bounds (already done by optimizer)
            low, high = limits[key]
            x_proj[offset] = np.clip(x[offset], low, high)
            offset += 1

    return x_proj


class BayesianOptimizer(Propagator):
    def __init__(
        self,
        limits: Dict[str, Union[Tuple[float, float], Tuple[int, int], Tuple[str, ...]]],
        rank: int,
        world_size: int = 1,
        fitter: Optional[SurrogateFitter] = None,
        optimizer: Optional[Any] = None,
        kernel: Optional[Kernel] = None,
        optimize_hyperparameters: bool = True,
        acquisition_type: str = "EI",
        acquisition_params: Optional[Dict[str, float]] = None,
        rank_stretch: bool = True,
        factor_min: float = 0.5,
        factor_max: float = 2.0,
        sparse: bool = False,
        sparse_params: Optional[Dict[str, int]] = None,
        rng: Optional[random.Random] = None,
        # Initial design parameters
        n_initial: Optional[int] = None,
        initial_design: str = "sobol",  # "sobol" | "random" | "lhs" (lhs falls back to sobol-like behavior per-call)
        # Exploration schedule (epsilon-greedy) parameters
        p_explore_start: float = 0.2,
        p_explore_end: float = 0.02,
        p_explore_tau: float = 150.0,
        anneal_acquisition: bool = True,
        # Optional dynamic acquisition switching
        second_acquisition_type: Optional[str] = None,
        acq_switch_generation: Optional[int] = None,
        second_acquisition_params: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize Bayesian Optimizer with Gaussian Process surrogate.

        Supports mixed search spaces with float, integer, and categorical parameters
        through continuous relaxation and projection.

        Parameters
        ----------
        limits : Dict[str, Union[Tuple[float, float], Tuple[int, int], Tuple[int, ...], Tuple[str, ...]]]
            Search space limits. Each parameter can be:
            - Float: Tuple of (low, high) floats for continuous parameters
            - Int range: Tuple of (low, high) ints for discrete parameters
            - Int ordinal: Tuple of distinct int values in ascending order
            - Categorical: Tuple of strings for categorical parameters

            Example:
                {
                    "learning_rate": (0.001, 0.1),           # continuous
                    "num_layers": (1, 10),                    # discrete range
                    "batch_size": (16, 32, 64, 128),          # discrete ordinal
                    "activation": ("relu", "tanh", "sigmoid") # categorical
                }

        rank : int
            MPI rank of this optimizer instance for parallel optimization
        world_size : int, optional
            Total number of MPI ranks (default: 1). Required for rank stretching
            to diversify acquisition parameters across parallel workers.
        fitter : SurrogateFitter, optional
            Surrogate model fitter (default: SingleCPUFitter)
        optimizer : Any, optional
            Acquisition function optimizer (default: MultiStartAcquisitionOptimizer)
        kernel : Kernel, optional
            GP kernel (default: Constant * Matern + WhiteKernel)
        optimize_hyperparameters : bool, optional
            Whether to optimize GP hyperparameters (default: True)
        acquisition_type : str, optional
            Acquisition function type: "EI", "PI", or "UCB" (default: "EI")
        acquisition_params : Dict[str, float], optional
            Acquisition function parameters (e.g., {"xi": 0.01} for EI/PI, {"kappa": 1.96} for UCB)
        rank_stretch : bool, optional
            Scale acquisition parameters across MPI ranks for diversity (default: True)
        factor_min : float, optional
            Minimum scaling factor for rank stretching (default: 0.5)
        factor_max : float, optional
            Maximum scaling factor for rank stretching (default: 2.0)
        sparse : bool, optional
            Enable sparse GP fitting for large datasets (default: False)
        sparse_params : Dict[str, int], optional
            Sparse fitting parameters: {"max_points": int, "top_m": int}
        rng : random.Random, optional
            Random number generator
        n_initial : int, optional
            Number of initial design points (default: min(10, 10 * num_params))
        initial_design : str, optional
            Initial design method: "sobol", "random", or "lhs" (default: "sobol")
        p_explore_start : float, optional
            Initial epsilon-greedy exploration probability (default: 0.2)
        p_explore_end : float, optional
            Final epsilon-greedy exploration probability (default: 0.02)
        p_explore_tau : float, optional
            Decay rate for exploration probability (default: 150.0)
        anneal_acquisition : bool, optional
            Anneal acquisition parameters over time (default: True)
        second_acquisition_type : str, optional
            Second acquisition function to switch to after acq_switch_generation
        acq_switch_generation : int, optional
            Generation at which to switch acquisition functions
        second_acquisition_params : Dict[str, float], optional
            Parameters for second acquisition function

        Notes
        -----
        Integer and categorical parameters are handled via continuous relaxation:
        - Integers: Optimized as continuous, then rounded to nearest integer
        - Categoricals: One-hot encoded, optimized continuously, projected via argmax

        The GP surrogate operates in position space, which may have higher dimension
        than the number of parameters due to one-hot encoding of categorical variables.

        Examples
        --------
        >>> # Float-only optimization
        >>> limits = {"x": (0.0, 10.0), "y": (-5.0, 5.0)}
        >>> opt = BayesianOptimizer(limits=limits, rank=0)

        >>> # Mixed-type optimization
        >>> limits = {
        ...     "learning_rate": (0.001, 0.1),           # continuous
        ...     "num_layers": (1, 10),                    # discrete
        ...     "activation": ("relu", "tanh", "sigmoid") # categorical
        ... }
        >>> opt = BayesianOptimizer(limits=limits, rank=0, n_initial=20)
        """
        super().__init__(parents=-1, offspring=1, rng=rng)
        self.limits = limits

        # Validate limits before any state initialization
        if not limits:
            raise ValueError("limits cannot be empty")

        for key, bounds in limits.items():
            if len(bounds) < 2:
                raise ValueError(f"Parameter '{key}' must have at least 2 bounds/categories")

            if isinstance(bounds[0], (int, float, np.integer, np.floating)):
                if isinstance(bounds[0], (int, np.integer)) and len(bounds) > 2:
                    # Ordinal integer: explicit set of values like (16, 32, 64, 128)
                    if len(set(bounds)) != len(bounds):
                        raise ValueError(f"Ordinal parameter '{key}' has duplicate values")
                    if sorted(bounds) != list(bounds):
                        raise ValueError(f"Ordinal parameter '{key}' values must be in ascending order")
                else:
                    if len(bounds) != 2:
                        raise ValueError(f"Numeric parameter '{key}' must have exactly 2 bounds")
                    if bounds[0] >= bounds[1]:
                        raise ValueError(f"Parameter '{key}': lower bound must be < upper bound")
            elif isinstance(bounds[0], str):
                if len(set(bounds)) != len(bounds):
                    raise ValueError(f"Categorical parameter '{key}' has duplicate categories")
            else:
                raise TypeError(f"Parameter '{key}' has unsupported type: {type(bounds[0])}")

        # Type detection for mixed-type support (normalizes numpy scalar types)
        self.param_types = {key: _normalize_param_type(limits[key][0]) for key in limits}

        # Calculate position dimension (accounting for one-hot encoding)
        self.position_dim = sum(
            len(limits[k]) if isinstance(limits[k][0], str) else 1
            for k in limits
        )

        # Precompute dimension layout for decoupled initial design, sparse selection,
        # and dimension-aware kernel initialization.
        self._continuous_dims: List[int] = []       # position indices of float/int params
        self._categorical_blocks: List[Tuple[int, int]] = []  # (offset, n_categories) per categorical
        _offset = 0
        for key in limits:
            if self.param_types[key] is str:
                n_cat = len(limits[key])
                self._categorical_blocks.append((_offset, n_cat))
                _offset += n_cat
            else:
                self._continuous_dims.append(_offset)
                _offset += 1
        self._n_continuous = len(self._continuous_dims)

        # Number of parameters (not position dimensions)
        self.dim = len(limits)

        # Create continuous bounds for L-BFGS-B optimization
        # Float/int: use actual bounds
        # Categorical: [0, 1] for each one-hot dimension
        self._continuous_lows = []
        self._continuous_highs = []
        for key in limits:
            if isinstance(limits[key][0], str):
                # Categorical: bounds [0, 1] for each one-hot dimension
                self._continuous_lows.extend([0.0] * len(limits[key]))
                self._continuous_highs.extend([1.0] * len(limits[key]))
            else:
                # Numeric (int or float): use min/max of bounds
                # For ordinal integers like (16, 32, 64, 128) this gives [16, 128]
                self._continuous_lows.append(float(min(limits[key])))
                self._continuous_highs.append(float(max(limits[key])))

        self._continuous_lows = np.array(self._continuous_lows, dtype=float)
        self._continuous_highs = np.array(self._continuous_highs, dtype=float)

        # Use continuous bounds for internal optimization
        self.limits_arr = np.array([self._continuous_lows, self._continuous_highs])

        # Warn about high-dimensional position spaces
        if self.position_dim > 100:
            import warnings
            warnings.warn(
                f"Position dimension ({self.position_dim}) is large due to one-hot encoding. "
                f"This may impact GP performance. Consider reducing categorical cardinality "
                f"or enabling sparse selection (sparse=True).",
                UserWarning
            )

        if optimizer is None:
            # dimension-aware defaults for acquisition search
            optimizer = MultiStartAcquisitionOptimizer(
                n_candidates=max(256, 64 * self.position_dim),
                n_restarts=max(5, min(20, 2 * self.position_dim)),
            )
        self.optimizer = optimizer
        # If no fitter is provided, default to a single-CPU sklearn-based fitter
        self.fitter = fitter if fitter is not None else SingleCPUFitter()
        self.optimize_hyperparameters = optimize_hyperparameters
        self.sparse = sparse
        self.max_points = (sparse_params or {}).get("max_points", 2000)
        # Always keep top-M elite points; default within 100-200 range
        self.top_m = max(0, min((sparse_params or {}).get("top_m", 150), self.max_points))
        self.rank = rank
        self.world_size = world_size
        # Initial design config/state
        self.n_initial = n_initial if n_initial is not None else max(10, 2 * self.dim)
        self.initial_design = initial_design.lower()
        self._qmc_engine = None  # type: ignore
        self._lhs_cache: Optional[np.ndarray] = None  # pre-generated LHS points
        self._lhs_index: int = 0  # next index into _lhs_cache
        self._hp_fit_calls = 0

        # Kernel for surrogate
        if kernel is None:
            # For this PR, only SingleCPUFitter is supported; default to sklearn kernel
            kernel = get_default_kernel_sklearn(self.position_dim, self._categorical_blocks)
        self.kernel = kernel

        # Store acquisition config; we'll instantiate dynamically each call to allow schedules
        self.acquisition_type = acquisition_type
        self.acquisition_params_base = dict(acquisition_params or {})
        self.rank_stretch = rank_stretch
        self.factor_min = factor_min
        self.factor_max = factor_max
        # Exploration schedule parameters (epsilon-greedy)
        self._p_explore_start = p_explore_start
        self._p_explore_end = p_explore_end
        self._p_explore_tau = p_explore_tau
        # Control whether to anneal xi/kappa automatically
        self.anneal_acquisition = anneal_acquisition
        # Acquisition switching config
        self.second_acquisition_type = (
            second_acquisition_type.upper() if second_acquisition_type else None
        )
        self.acq_switch_generation = acq_switch_generation
        self.second_acquisition_params = dict(second_acquisition_params or {})

    def _warm_start(self, inds: List[Individual]) -> Individual:
        """Generate an initial design point using Sobol, LHS, or random sampling."""
        lows, highs = self.limits_arr
        x0 = np.zeros(self.position_dim)

        # Continuous/integer dims: Sobol/LHS/random over only these dimensions
        if self._n_continuous > 0:
            if self.initial_design == "sobol":
                if self._qmc_engine is None:
                    seed = self.rng.randrange(0, 2**32 - 1)
                    self._qmc_engine = qmc.Sobol(
                        d=self._n_continuous, scramble=True,
                        seed=np.random.default_rng(seed),
                    )
                u = self._qmc_engine.random(1)[0]
            elif self.initial_design == "lhs":
                if self._lhs_cache is None:
                    seed = self.rng.randrange(0, 2**32 - 1)
                    engine = qmc.LatinHypercube(
                        d=self._n_continuous, seed=np.random.default_rng(seed),
                    )
                    self._lhs_cache = engine.random(self.n_initial)
                    self._lhs_index = 0
                if self._lhs_index < len(self._lhs_cache):
                    u = self._lhs_cache[self._lhs_index]
                    self._lhs_index += 1
                else:
                    # Fallback to random if more points requested than cached
                    u = np.array([self.rng.random() for _ in range(self._n_continuous)])
            else:  # random
                u = np.array([self.rng.random() for _ in range(self._n_continuous)])
            for i, dim_idx in enumerate(self._continuous_dims):
                x0[dim_idx] = lows[dim_idx] + u[i] * (highs[dim_idx] - lows[dim_idx])

        # Categorical dims: uniform random category selection (independent of QMC)
        for cat_offset, n_cat in self._categorical_blocks:
            chosen = self.rng.randrange(n_cat)
            x0[cat_offset:cat_offset + n_cat] = 0.0
            x0[cat_offset + chosen] = 1.0

        x0 = _project_to_discrete(x0, self.limits, self.param_types)
        x0 = np.clip(x0, lows, highs)
        gen0 = 0 if len(inds) == 0 else max(ind.generation for ind in inds) + 1
        return Individual(x0, self.limits, generation=gen0, rank=self.rank)

    def _subsample(self, inds: List[Individual]) -> Tuple[np.ndarray, np.ndarray, List[Individual]]:
        """Apply sparse selection and extract finite training arrays.

        Returns (X, y, inds) where inds may be a subsampled copy.
        """
        finite_inds = [ind for ind in inds if np.isfinite(ind.loss)]

        if self.sparse and len(finite_inds) > self.max_points:
            X_all = np.vstack([ind.position for ind in finite_inds])
            y_all = np.array([ind.loss for ind in finite_inds], dtype=float)

            lows, highs = self.limits_arr
            sel_idx = _sparse_select_indices(
                X_all,
                y_all,
                lows,
                highs,
                max_points=self.max_points,
                top_m=self.top_m,
                continuous_dims=self._continuous_dims,
                categorical_blocks=self._categorical_blocks,
            )
            finite_inds = [finite_inds[int(j)] for j in sel_idx]

        inds = finite_inds
        if len(inds) == 0:
            X_empty = np.empty((0, self.position_dim), dtype=float)
            y_empty = np.empty((0,), dtype=float)
            return X_empty, y_empty, inds

        X = np.vstack([ind.position for ind in inds])
        y = np.array([ind.loss for ind in inds], dtype=float)
        return X, y, inds

    def _fit_surrogate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        current_generation: int,
    ) -> Union[GaussianProcessRegressor, SupportsPredict]:
        """Normalize data, fit the GP surrogate, log diagnostics, and warm-start the kernel."""
        lows, highs = self.limits_arr
        scale = np.where((highs - lows) == 0.0, 1.0, (highs - lows))
        Xs = (X - lows) / scale

        # Hyperparameter optimization schedule:
        # - Once we have enough samples, optimize on the first few fits
        # - Then decimate to every k generations to save time
        n_min_samples = max(2 * self.dim, self.n_initial // 2)
        enough_samples = (X.shape[0] >= n_min_samples)
        optimize_hyperparameters_now = False
        if self.optimize_hyperparameters and enough_samples:
            if self._hp_fit_calls < 3:
                optimize_hyperparameters_now = True
            else:
                optimize_hyperparameters_now = (current_generation % 5 == 0)

        model = self.fitter.fit(kernel=self.kernel,
                                X=Xs,
                                y=y,
                                optimize_hyperparameters=optimize_hyperparameters_now)
        # Warm-start next fit (sklearn clones & uses .kernel_ as init)
        # Only sklearn GP models expose kernel_
        if isinstance(model, GaussianProcessRegressor) and hasattr(model, "kernel_"):
            self.kernel = model.kernel_
        self._hp_fit_calls += 1

        if isinstance(model, GaussianProcessRegressor):
            _log_gp_diagnostics(model, y, self.rank, current_generation, optimize_hyperparameters_now)

        return model

    def _select_next(
        self,
        model: Union[GaussianProcessRegressor, SupportsPredict],
        f_best: float,
        current_generation: int,
    ) -> np.ndarray:
        """Set up the acquisition function, optimize it, and return the next candidate in raw space."""
        lows, highs = self.limits_arr
        scale = np.where((highs - lows) == 0.0, 1.0, (highs - lows))
        t = float(current_generation + 1)

        # Decide which acquisition config to use (allow switch after threshold generation)
        use_second = (
            self.second_acquisition_type is not None
            and self.acq_switch_generation is not None
            and current_generation + 1 >= self.acq_switch_generation
        )
        if use_second:
            active_acq_type = cast(str, self.second_acquisition_type)
            params = dict(self.second_acquisition_params)
        else:
            active_acq_type = self.acquisition_type
            params = dict(self.acquisition_params_base)
        # Anneal only relevant parameter and drop the other to match constructor signature
        active_upper = active_acq_type.upper()
        if self.anneal_acquisition:
            if active_upper in ("EI", "PI"):
                if "xi" in params:
                    xi0 = float(params.get("xi", 0.01))
                    params["xi"] = max(1e-4, xi0 / np.sqrt(1.0 + 0.05 * t))
                params.pop("kappa", None)
            else:  # UCB
                if "kappa" in params:
                    k0 = float(params.get("kappa", 1.96))
                    params["kappa"] = max(0.1, k0 / np.sqrt(1.0 + 0.05 * t))
                params.pop("xi", None)
        else:
            if active_upper in ("EI", "PI"):
                params.pop("kappa", None)
            else:
                params.pop("xi", None)

        acquisition = create_acquisition(
            active_acq_type,
            rank_stretch=self.rank_stretch,
            rank=self.rank,
            size=self.world_size,
            factor_min=self.factor_min,
            factor_max=self.factor_max,
            **params,
        )

        # Acquisition wrapper — optimizer works in [0,1]^d (same space as the GP),
        # so no normalization needed inside the wrapper.
        def acq_func(x: np.ndarray) -> float:
            val = acquisition.evaluate(x, model, f_best)
            # For EI / PI we must *maximize* the acquisition, but our optimizer performs minimization.
            # Negate those values so that minimizing corresponds to maximizing the true acquisition.
            if active_acq_type.upper() in ("EI", "PI"):
                return -val
            # For (L)UCB (our implementation), minimizing mu - kappa*sigma is correct.
            return val

        # Optimize acquisition in normalized [0,1]^d space for better L-BFGS-B conditioning
        unit_bounds = np.array([np.zeros(self.position_dim), np.ones(self.position_dim)])
        x_new_unit = self.optimizer.optimize(
            acq_func,
            bounds=unit_bounds,
            rng=self.rng,
        )
        # Denormalize back to raw parameter space
        return lows + x_new_unit * scale

    def __call__(
        self,
        inds: List[Individual]
    ) -> Union[Individual, List[Individual]]:
        # Phase 1: initial design
        if len(inds) < self.n_initial:
            return self._warm_start(inds)

        # Track generation from the full incoming population (rank-local),
        # not from any sparse-selected training subset.
        current_generation = max([ind.generation for ind in inds if ind.rank == self.rank] or [0])

        # Phase 2: sparse selection + data extraction + NaN filtering
        X, y, inds = self._subsample(inds)

        # If we don't have enough valid data, fall back to random exploration
        if X.shape[0] < max(2, self.dim):
            lows, highs = self.limits_arr
            x_new = np.array([self.rng.uniform(l, h) for l, h in zip(lows, highs)], dtype=float)
            x_new = np.clip(x_new, lows, highs)
            x_new = _project_to_discrete(x_new, self.limits, self.param_types)
            return Individual(x_new, self.limits, generation=current_generation + 1, rank=self.rank)

        # Phase 3: fit GP surrogate
        model = self._fit_surrogate(X, y, current_generation)

        # Epsilon-greedy random explore with decaying probability.
        # Checked here (after GP fitting but before acquisition optimization) so that
        # exploration rounds skip the expensive multi-start L-BFGS-B while still
        # benefiting from the GP warm-start above.
        lows, highs = self.limits_arr
        t = float(current_generation + 1)
        p_explore = self._p_explore_end + (self._p_explore_start - self._p_explore_end) * np.exp(-t / self._p_explore_tau)
        if self.rng.random() < p_explore:
            x_new = np.array([self.rng.uniform(l, h) for l, h in zip(lows, highs)], dtype=float)
            x_new = np.clip(x_new, lows, highs)
            x_new = _project_to_discrete(x_new, self.limits, self.param_types)
            return Individual(x_new, self.limits, generation=current_generation + 1, rank=self.rank)

        # Phase 4: acquisition optimization + projection
        f_best = float(np.min(y))
        x_new = self._select_next(model, f_best, current_generation)

        # Clip and project continuous result to valid discrete/categorical values
        x_new = np.clip(x_new, lows, highs)
        x_new = _project_to_discrete(x_new, self.limits, self.param_types)

        return Individual(x_new,
                          self.limits,
                          generation=current_generation + 1,
                          rank=self.rank)
