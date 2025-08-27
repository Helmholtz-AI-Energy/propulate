from abc import ABC, abstractmethod
import random
from enum import Enum
import importlib
from typing import Dict, List, Optional, Tuple, Union, Protocol

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, RBF, ConstantKernel as C, WhiteKernel
from mpi4py import MPI

from ..propagators import Propagator
from ..population import Individual


def get_default_kernel(dim: int) -> Kernel:
    kernel = (
        C(1.0, (1e-2, 1e2)) *
        RBF(length_scale=np.ones(dim), length_scale_bounds=(1e-2, 10.0))
        + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-8, 1e-2))
    )
    return kernel

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
        Z = improvement / sigma
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
        model: Union[GaussianProcessRegressor, 'GPyTorchModel'],
        f_best: float,
    ) -> float:
        """
        Evaluate acquisition at x given surrogate and current best.
        """
        ...

    def _predict(
        self,
        x: np.ndarray,
        model: Union[GaussianProcessRegressor, 'GPyTorchModel'],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and std at x using the surrogate model.
        """
        x = np.atleast_2d(x)
        mu, sigma = model.predict(x, return_std=True)
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
    """
    Upper Confidence Bound acquisition (minimization via negative UCB).
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
        return float(-ucb[0])


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


def serialize_gpr(gpr):
    return {
        "kernel":       gpr.kernel_,
        "X_train":      gpr.X_train_,
        "alpha":        gpr.alpha_,
        "L":            gpr.L_,
        "normalize_y":  gpr.normalize_y,
        # only include these if normalize_y is True
        "y_mean":       getattr(gpr, "_y_train_mean", None),
        "y_std":        getattr(gpr, "_y_train_std",  None),
    }

def deserialize_gpr(state):
    gpr = GaussianProcessRegressor(
        kernel=state["kernel"],
        optimizer=None,
        normalize_y=state["normalize_y"],
        copy_X_train=True,
    )
    gpr.kernel_ = state["kernel"]
    gpr.X_train_ = state["X_train"]
    gpr.alpha_   = state["alpha"]
    gpr.L_       = state["L"]
    if state["normalize_y"]:
        gpr._y_train_mean = state["y_mean"]
        gpr._y_train_std  = state["y_std"]
    return gpr


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
    ) -> Union[GaussianProcessRegressor, SupportsPredict]:
        """
        Fit and return a surrogate model trained on (X, y).
        """
        ...


class SingleCPUFitter(SurrogateFitter):
    def __init__(self, optimize_hyperparameters: bool = True,
                 n_restarts: int = 0, random_state: Optional[int] = None,
                 alpha: float = 1e-6):
        self.optimize_hyperparameters = optimize_hyperparameters
        self.n_restarts = n_restarts
        self.random_state = random_state
        self.alpha = alpha

    def fit(self, kernel, X, y, **kwargs):
        flag = kwargs.get("optimize_hyperparameters", self.optimize_hyperparameters)
        opt = "fmin_l_bfgs_b" if flag else None
        nre = self.n_restarts if flag else 0
        gp = GaussianProcessRegressor(
            kernel=kernel, optimizer=opt, n_restarts_optimizer=nre,
            normalize_y=True, random_state=self.random_state, alpha=self.alpha
        )
        gp.fit(X, y)
        return gp

class MultiCPUFitter(SurrogateFitter):
    def __init__(self, comm, n_restarts_per_rank: int = 1, seed: int = 42,
                 optimize_hyperparameters: bool = True, alpha: float = 1e-6):
        self.comm = comm
        self.rank = comm.Get_rank(); self.size = comm.Get_size()
        self.seed = seed + self.rank
        self.n_restarts_per_rank = n_restarts_per_rank
        self.optimize_hyperparameters = optimize_hyperparameters
        self.alpha = alpha

    def fit(self, kernel, X, y, **kwargs):
        best_ll = -float("inf")
        best_model = None
        
        flag = kwargs.get("optimize_hyperparameters", self.optimize_hyperparameters)
        opt = "fmin_l_bfgs_b" if flag else None
        nre = self.n_restarts_per_rank if flag else 0
        gp = GaussianProcessRegressor(
            kernel=kernel, optimizer=opt, n_restarts_optimizer=nre,
            normalize_y=True, random_state=self.seed, alpha=self.alpha
        )
        gp.fit(X, y)
        ll = gp.log_marginal_likelihood_value_
        if ll > best_ll:
            best_ll = ll
            best_model = gp

        # Gather (ll, model) from all ranks to rank 0
        # Serialize the model to avoid issues with MPI
        best_model = serialize_gpr(best_model)
        all_results = self.comm.gather((best_ll, best_model), root=0 )

        if self.rank == 0:
            # pick the overall best
            global_best_model = max(all_results, key=lambda t: t[0])[1]
        else:
            global_best_model = None

        # Broadcast the chosen model to all ranks
        global_best_model = self.comm.bcast(global_best_model, root=0)
        # Deserialize the model back to GaussianProcessRegressor
        global_best_model = deserialize_gpr(global_best_model)

        return global_best_model


class SingleGPUFitter(SurrogateFitter):
    """
    Uses GPyTorch to train a GP surrogate on a single GPU (or CPU fallback if allowed).
    """
    def __init__(
        self,
        learning_rate: float = 0.1,
        num_epochs: int = 50,
        device: Optional[str] = None,
        require_cuda: bool = True,
    ):
        """
        learning_rate: Learning rate for Adam optimizer.
        num_epochs: Number of training epochs.
        device: Torch device string (e.g., 'cuda:0' or 'cpu'). If None, chooses
                'cuda:0' if available, else 'cpu' (unless require_cuda=True).
        require_cuda: If True and CUDA isn't available, raise a RuntimeError.
        """
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        try:
            self._torch = importlib.import_module("torch")
            self._gpytorch = importlib.import_module("gpytorch")
        except ImportError as e:
            raise ImportError(
                "SingleGPUFitter requires optional dependencies 'torch' and 'gpytorch'. "
                "Install them with: pip install torch gpytorch"
            ) from e

        # Configure device
        torch = self._torch
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            else:
                if require_cuda:
                    raise RuntimeError(
                        "CUDA is not available but a GPU is required. "
                        "Install a CUDA-enabled PyTorch build or pass device='cpu' or require_cuda=False."
                    )
                device = "cpu"
        self.device = torch.device(device)

    def fit(self, kernel: Kernel, X: np.ndarray, y: np.ndarray) -> 'GPyTorchModel':
        torch = self._torch
        gpytorch = self._gpytorch

        # Ensure tensors are the right dtype/shape and on the configured device
        train_x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        if y.ndim > 1:
            y = y.squeeze(-1)
        train_y = torch.as_tensor(y, dtype=torch.float32, device=self.device)

        # Inner model definition
        class GPyTorchModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood, kernel_mod):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = kernel_mod

            def forward(self, x):
                return gpytorch.distributions.MultivariateNormal(
                    self.mean_module(x), self.covar_module(x)
                )

        # Move kernel if it supports .to(...)
        kernel_mod = kernel.to(self.device) if hasattr(kernel, "to") else kernel

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        model = GPyTorchModel(train_x, train_y, likelihood, kernel_mod).to(self.device)

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for _ in range(self.num_epochs):
            optimizer.zero_grad(set_to_none=True)
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        model.eval()
        likelihood.eval()

        # Make likelihood accessible to callers if they need posterior predictions
        model.likelihood = likelihood
        return model



class MultiGPUFitter(SurrogateFitter):
    """
    Distributed GP training across multiple GPUs using PyTorch Distributed or DataParallel.
    """
    def fit(self, kernel: Kernel, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.1, num_epochs: int = 50) -> 'GPyTorchModel':
        """
        Placeholder: multi-GPU surrogate fitting not implemented.
        """
        raise NotImplementedError("MultiGPUFitter is not implemented yet.")

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
        return SingleCPUFitter()
    elif ft == FitterType.MULTI_CPU:
        return MultiCPUFitter(**kwargs)
    elif ft == FitterType.SINGLE_GPU:
        return SingleGPUFitter()
    elif ft == FitterType.MULTI_GPU:
        return MultiGPUFitter()
    else:
        raise ValueError(f"Unsupported fitter type '{fitter_type}'.")


class BayesianOptimizer(Propagator):
    def __init__(
        self,
        limits: Dict[str, Tuple[float, float]],
        optimizer,
        rank: int,
        fitter: SurrogateFitter,
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
    ) -> None:
        super().__init__(parents=-1, offspring=1, rng=rng)
        self.limits = limits
        self.dim = len(limits)
        self.limits_arr = np.array(list(limits.values())).T
        self.optimizer = optimizer
        self.fitter = fitter
        self.optimize_hyperparameters = optimize_hyperparameters
        self.sparse = sparse
        self.max_points = (sparse_params or {}).get("max_points", 500)
        self.rank = rank

        # Kernel for surrogate
        if kernel is None:
            kernel = get_default_kernel(self.dim)
        self.kernel = kernel

        # Instantiate acquisition
        self.acquisition = create_acquisition(
            acquisition_type,
            rank_stretch=rank_stretch,
            rank=self.rank,
            size=getattr(self.fitter, "size", None),
            factor_min=factor_min,
            factor_max=factor_max,
            **(acquisition_params or {})
        )

    def __call__(
        self,
        inds: List[Individual]
    ) -> Union[Individual, List[Individual]]:
        # Cold start if no data
        if len(inds) == 0:
            x_rand = np.array([
                self.rng.uniform(l, u) for l, u in self.limits.values()
            ])
            return Individual(x_rand, self.limits, generation=0, rank=self.rank)

        # Sparse subsample
        if self.sparse and len(inds) > self.max_points:
            inds = self.rng.sample(inds, self.max_points)

        # Prepare training data
        X = np.vstack([ind.position for ind in inds])
        y = np.array([ind.loss for ind in inds])
        
        lows, highs = self.limits_arr
        Xs = (X - lows) / (highs - lows)  # scale to unit cube
        n, d = X.shape
        optimize_hyperparameters_now = self.optimize_hyperparameters and (n >= 5 * d) # rule of thumb
        model = self.fitter.fit(kernel=self.kernel, 
                                X=Xs,
                                y=y,
                                optimize_hyperparameters=optimize_hyperparameters_now)
        

        # Determine current best
        f_best = float(np.min(y))

        # Acquisition wrapper
        def acq_func(x: np.ndarray) -> float:
            xs = (x - lows) / (highs - lows)
            return self.acquisition.evaluate(xs, model, f_best)

        # Optimize acquisition
        x_new = self.optimizer.optimize(
            acq_func,
            bounds=self.limits_arr,
            rng=self.rng,
        )

        # Create new individual
        gen = max(ind.generation for ind in inds) + 1
        return Individual(x_new, self.limits, generation=gen, rank=self.rank)
