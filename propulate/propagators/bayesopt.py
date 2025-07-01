from abc import ABC, abstractmethod
import random
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, RBF
from mpi4py import MPI

from ..propagators import Propagator
from ..population import Individual

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
        x = np.atleast_2d(x)
        mu, sigma = self._predict(x, model)
        ei = expected_improvement(mu, sigma, f_best, minimize=True, xi=self.xi)
        return float(ei[0])


class ProbabilityImprovement(AcquisitionFunction):
    """
    Probability of Improvement acquisition for minimization.
    """
    def __init__(self, xi: float = 0.01):
        self.xi = xi

    def evaluate(
        self,
        x: np.ndarray,
        model,
        f_best: float,
    ) -> float:
        x = np.atleast_2d(x)
        mu, sigma = model.predict(x, return_std=True)
        imp = f_best - mu - self.xi
        Z = imp / sigma
        pi = norm.cdf(Z)
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
        x = np.atleast_2d(x)
        mu, sigma = model.predict(x, return_std=True)
        ucb = mu - self.kappa * sigma
        return float(-ucb[0])


class AcquisitionType(Enum):
    EI = "EI"
    PI = "PI"
    UCB = "UCB"


def create_acquisition(
    acq_type: str,
    **params
) -> AcquisitionFunction:
    """
    Factory to create an acquisition function by name.
    """
    try:
        at = AcquisitionType(acq_type.upper())
    except ValueError:
        valid = [e.value for e in AcquisitionType]
        raise ValueError(f"Unknown acquisition type '{acq_type}'. Valid types: {valid}")

    if at == AcquisitionType.EI:
        return ExpectedImprovement(**params)
    elif at == AcquisitionType.PI:
        return ProbabilityImprovement(**params)
    elif at == AcquisitionType.UCB:
        return UpperConfidenceBound(**params)
    else:
        valid = [e.value for e in AcquisitionType]
        raise ValueError(f"Unknown acquisition type '{acq_type}'. Valid types: {valid}")


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
    ) -> Union[GaussianProcessRegressor, 'GPyTorchModel']:
        """
        Fit and return a surrogate model trained on (X, y).
        """
        ...


class SingleCPUFitter(SurrogateFitter):
    """
    Fits a GaussianProcessRegressor on a single CPU using scikit-learn.
    """
    def fit(self, kernel: Kernel, X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
        opt = "fmin_l_bfgs_b"
        model = GaussianProcessRegressor(
            kernel=kernel,
            optimizer=opt,
            normalize_y=True,
        )
        model.fit(X, y)
        return model


class MultiCPUFitter(SurrogateFitter):
    """
    Parallel hyperparameter optimization across MPI ranks.
    Each rank does a subset of random restarts, then they
    all reduce to pick the best model.
    """
    def __init__(self, comm: MPI.Comm, n_restarts_per_rank: int = 1, seed: int = 42) -> None:
        """
        comm: an mpi4py communicator (e.g. MPI.COMM_WORLD)
        n_restarts_per_rank: number of random restarts per rank
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.seed = seed + self.rank  # Ensure different seeds per rank
        self.n_restarts_per_rank = n_restarts_per_rank

    def fit(self, kernel: Kernel, X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
        best_ll = -float("inf")
        best_model = None

        # Each rank fits its subset
        
        gp = GaussianProcessRegressor(
            kernel=kernel,
            optimizer="fmin_l_bfgs_b",
            normalize_y=True,
            n_restarts_optimizer=self.n_restarts_per_rank,
            random_state=self.seed
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
    Uses GPyTorch to train a GP surrogate on a single GPU.
    """
    def __init__(self, learning_rate: float = 0.1, num_epochs: int = 50):
        """
        learning_rate: Learning rate for Adam optimizer
        num_epochs: Number of training epochs
        """
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs


    def fit(self, kernel: Kernel, X: np.ndarray, y: np.ndarray) -> 'GPyTorchModel':
        import torch
        import gpytorch
        # Construct a GP model in GPyTorch, move data and model to GPU
        # Train with Adam or LBFGS on cuda
        class GPyTorchModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood, kernel):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = kernel

            def forward(self, x):
                return gpytorch.distributions.MultivariateNormal(
                    self.mean_module(x), self.covar_module(x)
                )

        device = torch.device("cuda:0")
        train_x = torch.from_numpy(X).to(device)
        train_y = torch.from_numpy(y).to(device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = GPyTorchModel(train_x, train_y, likelihood, kernel.to(device))
        model.train(); likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for _ in range(self.num_epochs):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
        model.eval(); likelihood.eval()
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


class BayesOpt(Propagator):
    """
    Asynchronous Bayesian Optimization propagator for Propulate.

    Plug in compute-specific surrogate fitters and acquisition functions.
    """

    def __init__(
        self,
        limits: Dict[str, Tuple[float, float]],
        optimizer,
        rank: int,
        fitter: SurrogateFitter,
        kernel: Kernel = RBF(),
        optimize_hyperparameters: bool = True,
        acquisition_type: str = "EI",
        acquisition_params: Optional[Dict[str, float]] = None,
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
        self.kernel = kernel

        # Instantiate acquisition
        self.acquisition = create_acquisition(
            acquisition_type, **(acquisition_params or {})
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

        # Fit surrogate with appropriate resource
        model = self.fitter.fit(
            kernel=self.kernel,
            X=X,
            y=y
        )

        # Determine current best
        f_best = float(np.min(y))

        # Acquisition wrapper
        def acq_func(x: np.ndarray) -> float:
            return self.acquisition.evaluate(x, model, f_best)

        # Optimize acquisition
        x_new = self.optimizer.optimize(
            acq_func,
            bounds=self.limits_arr,
            rng=self.rng,
        )

        # Create new individual
        gen = max(ind.generation for ind in inds) + 1
        return Individual(x_new, self.limits, generation=gen, rank=self.rank)
