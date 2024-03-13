import random
from typing import List, Dict, Optional, Tuple

import numpy as np

from .base import Propagator, SelectMax, SelectMin, SelectUniform
from ..population import Individual


class CMAParameter:
    """
    Handle and store all basic/active CMA-related constants/variables and strategy parameters.

    Attributes
    ----------
    b_matrix : numpy.ndarray
        The B matrix in the covariance matrix decomposition.
    c_1 : float
        The learning rate for the rank-one update of the covariance matrix update.
    c_c : float
        The decay rate for evolution path for the rank-one update of the covariance matrix.
    c_mu : float
        The learning rate for the rank-mu update of the covariance matrix update.
    c_sigma : float
        A step-size control parameter.
    chi_n : float
        The expectation value of ||N(0,I)||.
    covariance_inv_sqrt : numpy.ndarray
        Square root of the inverse of the covariance matrix: C^-1/2 = B*D^(-1)*B^T
    covariance_matrix : numpy.ndarray
        The covariance matrix.
    condition_limit : float
        The maximum allowed condition of the covariance matrix to ensure numerical stability.
    constant_trace : bool
        Whether to keep the trace (sum of diagonal elements) of ``self.covariance_matrix`` constant.
    count_eval : int
        The number of individuals evaluated.
    d_matrix : numpy.ndarray
        The D matrix in the covariance matrix decomposition.
    d_sigma : float
        A step-size control parameter.
    eigen_eval : int
        The Number of individuals evaluated when the covariance matrix was last decomposed into B and D.
    exploration : bool
        If True decompose covariance matrix for each generation (worse runtime, less exploitation, more
        ``decompose_in_each_generation``); else decompose covariance matrix only after a certain number of
        individuals evaluated (better runtime, more exploitation, less ``decompose_in_each_generation``).
    lambd : int
        The number of individuals considered for each generation.
    limits : Dict[str, float]
        The limits of the search space.
    mean : numpy.ndarray
        The distribution's mean.
    mu : int
        The number of positive recombination weights.
    mu_eff : float
        The variance effective selection mass.
    old_mean : numpy.ndarray
        The mean of the last generation.
    p_c : numpy.ndarray
        A dynamic strategy variable.
    p_sigma : numpy.ndarray
        A dynamic strategy variable.
    problem_dimension : int
        The number of dimensions in the search space.
    sigma : float
        The distribution's standard deviation.
    weights : numpy.ndarray
        The recombination weights.

    Methods
    -------
    update_mean()
        Update mean and old mean.
    update_covariance_matrix()
        Update the covariance matrix.
    mahalanobis_norm()
        Compute the Mahalanobis distance.
    """

    def __init__(
        self,
        lambd: int,
        mu: int,
        problem_dimension: int,
        weights: np.ndarray,
        mu_eff: float,
        c_c: float,
        c_1: float,
        c_mu: float,
        limits: Dict,
        initial_mean: np.ndarray,
        exploration: bool,
    ) -> None:
        """
        Instantiate a ``CMAParameter`` object.

        Parameters
        ----------
        lambd : int
            The number of individuals considered for each generation.
        mu : int
            The number of positive recombination weights.
        problem_dimension : int
            The number of dimensions in the search space.
        weights : numpy.ndarray
            The recombination weights.
        mu_eff : float
            The variance effective selection mass.
        c_c : float
            The decay rate for the evolution path for the rank-one update of the covariance matrix.
        c_1 : float
            The learning rate for the rank-one update of the covariance matrix update.
        c_mu : float
            The learning rate for the rank-mu update of the covariance matrix update.
        limits : dict
            The limits of the search space.
        initial_mean : np.ndarray
            The initial mean of the distribution.
        exploration : bool
            If True decompose covariance matrix for each generation (worse runtime, less exploitation, more
            ``decompose_in_each_generation``); else decompose covariance matrix only after a certain number of
            individuals evaluated (better runtime, more exploitation, less ``decompose_in_each_generation``).
        """
        self.problem_dimension = problem_dimension
        self.limits = limits
        self.lambd = lambd
        self.mu = mu
        self.weights = weights
        self.mu_eff = mu_eff
        self.c_c = c_c
        self.c_1 = c_1
        self.c_mu = c_mu

        # Step-size control parameters
        self.c_sigma = (mu_eff + 2) / (problem_dimension + mu_eff + 5)
        self.d_sigma = (
            1
            + 2 * max(0, np.sqrt((mu_eff - 1) / (problem_dimension + 1)) - 1)
            + self.c_sigma
        )

        # Initialize dynamic strategy variables.
        self.p_sigma = np.zeros((problem_dimension, 1))
        self.p_c = np.zeros((problem_dimension, 1))

        # Prevent equal eigenvalues, hack from https://github.com/CMA-ES/pycma/blob/development/cma/sampler.py
        self.covariance_matrix = np.diag(
            np.ones(problem_dimension)
            * np.exp(
                (1e-4 / self.problem_dimension) * np.arange(self.problem_dimension)
            )
        )
        self.b_matrix = np.eye(self.problem_dimension)
        # Assume ``self.covariance_matrix`` to be initialized as a diagonal matrix.
        self.d_matrix = np.diag(self.covariance_matrix) ** 0.5
        # Sort eigenvalues in ascending order
        indices_eig = self.d_matrix.argsort()
        self.d_matrix = self.d_matrix[indices_eig]
        self.b_matrix = self.b_matrix[:, indices_eig]
        # Square root of the inverse of the covariance matrix: C^-1/2 = B*D^(-1)*B^T
        self.covariance_inv_sqrt = (
            self.b_matrix @ np.diag(self.d_matrix ** (-1)) @ self.b_matrix.T
        )
        # Maximum allowed condition of the covariance matrix to ensure numerical stability
        self.condition_limit = 1e5 - 1
        # Whether to keep the trace (sum of diagonal elements) of ``self.covariance_matrix`` constant.
        self.constant_trace = False

        # Use this initial mean when using multiple islands.
        self.mean = initial_mean
        # 0.3 instead of 0.2 is also often used for greater initial step size
        self.sigma = 0.2 * (
            (max(max(limits[i]) for i in limits)) - min(min(limits[i]) for i in limits)
        )

        # Mean of the last generation
        self.old_mean = None
        self.exploration = exploration

        # Number of individuals evaluated when the covariance matrix was last decomposed into B and D.
        self.eigen_eval = 0
        # Number of individuals evaluated
        self.count_eval = 0

        # Expectation value of ||N(0,I)||
        self.chi_n = problem_dimension**0.5 * (
            1 - 1.0 / (4 * problem_dimension) + 1.0 / (21 * problem_dimension**2)
        )

    def update_mean(self, new_mean: np.ndarray) -> None:
        """
        Update mean and old mean property.

        Parameters
        ----------
        new_mean : numpy.ndarray
            The new mean.
        """
        self.old_mean = self.mean
        self.mean = new_mean

    def update_covariance_matrix(self, new_covariance_matrix: np.ndarray) -> None:
        """
        Update the covariance matrix.

        Computes new values for ``b_matrix``, ``d_matrix``, and ``covariance_inv_sqrt``. Decomposition of
        ``covariance_matrix`` is O(n^3), hence the possibility of lazy updating ``b_matrix`` and ``d_matrix``.

        Parameters
        ----------
        new_covariance_matrix : numpy.ndarray
            The new covariance matrix.
        """
        # Update b and d matrix and covariance_inv_sqrt only after certain number of evaluations to ensure 0(n^2).
        # Also, trade-off decompose_in_each_generation or not.
        if self.exploration or (
            self.count_eval - self.eigen_eval
            > self.lambd / (self.c_1 + self.c_mu) / self.problem_dimension / 10
        ):
            self.eigen_eval = self.count_eval
            self._decompose_co_matrix(new_covariance_matrix)
            self.covariance_inv_sqrt = (
                self.b_matrix @ np.diag(self.d_matrix ** (-1)) @ self.b_matrix.T
            )
            # Ensure symmetry.
            self.covariance_inv_sqrt = (
                self.covariance_inv_sqrt + self.covariance_inv_sqrt.T
            ) / 2

    def _decompose_co_matrix(self, new_co_matrix: np.ndarray) -> None:
        """
        Eigen-decomposition of the covariance matrix into eigenvalues (d_matrix) and eigenvectors (columns of b_matrix).

        Parameters
        ----------
        new_co_matrix : numpy.ndarray
            The new covariance matrix that should be decomposed.
        """
        # Enforce symmetry.
        self.covariance_matrix = np.triu(new_co_matrix) + np.triu(new_co_matrix, 1).T
        d_matrix_old = self.d_matrix
        try:
            self.d_matrix, self.b_matrix = np.linalg.eigh(self.covariance_matrix)
            if np.any(self.d_matrix <= 0):
                # Covariance matrix eigen-decomposition failed, consider reformulating objective function.
                raise ValueError("Covariance matrix not positive definite.")
        except Exception as _:
            # Add min(eigenvalues(self.co_matrix_old)) to diag(self.covariance_matrix) and try again
            min_eig_old = min(d_matrix_old) ** 2
            for i in range(self.problem_dimension):
                self.covariance_matrix[i, i] += min_eig_old
            # Replace eigenvalues with standard deviations
            self.d_matrix = (d_matrix_old**2 + min_eig_old) ** 0.5
            self._decompose_co_matrix(self.covariance_matrix)
        else:
            assert all(np.isfinite(self.d_matrix))
            self._sort_b_d_matrix()
            if self.condition_limit is not None:
                self._limit_condition(self.condition_limit)
            if self.constant_trace:
                s = 1 / np.mean(
                    self.d_matrix
                )  # normalize covariance_matrix to control overall magnitude
                self.covariance_matrix *= s
                self.d_matrix *= s
            self.d_matrix **= 0.5

    def _limit_condition(self, limit: float) -> None:
        """
        Limit the condition (square of ratio largest to smallest eigenvalue) of the covariance matrix if it exceeds a
        threshold.

        Credits on how to limit the condition: https://github.com/CMA-ES/pycma/blob/development/cma/sampler.py

        Parameters
        ----------
        limit : float
            The threshold for the condition of the matrix.
        """
        # Check if condition number of matrix is too big.
        if (self.d_matrix[-1] / self.d_matrix[0]) ** 2 > limit:
            eps = (self.d_matrix[-1] ** 2 - limit * self.d_matrix[0] ** 2) / (limit - 1)
            for i in range(self.problem_dimension):
                # Decrease ratio of largest to smallest eigenvalue, absolute difference remains.
                self.covariance_matrix[i, i] += eps
            # Eigenvalues are definitely positive now.
            self.d_matrix **= 2
            self.d_matrix += eps
            self.d_matrix **= 0.5

    def _sort_b_d_matrix(self) -> None:
        """Sort columns of ``b_matrix`` and ``d_matrix`` according to the eigenvalues in ``d_matrix``."""
        indices_eig = np.argsort(self.d_matrix)
        self.d_matrix = self.d_matrix[indices_eig]
        self.b_matrix = self.b_matrix[:, indices_eig]
        assert (min(self.d_matrix), max(self.d_matrix)) == (
            self.d_matrix[0],
            self.d_matrix[-1],
        )

    def mahalanobis_norm(self, dx: np.ndarray) -> np.ndarray:
        """
        Compute the Mahalanobis distance using C^(-1/2) and the difference vector of a point to the distribution's mean.

        Parameters
        ----------
        dx : numpy.ndarray
            The difference vector.

        Returns
        -------
        numpy.ndarray
            The resulting Mahalanobis distance.
        """
        return np.linalg.norm(np.dot(self.covariance_inv_sqrt, dx))


class CMAAdapter:
    """
    Abstract base class for the adaption of strategy parameters of CMA-ES.

    Strategy class from the viewpoint of the strategy design pattern.


    Methods
    -------
    update_mean()
        Abstract method for updating of mean in CMA-ES variants.
    update_step_size()
        Update step-size in CMA-ES variants.
    update_covariance_matrix()
        Abstract method for the adaptation of the covariance matrix of CMA-ES variants.
    compute_weights()
        Abstract method for computing the recombination weights of a CMA-ES variant.
    compute_learning_rates()
        Compute the learning rates for the CMA-variants.
    """

    def update_mean(self, par: CMAParameter, arx: np.ndarray) -> None:
        """
        Abstract method for updating of mean in CMA-ES variants.

        Parameters
        ----------
        par : CMAParameter
            The parameter object of the CMA-ES propagation.
        arx : numpy.ndarray
            The individuals of the distribution.

        Raises
        ------
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError

    @staticmethod
    def update_step_size(par: CMAParameter) -> None:
        """
        Update step-size in CMA-ES variants. Calculate the current evolution path for the step-size adaption.

        Parameters
        ----------
        par : CMAParameter
            The parameter object of the CMA-ES propagation.
        """
        par.p_sigma = (1 - par.c_sigma) * par.p_sigma + np.sqrt(
            par.c_sigma * (2 - par.c_sigma) * par.mu_eff
        ) * par.covariance_inv_sqrt @ (par.mean - par.old_mean) / par.sigma

        par.sigma = par.sigma * np.exp(
            (par.c_sigma / par.d_sigma)
            * (np.linalg.norm(par.p_sigma, ord=2) / par.chi_n - 1)
        )

    def update_covariance_matrix(self, par: CMAParameter, arx: np.ndarray) -> None:
        """
        Abstract method for the adaptation of the covariance matrix of CMA-ES variants.

        Parameters
        ----------
        par : CMAParameter
            The parameter object of the CMA-ES propagation.
        arx : numpy.ndarray
            The individuals of the distribution.

        Raises
        ------
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError

    def compute_weights(
        self, mu: int, lamb: int, problem_dimension: int
    ) -> Tuple[np.ndarray, float, float, float, float]:
        """
        Abstract method for computing the recombination weights of a CMA-ES variant.

        Parameters
        ----------
        mu : int
            The number of positive recombination weights.
        lamb : int
            The number of individuals considered for each generation.
        problem_dimension : int
            The number of dimensions in the search space.

        Returns
        -------
        tuple[np.ndarray, float, float, float, float]
            tuple of the weights, mu_eff, c_1, c_c and c_mu

        Raises
        ------
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError

    @staticmethod
    def compute_learning_rates(
        mu_eff: float, problem_dimension: int
    ) -> Tuple[float, float, float]:
        """
        Compute the learning rates for the CMA-variants.

        Parameters
        ----------
        mu_eff : float
            The variance effective selection mass.
        problem_dimension : int
            The number of dimensions in the search space.

        Returns
        -------
        float
            The decay rate for evolution path for the rank-one update of the covariance matrix, ``c_c``.
        float
            The learning rate for the rank-one update of the covariance matrix update, ``c_1``.
        float
            The learning rate for the rank-mu update of the covariance matrix update, ``c_mu``.
        """
        c_c = (4 + mu_eff / problem_dimension) / (
            problem_dimension + 4 + 2 * mu_eff / problem_dimension
        )
        c_1 = 2 / ((problem_dimension + 1.3) ** 2 + mu_eff)
        c_mu = min(
            1 - c_1,
            2 * (mu_eff - 2 + (1 / mu_eff)) / ((problem_dimension + 2) ** 2 + mu_eff),
        )
        return c_c, c_1, c_mu


class BasicCMA(CMAAdapter):
    """
    Adaption of strategy parameters of CMA-ES according to the original CMA-ES algorithm.

    Concrete strategy class from the viewpoint of the strategy design pattern.

    Notes
    -----
    The ``BasicCMA`` class inherits all methods and attributes from the ``CMAAdapter`` class.

    See Also
    --------
    :class:`CMAAdapter` : The parent class.
    """

    def compute_weights(
        self, mu: int, lamb: int, problem_dimension: int
    ) -> Tuple[np.ndarray, float, float, float, float]:
        """
        Compute the recombination weights for basic CMA-ES.

        Parameters
        ----------
        mu : int
            The number of positive recombination weights
        lamb : int
            The number of individuals considered for each generation
        problem_dimension : int
            The number of dimensions in the search space

        Returns
        -------
        numpy.ndarray
            The weights.
        float
            The variance effective selection mass, ``mu_eff``.
        float
            The learning rate for the rank-one update of the covariance matrix update, ``c_1``.
        float
            The decay rate for evolution path for the rank-one update of the covariance matrix, ``c_c``.
        float
            The learning rate for the rank-mu update of the covariance matrix update, ``c_mu``.
        """
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = np.sum(weights) ** 2 / np.sum(weights**2)
        c_c, c_1, c_mu = BasicCMA.compute_learning_rates(mu_eff, problem_dimension)
        return weights, mu_eff, c_c, c_1, c_mu

    def update_mean(self, par: CMAParameter, arx: np.ndarray) -> None:
        """
        Update the mean in basic CMA-ES.

        Parameters
        ----------
        par : CMAParameter
            The parameter object of the CMA-ES propagation.
        arx : numpy.ndarray
            The individuals of the distribution.
        """
        # Matrix vector multiplication (reshape weights to column vector)
        par.update_mean(arx @ par.weights.reshape(-1, 1))

    def update_covariance_matrix(self, par: CMAParameter, arx: np.ndarray) -> None:
        """
        Adapt the covariance matrix of basic CMA-ES.

        Parameters
        ----------
        par : CMAParameter
            The parameter object of the CMA-ES propagation.
        arx : numpy.ndarray
            The individuals of the distribution.
        """
        # Turn off rank-one accumulation when sigma increases quickly.
        h_sig = np.sum(par.p_sigma**2) / (
            1 - (1 - par.c_sigma) ** (2 * (par.count_eval / par.lambd))
        ) / par.problem_dimension < 2 + 4.0 / (par.problem_dimension + 1)
        # Update evolution path.
        par.p_c = (1 - par.c_c) * par.p_c + h_sig * np.sqrt(
            par.c_c * (2 - par.c_c) * par.mu_eff
        ) * (par.mean - par.old_mean) / par.sigma

        # Use ``h_sig`` to the power of two (unlike in paper) for the variance loss from ``h_sig``.
        ar_tmp = (1 / par.sigma) * (
            arx[:, : par.mu] - np.tile(par.old_mean, (1, par.mu))
        )
        new_co_matrix = (
            (1 - par.c_1 - par.c_mu) * par.covariance_matrix
            + par.c_1
            * (
                par.p_c @ par.p_c.T
                + (1 - h_sig) * par.c_c * (2 - par.c_c) * par.covariance_matrix
            )
            + par.c_mu * ar_tmp @ (par.weights * ar_tmp).T
        )
        par.update_covariance_matrix(new_co_matrix)


class ActiveCMA(CMAAdapter):
    """
    Adaption of strategy parameters of CMA-ES according to the active CMA-ES algorithm.

    Differently from the original CMA-ES algorithm, active CMA-ES uses negative recombination weights (only for the
    covariance matrix adaptation) for individuals with relatively low fitness.
    Concrete strategy class from the viewpoint of the strategy design pattern.

    Notes
    -----
    The ``ActiveCMA`` class inherits all methods and attributes from the ``CMAAdapter`` class.

    See Also
    --------
    :class:`CMAAdapter` : The parent class.
    """

    def compute_weights(
        self, mu: int, lamb: int, problem_dimension: int
    ) -> Tuple[np.ndarray, float, float, float, float]:
        """
        Compute the recombination weights for active CMA-ES.

        Parameters
        ----------
        mu : int
            The number of positive recombination weights.
        lamb : int
            The number of individuals considered for each generation.
        problem_dimension : int
            The number of dimensions in the search space.

        Returns
        -------
        numpy.ndarray
            The weights.
        float
            The variance effective selection mass, ``mu_eff``.
        float
            The learning rate for the rank-one update of the covariance matrix update, ``c_1``.
        float
            The decay rate for evolution path for the rank-one update of the covariance matrix, ``c_c``.
        float
            The learning rate for the rank-mu update of the covariance matrix update, ``c_mu``.
        """
        weights_preliminary = np.log(lamb / 2 + 0.5) - np.log(np.arange(1, lamb + 1))
        mu_eff = np.sum(weights_preliminary[:mu]) ** 2 / np.sum(
            weights_preliminary[:mu] ** 2
        )
        c_c, c_1, c_mu = ActiveCMA.compute_learning_rates(mu_eff, problem_dimension)
        # Compute final weights.
        mu_eff_minus = np.sum(weights_preliminary[mu:]) ** 2 / np.sum(
            weights_preliminary[mu:] ** 2
        )
        alpha_mu_minus = 1 + c_1 / c_mu
        alpha_mu_eff_minus = 1 + 2 * mu_eff_minus / (mu_eff + 2)
        alpha_pos_def_minus = (1 - c_1 - c_mu) / problem_dimension * c_mu
        weights = weights_preliminary
        weights[:mu] /= np.sum(weights_preliminary[:mu])
        weights[mu:] *= (
            min(alpha_mu_minus, alpha_mu_eff_minus, alpha_pos_def_minus)
            / np.sum(weights_preliminary[mu:])
            * -1
        )
        return weights, mu_eff, c_c, c_1, c_mu

    def update_mean(self, par: CMAParameter, arx: np.ndarray) -> None:
        """
        Update the mean in active CMA-ES.

        Parameters
        ----------
        par : CMAParameter
            The parameter object of the CMA-ES propagation.
        arx : numpy.ndarray
            The individuals of the distribution.
        """
        # Matrix vector multiplication (reshape weights to column vector)
        # Only consider positive weights.
        par.update_mean(arx @ par.weights[: par.mu].reshape(-1, 1))

    def update_covariance_matrix(self, par: CMAParameter, arx: np.ndarray) -> None:
        """
        Adapt the covariance matrix of active CMA-ES.

        Parameters
        ----------
        par : CMAParameter
            The parameter object of the CMA-ES propagation.
        arx : numpy.ndarray
            The individuals of the distribution.
        """
        # Turn off rank-one accumulation when sigma increases quickly.
        h_sig = np.sum(par.p_sigma**2) / (
            1 - (1 - par.c_sigma) ** (2 * (par.count_eval / par.lambd))
        ) / par.problem_dimension < 2 + 4.0 / (par.problem_dimension + 1)
        # Update evolution path.
        par.p_c = (1 - par.c_c) * par.p_c + h_sig * np.sqrt(
            par.c_c * (2 - par.c_c) * par.mu_eff
        ) * (par.mean - par.old_mean) / par.sigma

        weights_circle = np.zeros((par.lambd,))
        for i, w_i in enumerate(par.weights):
            # Guarantee positive definiteness.
            weights_circle[i] = w_i
            if w_i < 0:
                weights_circle[i] *= (
                    par.problem_dimension
                    * (
                        par.sigma
                        / par.mahalanobis_norm(arx[:, i] - par.old_mean.ravel())
                    )
                    ** 2
                )
        # Use ``h_sig`` to the power of two (unlike in paper) for the variance loss from ``h_sig``.
        ar_tmp = (1 / par.sigma) * (arx - np.tile(par.old_mean, (1, par.lambd)))
        new_co_matrix = (
            (1 - par.c_1 - par.c_mu) * par.covariance_matrix
            + par.c_1
            * (
                par.p_c @ par.p_c.T
                + (1 - h_sig) * par.c_c * (2 - par.c_c) * par.covariance_matrix
            )
            + par.c_mu * ar_tmp @ (weights_circle * ar_tmp).T
        )
        par.update_covariance_matrix(new_co_matrix)


class CMAPropagator(Propagator):
    """
    CMA-ES propagator.

    Uses ``CMAAdapter`` to adapt strategy parameters like mean, step-size, and covariance matrix and stores them in a
    ``CMAParameter`` object. The context class from the viewpoint of the strategy design pattern.

    Attributes
    ----------
    adapter : CMAAdapter
        The adaptation strategy of CMA-ES.    par
    pool_size : int
        The size of the pool of individuals pre-selected before selecting the best from this pool.
    select_single_best : SelectMin
        Selection operator to select the best individual.
    select_from_pool : SelectUniform
        Select randomly from breeding pool.
    select_pool : SelectMin
        Select the best individuals from available ones as breeding pool.
    select_worst : SelectMax
        Select a specified number of the worst individuals.
    select_worst_all_time : bool
        If True, use the worst individuals for negative recombination weights in active CMA-ES, else use the worst
        (lambda - mu) individuals of the best lambda individuals.
    rng : random.Random
        The separate random number generator for the Propulate optimization.

    Notes
    -----
    The ``CMAPropagator`` class inherits all methods and attributes from the ``Propagator`` class.

    See Also
    --------
    :class:`Propagator` : The parent class.
    """

    def __init__(
        self,
        adapter: CMAAdapter,
        limits: Dict,
        decompose_in_each_generation: bool = False,
        select_worst_all_time: bool = False,
        pop_size: Optional[int] = None,
        pool_size: int = 3,
        rng: Optional[random.Random] = None,
    ) -> None:
        """
        Instantiate a CMA-ES propagator.

        Parameters
        ----------
        adapter : CMAAdapter
            The adaptation strategy of CMA-ES.
        limits : Dict[str, float]
            The limits of the search space.
        decompose_in_each_generation : bool, optional
            If True, decompose covariance matrix for each generation (worse runtime, less exploitation, more
            exploration); else decompose covariance matrix only after a certain number of individuals evaluated
            (better runtime, more exploitation, less exploration). Default is False.
        select_worst_all_time : bool, optional
            If True, use the worst individuals for negative recombination weights in active CMA-ES, else use the worst
            (lambda - mu) individuals of the best lambda individuals. If ``BasicCMA`` is used, the given value is
            irrelevant regarding functionality. Default is False.
        pop_size : int, optional
            The number of individuals to be considered in each generation.
        pool_size : int, optional
            The size of the pool of individuals pre-selected before selecting the best from this pool. Default is 3.
        rng: random.Random, optional
            The separate random number generator for the Propulate optimization.
        """
        self.adapter = adapter
        problem_dimension = len(limits)
        # Number of individuals considered for each generation
        lambd = (
            pop_size if pop_size else 4 + int(np.floor(3 * np.log(problem_dimension)))
        )
        super().__init__(lambd, 1, rng=rng)
        self.numpy_rng = np.random.default_rng(
            seed=self.rng.randint(a=0, b=np.iinfo(np.int32).max)
        )
        # Number of positive recombination weights
        mu = lambd // 2
        self.select_worst = SelectMax(lambd - mu)
        self.select_worst_all_time = select_worst_all_time

        # CMA-ES variant specific weights and learning rates
        weights, mu_eff, c_c, c_1, c_mu = adapter.compute_weights(
            mu, lambd, problem_dimension
        )

        initial_mean = np.array(
            [[self.numpy_rng.uniform(*limits[limit]) for limit in limits]]
        ).reshape((problem_dimension, 1))
        # 0.3 instead of 0.2 is also often used for greater initial step size

        self.par = CMAParameter(
            lambd,
            mu,
            problem_dimension,
            weights,
            mu_eff,
            c_c,
            c_1,
            c_mu,
            limits,
            initial_mean,
            decompose_in_each_generation,
        )
        self.pool_size = int(pool_size) if int(pool_size) >= 1 else 3
        self.select_pool = SelectMin(self.pool_size * lambd)
        self.select_from_pool = SelectUniform(mu - 1, rng=rng)
        self.select_single_best = SelectMin(1)

    def __call__(self, inds: List[Individual]) -> Individual:
        """
        The skeleton of the CMA-ES algorithm using the template method design pattern.

        Sampling individuals and adapting the strategy parameters. Template methods are ``update_mean``,
        ``update_covariance_matrix``, and ``update_step_size``.

        Parameters
        ----------
        inds: List[propulate.Individual]
            Available individuals.

        Returns
        -------
        new_ind : propulate.Individual
            The newly sampled individual.
        """
        num_inds = len(inds)
        # Add individuals from different workers to ``eval_count``.
        self.par.count_eval += num_inds - self.par.count_eval
        # Sample new individual.
        new_ind = self._sample_cma()
        # Check if ``len(inds)`` >= or < ``pool_size * lambda`` and make sample or sample + update.
        if num_inds >= self.pool_size * self.par.lambd:
            inds_pooled = self.select_pool(inds)
            best = self.select_single_best(inds_pooled)
            if not self.select_worst_all_time:
                worst = self.select_worst(inds_pooled)
            else:
                worst = self.select_worst(inds)

            inds_filtered = [
                ind for ind in inds_pooled if ind not in best and ind not in worst
            ]
            arx = self._transform_individuals_to_matrix(
                best + self.select_from_pool(inds_filtered) + worst
            )

            # Update mean.
            self.adapter.update_mean(self.par, arx[:, : self.par.mu])
            # Update covariance matrix.
            self.adapter.update_covariance_matrix(self.par, arx)
            # Update step size.
            self.adapter.update_step_size(self.par)
        return new_ind

    def _transform_individuals_to_matrix(self, inds: List[Individual]) -> np.ndarray:
        """
        Take a list of individuals and transform it to a numpy array for easier subsequent computation.

        Parameters
        ----------
        inds : list[propulate.Individual]
            The list of individuals.

        Returns
        -------
        arx : numpy.ndarray
            Array of shape [problem_dimension, len(inds)].
        """
        arx = np.zeros((self.par.problem_dimension, len(inds)))
        for k, ind in enumerate(inds):
            for i, (dim, _) in enumerate(self.par.limits.items()):
                arx[i, k] = ind[dim]
        return arx

    def _sample_cma(self) -> Individual:
        """
        Sample new individuals according to CMA-ES.

        Returns
        -------
        new_ind : propulate.Individual
            The newly sampled individual.
        """
        # Generate new offspring
        random_vector = self.numpy_rng.standard_normal((self.par.problem_dimension, 1))
        try:
            new_x = self.par.mean + self.par.sigma * self.par.b_matrix @ (
                self.par.d_matrix * random_vector
            )
        except (RuntimeWarning, Exception) as _:
            raise ValueError(
                "Failed to generate new offsprings, probably due to not well defined target function."
            )
        self.par.count_eval += 1
        # Remove problem_dim.
        new_ind = Individual()

        for i, (dim, _) in enumerate(self.par.limits.items()):
            new_ind[dim] = new_x[i, 0]
        return new_ind
