import copy
import random
from typing import Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

from ..population import Individual
from .base import Stochastic


class PointMutation(Stochastic):
    """
    Point-mutate given number of traits with given probability.

    Attributes
    ----------
    points : int
        The number of points to mutate.
    limits : Dict[str, Tuple[float, float]] | Dict[str, Tuple[int, int]] | Dict[str, Tuple[str, ...]]
        The search space, i.e., the limits of (hyper-)parameters to be optimized.

    Notes
    -----
    The ``PointMutation`` class inherits all methods and attributes from the ``Stochastic`` class.

    See Also
    --------
    :class:`Stochastic` : The parent class.
    """

    def __init__(
        self,
        limits: Mapping[str, Union[Tuple[float, float], Tuple[int, int], Tuple[str, ...]]],
        points: int = 1,
        probability: float = 1.0,
        rng: Optional[random.Random] = None,
    ) -> None:
        """
        Initialize point-mutation propagator.

        Parameters
        ----------
        limits : Dict[str, Tuple[float, float]] | Dict[str, Tuple[int, int]] | Dict[str, Tuple[str, ...]]
            The search space, i.e., the limits of the (hyper-)parameters to be optimized.
        points : int, optional
            The number of points to mutate. Default is 1.
        probability : float, optional
            The probability of application. Default is 1.0.
        rng : random.Random, optional
            The separate random number generator of the Propulate optimization.

        Raises
        ------
        ValueError
            If the requested number of points to mutate is greater than the number of traits.
        """
        super().__init__(1, 1, probability, rng)
        self.points = points
        self.limits = limits
        if len(limits) < points:
            raise ValueError(f"Too many points to mutate for individual with {len(limits)} traits.")

    def __call__(self, ind: Individual) -> Individual:  # type: ignore[override]
        """
        Apply the point-mutation propagator.

        Parameters
        ----------
        ind : propulate.population.Individual
            The individual the propagator is applied to.

        Returns
        -------
        propulate.population.Individual
            The possibly point-mutated individual after application of the propagator.
        """
        if self.rng.random() < self.probability:  # Apply propagator only with specified probability
            ind = copy.deepcopy(ind)
            ind.loss = float("inf")  # Initialize individual's loss attribute.
            # Determine traits to mutate via random sampling.
            # Return `self.points` length list of unique elements chosen from `ind.keys()`.
            # Used for random sampling without replacement.
            to_mutate = self.rng.sample(sorted(ind.keys()), self.points)
            # Point-mutate `self.points` randomly chosen traits of individual `ind`.
            for i in to_mutate:
                if isinstance(ind[i], int):
                    # Return randomly selected element from int range(start, stop, step).
                    ind[i] = self.rng.randint(*self.limits[i])
                elif isinstance(ind[i], float):
                    # Return random floating point number within limits.
                    ind[i] = self.rng.uniform(*self.limits[i])
                elif isinstance(ind[i], str):
                    # Return random element from non-empty sequence.
                    ind[i] = self.rng.choice(self.limits[i])
        return ind  # Return point-mutated individual.


class RandomPointMutation(Stochastic):
    """
    Point-mutate random number of traits with given probability.

    Attributes
    ----------
    min_points : int
        The minimum number of points to mutate.
    max_points : int
        The maximum number of points to mutate.
    limits : Dict[str, Tuple[float, float]] | Dict[str, Tuple[int, int]] | Dict[str, Tuple[str, ...]]
        The search space, i.e., the limits of (hyper-)parameters to be optimized.

    Notes
    -----
    The ``RandomPointMutation`` class inherits all methods and attributes from the ``Stochastic`` class.

    See Also
    --------
    :class:`Stochastic` : The parent class.
    """

    def __init__(
        self,
        limits: Union[
            Dict[str, Tuple[float, float]],
            Dict[str, Tuple[int, int]],
            Dict[str, Tuple[str, ...]],
        ],
        min_points: int = 1,
        max_points: int = 1,
        probability: float = 1.0,
        rng: Optional[random.Random] = None,
    ) -> None:
        """
        Initialize random point-mutation propagator.

        Parameters
        ----------
        limits : Dict[str, Tuple[float, float]] | Dict[str, Tuple[int, int]] | Dict[str, Tuple[str, ...]]
            The limits of the parameters to optimize, i.e., the search space.
        min_points : int, optional
            minimum number of points to mutate. Default is 1.
        max_points : int, optional
            maximum number of points to mutate. Default is 1.
        probability : float, optional
            probability of application. Default is 1.0.
        rng : random.Random, optional
            random number generator

        Raises
        ------
        ValueError
            If no or a negative number of points shall be mutated.
            If there are fewer traits than requested number of points to mutate.
            If the requested minimum number of points to mutate is greater than the requested maximum number.
        """
        super().__init__(1, 1, probability, rng)
        if min_points <= 0:
            raise ValueError(f"Minimum number of points to mutate must be > 0 but was {min_points}.")
        if len(limits) < max_points:
            raise ValueError(f"Too many points to mutate for individual with {len(limits)} traits.")
        if min_points > max_points:
            raise ValueError(
                f"Minimum number of traits to mutate must be <= respective maximum number "
                f"but min_points = {min_points} > {max_points} = max_points."
            )
        self.min_points = min_points
        self.max_points = max_points
        self.limits = limits

    def __call__(self, ind: Individual) -> Individual:  # type: ignore[override]
        """
        Apply the random-point-mutation propagator.

        Parameters
        ----------
        ind : propulate.population.Individual
            The individual the propagator is applied to.

        Returns
        -------
        propulate.population.Individual
            The possibly point-mutated individual after application of the propagator.
        """
        if self.rng.random() < self.probability:  # Apply propagator only with specified probability.
            ind = copy.deepcopy(ind)
            ind.loss = float("inf")  # Initialize individual's loss attribute.
            # Determine traits to mutate via random sampling.
            # Return `self.points` length list of unique elements chosen from `ind.keys()`.
            # Used for random sampling without replacement.
            points = self.rng.randint(self.min_points, self.max_points)
            to_mutate = self.rng.sample(sorted(ind.keys()), points)
            # Point-mutate `points` randomly chosen traits of individual `ind`.
            for i in to_mutate:
                if isinstance(ind[i], int):
                    # Return randomly selected element from int range(start, stop, step).
                    ind[i] = self.rng.randint(*self.limits[i])
                elif isinstance(ind[i], float):
                    # Return random floating point number N within limits.
                    ind[i] = self.rng.uniform(*self.limits[i])
                elif isinstance(ind[i], str):
                    # Return random element from non-empty sequence.
                    ind[i] = self.rng.choice(self.limits[i])

        return ind  # Return point-mutated individual.


class IntervalMutationNormal(Stochastic):
    """
    Mutate a number of traits according to a Gaussian distribution around the current value with the given probability.

    Attributes
    ----------
    sigma_factor : float
        The scaling factor for the interval width to obtain standard deviation.
    points : int
        The number of points to mutate.
    limits : Dict[str, Tuple[float, float]] | Dict[str, Tuple[int, int]] | Dict[str, Tuple[str, ...]]
        The search space, i.e., the limits of (hyper-)parameters to be optimized.

    Notes
    -----
    The ``IntervalMutationNormal`` class inherits all methods and attributes from the ``Stochastic`` class.

    See Also
    --------
    :class:`Stochastic` : The parent class.
    """

    def __init__(
        self,
        limits: Mapping[str, Union[Tuple[float, float], Tuple[int, int], Tuple[str, ...]]],
        sigma_factor: float = 0.1,
        points: int = 1,
        probability: float = 1.0,
        rng: Optional[random.Random] = None,
    ) -> None:
        """
        Initialize interval-mutation propagator.

        Parameters
        ----------
        limits : Dict[str, Tuple[float, float]] | Dict[str, Tuple[int, int]] | Dict[str, Tuple[str, ...]]
            The limits of the (hyper-)parameters to be optimized, i.e., the search space.
        sigma_factor : float, optional
            The scaling factor for the interval width to obtain the standard deviation. Default is 0.1.
        points : int, optional
            The number of points to mutate. Default is 1.
        probability : float, optional
            The probability of application, Default is 1.0
        rng : random.Random, optional
            The separate random number generator for the Propulate optimization.

        Raises
        ------
        ValueError
            If the individuals has fewer continuous traits than the requested number of points to mutate.
        """
        super().__init__(1, 1, probability, rng)
        self.points = points  # Number of traits to point-mutate
        self.limits = limits
        self.sigma_factor = sigma_factor
        n_interval_traits = len([x for x in limits if isinstance(limits[x][0], float)])
        if n_interval_traits < points:
            raise ValueError(f"Too many points to mutate ({points}) for individual with {n_interval_traits} continuous traits.")

    def __call__(self, ind: Individual) -> Individual:  # type: ignore[override]
        """
        Apply the interval-mutation propagator.

        Parameters
        ----------
        ind : propulate.population.Individual
            The input individual the propagator is applied to.

        Returns
        -------
        propulate.population.Individual
            The possibly interval-mutated output individual after application of the propagator.
        """
        if self.rng.random() < self.probability:  # Apply propagator only with specified probability.
            ind = copy.deepcopy(ind)
            ind.loss = float("inf")  # Initialize individual's loss attribute.
            # Determine traits of type float.
            interval_keys: List[str] = [x for x in ind.keys() if isinstance(ind[x], float)]
            # Determine ´self.points` traits to mutate.
            to_mutate: List[str] = self.rng.sample(interval_keys, self.points)
            # Mutate traits by sampling from Gaussian distribution centered around current value
            # with `sigma_factor` scaled interval width as standard distribution.
            for key in to_mutate:
                min_val, max_val = self.limits[key]  # Determine interval boundaries.
                sigma = (
                    float(max_val) - float(min_val)
                ) * self.sigma_factor  # Determine std from interval boundaries and sigma factor.
                ind[key] = self.rng.gauss(float(ind[key]), sigma)  # Sample new value from Gaussian centered around current value.
                ind[key] = min(max_val, ind[key])  # Make sure new value is within specified limits.
                ind[key] = max(min_val, ind[key])

        return ind  # Return point-mutated individual.


class CrossoverUniform(Stochastic):  # uniform crossover
    """
    Generate new individual by uniform crossover of two parents with specified relative parent contribution.

    Attributes
    ----------
    rel_parent_contrib : float
        The relative parent contribution with respect to the first parent.

    Notes
    -----
    The ``CrossoverUniform`` class inherits all methods and attributes from the ``Stochastic`` class.

    See Also
    --------
    :class:`Stochastic` : The parent class.
    """

    def __init__(
        self,
        relative_parent_contribution: float = 0.5,
        probability: float = 1.0,
        rng: Optional[random.Random] = None,
    ) -> None:
        """
        Initialize uniform crossover propagator.

        Parameters
        ----------
        relative_parent_contribution : float, optional
            The relative parent contribution with respect to the first parent. Default is 0.5.
        probability: float, optional
            The probability of application. Default is 1.0.
        rng: random.Random, optional
            The separate random number generator for the Propulate optimization.

        Raises
        ------
        ValueError
            If the relative parent contribution is not within [0, 1].
        """
        super().__init__(2, 1, probability, rng)  # Breed 1 offspring from 2 parents.
        if relative_parent_contribution <= 0 or relative_parent_contribution >= 1:
            raise ValueError(f"Relative parent contribution must be within (0, 1) but was {relative_parent_contribution}.")
        self.rel_parent_contrib = relative_parent_contribution

    def __call__(self, inds: List[Individual]) -> Individual:
        """
        Apply the uniform-crossover propagator.

        Parameters
        ----------
        inds : List[propulate.population.Individual]
            The individuals the propagator is applied to.

        Returns
        -------
        propulate.population.Individual
            The possibly cross-bred individual after application of the propagator.
        """
        ind = copy.deepcopy(inds[0])  # Consider 1st parent.
        ind.loss = float("inf")  # Initialize individual's loss attribute.
        if self.rng.random() < self.probability:  # Apply propagator only with specified `probability`.
            # Replace traits in first parent with values of second parent with specified relative parent contribution.
            for k in ind.keys():
                if self.rng.random() > self.rel_parent_contrib:
                    ind[k] = inds[1][k]
        return ind  # Return offspring.


class CrossoverMultiple(Stochastic):  # uniform crossover
    """
    Breed new individual by uniform crossover of multiple parents.

    Notes
    -----
    The ``CrossoverMultiple`` class inherits all methods and attributes from the ``Stochastic`` class.

    See Also
    --------
    :class:`Stochastic` : The parent class.
    """

    def __init__(
        self,
        parents: int = -1,
        probability: float = 1.0,
        rng: Optional[random.Random] = None,
    ) -> None:
        """
        Initialize a multiple-crossover propagator.

        Parameters
        ----------
        probability : float, optional
            The probability of application. Default is 1.0.
        parents: int, optional
            The number of parents (not used) here. Default is -1.
        rng : random.Random, optional
            The separate random number generator for the Propulate optimization.
        """
        super().__init__(parents, 1, probability, rng)  # Breed 1 offspring from specified number of parents.

    def __call__(self, inds: List[Individual]) -> Individual:
        """
        Apply the multi-crossover propagator.

        Parameters
        ----------
        inds : list[propulate.population.Individual]
            The individuals the propagator is applied to.

        Returns
        -------
        propulate.population.Individual
            The possibly cross-bred individual after application of propagator
        """
        ind = copy.deepcopy(inds[0])  # Consider 1st parent.
        ind.loss = float("inf")  # Initialize individual's loss attribute.
        if self.rng.random() < self.probability:  # Apply propagator only with specified `probability`.
            # Choose traits from all parents with uniform probability.
            for k in ind.keys():
                ind[k] = self.rng.choice([parent[k] for parent in inds])
        return ind  # Return offspring.


class CrossoverSigmoid(Stochastic):
    """
    Generate new individual by crossover of two parents according to Boltzmann sigmoid probability.

    Consider two parent individuals with fitness values f1 and f2. Let f1 <= f2. For each trait,
    the better parent's value is accepted with the probability sigmoid(- (f1-f2) / temperature).

    Attributes
    ----------
    temperature : float
        The temperature hyperparameter of the Boltzmann distribution.

    Notes
    -----
    The ``CrossoverSigmoid`` class inherits all methods and attributes from the ``Stochastic`` class.

    See Also
    --------
    :class:`Stochastic` : The parent class.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        probability: float = 1.0,
        rng: Optional[random.Random] = None,
    ) -> None:
        """
        Initialize a sigmoid-crossover propagator.

        Parameters
        ----------
        temperature : float, optional
            The temperature in the Boltzmann factor of the sigmoid probability. Default is 1.0.
        probability : float, optional
            The probability of application. Default is 1.0.
        rng : random.Random, optional
            The separate random number generator for the Propulate optimization.
        """
        super(CrossoverSigmoid, self).__init__(2, 1, probability, rng)  # Breed 1 offspring from 2 parents.
        self.temperature = temperature

    def __call__(self, inds: List[Individual]) -> Individual:
        """
        Apply the sigmoid-crossover propagator.

        Parameters
        ----------
        inds : List[propulate.population.Individual]
            The individuals the propagator is applied to.

        Returns
        -------
        propulate.population.Individual
            The possibly cross-bred individual after application of the propagator.
        """
        ind = copy.deepcopy(inds[0])  # Consider 1st parent.
        ind.loss = float("inf")  # Initialize individual's loss attribute.
        if inds[0].loss <= inds[1].loss:
            delta = inds[0].loss - inds[1].loss
            fraction = 1 / (1 + np.exp(-delta / self.temperature))
        else:
            delta = inds[1].loss - inds[0].loss
            fraction = 1 - 1 / (1 + np.exp(-delta / self.temperature))

        if self.rng.random() < self.probability:  # Apply propagator only with specified `probability`.
            # Replace traits in 1st parent with values of 2nd parent with Boltzmann probability.
            for k in inds[1].keys():
                if self.rng.random() > fraction:
                    ind[k] = inds[1][k]
        return ind  # Return offspring.
