import copy
import random
from typing import List, Dict, Union, Tuple

import numpy as np
from abc import ABC, abstractmethod

from ..population import Individual


def _check_compatible(out1: int, in2: int) -> bool:
    """
    Check compatibility of two propagators for stacking them together sequentially with `Compose`.

    Parameters
    ----------
    out1: int
          number of output individuals returned by first propagator
    in2: int
         number of input individuals taken by second propagator

    Returns
    -------
    bool
        True if propagators can be stacked, False if not.
    """
    return out1 == in2 or in2 == -1


class Propagator:
    """
    Abstract base class for all propagators, i.e., evolutionary operators.

    A propagator takes a collection of individuals and uses them to breed a new collection of individuals.
    """

    def __init__(
        self, parents: int = 0, offspring: int = 0, rng: random.Random = None
    ) -> None:
        """
        Initialize a propagator with given parameters.

        Parameters
        ----------
        parents: int
                 number of input individuals (-1 for any)
        offspring: int
                   number of output individuals
        rng: random.Random
             random number generator

        Raises
        ------
        ValueError
            If the number of offspring to breed is zero.
        """
        if offspring == 0:
            raise ValueError("Propagator has to sire more than 0 offspring.")
        self.offspring = offspring  # Number of offspring individuals to breed
        self.parents = parents  # Number of parent individuals
        self.rng = rng  # Random number generator

    def __call__(self, inds: List[Individual]) -> Union[List[Individual], Individual]:
        """
        Apply the propagator (not implemented for abstract base class).

        Parameters
        ----------
        inds: list[propulate.individual.Individual]
              input individuals the propagator is applied to

        Returns
        -------
        list[Individual] | Individual
            individual(s) bred by applying the propagator
            While this abstract base class method actually returns ``None``, each concrete child class
            of ``Propagator`` should return an individual or a list of individuals.

        Raises
        ------
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError()


class Stochastic(Propagator):
    """
    Apply a propagator with a given probability.

    If the propagator is not applied, the output still has to adhere to the defined number of offspring.
    """

    def __init__(
        self,
        parents: int = 0,
        offspring: int = 0,
        probability: float = 1.0,
        rng: random.Random = None,
    ) -> None:
        """
        Initialize a stochastic propagator that is only applied with a specified probability.

        Parameters
        ----------
        parents: int
                 number of input individuals (-1 for any)
        offspring: int
                   number of output individuals
        probability: float
                     probability of application
        rng: random.Random
             random number generator

        Raises
        ------
        ValueError
            If the number of offspring to breed is zero.
        """
        super(Stochastic, self).__init__(parents, offspring, rng)
        self.probability = probability
        if offspring == 0:
            raise ValueError("Propagator has to sire more than 0 offspring.")


class Conditional(Propagator):
    """
    Apply different propagators depending on whether the breeding population is already large enough or not.

    If the population consists of the specified number of individuals required for breeding (or more),
    a different propagator is applied than if not.
    """

    def __init__(
        self,
        pop_size: int,
        true_prop: Propagator,
        false_prop: Propagator,
        parents: int = -1,
        offspring: int = -1,
    ) -> None:
        """
        Initialize the conditional propagator.

        Parameters
        ----------
        pop_size: int
                  breeding population size
        true_prop: propulate.propagators.Propagator
                   propagator applied if size of current population >= pop_size.
        false_prop: propulate.propagators.Propagator
                    propagator applied if size of current population < pop_size.
        parents: int
                 number of input individuals (-1 for any)
        offspring: int
                   number of output individuals
        """
        super(Conditional, self).__init__(parents, offspring)
        self.pop_size = pop_size
        self.true_prop = true_prop
        self.false_prop = false_prop

    def __call__(self, inds: List[Individual]) -> List[Individual]:
        """
        Apply conditional propagator.

        Parameters
        ----------
        inds: list[propulate.individual.Individual]
              input individuals the propagator is applied to

        Returns
        -------
        list[propulate.individual.Individual]
            output individuals returned by the conditional propagator
        """
        if (
            len(inds) >= self.pop_size
        ):  # If number of evaluated individuals >= pop_size apply true_prop.
            return self.true_prop(inds)
        else:  # Else apply false_prop.
            return self.false_prop(inds)


class Compose(Propagator):
    """
    Stack propagators together sequentially for successive application.
    """

    def __init__(self, propagators: List[Propagator]) -> None:
        """
        Initialize composed propagator.

        Parameters
        ----------
        propagators: list[propulate.propagators.Propagator]
                     propagators to be stacked together sequentially

        Raises
        ------
        ValueError
            If propagators to stack are incompatible in terms of number of input and output individuals.
        """
        if len(propagators) < 1:
            raise ValueError(
                f"Not enough Propagators given ({len(propagators)}). At least 1 is required."
            )
        super(Compose, self).__init__(propagators[0].parents, propagators[-1].offspring)
        for i in range(len(propagators) - 1):
            # Check compatibility of consecutive propagators in terms of number of parents + offsprings.
            if not _check_compatible(
                propagators[i].offspring, propagators[i + 1].parents
            ):
                outp = propagators[i]
                inp = propagators[i + 1]
                outd = outp.offspring
                ind = inp.parents

                raise ValueError(
                    f"Incompatible combination of {outd} output individuals "
                    f"of {outp} and {ind} input individuals of {inp}."
                )
        self.propagators = propagators

    def __call__(self, inds: List[Individual]) -> List[Individual]:
        """
        Apply composed propagator.

        Parameters
        ----------
        inds: list[propulate.individual.Individual]
              input individuals the propagator is applied to

        Returns
        -------
        list[propulate.individual.Individual]
            output individuals after application of propagator
        """
        for p in self.propagators:
            inds = p(inds)
        return inds


class PointMutation(Stochastic):
    """
    Point-mutate given number of traits with given probability.
    """

    def __init__(
        self,
        limits: Union[
            Dict[str, Tuple[float, float]],
            Dict[str, Tuple[int, int]],
            Dict[str, Tuple[str, ...]],
        ],
        points: int = 1,
        probability: float = 1.0,
        rng: random.Random = None,
    ) -> None:
        """
        Initialize point-mutation propagator.

        Parameters
        ----------
        limits: dict[str, tuple[float, float]] | dict[str, tuple[int, int]] | dict[str, tuple[str, ...]]
                 limits of (hyper-)parameters to be optimized
        points: int
                number of points to mutate
        probability: float
                     probability of application
        rng: random.Random
             random number generator

        Raises
        ------
        ValueError
            If the requested number of points to mutate is greater than the number of traits.
        """
        super(PointMutation, self).__init__(1, 1, probability, rng)
        self.points = points
        self.limits = limits
        if len(limits) < points:
            raise ValueError(
                f"Too many points to mutate for individual with {len(limits)} traits."
            )

    def __call__(self, ind: Individual) -> Individual:
        """
        Apply point-mutation propagator.

        Parameters
        ----------
        ind: propulate.individual.Individual
             individual the propagator is applied to

        Returns
        -------
        propulate.individual.Individual
            possibly point-mutated individual after application of propagator
        """
        if (
            self.rng.random() < self.probability
        ):  # Apply propagator only with specified probability
            ind = copy.deepcopy(ind)
            ind.loss = None  # Initialize individual's loss attribute.
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
        rng: random.Random = None,
    ) -> None:
        """
        Initialize random point-mutation propagator.

        Parameters
        ----------
        limits: dict[str, tuple[float, float]] | dict[str, tuple[int, int]] | dict[str, tuple[str, ...]]
                limits of parameters to optimize, i.e., search space
        min_points: int
                    minimum number of points to mutate
        max_points: int
                    maximum number of points to mutate
        probability: float
                     probability of application
        rng: random.Random
             random number generator

        Raises
        ------
        ValueError
            If no or a negative number of points shall be mutated.
        ValueError
            If there are fewer traits than requested number of points to mutate.
        ValueError
            If the requested minimum number of points to mutate is greater than the requested maximum number.
        """
        super(RandomPointMutation, self).__init__(1, 1, probability, rng)
        if min_points <= 0:
            raise ValueError(
                f"Minimum number of points to mutate must be > 0 but was {min_points}."
            )
        if len(limits) < max_points:
            raise ValueError(
                f"Too many points to mutate for individual with {len(limits)} traits."
            )
        if min_points > max_points:
            raise ValueError(
                f"Minimum number of traits to mutate must be <= respective maximum number "
                f"but min_points = {min_points} > {max_points} = max_points."
            )
        self.min_points = min_points
        self.max_points = max_points
        self.limits = limits

    def __call__(self, ind: Individual) -> Individual:
        """
        Apply random-point-mutation propagator.

        Parameters
        ----------
        ind: propulate.population.Individual
             individual the propagator is applied to

        Returns
        -------
        propulate.population.Individual
            possibly point-mutated individual after application of propagator
        """
        if (
            self.rng.random() < self.probability
        ):  # Apply propagator only with specified probability.
            ind = copy.deepcopy(ind)
            ind.loss = None  # Initialize individual's loss attribute.
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
    Mutate given number of traits according to Gaussian distribution around current value with given probability.
    """

    def __init__(
        self,
        limits: Union[
            Dict[str, Tuple[float, float]],
            Dict[str, Tuple[int, int]],
            Dict[str, Tuple[str, ...]],
        ],
        sigma_factor: float = 0.1,
        points: int = 1,
        probability: float = 1.0,
        rng: random.Random = None,
    ) -> None:
        """
        Initialize interval-mutation propagator.

        Parameters
        ----------
        limits: dict[str, tuple[float, float]] | dict[str, tuple[int, int]] | dict[str, tuple[str, ...]]
                limits of (hyper-)parameters to be optimized, i.e., search space
        sigma_factor: float
                      scaling factor for interval width to obtain standard deviation
        points: int
                number of points to mutate
        probability: float
                     probability of application
        rng: random.Random
             random number generator

        Raises
        ------
        ValueError
            If the individuals has fewer continuous traits than the requested number of points to mutate.
        """
        super(IntervalMutationNormal, self).__init__(1, 1, probability, rng)
        self.points = points  # number of traits to point-mutate
        self.limits = limits
        self.sigma_factor = sigma_factor
        n_interval_traits = len([x for x in limits if isinstance(limits[x][0], float)])
        if n_interval_traits < points:
            raise ValueError(
                f"Too many points to mutate ({points}) for individual with {n_interval_traits} continuous traits."
            )

    def __call__(self, ind: Individual) -> Individual:
        """
        Apply interval-mutation propagator.

        Parameters
        ----------
        ind: propulate.individual.Individual
             input individual the propagator is applied to

        Returns
        -------
        propulate.individual.Individual
            possibly interval-mutated output individual after application of propagator
        """
        if (
            self.rng.random() < self.probability
        ):  # Apply propagator only with specified probability.
            ind = copy.deepcopy(ind)
            ind.loss = None  # Initialize individual's loss attribute.
            # Determine traits of type float.
            interval_keys = [x for x in ind.keys() if isinstance(ind[x], float)]
            # Determine ´self.points` traits to mutate.
            to_mutate = self.rng.sample(interval_keys, self.points)
            # Mutate traits by sampling from Gaussian distribution centered around current value
            # with `sigma_factor` scaled interval width as standard distribution.
            for i in to_mutate:
                min_val, max_val = self.limits[i]  # Determine interval boundaries.
                sigma = (
                    max_val - min_val
                ) * self.sigma_factor  # Determine std from interval boundaries and sigma factor.
                ind[i] = self.rng.gauss(
                    ind[i], sigma
                )  # Sample new value from Gaussian centered around current value.
                ind[i] = min(
                    max_val, ind[i]
                )  # Make sure new value is within specified limits.
                ind[i] = max(min_val, ind[i])

        return ind  # Return point-mutated individual.


class MateUniform(Stochastic):  # uniform crossover
    """
    Generate new individual by uniform crossover of two parents with specified relative parent contribution.
    """

    def __init__(
        self,
        rel_parent_contrib: float = 0.5,
        probability: float = 1.0,
        rng: random.Random = None,
    ) -> None:
        """
        Initialize uniform crossover propagator.

        Parameters
        ----------
        rel_parent_contrib: float
                            relative parent contribution w.r.t. first parent
        probability: float
                     probability of application
        rng: random.Random
             random number generator

        Raises
        ------
        ValueError
            If the relative parent contribution is not within [0, 1].
        """
        super(MateUniform, self).__init__(
            2, 1, probability, rng
        )  # Breed 1 offspring from 2 parents.
        if rel_parent_contrib <= 0 or rel_parent_contrib >= 1:
            raise ValueError(
                f"Relative parent contribution must be within (0, 1) but was {rel_parent_contrib}."
            )
        self.rel_parent_contrib = rel_parent_contrib

    def __call__(self, inds: List[Individual]) -> Individual:
        """
        Apply uniform-crossover propagator.

        Parameters
        ----------
        inds: List[propulate.individual.Individual]
              individuals the propagator is applied to

        Returns
        -------
        propulate.individual.Individual
            possibly cross-bred individual after application of propagator
        """
        ind = copy.deepcopy(inds[0])  # Consider 1st parent.
        ind.loss = None  # Initialize individual's loss attribute.
        if (
            self.rng.random() < self.probability
        ):  # Apply propagator only with specified `probability`.
            # Replace traits in first parent with values of second parent with specified relative parent contribution.
            for k in ind.keys():
                if self.rng.random() > self.rel_parent_contrib:
                    ind[k] = inds[1][k]
        return ind  # Return offspring.


class MateMultiple(Stochastic):  # uniform crossover
    """
    Breed new individual by uniform crossover of multiple parents.
    """

    def __init__(
        self, parents: int = -1, probability: float = 1.0, rng: random.Random = None
    ) -> None:
        """
        Initialize multiple-crossover propagator.

        Parameters
        ----------
        probability: float
                     probability of application
        parents: int
                 number of parents
        rng: random.Random
             random number generator
        """
        super(MateMultiple, self).__init__(
            parents, 1, probability, rng
        )  # Breed 1 offspring from specified number of parents.

    def __call__(self, inds: List[Individual]) -> Individual:
        """
        Apply multiple-crossover propagator.

        Parameters
        ----------
        inds: list[propulate.individual.Individual]
              individuals the propagator is applied to

        Returns
        -------
        propulate.individual.Individual
            possibly cross-bred individual after application of propagator
        """
        ind = copy.deepcopy(inds[0])  # Consider 1st parent.
        ind.loss = None  # Initialize individual's loss attribute.
        if (
            self.rng.random() < self.probability
        ):  # Apply propagator only with specified `probability`.
            # Choose traits from all parents with uniform probability.
            for k in ind.keys():
                ind[k] = self.rng.choice([parent[k] for parent in inds])
        return ind  # Return offspring.


class MateSigmoid(Stochastic):
    """
    Generate new individual by crossover of two parents according to Boltzmann sigmoid probability.

    Consider two parent individuals with fitness values f1 and f2. Let f1 <= f2. For each trait,
    the better parent's value is accepted with the probability sigmoid(- (f1-f2) / temperature).
    """

    def __init__(
        self,
        temperature: float = 1.0,
        probability: float = 1.0,
        rng: random.Random = None,
    ) -> None:
        """
        Initialize sigmoid-crossover propagator.

        Parameters
        ----------
        temperature: float
                     temperature for Boltzmann factor in sigmoid probability
        probability: float
                     probability of application
        rng: random.Random
             random number generator
        """
        super(MateSigmoid, self).__init__(
            2, 1, probability, rng
        )  # Breed 1 offspring from 2 parents.
        self.temperature = temperature

    def __call__(self, inds: List[Individual]) -> Individual:
        """
        Apply sigmoid-crossover propagator.

        Parameters
        ----------
        inds: list[propulate.individual.Individual]
              individuals the propagator is applied to

        Returns
        -------
        propulate.individual.Individual
            possibly cross-bred individual after application of propagator
        """
        ind = copy.deepcopy(inds[0])  # Consider 1st parent.
        ind.loss = None  # Initialize individual's loss attribute.
        if inds[0].loss <= inds[1].loss:
            delta = inds[0].loss - inds[1].loss
            fraction = 1 / (1 + np.exp(-delta / self.temperature))
        else:
            delta = inds[1].loss - inds[0].loss
            fraction = 1 - 1 / (1 + np.exp(-delta / self.temperature))

        if (
            self.rng.random() < self.probability
        ):  # Apply propagator only with specified `probability`.
            # Replace traits in 1st parent with values of 2nd parent with Boltzmann probability.
            for k in inds[1].keys():
                if self.rng.random() > fraction:
                    ind[k] = inds[1][k]
        return ind  # Return offspring.


class SelectMin(Propagator):
    """
    Select specified number of best performing individuals as evaluated by their losses.
    i.e., those individuals with minimum losses.
    """

    def __init__(
        self,
        offspring: int,
    ) -> None:
        """
        Initialize elitist selection propagator.

        Parameters
        ----------
        offspring: int
                   number of offsprings (individuals to be selected)
        """
        super(SelectMin, self).__init__(-1, offspring)

    def __call__(self, inds: List[Individual]) -> List[Individual]:
        """
        Apply elitist-selection propagator.

        Parameters
        ----------
        inds: list[propulate.individual.Individual]
              input individuals the propagator is applied to

        Returns
        -------
        list[propulate.individual.Individual]
            selected output individuals after application of the propagator

        Raises
        ------
        ValueError
            If more individuals than put in shall be selected.
        """
        if len(inds) < self.offspring:
            raise ValueError(
                f"Has to have at least {self.offspring} individuals to select the {self.offspring} best ones."
            )
        # Sort elements of given iterable in specific order + return as list.
        return sorted(inds, key=lambda ind: ind.loss)[
            : self.offspring
        ]  # Return `self.offspring` best individuals in terms of loss.


class SelectMax(Propagator):
    """
    Select specified number of worst performing individuals as evaluated by their losses,
    i.e., those individuals with maximum losses.
    """

    def __init__(
        self,
        offspring: int,
    ) -> None:
        """
        Initialize anti-elitist propagator.

        Parameters
        ----------
        offspring: int
                   number of offspring (individuals to be selected)
        """
        super(SelectMax, self).__init__(-1, offspring)

    def __call__(self, inds: List[Individual]) -> List[Individual]:
        """
        Apply anti-elitist-selection propagator.

        Parameters
        ----------
        inds: list[propulate.individual.Individual]
              individuals the propagator is applied to

        Returns
        -------
        list[propulate.individual.Individual]
            selected individuals after application of the propagator

        Raises
        ------
        ValueError: If more individuals than put in shall be selected.
        """
        if len(inds) < self.offspring:
            raise ValueError(
                f"Has to have at least {self.offspring} individuals to select the {self.offspring} worst ones."
            )
        # Sort elements of given iterable in specific order + return as list.
        return sorted(inds, key=lambda ind: -ind.loss)[
            : self.offspring
        ]  # Return the `self.offspring` worst individuals in terms of loss.


class SelectUniform(Propagator):
    """
    Select specified number of individuals randomly.
    """

    def __init__(self, offspring: int, rng: random.Random = None) -> None:
        """
        Initialize random-selection propagator.

        Parameters
        ----------
        offspring: int
                   number of offspring (individuals to be selected)
        rng: random.Random
             random number generator
        """
        super(SelectUniform, self).__init__(-1, offspring, rng)

    def __call__(self, inds: List[Individual]) -> List[Individual]:
        """
        Apply uniform-selection propagator.

        Parameters
        ----------
        inds: list[propulate.individual.Individual]
              individuals the propagator is applied to

        Returns
        -------
        list[propulate.individual.Individual]
            selected individuals after application of propagator

        Raises
        ------
        ValueError: If more individuals than put in shall be selected.
        """
        if len(inds) < self.offspring:
            raise ValueError(
                f"Has to have at least {self.offspring} individuals to select {self.offspring} from them."
            )
        # Return a `self.offspring` length list of unique elements chosen from `particles`.
        # Used for random sampling without replacement.
        return self.rng.sample(inds, self.offspring)


# TODO parents should be fixed to one NOTE see utils reason why it is not right now
class InitUniform(Stochastic):
    """
    Initialize individual by uniformly sampling specified limits for each trait.
    """

    def __init__(
        self,
        limits: Union[
            Dict[str, Tuple[float, float]],
            Dict[str, Tuple[int, int]],
            Dict[str, Tuple[str, ...]],
        ],
        parents: int = 0,
        probability: float = 1.0,
        rng: random.Random = None,
    ) -> None:
        """
        Initialize random-initialization propagator.

        Parameters
        ----------
        limits: dict[str, tuple[float, float]] | dict[str, tuple[int, int]] | dict[str, tuple[str, ...]]
                search space, i.e., limits of (hyper-)parameters to be optimized
        parents: int
                 number of parents
        probability: float
                     probability of application
        rng: random.Random
             random number generator
        """
        super(InitUniform, self).__init__(parents, 1, probability, rng)
        self.limits = limits

    def __call__(self, *inds: Individual) -> Individual:
        """
        Apply uniform-initialization propagator.

        Parameters
        ----------
        inds: propulate.individual.Individual
              individuals the propagator is applied to

        Returns
        -------
        propulate.individual.Individual
            output individual after application of propagator

        Raises
        ------
        ValueError
            If a parameter's type is invalid, i.e., not float (continuous), int (ordinal), or str (categorical).
        """
        if (
            self.rng.random() < self.probability
        ):  # Apply only with specified probability.
            ind = Individual()  # Instantiate new individual.
            for (
                limit
            ) in self.limits:  # Randomly sample from specified limits for each trait.
                if isinstance(
                    self.limits[limit][0], int
                ):  # If ordinal trait of type integer.
                    ind[limit] = self.rng.randint(*self.limits[limit])
                elif isinstance(
                    self.limits[limit][0], float
                ):  # If interval trait of type float.
                    ind[limit] = self.rng.uniform(*self.limits[limit])
                elif isinstance(
                    self.limits[limit][0], str
                ):  # If categorical trait of type string.
                    ind[limit] = self.rng.choice(self.limits[limit])
                else:
                    raise ValueError(
                        "Unknown type of limits. Has to be float for interval, "
                        "int for ordinal, or string for categorical."
                    )
        else:  # Return first input individual w/o changes otherwise.
            ind = inds[0]
        return ind


class CMAParameter:
    """
    Handles and stores all Basic/Active CMA related constants/variables and strategy parameters.
    """

    def __init__(
        self,
        lamb: int,
        mu: int,
        problem_dimension: int,
        weights: np.ndarray,
        mu_eff: float,
        c_c: float,
        c_1: float,
        c_mu: float,
        limits: Dict,
        exploration: bool,
    ) -> None:
        """
        Initializes a CMAParameter object.
        Parameters
        ----------
        lamb : the number of individuals considered for each generation
        mu : number of positive recombination weights
        problem_dimension: the number of dimensions in the search space
        weights : recombination weights
        mu_eff : variance effective selection mass
        c_c : decay rate for evolution path for the rank-one update of the covariance matrix
        c_1 : learning rate for the rank-one update of the covariance matrix update
        c_mu : learning rate for the rank-mu update of the covariance matrix update
        limits : limits of search space
        exploration : if true decompose covariance matrix for each generation (worse runtime, less exploitation, more decompose_in_each_generation)), else decompose covariance matrix only after a certain number of individuals evaluated (better runtime, more exploitation, less decompose_in_each_generation)
        """
        self.problem_dimension = problem_dimension
        self.limits = limits
        self.lamb = lamb
        self.mu = mu
        self.weights = weights
        # self.c_m = c_m
        self.mu_eff = mu_eff
        self.c_c = c_c
        self.c_1 = c_1
        self.c_mu = c_mu

        # Step-size control params
        self.c_sigma = (mu_eff + 2) / (problem_dimension + mu_eff + 5)
        self.d_sigma = (
            1
            + 2 * max(0, np.sqrt((mu_eff - 1) / (problem_dimension + 1)) - 1)
            + self.c_sigma
        )

        # Initialize dynamic strategy variables
        self.p_sigma = np.zeros((problem_dimension, 1))
        self.p_c = np.zeros((problem_dimension, 1))

        # prevent equal eigenvals, hack from https://github.com/CMA-ES/pycma/blob/development/cma/sampler.py
        self.co_matrix = np.diag(
            np.ones(problem_dimension)
            * np.exp(
                (1e-4 / self.problem_dimension) * np.arange(self.problem_dimension)
            )
        )
        self.b_matrix = np.eye(self.problem_dimension)
        # assuming here self.co_matrix is initialized to be diagonal
        self.d_matrix = np.diag(self.co_matrix) ** 0.5
        # sort eigenvalues in ascending order
        indices_eig = self.d_matrix.argsort()
        self.d_matrix = self.d_matrix[indices_eig]
        self.b_matrix = self.b_matrix[:, indices_eig]
        # the square root of the inverse of the covariance matrix: C^-1/2 = B*D^(-1)*B^T
        self.co_inv_sqrt = (
            self.b_matrix @ np.diag(self.d_matrix ** (-1)) @ self.b_matrix.T
        )
        # the maximum allowed condition of the covariance matrix to ensure numerical stability
        self.condition_limit = 1e5 - 1
        # whether to keep the trace (sum of diagonal elements) of self.co_matrix constant
        self.constant_trace = False

        # use this initial mean when using multiple islands?
        self.mean = np.array(
            [[np.random.uniform(*limits[limit]) for limit in limits]]
        ).reshape((problem_dimension, 1))
        # 0.3 instead of 0.2 is also often used for greater initial step size
        self.sigma = 0.2 * (
            (max(max(limits[i]) for i in limits)) - min(min(limits[i]) for i in limits)
        )

        # the mean of the last generation
        self.old_mean = None
        self.exploration = exploration

        # the number of individuals evaluated when the covariance matrix was last decomposed into B and D
        self.eigen_eval = 0
        # the number of individuals evaluated
        self.count_eval = 0

        # expectation value of ||N(0,I)||
        self.chiN = problem_dimension**0.5 * (
            1 - 1.0 / (4 * problem_dimension) + 1.0 / (21 * problem_dimension**2)
        )

    def set_mean(self, new_mean: np.ndarray) -> None:
        """
        Setter for mean property. Updates the old mean as well.
        Parameters
        ----------
        new_mean : the new mean
        """
        self.old_mean = self.mean
        self.mean = new_mean

    def set_p_sigma(self, new_p_sigma: np.ndarray) -> None:
        """
        Setter for evolution path of step-size adatpiton
        Parameters
        ----------
        new_p_sigma : the new evolution path
        """
        self.p_sigma = new_p_sigma

    def set_p_c(self, new_p_c: np.ndarray) -> None:
        """
        Setter for evolution path of covariance matrix adaption
        Parameters
        ----------
        new_p_c : the new evolution path
        """
        self.p_c = new_p_c

    def set_sigma(self, new_sigma: float) -> None:
        """
        Setter for step-size
        Parameters
        ----------
        new_sigma : the new step-size
        """
        self.sigma = new_sigma

    # version without condition handling
    """def set_co_matrix_depr(self, new_co_matrix: np.ndarray) -> None:
        Setter for the covariance matrix. Computes new values for b_matrix, d_matrix and co_inv_sqrt as well
        Parameters
        ----------
        new_co_matrix : the new covariance matrix
         Update b and d matrix and co_inv_sqrt only after certain number of evaluations to ensure 0(n^2)
         Also trade-Off decompose_in_each_generation or not
        if self.decompose_in_each_generation or (
                self.count_eval - self.eigen_eval
                > self.lamb / (self.c_1 + self.c_mu) / self.problem_dimension / 10
        ):
            self.eigen_eval = self.count_eval
            c = np.triu(new_co_matrix) + np.triu(new_co_matrix, 1).T  # Enforce symmetry
            d, self.b_matrix = np.linalg.eigh(c)  # Eigen decomposition
            self.d_matrix = np.sqrt(d)  # Replace eigenvalues with standard deviations
            self.co_matrix = c
            self.co_inv_sqrt = (
                    self.b_matrix @ np.diag(self.d_matrix ** (-1)) @ self.b_matrix.T
            )
            self.co_inv_sqrt = (self.co_inv_sqrt + self.co_inv_sqrt.T) / 2 # ensure symmetry
            self._sort_b_d_matrix()"""

    def set_co_matrix(self, new_co_matrix: np.ndarray) -> None:
        """
        Setter for the covariance matrix. Computes new values for b_matrix, d_matrix and co_inv_sqrt as well
        Decomposition of co_matrix in O(n^3), hence why the possibility of lazy updating b_matrix and d_matrix.
        Parameters
        ----------
        new_co_matrix : the new covariance matrix
        """
        # Update b and d matrix and co_inv_sqrt only after certain number of evaluations to ensure 0(n^2)
        # Also trade-Off decompose_in_each_generation or not
        if self.exploration or (
            self.count_eval - self.eigen_eval
            > self.lamb / (self.c_1 + self.c_mu) / self.problem_dimension / 10
        ):
            self.eigen_eval = self.count_eval
            self._decompose_co_matrix(new_co_matrix)
            self.co_inv_sqrt = (
                self.b_matrix @ np.diag(self.d_matrix ** (-1)) @ self.b_matrix.T
            )
            # ensure symmetry
            self.co_inv_sqrt = (self.co_inv_sqrt + self.co_inv_sqrt.T) / 2

    def _decompose_co_matrix(self, new_co_matrix: np.ndarray) -> None:
        """
        Eigendecomposition of the covariance matrix into eigenvalues (d_matrix) and eigenvectors (columns of b_matrix)
        Parameters
        ----------
        new_co_matrix: the new covariance matrix that should be decomposed
        """
        # Enforce symmetry
        self.co_matrix = np.triu(new_co_matrix) + np.triu(new_co_matrix, 1).T
        d_matrix_old = self.d_matrix
        try:
            self.d_matrix, self.b_matrix = np.linalg.eigh(self.co_matrix)
            if any(self.d_matrix <= 0):
                # covariance matrix eigen decomposition failed, consider reformulating objective function
                raise ValueError("covariance matrix was not positive definite")
        except Exception as _:
            # add min(eigenvalues(self.co_matrix_old)) to diag(self.co_matrix) and try again
            min_eig_old = min(d_matrix_old) ** 2
            for i in range(self.problem_dimension):
                self.co_matrix[i, i] += min_eig_old
            # Replace eigenvalues with standard deviations
            self.d_matrix = (d_matrix_old**2 + min_eig_old) ** 0.5
            self._decompose_co_matrix(self.co_matrix)
        else:
            assert all(np.isfinite(self.d_matrix))
            self._sort_b_d_matrix()
            if self.condition_limit is not None:
                self._limit_condition(self.condition_limit)
            if self.constant_trace:
                s = 1 / np.mean(
                    self.d_matrix
                )  # normalize co_matrix to control overall magnitude
                self.co_matrix *= s
                self.d_matrix *= s
            self.d_matrix **= 0.5

    def _limit_condition(self, limit) -> None:
        """
        Limit the condition (square of ratio largest to smallest eigenvalue) of the covariance matrix if it exceeds a limit.
        Credits on how to limit the condition: https://github.com/CMA-ES/pycma/blob/development/cma/sampler.py
        Parameters
        ----------
        limit: the treshold for the condition of the matrix
        """
        # check if condition number of matrix is to big
        if (self.d_matrix[-1] / self.d_matrix[0]) ** 2 > limit:
            eps = (self.d_matrix[-1] ** 2 - limit * self.d_matrix[0] ** 2) / (limit - 1)
            for i in range(self.problem_dimension):
                # decrease ratio of largest to smallest eigenvalue, absolute difference remains
                self.co_matrix[i, i] += eps
            # eigenvalues are definitely positive now
            self.d_matrix **= 2
            self.d_matrix += eps
            self.d_matrix **= 0.5

    def _sort_b_d_matrix(self) -> None:
        """
        Sort columns of b_matrix and d_matrix according to the eigenvalues in d_matrix
        """
        indices_eig = np.argsort(self.d_matrix)
        self.d_matrix = self.d_matrix[indices_eig]
        self.b_matrix = self.b_matrix[:, indices_eig]
        assert (min(self.d_matrix), max(self.d_matrix)) == (
            self.d_matrix[0],
            self.d_matrix[-1],
        )

    def mahalanobis_norm(self, dx: np.ndarray) -> np.ndarray:
        """
        Computes the mahalanobis distance by using C^(-1/2) and the difference vector of a point to the mean of a distribution.
        Parameters
        ----------
        dx : the difference vector

        Returns
        -------
        the resulting mahalanobis distance
        """
        return np.linalg.norm(np.dot(self.co_inv_sqrt, dx))


class CMAAdapter(ABC):
    """
    Abstract base class for the adaption of strategy parameters of CMA-ES. Strategy class from the viewpoint of the strategy desing pattern.
    """

    @abstractmethod
    def update_mean(self, par: CMAParameter, arx: np.ndarray) -> None:
        """
        Abstract method for updating of mean in CMA-ES variants.
        Parameters
        ----------
        par : the parameter object of the CMA-ES propagation
        arx : the individuals of the distribution
        """
        pass

    def update_step_size(self, par: CMAParameter) -> None:
        """
        Method for updating step-size in CMA-ES variants. Calculates the current evolution path for the step-size adaption.
        Parameters
        ----------
        par : the parameter object of the CMA-ES propagation
        """
        par.set_p_sigma(
            (1 - par.c_sigma) * par.p_sigma
            + np.sqrt(par.c_sigma * (2 - par.c_sigma) * par.mu_eff)
            * par.co_inv_sqrt
            @ (par.mean - par.old_mean)
            / par.sigma
        )
        par.set_sigma(
            par.sigma
            * np.exp(
                (par.c_sigma / par.d_sigma)
                * (np.linalg.norm(par.p_sigma, ord=2) / par.chiN - 1)
            )
        )

    @abstractmethod
    def update_covariance_matrix(self, par: CMAParameter, arx: np.ndarray) -> None:
        """
        Abstract method for the adaption of the covariance matrix of CMA-ES variants.
        Parameters
        ----------
        par : the parameter object of the CMA-ES propagation
        arx : the individuals of the distribution
        """
        pass

    @abstractmethod
    def compute_weights(
        self, mu: int, lamb: int, problem_dimension: int
    ) -> Tuple[np.ndarray, float, float, float, float]:
        """
        Abstract method for computing the recombination weights of a CMA-ES variant.
        Parameters
        ----------
        mu : the number of positive recombination weights
        lamb : the number of individuals considered for each generation
        problem_dimension : the number of dimensions in the search space

        Returns
        -------
        A Tuple of the weights, mu_eff, c_1, c_c and c_mu
        """
        pass

    @staticmethod
    def compute_learning_rates(
        mu_eff: float, problem_dimension: int
    ) -> Tuple[float, float, float]:
        """
        Computes the learning rates for the CMA-variants.
        Parameters
        ----------
        mu_eff : the variance effective selection mass
        problem_dimension : the number of dimensions in the search space

        Returns
        -------
        A Tuple of c_c, c_1, c_mu
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
    Adaption of strategy parameters of CMA-ES according to the original CMA-ES algorithm. Concrete strategy class from the viewpoint of the strategy design pattern.
    """

    def compute_weights(
        self, mu: int, lamb: int, problem_dimension: int
    ) -> Tuple[np.ndarray, float, float, float, float]:
        """
        Computes the recombination weights for Basic CMA-ES
        Parameters
        ----------
        mu : the number of positive recombination weights
        lamb : the number of individuals considered for each generation
        problem_dimension : the number of dimensions in the search space

        Returns
        -------
        A Tuple of the weights, mu_eff, c_1, c_c and c_mu.
        """
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = np.sum(weights) ** 2 / np.sum(weights**2)
        c_c, c_1, c_mu = BasicCMA.compute_learning_rates(mu_eff, problem_dimension)
        return weights, mu_eff, c_c, c_1, c_mu

    def update_mean(self, par: CMAParameter, arx: np.ndarray) -> None:
        """
        Updates the mean in Basic CMA-ES.
        Parameters
        ----------
        par : the parameter object of the CMA-ES propagation
        arx : the individuals of the distribution

        """
        # matrix vector multiplication (reshape weights to column vector)
        par.set_mean(arx @ par.weights.reshape(-1, 1))

    def update_covariance_matrix(self, par: CMAParameter, arx: np.ndarray) -> None:
        """
        Adapts the covariance matrix of Basic CMA-ES.
        Parameters
        ----------
        par : the parameter object of the CMA-ES propagation
        arx : the individuals of the distribution
        """
        # turn off rank-one accumulation when sigma increases quickly
        h_sig = np.sum(par.p_sigma**2) / (
            1 - (1 - par.c_sigma) ** (2 * (par.count_eval / par.lamb))
        ) / par.problem_dimension < 2 + 4.0 / (par.problem_dimension + 1)
        # update evolution path
        par.set_p_c(
            (1 - par.c_c) * par.p_c
            + h_sig
            * np.sqrt(par.c_c * (2 - par.c_c) * par.mu_eff)
            * (par.mean - par.old_mean)
            / par.sigma
        )
        # use h_sig to the power of two (unlike in paper) for the variance loss from h_sig
        ar_tmp = (1 / par.sigma) * (
            arx[:, : par.mu] - np.tile(par.old_mean, (1, par.mu))
        )
        new_co_matrix = (
            (1 - par.c_1 - par.c_mu) * par.co_matrix
            + par.c_1
            * (
                par.p_c @ par.p_c.T
                + (1 - h_sig) * par.c_c * (2 - par.c_c) * par.co_matrix
            )
            + par.c_mu * ar_tmp @ (par.weights * ar_tmp).T
        )
        par.set_co_matrix(new_co_matrix)


class ActiveCMA(CMAAdapter):
    """
    Adaption of strategy parameters of CMA-ES according to the Active CMA-ES algorithm. Different to the original CMA-ES algorithm Active CMA-ES uses negative recombination weights (only for the covariance matrix adaption) for individuals with relatively low fitness. Concrete strategy class from the viewpoint of the strategy design pattern.
    """

    def compute_weights(
        self, mu: int, lamb: int, problem_dimension: int
    ) -> Tuple[np.ndarray, float, float, float, float]:
        """
        Computes the recombination weights for Active CMA-ES
        Parameters
        ----------
        mu : the number of positive recombination weights
        lamb : the number of individuals considered for each generation
        problem_dimension : the number of dimensions in the search space

        Returns
        -------
        A Tuple of the weights, mu_eff, c_1, c_c and c_mu.
        """
        weights_preliminary = np.log(lamb / 2 + 0.5) - np.log(np.arange(1, lamb + 1))
        mu_eff = np.sum(weights_preliminary[:mu]) ** 2 / np.sum(
            weights_preliminary[:mu] ** 2
        )
        c_c, c_1, c_mu = ActiveCMA.compute_learning_rates(mu_eff, problem_dimension)
        # now compute final weights
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
        Updates the mean in Active CMA-ES.
        Parameters
        ----------
        par : the parameter object of the CMA-ES propagation
        arx : the individuals of the distribution
        """
        # matrix vector multiplication (reshape weights to column vector)
        # Only consider positive weights
        par.set_mean(arx @ par.weights[: par.mu].reshape(-1, 1))

    def update_covariance_matrix(self, par: CMAParameter, arx: np.ndarray) -> None:
        """
        Adapts the covariance matrix of Basic CMA-ES.
        Parameters
        ----------
        par : the parameter object of the CMA-ES propagation
        arx : the individuals of the distribution
        """
        # turn off rank-one accumulation when sigma increases quickly
        h_sig = np.sum(par.p_sigma**2) / (
            1 - (1 - par.c_sigma) ** (2 * (par.count_eval / par.lamb))
        ) / par.problem_dimension < 2 + 4.0 / (par.problem_dimension + 1)
        # update evolution path
        par.set_p_c(
            (1 - par.c_c) * par.p_c
            + h_sig
            * np.sqrt(par.c_c * (2 - par.c_c) * par.mu_eff)
            * (par.mean - par.old_mean)
            / par.sigma
        )
        weights_circle = np.zeros((par.lamb,))
        for i, w_i in enumerate(par.weights):
            # guaranty positive definiteness
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
        # use h_sig to the power of two (unlike in paper) for the variance loss from h_sig?
        ar_tmp = (1 / par.sigma) * (arx - np.tile(par.old_mean, (1, par.lamb)))
        new_co_matrix = (
            (1 - par.c_1 - par.c_mu) * par.co_matrix
            + par.c_1
            * (
                par.p_c @ par.p_c.T
                + (1 - h_sig) * par.c_c * (2 - par.c_c) * par.co_matrix
            )
            + par.c_mu * ar_tmp @ (weights_circle * ar_tmp).T
        )
        par.set_co_matrix(new_co_matrix)


class CMAPropagator(Propagator):
    """
    Propagator of CMA-ES. Uses CMAAdapter to adapt strategy parameters like mean, step-size and covariance matrix and stores them in a CMAParameter object.
    The context class from the viewpoint of the strategy design pattern.
    """

    def __init__(
        self,
        adapter: CMAAdapter,
        limits: Dict,
        rng,
        decompose_in_each_generation=False,
        select_worst_all_time=False,
        pop_size=None,
        pool_size=3,
    ) -> None:
        """
        Constructor of CMAPropagator.
        Parameters
        ----------
        adapter : the adaption strategy of CMA-ES
        limits : the limits of the search space
        decompose_in_each_generation : if true decompose covariance matrix for each generation (worse runtime, less exploitation, more exploration)), else decompose covariance matrix only after a certain number of individuals evaluated (better runtime, more exploitation, less exploration)
        select_worst_all_time : if true use the worst individuals for negative recombination weights in active CMA-ES, else use the worst (lambda - mu) individuals of the best lambda individuals. If BasicCMA is used the given value is irrelevant with regards to functionality.
        pop_size: the number of individuals to be considered in each generation
        pool_size: the size of the pool of individuals preselected before selecting the best of this pool
        """
        self.adapter = adapter
        problem_dimension = len(limits)
        # The number of individuals considered for each generation
        lamb = (
            pop_size if pop_size else 4 + int(np.floor(3 * np.log(problem_dimension)))
        )
        super(CMAPropagator, self).__init__(lamb, 1)

        # Number of positive recombination weights
        mu = lamb // 2
        self.select_worst = SelectMax(lamb - mu)
        self.select_worst_all_time = select_worst_all_time

        # CMA-ES variant specific weights and learning rates
        weights, mu_eff, c_c, c_1, c_mu = adapter.compute_weights(
            mu, lamb, problem_dimension
        )

        self.par = CMAParameter(
            lamb,
            mu,
            problem_dimension,
            weights,
            mu_eff,
            c_c,
            c_1,
            c_mu,
            limits,
            decompose_in_each_generation,
        )
        self.pool_size = int(pool_size) if int(pool_size) >= 1 else 3
        self.select_pool = SelectMin(self.pool_size * lamb)
        self.select_from_pool = SelectUniform(mu - 1, rng=rng)
        self.select_best_1 = SelectMin(1)

    def __call__(self, inds: List[Individual]) -> Individual:
        """
        The skeleton of the CMA-ES algorithm using the template method design pattern. Sampling individuals and adapting the strategy parameters.
        Template methods are "update_mean()", "update_covariance_matrix()" and "update_step_size()".
        Parameters
        ----------
        inds: list of individuals available

        Returns
        -------
        new_ind : the new sampled individual
        """
        num_inds = len(inds)
        # add individuals from different workers to eval_count
        self.par.count_eval += num_inds - self.par.count_eval
        # sample new individual
        new_ind = self._sample_cma()
        # check if len(inds) >= or < pool_size * lambda and make sample or sample + update
        if num_inds >= self.pool_size * self.par.lamb:
            inds_pooled = self.select_pool(inds)
            best = self.select_best_1(inds_pooled)
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

            # Update mean
            self.adapter.update_mean(self.par, arx[:, : self.par.mu])
            # Update Covariance Matrix
            self.adapter.update_covariance_matrix(self.par, arx)
            # Update step_size
            self.adapter.update_step_size(self.par)
        return new_ind

    def _transform_individuals_to_matrix(self, inds: List[Individual]) -> np.ndarray:
        """
        Takes a list of individuals and transform it to numpy matrix for easier subsequent computation
        Parameters
        ----------
        inds : list of individuals

        Returns
        -------
        arx : a numpy array of shape (problem_dimension, len(inds))
        """
        arx = np.zeros((self.par.problem_dimension, len(inds)))
        for k, ind in enumerate(inds):
            for i, (dim, _) in enumerate(self.par.limits.items()):
                arx[i, k] = ind[dim]
        return arx

    def _sample_cma(self) -> Individual:
        """
        Samples new individuals according to CMA-ES.
        Returns
        -------
        new_ind : the new sampled individual
        """
        new_x = None
        # Generate new offspring
        random_vector = np.random.randn(self.par.problem_dimension, 1)
        try:
            new_x = self.par.mean + self.par.sigma * self.par.b_matrix @ (
                self.par.d_matrix * random_vector
            )
        except (RuntimeWarning, Exception) as _:
            raise ValueError(
                "Failed to generate new offsprings, probably due to not well defined target function."
            )
        self.par.count_eval += 1
        # Remove problem_dim
        new_ind = Individual()

        for i, (dim, _) in enumerate(self.par.limits.items()):
            new_ind[dim] = new_x[i, 0]
        return new_ind

    def get_mean(self) -> np.ndarray:
        """
        Getter for mean attribute.
        Returns
        -------
        mean : the current cma-es mean of the best mu individuals
        """
        return self.par.mean

    def get_sigma(self) -> float:
        """
        Getter for step size.
        Returns
        -------
        sigma : the current step-size
        """
        return self.par.sigma

    def get_co_matrix(self) -> np.ndarray:
        """
        Getter for covariance matrix.
        Returns
        -------
        co_matrix : current covariance matrix
        """
        return self.par.co_matrix

    def get_evolution_path_sigma(self) -> np.ndarray:
        """
        Getter for evolution path of step-size adaption.
        Returns
        -------
        p_sigma : evolution path for step-size adaption
        """
        return self.par.p_sigma

    def get_evolution_path_co_matrix(self) -> np.ndarray:
        """
        Getter for evolution path of covariance matrix adpation.
        Returns
        -------
        p_c : evolution path for covariance matrix adaption
        """
        return self.par.p_c

