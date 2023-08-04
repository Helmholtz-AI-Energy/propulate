import copy

import numpy as np
from abc import ABC, abstractmethod

from typing import List, Dict, Tuple
from .population import Individual


def _check_compatible(out1, in2):
    """
    Check compability of two propagators for stacking them together sequentially with Compose().
    """
    return out1 == in2 or in2 == -1


class Propagator:
    """
    Abstract base class for all propagators, i.e., evolutionary operators, in Propulate.

    Take a collection of individuals and use them to breed a new collection of individuals.
    """

    def __init__(self, parents=0, offspring=0, rng=None):
        """
        Constructor of Propagator class.

        Parameters
        ----------
        parents : int
                  number of input individuals (-1 for any)
        offspring : int
                    number of output individuals
        rng : random.Random()
              random number generator
        """
        self.parents = parents
        self.rng = rng
        self.offspring = offspring
        if offspring == 0:
            raise ValueError("Propagator has to sire more than 0 offspring.")

    def __call__(self, inds):
        """
        Apply propagator (not implemented!).

        Parameters
        ----------
        inds: propulate.population.Individual
              individuals the propagator is applied to
        """
        raise NotImplementedError()


class Stochastic(Propagator):
    """
    Apply StochasticPropagator only with a given probability.

    If not applied, the output still has to adhere to the defined number of offsprings.
    """

    def __init__(self, parents=0, offspring=0, probability=1.0, rng=None):
        """
        Constructor of StochasticPropagator class.

        Parameters
        ----------
        parents : int
                  number of input individuals (-1 for any)
        offspring : int
                    number of output individuals
        probability : float
                      probability of application
        rng : random.Random()
              random number generator
        """
        super(Stochastic, self).__init__(parents, offspring, rng)
        self.probability = probability
        if offspring == 0:
            raise ValueError("Propagator has to sire more than 0 offspring.")


class Conditional(Propagator):
    """
    Apply different propagators depending on whether breeding population is complete or not.

    If population consists of less than the specified number of individuals required for breeding,
    a different propagator is applied than if this condition is fulfilled.
    """

    def __init__(self, pop_size, true_prop, false_prop, parents=-1, offspring=-1):
        """
        Constructor of Conditional class.

        Parameters
        ----------
        pop_size : int
                   breeding population size
        true_prop : propulate.propagators.Propagator
                    propagator applied if size of current population >= pop_size.
        false_prop : propulate.propagators.Propagator
                     propagator applied if size of current population < pop_size.
        parents : int
                  number of input individuals (-1 for any)
        offspring : int
                    number of output individuals
        """
        super(Conditional, self).__init__(parents, offspring)
        self.pop_size = pop_size
        self.true_prop = true_prop
        self.false_prop = false_prop

    def __call__(self, inds):
        """
        Apply conditional propagator.

        Parameters
        ----------
        inds: propulate.population.Individual
              individuals the propagator is applied to
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

    def __init__(self, propagators):
        """
        Constructor of Compose class.

        Parameters
        ----------
        propagators : list of propulate.propagators.Propagator objects
                      propagators to be stacked together sequentially
        """
        super(Compose, self).__init__(propagators[0].parents, propagators[-1].offspring)
        self.propagators = propagators
        for i in range(len(propagators) - 1):
            # Check compability of consecutive propagators in terms of number of parents + offsprings.
            if not _check_compatible(
                propagators[i].offspring, propagators[i + 1].parents
            ):
                outp = propagators[i]
                inp = propagators[i + 1]
                outd = outp.offspring
                ind = inp.parents

                raise ValueError(
                    f"Incompatible combination of {outd} output individuals of {outp} and {ind} input individuals of {inp}."
                )

    def __call__(
        self, inds
    ):  # Apply propagators sequentially as requested in Compose(...)
        """
        Apply Compose propagator.

        Parameters
        ----------
        inds: list of propulate.population.Individual objects
              individuals the propagator is applied to

        Returns
        -------
        inds: list of propulate.population.Individual objects
              individuals after application of propagator
        """
        for p in self.propagators:
            inds = p(inds)
        return inds


class PointMutation(Stochastic):
    """
    Point-mutate given number of traits with given probability.
    """

    def __init__(self, limits, points=1, probability=1.0, rng=None):
        """
        Constructor of PointMutation class.

        Parameters
        ----------
        limits : dict
                 limits of (hyper-)parameters to be optimized
        points : int
                 number of points to mutate
        probability: float
                     probability of application
        rng : random.Random()
              random number generator
        """
        super(PointMutation, self).__init__(1, 1, probability, rng)
        self.points = points
        self.limits = limits
        if len(limits) < points:
            raise ValueError(
                f"Too many points to mutate for individual with {len(limits)} traits."
            )

    def __call__(self, ind):
        """
        Apply point-mutation propagator.

        Parameters
        ----------
        ind: propulate.population.Individual
             individual the propagator is applied to

        Returns
        -------
        ind : propulate.population.Individual
              possibly point-mutated individual after application of propagator
        """
        if (
            self.rng.random() < self.probability
        ):  # Apply propagator only with specified `probability`
            ind = copy.deepcopy(ind)
            ind.loss = None  # Initialize individual's loss attribute.
            # Determine traits to mutate via random sampling.
            # Return `self.points` length list of unique elements chosen from `ind.keys()`.
            # Used for random sampling without replacement.
            to_mutate = self.rng.sample(sorted(ind.keys()), self.points)
            # Point-mutate `self.points` randomly chosen traits of individual `ind`.
            for i in to_mutate:
                if type(ind[i]) == int:
                    # Return randomly selected element from int range(start, stop, step).
                    ind[i] = self.rng.randint(*self.limits[i])
                elif type(ind[i]) == float:
                    # Return random floating point number N within limits.
                    ind[i] = self.rng.uniform(*self.limits[i])
                elif type(ind[i]) == str:
                    # Return random element from non-empty sequence.
                    ind[i] = self.rng.choice(self.limits[i])

        return ind  # Return point-mutated individual.


class RandomPointMutation(Stochastic):
    """
    Point-mutate random number of traits between min_points and max_points with given probability.
    """

    def __init__(self, limits, min_points, max_points, probability=1.0, rng=None):
        """
        Constructor of RandomPointMutation class.

        Parameters
        ----------
        limits : dict
                 limits of (hyper-)parameters to be optimized
        min_points : int
                     minimum number of points to mutate
        max_points : int
                     maximum number of points to mutate
        probability : float
                      probability of application
        rng : random.Random()
              random number generator
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
                f"Minimum number of traits to mutate must be <= respective maximum number but min_points = {min_points} > {max_points} = max_points."
            )
        self.min_points = int(min_points)
        self.max_points = int(max_points)
        self.limits = limits

    def __call__(self, ind):
        """
        Apply random-point-mutation propagator.

        Parameters
        ----------
        ind : propulate.population.Individual
              individual the propagator is applied to

        Returns
        -------
        ind : propulate.population.Individual
              possibly point-mutated individual after application of propagator
        """
        if (
            self.rng.random() < self.probability
        ):  # Apply propagator only with specified `probability`
            ind = copy.deepcopy(ind)
            ind.loss = None  # Initialize individual's loss attribute.
            # Determine traits to mutate via random sampling.
            # Return `self.points` length list of unique elements chosen from `ind.keys()`.
            # Used for random sampling without replacement.
            points = self.rng.randint(self.min_points, self.max_points)
            to_mutate = self.rng.sample(sorted(ind.keys()), points)
            # Point-mutate `points` randomly chosen traits of individual `ind`.
            for i in to_mutate:
                if type(ind[i]) == int:
                    # Return randomly selected element from int range(start, stop, step).
                    ind[i] = self.rng.randint(*self.limits[i])
                elif type(ind[i]) == float:
                    # Return random floating point number N within limits.
                    ind[i] = self.rng.uniform(*self.limits[i])
                elif type(ind[i]) == str:
                    # Return random element from non-empty sequence.
                    ind[i] = self.rng.choice(self.limits[i])

        return ind  # Return point-mutated individual.


class IntervalMutationNormal(Stochastic):
    """
    Mutate given number of traits according to Gaussian distribution around current value with given probability.
    """

    def __init__(self, limits, sigma_factor=0.1, points=1, probability=1.0, rng=None):
        """
        Constructor of IntervalMutationNormal class.

        Parameters
        ----------
        limits : dict
                 limits of (hyper-)parameters to be optimized
        sigma_factor : float
                       scaling factor for interval width to obtain std
        points : int
                 number of points to mutate
        probability : float
                      probability of application
        rng : random.Random()
              random number generator
        """
        super(IntervalMutationNormal, self).__init__(1, 1, probability, rng)
        self.points = points  # number of traits to point-mutate
        self.limits = limits
        self.sigma_factor = sigma_factor
        n_interval_traits = len([x for x in limits if type(limits[x][0]) == float])
        if n_interval_traits < points:
            raise ValueError(
                f"Too many points to mutate for individual with {n_interval_traits} interval traits"
            )

    def __call__(self, ind):
        """
        Apply interval-mutation propagator.

        Parameters
        ----------
        ind : propulate.population.Individual
              individual the propagator is applied to

        Returns
        -------
        ind : propulate.population.Individual
              possibly interval-mutated individual after application of propagator
        """
        if (
            self.rng.random() < self.probability
        ):  # Apply propagator only with specified `probability`.
            ind = copy.deepcopy(ind)
            ind.loss = None  # Initialize individual's loss attribute.
            # Determine traits of type float.
            interval_keys = [x for x in ind.keys() if type(ind[x]) == float]
            # Determine ´self.points` traits to mutate.
            to_mutate = self.rng.sample(interval_keys, self.points)
            # Mutate traits by sampling from Gaussian distribution centered around current value
            # with `sigma_factor` scaled interval width as standard distribution.
            for i in to_mutate:
                min_val, max_val = self.limits[i]  # Determine interval boundaries.
                mu = ind[i]  # Current value is mean.
                sigma = (
                    max_val - min_val
                ) * self.sigma_factor  # Determine std from interval boundaries + sigma factor.
                ind[i] = self.rng.gauss(
                    mu, sigma
                )  # Sample new value from Gaussian blob centered around current value.
                ind[i] = min(
                    max_val, ind[i]
                )  # Make sure new value is within specified limits.
                ind[i] = max(min_val, ind[i])

        return ind  # Return point-mutated individual.


class MateUniform(Stochastic):  # uniform crossover
    """
    Generate new individual by uniform crossover of two parents with specified relative parent contribution.
    """

    def __init__(self, rel_parent_contrib=0.5, probability=1.0, rng=None):
        """
        Constructor of MateUniform class.

        Parameters
        ----------
        rel_parent_contrib : float
                             relative parent contribution (w.r.t. 1st parent)
        probability : float
                      probability of application
        rng : random.Random()
              random number generator
        """
        super(MateUniform, self).__init__(
            2, 1, probability, rng
        )  # Breed 1 offspring from 2 parents.
        if rel_parent_contrib <= 0 or rel_parent_contrib >= 1:
            raise ValueError(
                f"Relative parent contribution must be within (0, 1) but was {rel_parent_contrib}."
            )
        self.rel_parent_contrib = rel_parent_contrib

    def __call__(self, inds):
        """
        Apply uniform-crossover propagator.

        Parameters
        ----------
        inds : list of propulate.population.Individual objects
               individuals the propagator is applied to

        Returns
        -------
        ind : propulate.population.Individual
              possibly cross-bred individual after application of propagator
        """
        ind = copy.deepcopy(inds[0])  # Consider 1st parent.
        if (
            self.rng.random() < self.probability
        ):  # Apply propagator only with specified `probability`.
            ind.loss = None  # Initialize individual's loss attribute.
            # Replace traits in 1st parent with values of 2nd parent with a probability of 0.5.
            for k in inds[1].keys():
                if self.rng.random() > self.rel_parent_contrib:
                    ind[k] = inds[1][k]
        return ind  # Return offspring.


class MateMultiple(Stochastic):  # uniform crossover
    """
    Generate new individual by uniform crossover of multiple parents.
    """

    def __init__(self, parents=-1, probability=1.0, rng=None):
        """
        Constructor of MateMultiple class.

        Parameters
        ----------
        rel_parent_contrib : float
                             relative parent contribution (w.r.t. 1st parent)
        probability : float
                      probability of application
        rng : random.Random()
              random number generator
        """
        super(MateMultiple, self).__init__(
            parents, 1, probability, rng
        )  # Breed 1 offspring from 2 parents.

    def __call__(self, inds):
        """
        Apply multiple-crossover propagator.

        Parameters
        ----------
        inds : list of propulate.population.Individual objects
               individuals the propagator is applied to

        Returns
        -------
        ind : propulate.population.Individual
              possibly cross-bred individual after application of propagator
        """
        ind = copy.deepcopy(inds[0])  # Consider 1st parent.
        if (
            self.rng.random() < self.probability
        ):  # Apply propagator only with specified `probability`.
            ind.loss = None  # Initialize individual's loss attribute.
            # Replace traits in 1st parent with values of 2nd parent with a probability of 0.5.
            for k in ind.keys():
                if self.rng.random() > self.rel_parent_contrib:
                    temp = self.rng.choice(inds)
                    ind[k] = temp[k]
        return ind  # Return offspring.


class MateSigmoid(
    Stochastic
):  # crossover according to sigmoid probability of fitnesses
    """
    Generate new individual by crossover of two parents according to Boltzmann sigmoid probability.

    Consider two parents `ind1` and `ind2` with fitnesses `f1` and `f2`. Let f1 <= f2. For each trait,
    the better parent's value is accepted with the probability sigmoid(- (f1-f2) / temperature).
    """

    def __init__(self, temperature=1.0, probability=1.0, rng=None):
        """
        Constructor of MateSigmoid class.

        Parameters
        ----------
        temperature : float
                      temperature for Boltzmann factor in sigmoid probability
        probability : float
                      probability of application
        rng : random.Random()
              random number generator
        """
        super(MateSigmoid, self).__init__(
            2, 1, probability, rng
        )  # Breed 1 offspring from 2 parents.
        self.temperature = temperature

    def __call__(self, inds):
        """
        Apply sigmoid-crossover propagator.

        Parameters
        ----------
        inds : list of propulate.population.Individual objects
               individuals the propagator is applied to

        Returns
        -------
        ind : propulate.population.Individual
              possibly cross-bred individual after application of propagator
        """
        ind = copy.deepcopy(inds[0])  # Consider 1st parent.
        if inds[0].loss <= inds[1].loss:
            delta = inds[0].loss - inds[1].loss
            fraction = 1 / (1 + np.exp(-delta / self.temperature))
        else:
            delta = inds[1].loss - inds[0].loss
            fraction = 1 - 1 / (1 + np.exp(-delta / self.temperature))

        if (
            self.rng.random() < self.probability
        ):  # Apply propagator only with specified `probability`.
            ind.loss = None  # Initialize individual's loss attribute.
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

    def __init__(self, offspring, rng=None):
        """
        Constructor of SelectMin class.

        Parameters
        ----------
        offspring : int
                    number of offsprings (individuals to be selected)
        """
        super(SelectMin, self).__init__(-1, offspring)

    def __call__(self, inds):
        """
        Apply elitist-selection propagator.

        Parameters
        ----------
        inds : list of propulate.population.Individual objects
               individuals the propagator is applied to

        Returns
        -------
        inds : list of propulate.population.Individual objects
              list of selected individuals after application of propagator
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

    def __init__(self, offspring, rng=None):
        """
        Constructor of SelectMax class.

        Parameters
        ----------
        offspring : int
                    number of offsprings (individuals to be selected)
        """
        super(SelectMax, self).__init__(-1, offspring)

    def __call__(self, inds):
        """
        Apply anti-elitist-selection propagator.

        Parameters
        ----------
        inds : list of propulate.population.Individual objects
               individuals the propagator is applied to

        Returns
        -------
        inds : list of propulate.population.Individual objects
              list of selected individuals after application of propagator
        """
        if len(inds) < self.offspring:
            raise ValueError(
                f"Has to have at least {self.offspring} individuals to select the {self.offspring} worst ones."
            )
        # Sort elements of given iterable in specific order + return as list.
        return sorted(inds, key=lambda ind: -ind.loss)[
            : self.offspring
        ]  # Return `self.offspring` worst individuals in terms of loss.


class SelectUniform(Propagator):
    """
    Select specified number of individuals randomly.
    """

    def __init__(self, offspring, rng=None):
        """
        Constructor of SelectUniform class.

        Parameters
        ----------
        offspring : int
                    number of offsprings (individuals to be selected)
        rng : random.Random()
              random number generator
        """
        super(SelectUniform, self).__init__(-1, offspring, rng)

    def __call__(self, inds):
        """
        Apply random-selection propagator.

        Parameters
        ----------
        inds : list of propulate.population.Individual objects
               individuals the propagator is applied to

        Returns
        -------
        ind : propulate.population.Individual
              list of selected individuals after application of propagator
        """
        if len(inds) < self.offspring:
            raise ValueError(
                f"Has to have at least {self.offspring} individuals to select {self.offspring} from them."
            )
        # Return a `self.offspring` length list of unique elements chosen from `inds`.
        # Used for random sampling without replacement.
        return self.rng.sample(inds, self.offspring)


# TODO parents should be fixed to one NOTE see utils reason why it is not right now
class InitUniform(Stochastic):
    """
    Initialize individuals by uniformly sampling specified limits for each trait.
    """

    def __init__(self, limits, parents=0, probability=1.0, rng=None):
        """
        Constructor of InitUniform class.

        In case of parents > 0 and probability < 1., call returns input individual without change.

        Parameters
        ----------
        limits : dict
                 limits of (hyper-)parameters to be optimized
        offspring : int
                    number of offsprings (individuals to be selected)
        rng : random.Random()
              random number generator
        """
        super(InitUniform, self).__init__(parents, 1, probability, rng)
        self.limits = limits

    def __call__(self, *inds):
        """
        Apply uniform-initialization propagator.

        Parameters
        ----------
        inds : list of propulate.population.Individual objects
               individuals the propagator is applied to

        Returns
        -------
        ind : propulate.population.Individual
              list of selected individuals after application of propagator
        """
        if (
            self.rng.random() < self.probability
        ):  # Apply only with specified `probability`.
            ind = Individual()  # Instantiate new individual.
            for limit in self.limits:
                # Randomly sample from specified limits for each trait.
                if (
                    type(self.limits[limit][0]) == int
                ):  # If ordinal trait of type integer.
                    ind[limit] = self.rng.randint(*self.limits[limit])
                elif (
                    type(self.limits[limit][0]) == float
                ):  # If interval trait of type float.
                    ind[limit] = self.rng.uniform(*self.limits[limit])
                elif (
                    type(self.limits[limit][0]) == str
                ):  # If categorical trait of type string.
                    ind[limit] = self.rng.choice(self.limits[limit])
                else:
                    raise ValueError(
                        "Unknown type of limits. Has to be float for interval, int for ordinal, or string for categorical."
                    )
            return ind
        else:
            ind = inds[0]
            return ind  # Return 1st input individual w/o changes.


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
        exploration : if true decompose covariance matrix for each generation (worse runtime, less exploitation, more exploration)), else decompose covariance matrix only after a certain number of individuals evaluated (better runtime, more exploitation, less exploration)
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
        self.b_matrix = np.eye(problem_dimension)
        self.d_matrix = np.ones(
            (problem_dimension, 1)
        )  # Diagonal entries responsible for scaling co_matrix
        self.co_matrix = (
            self.b_matrix @ np.diag(self.d_matrix[:, 0] ** 2) @ self.b_matrix.T
        )
        # the square root of the inverse of the covariance matrix: C^-1/2 = B*D^(-1)*B^T
        self.co_inv_sqrt = (
            self.b_matrix @ np.diag(self.d_matrix[:, 0] ** -1) @ self.b_matrix.T
        )

        # use this initial mean when using multiple islands?
        # mean = np.array([[np.random.uniform(*limits[limit]) for limit in limits]]).reshape((problem_dimension, 1))
        # use this initial mean when using one island?
        self.mean = np.random.rand(problem_dimension, 1)
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

    def get_mean(self) -> np.ndarray:
        """
        Getter for mean attribute.
        Returns
        -------
        mean : the current cma-es mean of the best mu individuals
        """
        return self.mean

    def get_sigma(self) -> float:
        """
        Getter for step size.
        Returns
        -------
        sigma : the current step-size
        """
        return self.sigma

    def get_co_matrix(self) -> np.ndarray:
        """
        Getter for covariance matrix.
        Returns
        -------
        co_matrix : current covariance matrix
        """
        return self.co_matrix

    def get_evolution_path_sigma(self) -> np.ndarray:
        """
        Getter for evolution path of step-size adaption.
        Returns
        -------
        p_sigma : evolution path for step-size adaption
        """
        return self.p_sigma

    def get_evolution_path_co_matrix(self) -> np.ndarray:
        """
        Getter for evolution path of covariance matrix adpation.
        Returns
        -------
        p_c : evolution path for covariance matrix adaption
        """
        return self.p_c

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

    def set_co_matrix(self, new_co_matrix: np.ndarray) -> None:
        """
        Setter for the covariance matrix. Computes new values for b_matrix, d_matrix and co_inv_sqrt as well
        Parameters
        ----------
        new_co_matrix : the new covariance matrix
        """
        self.co_matrix = new_co_matrix
        # Update b and d matrix and co_inv_sqrt only after certain number of evaluations to ensure 0(n^2)
        # Also trade-Off exploitation vs exploration
        if self.exploration or (
            self.count_eval - self.eigen_eval
            > self.lamb / (self.c_1 + self.c_mu) / self.problem_dimension / 10
        ):
            self.eigen_eval = self.count_eval
            c = np.triu(new_co_matrix) + np.triu(new_co_matrix, 1).T  # Enforce symmetry
            d, self.b_matrix = np.linalg.eig(c)  # Eigen decomposition
            self.d_matrix = np.sqrt(d)  # Replace eigenvalues with standard deviations
            self.co_inv_sqrt = (
                self.b_matrix @ np.diag(self.d_matrix ** (-1)) @ self.b_matrix.T
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
        # new_co_matrix = (1 - par.c_1 - par.c_mu) * par.co_matrix + par.c_1 * (par.p_c @ par.p_c.T + (1 - h_sig) * par.c_c * par.c_1 * (2 - par.c_c) * par.co_matrix) + par.c_mu * ar_tmp @ (par.weights * ar_tmp).T
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
        # use h_sig to the power of two (unlike in paper) for the variance loss from h_sig
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
        # new_co_matrix = (1 - par.c_1 - par.c_mu) * par.co_matrix + par.c_1 * (par.p_c @ par.p_c.T + (1 - h_sig) * par.c_c * par.c_1 * (2 - par.c_c) * par.co_matrix) + par.c_mu * ar_tmp @ (par.weights * ar_tmp).T
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
        exploration=False,
        select_worst_all_time=False,
        pop_size=None,
    ) -> None:
        """
        Constructor of CMAPropagator.
        Parameters
        ----------
        adapter : the adaption strategy of CMA-ES
        limits : the limits of the search space
        exploration : if true decompose covariance matrix for each generation (worse runtime, less exploitation, more exploration)), else decompose covariance matrix only after a certain number of individuals evaluated (better runtime, more exploitation, less exploration)
        select_worst_all_time : if true use the worst individuals for negative recombination weights in active CMA-ES, else use the worst (lambda - mu) individuals of the best lambda individuals. If BasicCMA is used the given value is irrelevant with regards to functionality.
        pop_size: the number of individuals to be considered in each generation
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
        self.selectBestMu = SelectMin(mu)
        self.selectBestLambda = SelectMin(lamb)
        self.selectWorst = SelectMax(lamb - mu)
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
            exploration,
        )

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
        # check if len(inds) >= oder < lambda and make sample or sample + update
        if num_inds >= self.par.lamb:
            # Update mean
            self.adapter.update_mean(
                self.par, self._transform_individuals_to_matrix(self.selectBestMu(inds))
            )
            # Update Covariance Matrix
            if not self.select_worst_all_time:
                self.adapter.update_covariance_matrix(
                    self.par,
                    self._transform_individuals_to_matrix(self.selectBestLambda(inds)),
                )
            else:
                self.adapter.update_covariance_matrix(
                    self.par,
                    self._transform_individuals_to_matrix(
                        self.selectBestMu(inds) + self.selectWorst(inds)
                    ),
                )
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
        # Generate new offspring
        random_vector = np.random.randn(self.par.problem_dimension, 1)
        new_x = self.par.mean + self.par.sigma * self.par.b_matrix @ (
            self.par.d_matrix * random_vector
        )
        self.par.count_eval += 1

        new_ind = Individual()

        for i, (dim, _) in enumerate(self.par.limits.items()):
            new_ind[dim] = new_x[i, 0]
        return new_ind
