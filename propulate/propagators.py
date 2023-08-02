import copy

import numpy

from .population import Individual


def _check_compatible(out1, in2):
    """
    Check compability of two propagators for stacking them together sequentially with Cascade().
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

    def __call__(self, inds):
        """
        Apply stochastic propagator (not implemented!).

        Parameters
        ----------
        inds: propulate.population.Individual
              individuals the propagator is applied to
        """
        raise NotImplementedError()


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


class Cascade(Propagator):
    """
    Stack propagators together sequentially for successive application.
    """

    def __init__(self, propagators):
        """
        Constructor of Cascade class.

        Parameters
        ----------
        propagators : list of propulate.propagators.Propagator objects
                      propagators to be stacked together sequentially
        """
        super(Cascade, self).__init__(propagators[0].parents, propagators[-1].offspring)
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
    ):  # Apply propagators sequentially as requested in Cascade(...)
        """
        Apply Cascade propagator.

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
            to_mutate = self.rng.sample(ind.keys(), self.points)
            # Point-mutate `self.points` randomly chosen traits of individual `ind`.
            for i in to_mutate:
                if type(ind[i]) == int:
                    # Return randomly selected element from int range(start, stop, step).
                    ind[i] = self.rng.randrange(*self.limits[i])
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
            to_mutate = self.rng.sample(ind.keys(), points)
            # Point-mutate `points` randomly chosen traits of individual `ind`.
            for i in to_mutate:
                if type(ind[i]) == int:
                    # Return randomly selected element from int range(start, stop, step).
                    ind[i] = self.rng.randrange(*self.limits[i])
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
            fraction = 1 / (1 + numpy.exp(-delta / self.temperature))
        else:
            delta = inds[1].loss - inds[0].loss
            fraction = 1 - 1 / (1 + numpy.exp(-delta / self.temperature))

        if (
            self.rng.random() < self.probability
        ):  # Apply propagator only with specified `probability`.
            ind.loss = None  # Initialize individual's loss attribute.
            # Replace traits in 1st parent with values of 2nd parent with Boltzmann probability.
            for k in inds[1].keys():
                if self.rng.random() > fraction:
                    ind[k] = inds[1][k]
        return ind  # Return offspring.


class SelectBest(Propagator):
    """
    Select specified number of best performing individuals as evaluated by their losses.
    """

    def __init__(self, offspring, rng=None):
        """
        Constructor of SelectBest class.

        Parameters
        ----------
        offspring : int
                    number of offsprings (individuals to be selected)
        """
        super(SelectBest, self).__init__(-1, offspring)

    def __call__(self, inds):
        """
        Apply elitist-selection propagator.

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
                f"Has to have at least {self.offspring} individuals to select the {self.offspring} best ones."
            )
        # Sort elements of given iterable in specific order + return as list.
        return sorted(inds, key=lambda ind: ind.loss)[
            : self.offspring
        ]  # Return `self.offspring` best individuals in terms of loss.


class SelectWorst(Propagator):
    """
    Select specified number of worst performing individuals as evaluated by their losses.
    """

    def __init__(self, offspring, rng=None):
        """
        Constructor of SelectBest class.

        Parameters
        ----------
        offspring : int
                    number of offsprings (individuals to be selected)
        """
        super(SelectWorst, self).__init__(-1, offspring)

    def __call__(self, inds):
        """
        Apply anti-elitist-selection propagator.

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
        Constructor of SelectRandom class.

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
                    ind[limit] = self.rng.randrange(*self.limits[limit])
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

import numpy as np
"""
Bibliography:

individuals := x_k for k = 1,...,lambda
number of offsprings := lambda
translation := y_k for k = 1,..., lambda
weights := w_i for i = 1,..., lambda (recombination weights)
step_size := sigma
learning rate_mean := c_m
learning rate rone := c_1
learning rate rmu := c_mü
number of selected offsprings := mü
decay rate step := c_sigma
decay rate co := c_c
variance effective selection mass := mü_eff
damping factor := d_sigma
step of distribution mean := y_w
evolution path of step := p_sigma
evolution path of covariance matrix adaption := p_c
"""


class CMAParam:
    # if not all parameters will be used consider @property decorator style to reduce unnecessary computation
    def __init__(self, mean, sigma, lamb, mu, co_matrix, b_matrix, d_matrix, problem_dimension: int, weights,
                 p_sigma, c_sigma, mu_eff, d_sigma, p_c, c_c, generation, c_1, c_mu, limits):
        # TODO make inits without parameters here instead of propagator
        self.limits = limits
        self.mean = mean
        self.old_mean = None
        self.sigma = sigma
        self.lamb = lamb
        self.mu = mu
        self.co_matrix = co_matrix
        self.problem_dimension = problem_dimension
        self.b_matrix = b_matrix
        self.d_matrix = d_matrix
        self.co_inv_sqrt = b_matrix @ np.diag(d_matrix[:, 0]**-1) @ b_matrix.T
        self.weights = weights
        #self.c_m = c_m
        self.p_sigma = p_sigma
        self.c_sigma = c_sigma
        self.mu_eff = mu_eff
        self.d_sigma = d_sigma
        self.p_c = p_c
        self.c_c = c_c
        # TODO unnötig
        self.generation = generation
        self.eigen_eval = 0
        self.count_eval = 0
        self.problem_dimension = problem_dimension
        self.c_1 = c_1
        self.c_mu = c_mu
        self.chiN = problem_dimension**0.5 * (1 - 1. / (4 * problem_dimension) + 1. / (21 * problem_dimension**2))

    #TODO: Getter and Setter

    def set_mean(self, new_mean):
        self.old_mean = self.mean
        self.mean = new_mean

    def set_p_sigma(self, new_p_sigma):
        self.p_sigma = new_p_sigma

    def set_p_c(self, new_p_c):
        self.p_c = new_p_c

    def set_sigma(self, new_sigma):
        self.sigma = new_sigma

    def set_co_matrix(self, new_co_matrix):
        self.co_matrix = new_co_matrix
        # Update b and d matrix and co_inv_sqrt only after certain number of evaluations to ensure 0(n^2)
        if self.count_eval - self.eigen_eval > self.lamb / (self.c_1 + self.c_mu) / self.problem_dimension / 10:
            self.eigen_eval = self.count_eval
            c = np.triu(new_co_matrix) + np.triu(new_co_matrix, 1).T  # Enforce symmetry
            d, self.b_matrix = np.linalg.eig(c)  # Eigen decomposition
            self.d_matrix = np.sqrt(d)  # Replace eigenvalues with standard deviations
            self.co_inv_sqrt = self.b_matrix @ np.diag(self.d_matrix**(-1)) @ self.b_matrix.T

    """
    # TODO: store y?
    def compute_translation_of(self, inds):
        # create translations of the individuals
        y = np.zeros((self.problem_dimension, len(inds)))
        for k in range(len(inds)):
            for i, (dim, _) in enumerate(self.limits.items()):
                # translate array of individuals (dicts) to y
                y[i][k] = inds[k][dim] - inds[k]["mean"][i]  # TODO verify working should substract mean
        y /= self.sigma
        return y
    """
from abc import ABC, abstractmethod


# Abstract base class
class CMAAdapter(ABC):
    @abstractmethod
    def update_mean(self, params: CMAParam, arx):
        pass

    def update_step_size(self, par: CMAParam):
        new_p_sigma = (1 - par.c_sigma) * par.p_sigma + np.sqrt(
            par.c_sigma * (2 - par.c_sigma) * par.mu_eff) * par.co_inv_sqrt @ (par.mean - par.old_mean) / par.sigma
        par.set_p_sigma(new_p_sigma)
        par.set_sigma(par.sigma * np.exp((par.c_sigma / par.d_sigma) * (
                np.linalg.norm(par.p_sigma, ord=2) / par.chiN - 1)))

    @abstractmethod
    def update_covariance_matrix(self, par: CMAParam, arx):
        pass

    @abstractmethod
    def compute_weights(self, mu, lamb, problem_dimension):
        # TODO assert sum of weights equals 1
        pass

    @staticmethod
    def compute_learning_rates(mu_eff, problem_dimension):
        c_c = (4 + mu_eff / problem_dimension) / (problem_dimension + 4 + 2 * mu_eff / problem_dimension)
        c_1 = 2 / ((problem_dimension + 1.3)**2 + mu_eff)
        # TODO für active + 0.25 hinzfügen also: c_mu = min(1 - c_1, 2 * (0.25 + mu_eff - 2 + (1 / mu_eff)) / ((problem_dimension + 2)**2 + mu_eff))
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + (1 / mu_eff)) / ((problem_dimension + 2)**2 + mu_eff)) # wie in Bi-Population Paper
        return c_c, c_1, c_mu


class BasicCMA(CMAAdapter):
    def compute_weights(self, mu, lamb, problem_dimension):
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)) # TODO in Benchmarking BI-Population Paper macht man nur np.log(mu + 1)
        weights /= np.sum(weights)
        mu_eff = np.sum(weights)**2 / np.sum(weights**2)
        c_c, c_1, c_mu = BasicCMA.compute_learning_rates(mu_eff, problem_dimension)
        return weights, mu_eff, c_c, c_1, c_mu

    def update_mean(self, params: CMAParam, arx):
        # matrix vector multiplication (reshape weights to column vector)
        params.set_mean(arx @ params.weights.reshape(-1, 1))

    def update_covariance_matrix(self, par: CMAParam, arx):
        # turn off rank-one accumulation when sigma increases quickly
        # TODO: statt counteval / lamb generation + 1 nutzen?
        h_sig = np.sum(par.p_sigma**2) / (1 - (1 - par.c_sigma)**(2 * (par.count_eval / par.lamb))) / par.problem_dimension < 2 + 4. / (par.problem_dimension + 1)
        # update evolution path
        new_p_c = (1 - par.c_c) * par.p_c + h_sig * np.sqrt(par.c_c * (2 - par.c_c) * par.mu_eff) * (par.mean - par.old_mean) / par.sigma
        par.set_p_c(new_p_c)
        # TODO: use h_sig to the power of two (unlike in paper) for the variance loss from h_sig
        # TODO: MÖGLICHER FEHLER, im paper wird hier immer xi - m(t) gemacht, aber ist das denn sinvoll mit xi die nicht vom letzen mi abstammen?
        ar_tmp = (1 / par.sigma) * (arx[:, :par.mu] - np.tile(par.old_mean, (1, par.mu)))
        new_co_matrix = (1 - par.c_1 - par.c_mu) * par.co_matrix + par.c_1 * (par.p_c @ par.p_c.T + (1 - h_sig) * par.c_c * (2 - par.c_c) * par.co_matrix) + par.c_mu * ar_tmp @ (par.weights * ar_tmp).T
        # TODO test this with additional c1 (like in paper)
        # new_co_matrix = (1 - par.c1 - par.c_mu) * par.co_matrix + par.c_1 * (par.p_c @ par.p_c.T + (1 - h_sig) * par.c_c * par.c_1 * (2 - par.c_c) * par.co_matrix) + par.c_mu * ar_tmp @ (par.weights * ar_tmp).T
        par.set_co_matrix(new_co_matrix)


class ActiveCMA(CMAAdapter):
    def compute_weights(self, mu, lamb, problem_dimension):
        weights_preliminary = np.log(lamb/2 + 0.5) - np.log(np.arange(1, lamb + 1))
        mu_eff = np.sum(weights_preliminary[:mu])**2 / np.sum(weights_preliminary[:mu]**2)
        c_c, c_1, c_mu = ActiveCMA.compute_learning_rates(mu_eff, problem_dimension)
        # now compute final weights
        mu_eff_minus = np.sum(weights_preliminary[mu:])**2 / np.sum(weights_preliminary[mu:]**2)
        alpha_mu_minus = 1 + c_1 / c_mu
        alpha_mu_eff_minus = 1 + 2 * mu_eff_minus / (mu_eff + 2)
        alpha_pos_def_minus = (1 - c_1 - c_mu) / problem_dimension * c_mu
        weights = weights_preliminary
        weights[:mu] /= np.sum(weights_preliminary[:mu])
        weights[mu:] *= min(alpha_mu_minus, alpha_mu_eff_minus, alpha_pos_def_minus) / np.sum(weights_preliminary[mu:]) * -1
        return weights, mu_eff, c_c, c_1, c_mu

    def update_mean(self, params: CMAParam, arx):
        # matrix vector multiplication (reshape weights to column vector)
        # Only consider positive weights
        params.set_mean(arx @ params.weights[:params.mu].reshape(-1, 1))

    # TODO ACTIVE VS BASIC WEIL ARX SIEHT DANN ANDERS AUS
    def update_covariance_matrix(self, par: CMAParam, arx):
        pass
        # turn off rank-one accumulation when sigma increases quickly
       # h_sig = (np.sum(par.p_sigma ** 2) / (
       #            1 - (1 - par.c_sigma) ** (2 * (par.count_eval / par.lamb))) / par.problem_dimension < 2 + 4. / (
       #                      par.problem_dimension + 1)
       #          # update evolution path
        #         new_p_c = (1 - par.c_c) * par.p_c + h_sig * np.sqrt(par.c_c * (2 - par.c_c) * par.mu_eff) * (
         #                   par.mean - par.old_mean) / par.sigma
        #par.set_p_c(new_p_c)
        # TODO: use h_sig to the power of two (unlike in paper) for the variance loss from h_sig
       # artmp = (1 / par.sigma) * (arx[:, :par.mu] - np.tile(par.old_mean, (1, par.mu)))
        ## weights_circle = [w >= 0 ? 1 : params.problem_dimension / for w in params.weights]

        #updated_co_matrix = (1 - c1_a - params.c_mu * sum(params.weights)) * params.co_matrix + \
         #                   params.c_1 * np.dot(params.p_c, params.p_c.T) + params.c_mu * weights_circle_term
        ## updated_co_matrix = (1 + c1_a - params.c_1 - params.c_mu * sum(params.weights)) * params.co_matrix + params.c_1 * np.dot(params.p_c, params.p_c.T) + params.c_mu * weights_circle_term
        #params.set_co_matrix(updated_co_matrix)


# TODO initiale Population mitgeben
# TODO bestimmte parameter einstellbar machen
# TODO Updatebarkeit von B und D parametriesierbar wurzel der pop size oder so
class CMAPropagator(Propagator):
    def __init__(self, adapter: CMAAdapter, problem_dimension: int, limits, pop_size=None, offsprings=1):
        #self.last_num_inds = 0
        # TODO check input
        self.adapter = adapter
        generation = 0
        lamb = pop_size if pop_size else 4 + int(np.floor(3 * np.log(problem_dimension)))
        super(CMAPropagator, self).__init__(lamb, offsprings)

        # Selection and Recombination params and Covariance Matrix Adaption
        mu = lamb // 2
        self.selectBestMu = SelectBest(mu)
        self.selectBestLambda = SelectBest(lamb)
        weights, mu_eff, c_c, c_1, c_mu = adapter.compute_weights(mu, lamb, problem_dimension)
        #c_m = 1

        # Step-size control params
        c_sigma = (mu_eff + 2) / (problem_dimension + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (problem_dimension + 1)) - 1) + c_sigma

        # Initialize dynamic strategy variables
        p_sigma = np.zeros((problem_dimension, 1))
        p_c = np.zeros((problem_dimension, 1))
        b_matrix = np.eye(problem_dimension)
        d_matrix = np.ones((problem_dimension, 1)) # Diagonal entries responsible for scaling co_matrix
        co_matrix = b_matrix @ np.diag(d_matrix[:, 0] ** 2) @ b_matrix.T # TODO just identiy #TODO do it faster than numpy and consider: Different search intervals ∆si for different variables can be reflected by a different initialization of C, in that the diagonal elements of C obey cii = (∆si)2. However, the ∆si should not disagree by several orders of magnitude. Otherwise a scaling of the variables should be applied.

        # TODO choose mean better than random maybe and consider limits
        #mean = np.array([[np.random.uniform(*limits[limit]) for limit in limits]])
        mean = np.random.rand(problem_dimension, 1)
        # TODO different init step size?
        sigma = 0.2 * ((max(max(limits[i]) for i in limits)) - min(min(limits[i]) for i in limits))

        self.params = CMAParam(mean, sigma, lamb, mu, co_matrix, b_matrix, d_matrix, problem_dimension, weights,
                               p_sigma, c_sigma, mu_eff, d_sigma, p_c, c_c, generation, c_1, c_mu, limits)

    def __call__(self, inds):
        num_inds = len(inds)
        # TODO: möglicherweise falsch count eval so zu erhöhen?
        # add individuals from different workers to eval_count
        self.params.count_eval += num_inds - self.params.count_eval
        # sample new individual
        new_ind = self.__sample_cma()
        # check if len(inds) >= oder < lambda and make sample or sample + update
        if num_inds >= self.params.lamb:
            #self.last_num_inds = len(inds)
            # TODO Wir können hier new_ind nicht direkt evaluieren, daher können wir nur adaptieren ohne das new_ind und es erst im nächsten update berücksichtigen
            # Update mean
            self.adapter.update_mean(self.params, self.__transform_individuals_to_matrix(self.selectBestMu(inds)))
            # Update Covariance Matrix
            self.adapter.update_covariance_matrix(self.params,
                                                  self.__transform_individuals_to_matrix(self.selectBestLambda(inds)))
            # Update step_size
            self.adapter.update_step_size(self.params)
        return new_ind

    # Take bunch of individuals of type individual and transform to numpy matrix for easier subsequent computation
    def __transform_individuals_to_matrix(self, inds):
        arx = np.zeros((self.params.problem_dimension, len(inds)))
        for k, ind in enumerate(inds):
            for i, (dim, _) in enumerate(self.params.limits.items()):
                arx[i, k] = ind[dim]
        return arx

    def __sample_cma(self):
        # TODO par = self.params am Anfang
        # Generate new offspring
        random_vector = np.random.randn(self.params.problem_dimension, 1)
        new_x = self.params.mean + self.params.sigma * self.params.b_matrix @ (self.params.d_matrix * random_vector)
        self.params.count_eval += 1

        # TODO limits? Prüfen ob eingabe Float wie oben?
        new_ind = Individual()

        for i, (dim, _) in enumerate(self.params.limits.items()):
            new_ind[dim] = new_x[i, 0]
        return new_ind
