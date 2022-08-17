import random
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
                ind = indp.parents

                raise ValueError(
                    "Incompatible combination of {} output individuals of {} and {} input individuals of {}".format(
                        outd, outp, ind, inp
                    )
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
                "Too many points to mutate for individual with {} traits".format(
                    len(limits)
                )
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
                "Minimum number of points to mutate must be > 0 but was {}.".format(
                    min_points
                )
            )
        if len(limits) < max_points:
            raise ValueError(
                "Too many points to mutate for individual with {} traits.".format(
                    len(limits)
                )
            )
        if min_points > max_points:
            raise ValueError(
                "Minimum number of traits to mutate must be <= respective maximum number but min_points = {} > {} = max_points.".format(
                    min_points, max_points
                )
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
                "Too many points to mutate for individual with {} interval traits".format(
                    n_interval_traits
                )
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
                "Relative parent contribution must be within (0, 1) but was {}.".format(
                    rel_parent_contrib
                )
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
            fraction = 1 / (1 + numpy.exp(-delta / temperature))
        else:
            delta = inds[1].loss - inds[0].loss
            fraction = 1 - 1 / (1 + numpy.exp(-delta / temperature))

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
                "Has to have at least {} individuals to select the {} best ones.".format(
                    self.offspring, self.offspring
                )
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
                "Has to have at least {} individuals to select the {} worst ones.".format(
                    self.offspring, self.offspring
                )
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
