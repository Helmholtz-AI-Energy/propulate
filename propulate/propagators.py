import copy
import random
from typing import List, Dict, Union, Tuple

import numpy

from .population import Individual


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
        inds: list[propulate.population.Individual]
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
        inds: list[propulate.population.Individual]
              input individuals the propagator is applied to

        Returns
        -------
        list[propulate.population.Individual]
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
        inds: list[propulate.population.Individual]
              input individuals the propagator is applied to

        Returns
        -------
        list[propulate.population.Individual]
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
        ind: propulate.population.Individual
             individual the propagator is applied to

        Returns
        -------
        propulate.population.Individual
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
        ind: propulate.population.Individual
             input individual the propagator is applied to

        Returns
        -------
        propulate.population.Individual
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
        inds: List[propulate.population.Individual]
              individuals the propagator is applied to

        Returns
        -------
        propulate.population.Individual
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
        inds: list[propulate.population.Individual]
              individuals the propagator is applied to

        Returns
        -------
        propulate.population.Individual
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
        inds: list[propulate.population.Individual]
              individuals the propagator is applied to

        Returns
        -------
        propulate.population.Individual
            possibly cross-bred individual after application of propagator
        """
        ind = copy.deepcopy(inds[0])  # Consider 1st parent.
        ind.loss = None  # Initialize individual's loss attribute.
        if inds[0].loss <= inds[1].loss:
            delta = inds[0].loss - inds[1].loss
            fraction = 1 / (1 + numpy.exp(-delta / self.temperature))
        else:
            delta = inds[1].loss - inds[0].loss
            fraction = 1 - 1 / (1 + numpy.exp(-delta / self.temperature))

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
        inds: list[propulate.population.Individual]
              input individuals the propagator is applied to

        Returns
        -------
        list[propulate.population.Individual]
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
        inds: list[propulate.population.Individual]
              individuals the propagator is applied to

        Returns
        -------
        list[propulate.population.Individual]
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
        inds: list[propulate.population.Individual]
              individuals the propagator is applied to

        Returns
        -------
        list[propulate.population.Individual]
            selected individuals after application of propagator

        Raises
        ------
        ValueError: If more individuals than put in shall be selected.
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
        inds: propulate.population.Individual
              individuals the propagator is applied to

        Returns
        -------
        propulate.population.Individual
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
