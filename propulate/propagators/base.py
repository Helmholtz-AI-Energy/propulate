import copy
import random
from typing import List, Dict, Union, Tuple

import numpy as np
from abc import ABC, abstractmethod

from ..population import Individual


def _check_compatible(out1: int, in2: int) -> bool:
    """
    Check compatibility of two propagators for stacking them together sequentially with ``Compose``.

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
            While this abstract base class method actually returns ``None``, each concrete child class of ``Propagator``
            should return an ``Individual`` instance or a list of them.

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
                f"Not enough propagators given ({len(propagators)}). At least 1 is required."
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
        ValueError
            If more individuals than put in shall be selected.
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
        ValueError
            If more individuals than put in shall be selected.
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
