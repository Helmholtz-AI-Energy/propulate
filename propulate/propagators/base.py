import random
from typing import Dict, List, Mapping, MutableMapping, Optional, Tuple, Union

import numpy as np

from ..population import Individual


def _check_compatible(out1: int, in2: int) -> bool:
    """
    Check compatibility of two propagators for stacking them together sequentially with ``Compose``.

    Parameters
    ----------
    out1: int
        The number of output individuals returned by the first propagator.
    in2: int
        The number of input individuals taken by the second propagator.

    Returns
    -------
    bool
        True if the input propagators can be stacked: False if not.
    """
    return out1 == in2 or in2 == -1


class Propagator:
    """
    Abstract base class for all propagators, i.e., evolutionary operators.

    A propagator takes a collection of individuals and uses them to breed a new collection of individuals.

    Attributes
    ----------
    offspring : int
        The number of output individuals to breed.
    parents : int
        The number of input individuals to use as parents.
    rng : random.Random
        The separate random number generator for the Propulate optimization.

    Methods
    -------
    __call__()
        Apply the propagator.
    """

    def __init__(
        self, parents: int = 0, offspring: int = 0, rng: Optional[random.Random] = None
    ) -> None:
        """
        Initialize a propagator with given parameters.

        Parameters
        ----------
        parents : int, optional
            The number of input individuals (-1 for any). Default is 0 for abstract base class.
        offspring : int, optional
            The number of output individuals to breed. Default is 0 for abstract base class.
        rng : random.Random, optional
            The separate random number generator for the Propulate optimization.

        Raises
        ------
        ValueError
            If the number of offspring to breed is zero.
        """
        if offspring == 0:
            raise ValueError("Propagator has to sire more than 0 offspring.")
        self.offspring = offspring  # Number of offspring individuals to breed
        self.parents = parents  # Number of parent individuals
        if rng is None:
            rng = random.Random()
        self.rng = rng  # Random number generator

    def __call__(self, inds: List[Individual]) -> Union[List[Individual], Individual]:
        """
        Apply the propagator (not implemented for abstract base class).

        Parameters
        ----------
        inds : List[propulate.population.Individual]
            The input individuals the propagator is applied to.

        Returns
        -------
        List[propulate.population.Individual] | propulate.population.Individual
            The individual(s) bred by applying the propagator.
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

    Attributes
    ----------
    probability : float
        The probability of applying the propagator.

    Notes
    -----
    The ``Stochastic`` class inherits all methods and attributes from the ``Propagator`` class.

    See Also
    --------
    :class:`Propagator` : The parent class.
    """

    def __init__(
        self,
        parents: int = 0,
        offspring: int = 0,
        probability: float = 1.0,
        rng: Optional[random.Random] = random.Random(),
    ) -> None:
        """
        Initialize a stochastic propagator that is only applied with a specified probability.

        Parameters
        ----------
        parents : int, optional
            The number of input individuals (-1 for any). Default is 0.
        offspring : int, optional
            The number of output individuals. Default is 0.
        probability : float, optional
            The probability of application. Default is 0.0.
        rng : random.Random, optional
            The separate random number generator for the Propulate optimization.

        Raises
        ------
        ValueError
            If the number of offspring to breed is zero.
        """
        super().__init__(parents, offspring, rng)
        self.probability = probability
        if offspring == 0:
            raise ValueError("Propagator has to sire more than 0 offspring.")


class Conditional(Propagator):
    """
    Apply different propagators depending on whether the breeding population is already large enough or not.

    If the population consists of the specified number of individuals required for breeding (or more),
    a different propagator is applied than if not.

    Attributes
    ----------
    pop_size : int
        The breeding population size.
    true_prop : propulate.propagators.Propagator
        The propagator applied if the current population's size is greater than ``pop_size``.
    false_prop : propulate.propagators.Propagator
        The propagator applied if the current population's size equals at least ``pop_size``.

    Notes
    -----
    The ``Conditional`` class inherits all methods and attributes from the ``Propagator`` class.

    See Also
    --------
    :class:`Propagator` : The parent class.
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
        Initialize a conditional propagator.

        Parameters
        ----------
        pop_size : int
            The breeding population size.
        true_prop : propulate.propagators.Propagator
            The propagator applied if the current population's size equals at least ``pop_size``.
        false_prop : propulate.propagators.Propagator
            The propagator applied if the current population's size is less than ``pop_size``.
        parents : int, optional
            The number of input individuals (-1 for any). Default is -1.
        offspring : int
            The number of output individuals to breed. Default is -1.
        """
        super().__init__(parents, offspring)
        self.pop_size = pop_size
        self.true_prop = true_prop
        self.false_prop = false_prop

    def __call__(self, inds: List[Individual]) -> Union[List[Individual], Individual]:
        """
        Apply conditional propagator.

        Parameters
        ----------
        inds : List[propulate.population.Individual]
            The input individuals the propagator is applied to.

        Returns
        -------
        List[propulate.population.Individual]
            The output individuals returned by the conditional propagator.
        """
        if (
            len(inds) >= self.pop_size
        ):  # If number of evaluated individuals >= `pop_size`, apply `true_prop`.
            return self.true_prop(inds)
        else:  # Else apply `false_prop`.
            return self.false_prop(inds)


class Compose(Propagator):
    """
    Stack propagators together sequentially for successive application.

    Attributes
    ----------
    propagators : List[propulate.propagators.Propagator]
        The propagators to be stacked together.

    Notes
    -----
    The ``Compose`` class inherits all methods and attributes from the ``Propagator`` class.

    See Also
    --------
    :class:`Propagator` : The parent class.
    """

    def __init__(self, propagators: List[Propagator]) -> None:
        """
        Initialize a composed propagator.

        Parameters
        ----------
        propagators : List[propulate.propagators.Propagator]
            The propagators to be stacked together sequentially.

        Raises
        ------
        ValueError
            If the propagators to stack are incompatible in terms of number of input and output individuals.
        """
        if len(propagators) < 1:
            raise ValueError(
                f"Not enough propagators given ({len(propagators)}). At least 1 is required."
            )
        super().__init__(propagators[0].parents, propagators[-1].offspring)
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

    def __call__(self, inds: List[Individual]) -> Union[List[Individual], Individual]:
        """
        Apply the composed propagator.

        Parameters
        ----------
        inds : List[propulate.population.Individual]
            The input individuals the propagator is applied to.

        Returns
        -------
        List[propulate.population.Individual]
            The output individuals after application of the propagator.
        """
        for p in self.propagators:
            inds = p(inds)  # type: ignore
        return inds


class SelectMin(Propagator):
    """
    Select a specified number of best performing individuals in terms of their losses, i.e., with the smallest losses.

    Notes
    -----
    The ``SelectMin`` class inherits all methods and attributes from the ``Propagator`` class.

    See Also
    --------
    :class:`Propagator` : The parent class.
    """

    def __init__(
        self,
        offspring: int,
    ) -> None:
        """
        Initialize an elitist selection propagator.

        Parameters
        ----------
        offspring : int
            The number of offspring (individuals to be selected).
        """
        super().__init__(-1, offspring)

    def __call__(self, inds: List[Individual]) -> List[Individual]:
        """
        Apply the elitist-selection propagator.

        Parameters
        ----------
        inds : List[propulate.population.Individual]
            The input individuals the propagator is applied to.

        Returns
        -------
        List[propulate.population.Individual]
            The selected output individuals after application of the propagator.

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
        return sorted(inds, key=lambda ind: float(ind.loss))[
            : self.offspring
        ]  # Return `self.offspring` best individuals in terms of loss.


class SelectMax(Propagator):
    """
    Select a specified number of worst performing individuals in terms of their losses, i.e., with the greatest losses.

    Notes
    -----
    The ``SelectMax`` class inherits all methods and attributes from the ``Propagator`` class.

    See Also
    --------
    :class:`Propagator` : The parent class.
    """

    def __init__(
        self,
        offspring: int,
    ) -> None:
        """
        Initialize an anti-elitist propagator.

        Parameters
        ----------
        offspring : int
            The number of offspring (individuals to be selected).
        """
        super().__init__(-1, offspring)

    def __call__(self, inds: List[Individual]) -> List[Individual]:
        """
        Apply the anti-elitist-selection propagator.

        Parameters
        ----------
        inds : List[propulate.population.Individual]
            The individuals the propagator is applied to.

        Returns
        -------
        List[propulate.population.Individual]
            The selected individuals after application of the propagator.

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
    Select a specified number of individuals randomly.

    Notes
    -----
    The ``SelectUniform`` class inherits all methods and attributes from the ``Propagator`` class.

    See Also
    --------
    :class:`Propagator` : The parent class.
    """

    def __init__(self, offspring: int, rng: Optional[random.Random] = None) -> None:
        """
        Initialize a random-selection propagator.

        Parameters
        ----------
        offspring : int
            The number of offspring (individuals to be selected).
        rng : random.Random, optional
            The separate random number generator for the Propulate optimization.
        """
        super().__init__(-1, offspring, rng)

    def __call__(self, inds: List[Individual]) -> List[Individual]:
        """
        Apply the uniform-selection propagator.

        Parameters
        ----------
        inds : List[propulate.population.Individual]
            The individuals the propagator is applied to.

        Returns
        -------
        List[propulate.population.Individual]
            The selected individuals after application of the propagator.

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


class InitUniform(Stochastic):
    """
    Initialize an individual by uniformly sampling the specified limits for each trait.

    Attributes
    ----------
    limits : Dict[str, Tuple[float, float]] | Dict[str, Tuple[int, int]] | Dict[str, Tuple[str, ...]]
        The search space, i.e., the limits of (hyper-)parameters to be optimized.

    Notes
    -----
    The ``InitUniform`` class inherits all methods and attributes from the ``Stochastic`` class.

    See Also
    --------
    :class:`Stochastic` : The parent class.
    """

    def __init__(
        self,
        limits: Mapping[
            str, Union[Tuple[float, float], Tuple[int, int], Tuple[str, ...]]
        ],
        parents: int = 0,
        probability: float = 1.0,
        rng: Optional[random.Random] = random.Random(),
    ) -> None:
        """
        Initialize a random-initialization propagator.

        Parameters
        ----------
        limits : Dict[str, Tuple[float, float]] | Dict[str, Tuple[int, int]] | Dict[str, Tuple[str, ...]]
            The search space, i.e., the limits of (hyper-)parameters to be optimized.
        parents : int, optional
            The number of parents. Default is 0.
        probability : float, optional
            The probability of application. Default is 1.0.
        rng : random.Random, optional
            The separate random number generator for the Propulate optimization.
        """
        super().__init__(parents, 1, probability, rng)
        self.limits = limits

    def __call__(self, *inds: Individual) -> Individual:  # type: ignore[override]
        """
        Apply the uniform-initialization propagator.

        Parameters
        ----------
        inds : propulate.population.Individual
            The individuals the propagator is applied to.

        Returns
        -------
        propulate.population.Individual
            The output individual after application of the propagator.

        Raises
        ------
        ValueError
            If a parameter's type is invalid, i.e., not float (continuous), int (ordinal), or str (categorical).
        """
        position: MutableMapping[str, Union[int, float, str]] = {}
        if (
            self.rng.random() < self.probability
        ):  # Apply only with specified probability.
            for (
                limit
            ) in self.limits:  # Randomly sample from specified limits for each trait.
                if isinstance(
                    self.limits[limit][0], int
                ):  # If ordinal trait of type integer.
                    position[limit] = self.rng.randint(*self.limits[limit])
                elif isinstance(
                    self.limits[limit][0], float
                ):  # If interval trait of type float.
                    position[limit] = self.rng.uniform(*self.limits[limit])
                elif isinstance(
                    self.limits[limit][0], str
                ):  # If categorical trait of type string.
                    position[limit] = str(self.rng.choice(self.limits[limit]))
                else:
                    raise ValueError(
                        "Unknown type of limits. Has to be float for interval, "
                        "int for ordinal, or string for categorical."
                    )
            ind = Individual(position, self.limits)  # Instantiate new individual.
        else:  # Return first input individual w/o changes otherwise.
            ind = inds[0]
        return ind


class Gaussian(Propagator):
    """Sample a new individual from a multivariate gaussian distribution around an initial point."""

    def __init__(
        self,
        limits: Dict[str, Tuple[float, float]],
        scale: float,
        rng: np.random.Generator,
    ):
        """
        Initialize Gaussian propagator.

        Parameters
        ----------
        limits : Dict[str, Tuple[float, float]] | Dict[str, Tuple[int, int]] | Dict[str, Tuple[str, ...]]
            The search space, i.e., limits of (hyper-)parameters to be optimized.
        scale : float
            The standard deviation of the Gaussian distribution.
        rng : random.Random
            The separate random number generator for the Propulate optimization.

        """
        super().__init__(1, 1)
        self.limits = limits
        self.rng: np.random.Generator = rng  # type:ignore
        self.scale = scale

    def __call__(self, inds: List[Individual]) -> Individual:
        """
        Apply the Gaussian propagator.

        Parameters
        ----------
        inds : propulate.population.Individual
            The individuals the propagator is applied to.

        Returns
        -------
        propulate.population.Individual
            The output individual after application of the propagator.
        """
        position = np.array(inds[0].position)
        position += self.rng.normal(scale=self.scale, size=position.shape)
        return Individual(position, self.limits)
