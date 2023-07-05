from random import Random

import numpy as np
from particle import Particle

from propulate.population import Individual
from propulate.propagators import Stochastic


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
        if (self.rng.random() < self.probability):  # Apply only with specified `probability`.
            ind = Individual()  # Instantiate new individual.
            for limit in self.limits:
                # Randomly sample from specified limits for each trait.
                if (type(self.limits[limit][0]) == int):  # If ordinal trait of type integer.
                    ind[limit] = self.rng.randrange(*self.limits[limit])
                elif (type(self.limits[limit][0]) == float):  # If interval trait of type float.
                    ind[limit] = self.rng.uniform(*self.limits[limit])
                elif (type(self.limits[limit][0]) == str):  # If categorical trait of type string.
                    ind[limit] = self.rng.choice(self.limits[limit])
                else:
                    raise ValueError(
                        "Unknown type of limits. Has to be float for interval, int for ordinal, or string for "
                        "categorical.")
            return ind
        else:
            ind = inds[0]
            return ind  # Return 1st input individual w/o changes.


class PSOInitUniform(Stochastic):
    """
    Initialize individuals by uniformly sampling specified limits for each trait.
    """

    def __init__(self, limits: dict[str, tuple[float]], parents=0, probability=1.0, rng: Random = None):
        """
        Constructor of PSOInitUniform class.

        In case of parents > 0 and probability < 1., call returns input individual without change.

        Parameters
        ----------
        limits : dict
                 a named list of tuples representing the limits in which the search space resides and where
                 solutions can be expected to be found.
                 Limits of (hyper-)parameters to be optimized
        parents : int
                  number of input individuals (-1 for any)
        probability : float
                    the probability with which a completely new individual is created
        rng : random.Random()
              random number generator
        """
        super().__init__(parents, 1, probability, rng)
        self.limits = limits

    def __call__(self, *particles: list[Individual]):
        """
        Apply uniform-initialization propagator.

        Parameters
        ----------
        particles : list of propulate.population.Individual objects
               individuals the propagator is applied to

        Returns
        -------
        ind : propulate.population.Individual
              list of selected individuals after application of propagator
        """
        if self.rng.random() < self.probability:  # Apply only with specified `probability`.
            dim = len(self.limits)
            position = np.zeros(shape=dim)
            velocity = np.zeros(shape=dim)

            particle = Particle(position, velocity)  # Instantiate new particle.

            for index, limit in zip(range(dim),
                                    self.limits):  # Since Py 3.7, iterating over dicts is stable, so we can do the following.
                if type(self.limits[limit][0]) != float:  # Check search space for validity
                    raise TypeError("PSO only works on continuous search spaces!")

                # Randomly sample from specified limits for each trait.
                pos_on_limit = self.rng.uniform(*self.limits[limit])
                particle.position[index] = pos_on_limit
                particle[limit] = pos_on_limit
                particle.velocity[index] = self.rng.uniform(*self.limits[limit])

            return particle
        else:
            particle = particles[0]
            return particle  # Return 1st input individual w/o changes.
