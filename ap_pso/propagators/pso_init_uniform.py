"""
This file contains propagators, that can be used to initialize a population of either Individuals or Particles.
"""
from random import Random

import numpy as np

from ap_pso import Particle, make_particle
from propulate.population import Individual
from propulate.propagators import Stochastic


class PSOInitUniform(Stochastic):
    """
    Initialize individuals by uniformly sampling specified limits for each trait.
    """

    def __init__(self, limits: dict[str, tuple[float, float]], parents=0, probability=1.0, rng: Random = None, *,
                 v_init_limit: float | np.ndarray = 0.1):
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
        v_init_limit: float | np.ndarray
                      some multiplicative constant to reduce initial random velocity values.
        """
        super().__init__(parents, 1, probability, rng)
        self.limits = limits
        self.laa = np.array(list(limits.values())).T
        if isinstance(v_init_limit, np.ndarray):
            assert v_init_limit.shape[-1] == self.laa.shape[-1]
        self.v_limits = v_init_limit

    def __call__(self, *particles: Individual) -> Particle:
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

            position = self.rng.uniform(*self.laa)
            velocity = self.rng.uniform(*(self.v_limits * self.laa))

            particle = Particle(position, velocity)  # Instantiate new particle.

            for index, limit in enumerate(self.limits):
                # Since Py 3.7, iterating over dicts is stable, so we can do the following.

                if type(self.limits[limit][0]) != float:  # Check search space for validity
                    raise TypeError("PSO only works on continuous search spaces!")

                # Randomly sample from specified limits for each trait.
                particle[limit] = particle.position[index]
            return particle
        else:
            particle = particles[0]
            if isinstance(particle, Particle):
                return particle  # Return 1st input individual w/o changes.
            else:
                return make_particle(particle)
