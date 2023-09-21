"""
This file contains a prototype proof-of-concept propagator to run PSO in Propulate.
"""

from random import Random
from typing import Dict, Tuple, List

from ..propagators import Propagator
from ...population import Individual


class Stateless(Propagator):
    """
    This propagator performs PSO without the need of Particles, but as a consequence, also without velocity.
    Thus, it is called stateless.

    As this propagator works without velocity, there is also no inertia weight used.

    It uses only classes provided by vanilla Propulate.
    """

    def __init__(
        self,
        c_cognitive: float,
        c_social: float,
        rank: int,
        limits: Dict[str, Tuple[float, float]],
        rng: Random,
    ):
        """
        The class constructor.

        Parameters
        ----------
        c_cognitive : float
                      Constant cognitive factor to scale individual's personal best value with
        c_social : float
                   Constant social factor to scale swarm's global best value with
        rank : int
               The global rank of the worker the propagator is living on
        limits : Dict[str, Tuple[float, float]
                 A dict with str keys and 2-tuples of floats associated to each of them describing the borders of
                 the search domain.
        rng : random.Random
              The random number generator required for non-linearity of update.
        """
        super().__init__(parents=-1, offspring=1)
        self.c_social = c_social
        self.c_cognitive = c_cognitive
        self.rank = rank
        self.limits = limits
        self.rng = rng

    def __call__(self, individuals: List[Individual]) -> Individual:
        """
        Apply standard PSO update without inertia and old velocity.

        Parameters
        ----------
        individuals : List[Individual]
                      The individual that are used as data basis for the PSO update

        Returns
        -------
        propulate.population.Individual
            An updated Individual
        """
        if len(individuals) < self.offspring:
            raise ValueError("Not enough Particles")
        own_p = [x for x in individuals if x.rank == self.rank]
        if len(own_p) > 0:
            old_p = max(own_p, key=lambda p: p.generation)
        else:  # No own particle found in given parameters, thus creating new one.
            old_p = Individual(0, self.rank)
            for k in self.limits:
                old_p[k] = self.rng.uniform(*self.limits[k])
            return old_p
        g_best = min(individuals, key=lambda p: p.loss)
        p_best = min(own_p, key=lambda p: p.loss)
        new_p = Individual(generation=old_p.generation + 1)
        for k in self.limits:
            new_p[k] = (
                old_p[k]
                + self.rng.uniform(0, self.c_cognitive) * (p_best[k] - old_p[k])
                + self.rng.uniform(0, self.c_social) * (g_best[k] - old_p[k])
            )
        return new_p
