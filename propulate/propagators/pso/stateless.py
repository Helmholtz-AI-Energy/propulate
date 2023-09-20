"""
This file contains the first prototype of a propagator that runs PSO on Propulate.
"""

from random import Random
from typing import Dict, Tuple, List

from ..propagators import Propagator
from ...population import Individual


class Stateless(Propagator):
    """
    The first draft of a pso propagator. It uses the infrastructure brought to you by vanilla Propulate and nothing more.

    Thus, it won't deliver that interesting results.

    This propagator works on Propulate's Individual-class objects.
    """

    def __init__(
        self,
        w_k: float,
        c_cognitive: float,
        c_social: float,
        rank: int,
        limits: Dict[str, Tuple[float, float]],
        rng: Random,
    ):
        """

        :param w_k: The learning rate ... somehow - currently without effect
        :param c_cognitive: constant cognitive factor to scale p_best with
        :param c_social: constant social factor to scale g_best with
        :param rank: the rank of the worker the propagator is living on in MPI.COMM_WORLD
        :param limits: a dict with str keys and 2-tuples of floats associated to each of them
        :param rng: random number generator
        """
        super().__init__(parents=-1, offspring=1)
        self.c_social = c_social
        self.c_cognitive = c_cognitive
        self.w_k = w_k
        self.rank = rank
        self.limits = limits
        self.rng = rng

    def __call__(self, particles: List[Individual]) -> Individual:
        if len(particles) < self.offspring:
            raise ValueError("Not enough Particles")
        own_p = [x for x in particles if x.rank == self.rank]
        if len(own_p) > 0:
            old_p = max(own_p, key=lambda p: p.generation)
        else:  # No own particle found in given parameters, thus creating new one.
            old_p = Individual(0, self.rank)
            for k in self.limits:
                old_p[k] = self.rng.uniform(*self.limits[k])
            return old_p
        g_best = min(particles, key=lambda p: p.loss)
        p_best = min(own_p, key=lambda p: p.loss)
        new_p = Individual(generation=old_p.generation + 1)
        for k in self.limits:
            new_p[k] = (
                old_p[k]
                + self.rng.uniform(0, self.c_cognitive) * (p_best[k] - old_p[k])
                + self.rng.uniform(0, self.c_social) * (g_best[k] - old_p[k])
            )
        return new_p
