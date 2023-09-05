"""
This file contains a Propagator subclass providing constriction-flavoured pso.
"""
from random import Random
from typing import List, Dict, Tuple

import numpy as np

from basic_pso import BasicPSO
from propulate.particle import Particle


class Constriction(BasicPSO):
    """
    This propagator subclass features Constriction PSO as proposed by Clerc and Kennedy in 2002.

    Instead of an inertia factor that affects the old velocity value within the velocity update,
    there is a constriction factor, that is applied on the new velocity `after' the update.

    This propagator runs on Particle-class objects.
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
        Class constructor.
        Important note: `c_cognitive` and `c_social` have to sum up to something greater than 4!
        :param c_cognitive: constant cognitive factor to scale p_best with
        :param c_social: constant social factor to scale g_best with
        :param rank: the rank of the worker the propagator is living on in MPI.COMM_WORLD
        :param limits: a dict with str keys and 2-tuples of floats associated to each of them
        :param rng: random number generator
        """
        assert c_cognitive + c_social > 4, "c_cognitive + c_social < 4!"
        phi: float = c_cognitive + c_social
        chi: float = 2.0 / (phi - 2.0 + np.sqrt(phi * (phi - 4.0)))
        super().__init__(chi, c_cognitive, c_social, rank, limits, rng)

    def __call__(self, particles: List[Particle]) -> Particle:
        old_p, p_best, g_best = self._prepare_data(particles)

        new_velocity = self.w_k * (
            old_p.velocity
            + self.rng.uniform(0, self.c_cognitive) * (p_best.position - old_p.position)
            + self.rng.uniform(0, self.c_social) * (g_best.position - old_p.position)
        )
        new_position = old_p.position + new_velocity

        return self._make_new_particle(new_position, new_velocity, old_p.generation + 1)
