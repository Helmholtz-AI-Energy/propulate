"""
This file contains a Propagator subclass providing constriction-flavoured pso.
"""
from random import Random
from typing import List, Dict, Tuple

import numpy as np

from .basic import Basic
from ...population import Individual, Particle


class Constriction(Basic):
    """
    This propagator subclass features constriction PSO as proposed by Clerc and Kennedy in 2002.
    
    Original publication: Poli, R., Kennedy, J. & Blackwell, T. Particle swarm optimization. Swarm Intell 1, 33–57 (2007). https://doi.org/10.1007/s11721-007-0002-0

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

    def __call__(self, individuals: List[Individual]) -> Particle:
        """
        Applies the constriction PSO update rule.

        Returns a Particle object that contains the updated values of the youngest passed Particle or Individual that
        belongs to the worker the propagator is living on.

        Parameters
        ----------
        individuals: List[Individual]
                     A list of individuals that must at least contain one individual that belongs to the propagator.
                     This list is used to calculate personal and global best of the particle and the swarm and to
                     then update the particle based on the retrieved results.
                     Individuals that cannot be used as Particle class objects are copied to particles before going on.

        Returns
        -------
        propulate.population.Particle
            An updated Particle.
        """
        old_p, p_best, g_best = self._prepare_data(individuals)

        new_velocity = self.inertia * (
            old_p.velocity
            + self.rng.uniform(0, self.c_cognitive) * (p_best.position - old_p.position)
            + self.rng.uniform(0, self.c_social) * (g_best.position - old_p.position)
        )
        new_position = old_p.position + new_velocity

        return self._make_new_particle(new_position, new_velocity, old_p.generation + 1)
