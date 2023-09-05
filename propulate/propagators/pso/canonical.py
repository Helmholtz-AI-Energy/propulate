from typing import List

import numpy as np

from propulate.particle import Particle
from constriction import Constriction


class Canonical(Constriction):
    """
    This propagator subclass features a combination of constriction and velocity clamping.

    The velocity clamping uses a clamping factor of 1,
    the constriction is done as in the parental ``Constriction`` propagator.
    """

    def __init__(self, c_cognitive, c_social, rank, limits, rng):
        super().__init__(c_cognitive, c_social, rank, limits, rng)
        x_min, x_max = self.laa
        x_range = np.abs(x_max - x_min)
        self.v_cap: np.ndarray = np.array([-x_range, x_range])

    def __call__(self, particles: List[Particle]):
        # Abuse Constriction's update rule so I don't have to rewrite it.
        victim = super().__call__(particles)

        # Set new position and speed.
        v = victim.velocity.clip(*self.v_cap)
        p = victim.position - victim.velocity + v

        # create and return new particle.
        return self._make_new_particle(p, v, victim.generation)
