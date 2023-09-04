from typing import List

from propulate.particle import Particle
from propulate.population import Individual
from propulate.propagators import Compose


class PSOCompose(Compose):
    def __call__(self, particles: List[Particle]) -> Particle:
        """
        Returns the first element of the list of particles returned by the last Propagator in the list
        input upon creation of the object.

        This behaviour should change in near future, so that a list of Particles is returned,
        with hopefully only one member.
        """
        for p in self.propagators:
            tmp = p(particles)
            if isinstance(tmp, Individual):
                particles = [tmp]
            else:
                particles = tmp
        return particles[0]
