from ap_pso import Particle
from ap_pso.propagators import ConstrictionPropagator


class CanonicalPropagator(ConstrictionPropagator):
    def __call__(self, particles: list[Particle]):
        # Abuse Constriction's update rule so I don't have to rewrite it.
        victim = super().__call__(particles)

        # Set new position and speed.
        v = victim.velocity.clip(*self.laa)
        p = victim.position - victim.velocity + v

        # create and return new particle.
        return self._make_new_particle(p, v, victim.generation)
