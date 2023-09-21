from typing import List

import numpy as np

from .constriction import Constriction
from ...population import Individual, Particle


class Canonical(Constriction):
    """
    This propagator subclass features a combination of constriction PSO and velocity clamping.

    The velocity clamping uses with a clamping factor of 1, the constriction is done as in the parental ``Constriction``
    propagator.

    For information on the method parameters, please refer to the ``Constriction`` propagator.

    Original publications: Poli, R., Kennedy, J. & Blackwell, T. Particle swarm optimization. Swarm Intell 1, 33–57 (2007). https://doi.org/10.1007/s11721-007-0002-0
    R. C. Eberhart and Y. Shi, "Comparing inertia weights and constriction factors in particle swarm optimization," Proceedings of the 2000 Congress on Evolutionary Computation. CEC00 (Cat. No.00TH8512), La Jolla, CA, USA, 2000, pp. 84-88 vol.1, doi: 10.1109/CEC.2000.870279.
    """

    def __init__(self, c_cognitive, c_social, rank, limits, rng):
        """
        The class constructor.

        In theory, it should be of no problem to hand over numpy arrays instead of the float hyperparameters cognitive
        and social factor.
        Please note that in this case, you are on your own to ensure that the dimension of the passed arrays fits to the
        search domain.

        Parameters
        ----------
        c_cognitive : float
                      Constant cognitive factor to scale the distance to the particle's personal best value with
        c_social : float
                   Constant social factor to scale the distance to the swarm's global best value with
        rank : int
               The global rank of the worker the propagator is living on
        limits : Dict[str, Tuple[float, float]]
                 A dict with str keys and 2-tuples of floats associated to each of them. It describes the borders of
                 the search domain.
        rng : random.Random
              Random number generator for said non-linearity
        """
        super().__init__(c_cognitive, c_social, rank, limits, rng)
        x_min, x_max = self.limits_as_array
        x_range = np.abs(x_max - x_min)
        self.v_cap: np.ndarray = np.array([-x_range, x_range])

    def __call__(self, individuals: List[Individual]) -> Particle:
        """
        Applies the canonical PSO variant update rule.

        Returns a Particle object that contains the updated values of the youngest passed Particle or Individual that
        belongs to the worker the propagator is living on.

        Parameters
        ----------
        individuals: List[Individual]
                     A list of individuals that must at least contain one individual that belongs to the propagator.
                     This list is used to calculate personal and global best of the particle and the swarm and to
                     then update the particle based on the retrieved results.
                     Individuals that cannot be used as Particle class objects are copied to Particles before going on.

        Returns
        -------
        propulate.population.Particle
            An updated Particle.
        """
        # Abuse Constriction's update rule, so I don't have to rewrite it.
        victim = super().__call__(individuals)

        # Set new position and speed.
        v = victim.velocity.clip(*self.v_cap)
        p = victim.position - victim.velocity + v

        # create and return new particle.
        return self._make_new_particle(p, v, victim.generation)
