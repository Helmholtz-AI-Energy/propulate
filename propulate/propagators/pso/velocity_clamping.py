"""
This file contains a PSO propagator relying on the standard one but additionally performing velocity clamping.
"""
from random import Random
from typing import Dict, Tuple, Union, List

import numpy as np

from .basic import Basic
from ...population import Individual, Particle


class VelocityClamping(Basic):
    """
    This propagator implements velocity clamping PSO.

    In addition to the parameters known from the basic PSO
    propagator, it features a clamping factor within [0, 1] used to determine each parameter's maximum velocity value
    relative to its search-space limits.

    Based on these values, the velocities of the particles are cut down to a
    reasonable value.
    """

    def __init__(
        self,
        inertia: float,
        c_cognitive: float,
        c_social: float,
        rank: int,
        limits: Dict[str, Tuple[float, float]],
        rng: Random,
        v_limits: Union[float, np.ndarray],
    ):
        """
        The class constructor.

        Parameters
        ----------
        inertia : float
                  The particle's inertia factor
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
        v_limits : Union[float, np.ndarray]
                   This parameter is multiplied with the clamping limit in order to reduce it further in most cases
                   (this is, when the value is in (0; 1)).
                   If this parameter has float type, it is applied to all dimensions of the search domain, else,
                   each of its elements are applied to their corresponding dimension of the search domain.
        """
        super().__init__(inertia, c_cognitive, c_social, rank, limits, rng)
        x_min, x_max = self.limits_as_array
        x_range = abs(x_max - x_min)
        v_limits = abs(v_limits)
        self.v_cap: np.ndarray = np.array([-v_limits * x_range, v_limits * x_range])

    def __call__(self, individuals: List[Individual]) -> Particle:
        """
        Applies the standard PSO update rule with inertia, extended by cutting off too high velocities.

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

        new_velocity: np.ndarray = (
            self.inertia * old_p.velocity
            + self.rng.uniform(0, self.c_cognitive) * (p_best.position - old_p.position)
            + self.rng.uniform(0, self.c_social) * (g_best.position - old_p.position)
        ).clip(*self.v_cap)
        new_position: np.ndarray = old_p.position + new_velocity

        return self._make_new_particle(new_position, new_velocity, old_p.generation + 1)
