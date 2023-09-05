"""
This file contains a PSO propagator relying on the standard one but additionally performing velocity clamping.
"""
from random import Random
from typing import Dict, Tuple, Union, List

import numpy as np

from propulate.particle import Particle
from basic_pso import BasicPSO


class VelocityClamping(BasicPSO):
    """
    This propagator implements the Velocity Clamping pso variant.

    In addition to the parameters known from the basic PSO propagator, it features a parameter,
    via which relative values (best between 0 and 1) are passed to the propagator.

    Based on these values, the velocities of the particles are cut down to a reasonable value.
    """

    def __init__(
        self,
        w_k: float,
        c_cognitive: float,
        c_social: float,
        rank: int,
        limits: Dict[str, Tuple[float, float]],
        rng: Random,
        v_limits: Union[float, np.ndarray],
    ):
        """
        Class constructor.
        :param w_k: The particle's inertia factor
        :param c_cognitive: constant cognitive factor to scale p_best with
        :param c_social: constant social factor to scale g_best with
        :param rank: the rank of the worker the propagator is living on in MPI.COMM_WORLD
        :param limits: a dict with str keys and 2-tuples of floats associated to each of them
        :param rng: random number generator :param v_limits: a numpy array containing values that work as relative caps
                    for their corresponding search space dimensions.
                    If this is a float instead, it does its job for all axes.
        """
        super().__init__(w_k, c_cognitive, c_social, rank, limits, rng)
        x_min, x_max = self.laa
        x_range = np.abs(x_max - x_min)
        if v_limits < 0:
            v_limits *= -1
        self.v_cap: np.ndarray = np.array([-v_limits * x_range, v_limits * x_range])

    def __call__(self, particles: List[Particle]) -> Particle:
        old_p, p_best, g_best = self._prepare_data(particles)

        new_velocity: np.ndarray = (
            self.w_k * old_p.velocity
            + self.rng.uniform(0, self.c_cognitive) * (p_best.position - old_p.position)
            + self.rng.uniform(0, self.c_social) * (g_best.position - old_p.position)
        ).clip(*self.v_cap)
        new_position: np.ndarray = old_p.position + new_velocity

        return self._make_new_particle(new_position, new_velocity, old_p.generation + 1)
