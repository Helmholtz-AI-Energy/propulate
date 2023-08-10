"""
This file contains the first stateful PSO propagator for Propulate.
"""
from random import Random

import numpy as np

from ap_pso import Particle, make_particle
from propulate.propagators import Propagator


class BasicPSOPropagator(Propagator):

    def __init__(self, w_k: float, c_cognitive: float, c_social: float, rank: int,
                 limits: dict[str, tuple[float, float]], rng: Random):
        """
        Class constructor.
        :param w_k: The learning rate ... somehow
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
        self.laa = np.array(list(limits.values())).T

    def __call__(self, particles: list[Particle]) -> Particle:
        if len(particles) < self.offspring:
            raise ValueError("Not enough Particles")
        own_p = [x for x in particles if x.rank == self.rank]
        old_p = Particle(iteration=-1)
        for y in own_p:
            if y.generation > old_p.generation:
                old_p = y
        if not isinstance(old_p, Particle):
            old_p = make_particle(old_p)
            print(f"R{self.rank}, Iteration#{old_p.generation}: Type Error.")
        g_best = min(particles, key=lambda p: p.loss)
        p_best = min(own_p, key=lambda p: p.loss)
        new_velocity = self.w_k * old_p.velocity \
                       + self.c_cognitive * self.rng.uniform(*self.laa) * (p_best.position - old_p.position) \
                       + self.c_social * self.rng.uniform(*self.laa) * (g_best.position - old_p.position)
        new_position = old_p.position + new_velocity

        new_p = Particle(new_position, new_velocity, old_p.generation + 1, self.rank)
        for i, k in enumerate(self.limits):
            new_p[k] = new_p.position[i]
        return new_p
