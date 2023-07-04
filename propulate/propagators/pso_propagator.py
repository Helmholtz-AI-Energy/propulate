from random import Random

from propulate.population import Individual

from propulate.propagators import Propagator


class PSOPropagator(Propagator):

    def __init__(self, w_k: float, c_cognitive: float, c_social: float, rank: int,
                 limits: dict[str, tuple[float, float]], rng: Random):
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

    def __call__(self, particles: list[Individual]) -> Individual:
        if len(particles) < self.offspring:
            raise ValueError("Not enough Particles")
        own_p = [x for x in particles if x.rank == self.rank]
        old_p = Individual(generation=-1)
        for y in own_p:
            if y.generation > old_p.generation:
                old_p = y
        g_best = sorted(particles, key=lambda p: p.loss)[0]
        p_best = sorted(own_p, key=lambda p: p.loss)[0]
        new_p = Individual(generation=old_p.generation + 1)
        for k in self.limits:
            new_p[k] = self.c_cognitive * self.rng.uniform(*self.limits[k]) * (p_best[k] - old_p[k]) \
                        + self.c_social * self.rng.uniform(*self.limits[k]) * (g_best[k] - old_p[k])
        return new_p
