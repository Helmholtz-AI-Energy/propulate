"""
This file contains the original (stateful) PSO propagator for Propulate.
"""
import logging
from random import Random
from typing import Dict, Tuple, List

import numpy as np

from ..propagators import Propagator
from ...population import Particle, Individual
from ...utils import make_particle


class Basic(Propagator):
    """
    This propagator implements the most basic PSO variant one possibly could think of.

    It features an inertia factor applied to the old velocity in the velocity update,
    a social and a cognitive factor.

    With the help of the random number generator required as creation parameter, non-linearity is added to the particle
    update in order to not collapse to linear regression.

    This basic PSO propagator can only explore real-valued search spaces, i.e., continuous parameters.
    It works on ``Particle`` objects and serves as the foundation of all other PSO propagators.
    Further PSO propagators should be derived from this propagator or from one that is derived from this.

    This variant was first proposed in Y. Shi and R. Eberhart. “A modified particle swarm optimizer”, 1998,
    https://doi.org/10.1109/ICEC.1998.699146
    """

    def __init__(
        self,
        inertia: float,
        c_cognitive: float,
        c_social: float,
        rank: int,
        limits: Dict[str, Tuple[float, float]],
        rng: Random,
    ):
        """
        The class constructor.

        In theory, it should be of no problem to hand over numpy arrays instead of the float hyperparameters inertia,
        cognitive and social factor.
        Please note that in this case, you are on your own to ensure that the dimension of the passed arrays fits to the
        search domain.

        Parameters
        ----------
        inertia : float
                  The inertia weight.
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
        super().__init__(parents=-1, offspring=1)
        self.c_social = c_social
        self.c_cognitive = c_cognitive
        self.inertia = inertia
        self.rank = rank
        self.limits = limits
        self.rng = rng
        self.limits_as_array: np.ndarray = np.array(list(limits.values())).T

    def __call__(self, individuals: List[Individual]) -> Particle:
        """
        Applies the standard PSO update rule with inertia.

        Returns a Particle object that contains the updated values
        of the youngest passed Particle or Individual that belongs to the worker the propagator is living on.

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
        )
        new_position: np.ndarray = old_p.position + new_velocity

        return self._make_new_particle(new_position, new_velocity, old_p.generation + 1)

    def _prepare_data(
        self, individuals: List[Individual]
    ) -> Tuple[Particle, Particle, Particle]:
        """
        This method prepares the passed list of Individuals, that hopefully are Particles.
        If they are not, they are copied over to Particle objects to avoid handling issues.

        Parameters
        ----------
        individuals : List[Individual]
                      A list of Individual objects that shall be used as data basis for a PSO update step

        Returns
        -------
        Tuple[propulate.population.Particle, propulate.population.Particle, propulate.population.Particle]
            The following particles in this very order:

            1.  old_p: the current particle to be updated now
            2.  p_best: the personal best value of this particle
            3.  g_best: the global best value currently known
        """
        if len(individuals) < self.offspring:
            raise ValueError("Not enough Particles")

        particles = []
        for individual in individuals:
            if isinstance(individual, Particle):
                particles.append(individual)
            else:
                particles.append(make_particle(individual))
                logging.warning(
                    f"Got Individual instead of Particle. If this is on purpose, you can ignore this warning. "
                    f"Converted the Individual to Particle. Continuing."
                )

        own_p = [
            x
            for x in particles
            if (isinstance(x, Particle) and x.global_rank == self.rank)
            or x.rank == self.rank
        ]
        if len(own_p) > 0:
            old_p: Individual = max(own_p, key=lambda p: p.generation)
            if not isinstance(old_p, Particle):
                old_p = make_particle(old_p)

        else:
            victim = max(particles, key=lambda p: p.generation)
            old_p = self._make_new_particle(
                victim.position, victim.velocity, victim.generation
            )

        g_best = min(particles, key=lambda p: p.loss)
        p_best = min(own_p, key=lambda p: p.loss)

        return old_p, p_best, g_best

    def _make_new_particle(
        self, position: np.ndarray, velocity: np.ndarray, generation: int
    ) -> Particle:
        """
        Takes the necessary data to create a new Particle with the position dict set to the correct values.

        Parameters
        ----------
        position : np.ndarray
                   An array containing the position of the particle to be created
        velocity : np.ndarray
                   An array containing the velocity of the particle to be created
        generation : int
                     The generation of the new particle

        Returns
        -------
        propulate.population.Particle
            The newly created Particle object that results from the PSO update.
        """
        new_p = Particle(position, velocity, generation, self.rank)
        for i, k in enumerate(self.limits):
            new_p[k] = new_p.position[i]
        return new_p
