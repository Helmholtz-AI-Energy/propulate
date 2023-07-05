"""
This file contains all sort of more or less useful stuff.
"""
from typing import Iterable

import numpy as np

from ap_pso import Particle
from propulate.population import Individual


def make_particle(individual: Individual) -> Particle:
    """
    Makes particles out of individuals.

    Parameters
    ----------
    individual : An Individual that needs to be a particle
    """
    p = Particle(iteration=individual.generation)
    p.position = np.zeros(len(individual))
    p.velocity = np.zeros(len(individual))
    for i, k in enumerate(individual):
        p.position[i] = individual[k]
        p[k] = individual[k]
    return p
