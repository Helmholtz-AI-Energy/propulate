"""
This file contains the 'abstract base class' for all propagators of this project.
"""
from random import Random
from typing import Callable

import numpy as np

from ap_pso.particle import Particle


class Propagator:
    """
    Abstract base class for all propagators, i.e., evolutionary operators, in Propulate.

    Take a collection of individuals and use them to breed a new collection of individuals.
    """

    def __init__(self, loss_fn: Callable[[np.ndarray], float]):
        """
        Constructor of Propagator class.

        Parameters
        ----------
        loss_fn: Callable
            The function to be optimized by the particles. Should take a numpy array and return a float.
        """
        self.loss_fn = loss_fn

    def __call__(self, particle: Particle):
        """
        Apply propagator (not implemented!).

        Parameters
        ----------
        particle: Particle
              The particle on which the propagator shall perform a positional update.
        """
        raise NotImplementedError()

