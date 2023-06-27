"""
This file contains the 'abstract base class' for all propagators of this project.
"""
from random import Random

from particle import Particle


class Propagator:
    """
    Abstract base class for all propagators, i.e., evolutionary operators, in Propulate.

    Take a collection of individuals and use them to breed a new collection of individuals.
    """

    def __init__(self, parents: int = 0, offspring: int = 0, rng: Random = None):
        """
        Constructor of Propagator class.

        Parameters
        ----------
        parents : int
                  number of input individuals (-1 for any)
        offspring : int
                    number of output individuals
        rng : random.Random()
              random number generator
        """
        self.parents = parents
        self.rng = rng
        self.offspring = offspring
        if offspring == 0:
            raise ValueError("Propagator has to sire more than 0 offspring.")

    def __call__(self, particle: Particle):
        """
        Apply propagator (not implemented!).

        Parameters
        ----------
        particles: propulate.population.Individual
              individuals the propagator is applied to
        """
        raise NotImplementedError()
