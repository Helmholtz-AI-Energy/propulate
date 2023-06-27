"""
This file contains some random util functions, as, for example, get_default_propagator
"""
from random import Random


def get_default_propagator(pop_size: int, limits: dict, mate_prob: float, mut_prob: float, random_prob: float,
                           sigma_factor: float = 0.05, rng: Random = None):
    """
    Returns a generic, but working propagator to use on Swarm objects in order to update the particles.


    Parameters
    ----------
    pop_size : int
               number of individuals in breeding population
    limits : dict
    mate_prob : float
                uniform-crossover probability
    mut_prob : float
               point-mutation probability
    random_prob : float
                  random-initialization probability
    sigma_factor : float
                   scaling factor for obtaining std from search-space boundaries for interval mutation
    rng : random.Random()
          random number generator
    """
    _ = (pop_size, limits, mate_prob, mut_prob, random_prob, sigma_factor, rng)
    pass


class ExtendedPosition:

    def __init__(self, position: np.ndarray, loss: float):
        self.position = position
        self.loss = loss


TELL_TAG = 0
