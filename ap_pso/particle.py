"""
This file contains the Particle class, an extension of Propulate's Individual class.
"""
import numpy as np

from propulate.population import Individual


class Particle(Individual):
    """
    Extension of Individual class with additional properties necessary for full PSO.
    It also comes along with a numpy array to store positional information in.
    As Propulate rather relies on Individuals being dicts and using this property to work with, it is just for future use.

    Please keep in mind, that users of this class are responsible to ensure, that a Particle's position always
    matches their dict contents and vice versa.
    """

    def __init__(self,
                 position: np.ndarray = None,
                 velocity: np.ndarray = None,
                 iteration: int = None,
                 rank: int = None
                 ):
        super().__init__(generation=iteration, rank=rank)
        if position is not None and velocity is not None:
            assert position.shape == velocity.shape
        self.velocity = velocity
        self.position = position
