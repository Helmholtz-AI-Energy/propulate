"""
This file contains the Particle class, an extension of Propulate's Individual class.
"""
import numpy as np

from .individual import Individual


class Particle(Individual):
    """
    Child class of ``Individual`` with additional properties required for PSO, i.e., an array-type velocity field and a (redundant) array-type position field.
    Note that Propulate relies on ``Individual``s being ``dict``s.
    When defining new propagators, users of the ``Particle`` class thus need to ensure that a ``Particle``'s position always matches its dict contents and vice versa.
    """

    def __init__(
        self,
        position: np.ndarray = None,
        velocity: np.ndarray = None,
        generation: int = -1,
        rank: int = -1,
    ):
        super().__init__(generation=generation, rank=rank)
        if position is not None and velocity is not None:
            assert position.shape == velocity.shape
        self.velocity = velocity
        self.position = position
        self.g_rank = (
            rank  # necessary as Propulate splits up the COMM_WORLD communicator
        )
        # which leads to errors with normal rank in multi-island case.
