"""
This file contains the Particle class, an extension of Propulate's Individual class.
"""
import numpy as np

from .individual import Individual


class Particle(Individual):
    """
    Child class of ``Individual`` with additional properties required for PSO, i.e., an array-type velocity field and
    a (redundant) array-type position field.

    Note that Propulate relies on ``Individual``s being ``dict``s.

    When defining new propagators, users of the ``Particle`` class thus need to ensure that a ``Particle``'s position
    always matches its dict contents and vice versa.

    This class also contains an attribute field called ``global_rank``. It contains the global rank of the propagator
    that
    created it.
    This is for purposes of better (or at all) retrieval in multi swarm case.
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
        self.global_rank = rank  # The global rank of the creating propagator for later retrieval upon update.
