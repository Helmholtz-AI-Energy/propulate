from typing import Callable

import numpy as np
from numpy import ndarray

from ap_pso.utils import ExtendedPosition


class Particle:
    """
    This class resembles a single particle within the particle swarm solving tasks.
    """

    def __init__(self, position: np.ndarray, velocity: np.ndarray):
        assert position.shape == velocity.shape

        self.position = position
        self.velocity = velocity
        self.loss: float = None
        self.p_best: ExtendedPosition = None

    def update_p_best(self) -> None:
        if self.loss is None:
            return
        if self.p_best is None or self.loss < self.p_best.loss:
            self.p_best = ExtendedPosition(self.position, self.loss)
