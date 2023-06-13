from typing import Callable

import numpy as np
from numpy import ndarray


class Particle:
    """
    This class resembles a single particle within the particle swarm solving tasks.
    """

    def __init__(self, dimension: tuple[int], target_fn: Callable, position: np.ndarray = None,
                 velocity: np.ndarray = None, seed: int = None):
        """
        Constructor of Particle class.

        Parameters
        ----------
        dimension : tuple[int]
                    A tuple containing information on the dimensionality of the search space.
                    All values that have something to do with the search space, are tested against this value
                    to ensure usability. Only values > 0 are allowed.
        target_fn : callable[np.ndarray -> double]
                    The function given by this parameter is evaluated in each step of the algorithm
        position :  optional np.ndarray of shape `dimension`
                    The initial position of this Particle. Also, the initial value for p_best.
        velocity :  optional np.ndarray of shape `dimension`
                    The initial velocity of this Particle.
        seed :      optional int
                    The random number generator seed. If set, all values of position or velocity that are not set, will
                    be filled with random numbers generated via the given seed.
        """
        assert all((x > 0 for x in dimension))
        assert all((position is None or position.shape == dimension, velocity is None or velocity.shape == dimension))

        self._search_space_dim = dimension
        self._target_fn = target_fn

        self._rng = None
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._position = np.zeros(self._search_space_dim)
        if position is not None:
            self._position = position
        elif self._rng is not None:
            self._position = np.array(self._rng.random(self._search_space_dim))

        self._velocity = np.zeros(self._search_space_dim)
        if velocity is not None:
            self._velocity = velocity
        elif self._rng is not None:
            self._velocity = np.array(self._rng.random(self._search_space_dim))

        self._p_best = self._g_best = self._position

    def update(self, w_k: float, c_1: float, c_2: float, r_1: float, r_2: float) -> tuple[ndarray, ndarray]:
        """
        This method calculates the position and velocity of the particle for the next time step and returns them.
        :param w_k: particle inertia - how strong influences the current v the speed for the particle in next round
        :param c_1: cognitive factor - how good is the particle's brain
        :param c_2: social factor - how strong the particle believes in the swarm
        :param r_1: random factor to c_1
        :param r_2: random factor to c_2
        :return:
        """
        x: np.ndarray = self._position + self._velocity
        v: np.ndarray = w_k * self._velocity + c_1 * r_1 * (self._p_best - self._position) + c_2 * r_2 * (
                self._g_best - self._position)
        return x, v

    def eval(self) -> float:
        """
        Returns the value of the particles target function on the particle's current position.
        """
        return self._target_fn(self._position)

    @property
    def position(self) -> np.ndarray:
        return self._position

    @position.setter
    def position(self, value: np.ndarray) -> None:
        assert value.shape == self._search_space_dim
        self._position = value

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity

    @velocity.setter
    def velocity(self, value: np.ndarray) -> None:
        assert value.shape == self._search_space_dim
        self._velocity = value

    @property
    def g_best(self) -> np.ndarray:
        return self._g_best

    @g_best.setter
    def g_best(self, value: np.ndarray) -> None:
        assert value.shape == self._search_space_dim
        self._g_best = value
