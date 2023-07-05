"""
This file contains all sort of more or less useful stuff.
"""
from typing import Iterable

import numpy as np

from ap_pso import Particle


def get_dummy(shape: int | Iterable | tuple[int]) -> Particle:
    """
    Returns a dummy particle that is just for age comparisons

    Parameters
    ----------
    shape : The dimension(s) of the search space, as used to define numpy arrays.
    """
    values = np.zeros(shape=shape)
    return Particle(values, values, iteration=-1)
