# -*- coding: utf-8 -*-
import random
from typing import Dict, Union, Tuple

from .propagators import (
    Compose,
    Conditional,
    InitUniform,
    IntervalMutationNormal,
    MateUniform,
    PointMutation,
    Propagator,
    SelectMin,
    SelectUniform,
)


def get_default_propagator(
    pop_size: int,
    limits: Union[Dict[str, Tuple[float, float]], Dict[str, Tuple[int, int]], Dict[str, Tuple[str, ...]]],
    mate_prob: float,
    mut_prob: float,
    random_prob: float,
    sigma_factor: float = 0.05,
    rng: random.Random = None
) -> Propagator:
    """
    Get Propulate's default evolutionary optimization propagator.

    Parameters
    ----------
    pop_size: int
              number of individuals in breeding population
    limits: dict
            (hyper-)parameters to be optimized, i.e., search space
    mate_prob: float
               uniform-crossover probability
    mut_prob: float
              point-mutation probability
    random_prob: float
                 random-initialization probability
    sigma_factor: float
                  scaling factor for obtaining std from search-space boundaries for interval mutation
    rng: random.Random
         random number generator

    Returns
    -------
    propagators.Propagator: A basic evolutionary optimization propagator.
    """
    if any(isinstance(limits[x][0], float) for x in limits):  # Check for existence of at least one continuous trait.
        propagator = Compose([  # Compose propagator out of basic evolutionary operators with Compose(...).
            SelectMin(pop_size),
            SelectUniform(offspring=2, rng=rng),
            MateUniform(mate_prob, rng=rng),
            PointMutation(limits, probability=mut_prob, rng=rng),
            IntervalMutationNormal(limits, sigma_factor=sigma_factor, probability=1.0, rng=rng),
            InitUniform(limits, parents=1, probability=random_prob, rng=rng)
        ])
    else:
        propagator = Compose([  # Compose propagator out of basic evolutionary operators with Compose(...).
            SelectMin(pop_size),
            SelectUniform(offspring=2, rng=rng),
            MateUniform(mate_prob, rng=rng),
            PointMutation(limits, probability=mut_prob, rng=rng),
            InitUniform(limits, parents=1, probability=random_prob, rng=rng),
        ])

    init = InitUniform(limits, rng=rng)
    propagator = Conditional(pop_size, propagator, init)  # Initialize random if population size < specified `pop_size`.

    return propagator
