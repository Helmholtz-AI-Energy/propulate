from .propagators import (
    Conditional,
    Cascade,
    PointMutation,
    MateUniform,
    SelectBest,
    SelectUniform,
    InitUniform,
    IntervalMutationNormal,
)
from .population import Individual


def get_default_propagator(
    pop_size, limits, mate_prob, mut_prob, random_prob, sigma_factor=0.05, rng=None
):
    """
    Get propulate's default propagator.


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
    propagator = Cascade(  # Compose propagator out of basic evolutionary operators with Cascade(...).
        [
            SelectBest(pop_size),
            SelectUniform(2, rng=rng),
            MateUniform(mate_prob, rng=rng),
            PointMutation(limits, probability=mut_prob, rng=rng),
            IntervalMutationNormal(limits, sigma_factor=sigma_factor, probability=1, rng=rng),
            InitUniform(
                limits, parents=1, probability=random_prob, rng=rng
            ),  # TODO this should be put in a "forked" propagator?
        ]
    )

    init = InitUniform(limits, rng=rng)

    propagator = Conditional(
        pop_size, propagator, init
    )  # Initialize random if current population size < specified `pop_size`.
    return propagator


def get_default_propagator_select_random(
    pop_size, limits, mate_prob, mut_prob, random_prob, sigma_factor=0.1, rng=None):
    """
    Get propulate's default propagator.


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
    propagator = Cascade(  # Compose propagator out of basic evolutionary operators with Cascade(...).
        [
            SelectUniform(pop_size, rng=rng),
            SelectUniform(2, rng=rng),
            MateUniform(mate_prob, rng=rng),
            PointMutation(limits, probability=mut_prob, rng=rng),
            IntervalMutationNormal(limits, sigma_factor=sigma_factor, probability=1, rng=rng),
            InitUniform(
                limits, parents=1, probability=random_prob, rng=rng
            ),  # TODO this should be put in a "forked" propagator?
        ]
    )

    init = InitUniform(limits, rng=rng)

    propagator = Conditional(
        pop_size, propagator, init
    )  # Initialize random if current population size < specified `pop_size`.
    return propagator
