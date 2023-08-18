#!/usr/bin/env python3
import argparse
import random
import sys

from mpi4py import MPI

import numpy as np
from propulate import Islands
from propulate.propagators import SelectMin, SelectMax, CMAPropagator, BasicCMA, ActiveCMA
from propulate.utils import get_default_propagator
from propulate.population import Individual

# TODO Bisphere, birastrigin


def sphere(ind: Individual):
    """
    Sphere function: continuous, convex, separable, differentiable, unimodal

    Input domain: -5.12 <= xi <= 5.12
    Global minimum 0 at xi = 0 for all i in [1,N]
    Params
    ------
    ind: : Individual
             Individual to be evaluated
    Returns
    -------
    float : loss value
    """
    return np.sum(ind[i] ** 2 for i in range(ind.problem_dim))


def rosenbrock(ind: Individual):
    """
    Rosenbrock TODO function: continuous, convex, separable, differentiable, unimodal

    Input domain: -5 <= xi <= 10 TODO: Propulate paper ist es +- 2.048
    Global minimum 0 at xi = 1 for all i in [1,N]
    Params
    ------
    ind: : Individual
             Individual to be evaluated
    Returns
    -------
    float : loss value
    """
    return np.sum((1 - ind[i]) ** 2 + 100 * (ind[i + 1] - ind[i] ** 2) ** 2 for i in range(ind.problem_dim - 1))


def step(ind: Individual):
    """
    Step TODO function: continuous, convex, separable, differentiable, unimodal

    Input domain: -5.12 <= xi <= 5.12
    Global minimum -5*N at xi <= -5 for all i in [1,N]
    Params
    ------
    ind: : Individual
             Individual to be evaluated
    Returns
    -------
    float : loss value
    """
    return np.sum(int(ind[i]) for i in range(ind.problem_dim)) + (5 * ind.problem_dim)


def quartic(ind: Individual):
    """
    Quartic TODO function: continuous, convex, separable, differentiable, unimodal

    Input domain: -1.28 <= xi <= 1.28
    Global minimum f(0,...,0) = Sum Ni(0, 1) for all i in [1,N]
    Params
    ------
    ind: : Individual
             Individual to be evaluated
    Returns
    -------
    float : loss value
    """
    return np.sum(i * ind[i]**4 + np.random.randn() for i in range(ind.problem_dim))


def rastrigin(ind: Individual):
    """
    Rastrigin: continuous, non-convex, separable, differentiable, multimodal

    Input domain: -5.12 <= xi <= 5.12
    Global minimum 0 at xi = 1 for all i in [1,N]
    Params
    ------
    ind: : Individual
             Individual to be evaluated
    Returns
    -------
    float : loss value
    """
    return 10 * ind.problem_dim + np.sum(ind[i]**2 - 10 * np.cos(2 * np.pi * ind[i]) for i in range(ind.problem_dim))


def griewank(ind: Individual):
    """
    Griewank: TODO: continuous, non-convex, separable, differentiable, multimodal

    Input domain: -600 <= xi <= 600
    Global minimum 0 at xi = 0 for all i in [1,N]
    Params
    ------
    ind : Individual
        Individual to be evaluated

    Returns
    -------
    float : loss value
    """
    return 1 + np.sum([ind[i] ** 2 for i in range(ind.problem_dim)]) / 4000 - np.prod([np.cos(ind[i] / np.sqrt(i + 1)) for i in range(ind.problem_dim)])


def schwefel(ind: Individual):
    """
    Schwefel: TODO continuous, convex, separable, non-differentiable, non-multimodal

    Input domain: -500 <= xi <= 500
    Global minimum 0 at xi = 420.968746 for all i in [1,N]
    Params
    ------
    ind : Individual
        Individual to be evaluated

    Returns
    -------
    float : loss value
    """
    return 418.982887 * ind.problem_dim - np.sum(ind[i] * np.sin(np.sqrt(np.abs(ind[i]))) for i in range(ind.problem_dim))


def get_function_search_space(fname, problem_dimension):
    """
    Get search space limits and function from function name.

    Params
    ------
    fname : str
            function name
    problem_dimension: the number of dimensions

    Returns
    -------
    callable : function
    dict: search space
    """
    if fname == "rosenbrock":
        function = rosenbrock
        limits = {}
        for i in range(problem_dimension):
            limits[i] = (-5.0, 10.0)
    elif fname == "step":
        function = step
        limits = {}
        for i in range(problem_dimension):
            limits[i] = (-5.12, 5.12)
    elif fname == "quartic":
        function = quartic
        limits = {}
        for i in range(problem_dimension):
            limits[i] = (-1.28, 1.28)
    elif fname == "rastrigin":
        function = rastrigin
        limits = {}
        for i in range(problem_dimension):
            limits[i] = (-5.12, 5.12)
    elif fname == "griewank":
        function = griewank
        limits = {}
        for i in range(problem_dimension):
            limits[i] = (-600, 600)
    elif fname == "schwefel":
        function = schwefel
        limits = {}
        for i in range(problem_dimension):
            limits[i] = (-500, 500)
    elif fname == "sphere":
        function = sphere
        limits = {}
        for i in range(problem_dimension):
            limits[i] = (-5.12, 5.12)
    else:
        sys.exit("ERROR: Function undefined...exiting")

    return function, limits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark CMA-ES in Propulate")
    parser.add_argument("--function", type=str, default="sphere", help="The function to optimize.")
    parser.add_argument("--generation", type=int, default=100, help="The number of generations in the algorithm.")
    parser.add_argument("--pop_size", type=int, default=None, help="The size of the breeding population.")
    parser.add_argument("--num_isles", type=int, default=2, help="The number of isles for the Island model.")
    parser.add_argument("--migration_probability", type=float, default=0.9, help="The probability of migration for each generation.")
    parser.add_argument("--pollination", type=bool, default=False, help="Whether to use real migration or pollination.")
    parser.add_argument("--default_propulate", type=bool, default=False, help="Whether to use the default propulate propagator.")
    parser.add_argument("--active_cma", type=bool, default=False, help="Whether to use active CMA or the basic algorithm.")
    parser.add_argument("--exploration", type=bool, default=False, help="Whether to update the covariance matrix after each generation")
    parser.add_argument("--select_worst_all_time", type=bool, default=False, help="Whether to always use the worst individuals of all time in the case of active cma.")
    parser.add_argument("--dimension", type=int, default=2, help="The number of dimension in the search space.")
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoint_data/', help="The path for the checkpoint and logging data.")
    parser.add_argument("--pool_size", type=int, default=2, help="The pool size of the cma propagator.")

    args = parser.parse_args()

    # Set up migration topology.
    num_migrants = 1  # Set number of individuals migrating at once.
    migration_topology = num_migrants * np.ones(  # Set up fully connected migration topology.
        (args.num_isles, args.num_isles),
        dtype=int
    )
    np.fill_diagonal(migration_topology, 0)  # An island does not send migrants to itself.

    func, limits = get_function_search_space(
        args.function, args.dimension)  # Get callable function and search-space limits from function name.
    rng = random.Random(
        MPI.COMM_WORLD.rank)  # Set up separate random number generator for evolutionary optimization process.

    # Set up evolutionary operator.
    if args.default_propulate:
        propagator = get_default_propagator(                # Get default evolutionary operator.
                pop_size=4,                          # Breeding population size TODO Andere pop size als cma
                limits=limits,                              # Search-space limits
                mate_prob=0.7,                              # Crossover probability
                mut_prob=0.4,                               # Mutation probability
                random_prob=0.1,                            # Random-initialization probability
                rng=rng                                     # Random number generator
            )
    else:
        adapter = ActiveCMA() if args.active_cma else BasicCMA()
        propagator = CMAPropagator(adapter, limits, rng, exploration=args.exploration, select_worst_all_time=args.select_worst_all_time, pop_size=args.pop_size, pool_size=args.pool_size)

    # Set up island model.
    islands = Islands(
        func,  # Function to optimize
        propagator,  # Evolutionary operator
        rng,  # Random number generator
        generations=args.generation,  # Number of generations
        num_islands=args.num_isles,  # Number of separate evolutionary islands
        migration_topology=migration_topology,          # Migration topology
        checkpoint_path=args.checkpoint_path,  # Path to potentially read checkpoints from and write new checkpoints to
        migration_probability=args.migration_probability,  # Migration probability
        emigration_propagator=SelectMin,  # Emigration propagator (how to select migrants)
        immigration_propagator=SelectMax,
        # Immigration propagator (only relevant for pollination, how to choose individuals to be replaced by immigrants)
        pollination=args.pollination,  # Pollination or real migration?
    )

    # Run actual optimization.
    islands.evolve(
        top_n=1,
        # Top-n best individuals are returned and printed (whole population can be accessed from checkpoint file).
        logging_interval=1,  # Logging interval used for print-outs.
        DEBUG=2  # Debug / verbosity level
    )
