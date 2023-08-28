#!/usr/bin/env python3
import argparse
import os
import random
import sys
from typing import Dict
from mpi4py import MPI

import numpy as np
from propulate import Islands
from propulate.propagators import (
    SelectMin,
    SelectMax,
    CMAPropagator,
    BasicCMA,
    ActiveCMA,
    Propagator,
)
from propulate.utils import get_default_propagator
from propulate.population import Individual


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
    problem_dim = len(ind.values())
    return np.sum(ind[i] ** 2 for i in range(problem_dim))


def bisphere(ind: Individual) -> float:
    """
    Lunacek's double-sphere benchmark function.

    Lunacek, M., Whitley, D., & Sutton, A. (2008, September).
    The impact of global structure on search.
    In International Conference on Parallel Problem Solving from Nature
    (pp. 498-507). Springer, Berlin, Heidelberg.

    This function's landscape structure is the minimum of two quadratic functions, each creating a single funnel in the
    search space. The spheres are placed along the positive search-space diagonal, with the optimal and sub-optimal
    sphere in the middle of the positive and negative quadrant, respectively. Their distance and the barrier's height
    increase with dimensionality, creating a globally non-separable underlying surface.

    Input domain: -5.12 <= x_i <= 5.12, i = 1,...,N
    Global minimum 0 at (x_i)_N = (µ_1)_N with µ_1 = 2.5
    The Propulate paper uses N = 30.

    Parameters
    ----------
    ind: : Individual
             Individual to be evaluated

    Returns
    -------
    float: function value
    """
    params = np.array(list(ind.values()))
    n = len(params)
    d = 1
    s = 1 - np.sqrt(1 / (2 * np.sqrt(n + 20) - 8.2))
    mu1 = 2.5
    mu2 = -np.sqrt((mu1**2 - d) / s)
    return min(np.sum((params - mu1) ** 2), d * n + s * np.sum((params - mu2) ** 2))


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
    problem_dim = len(ind.values())
    return np.sum(
        (1 - ind[i]) ** 2 + 100 * (ind[i + 1] - ind[i] ** 2) ** 2
        for i in range(problem_dim - 1)
    )


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
    problem_dim = len(ind.values())
    return np.abs(np.sum(int(ind[i]) for i in range(problem_dim)) + (5 * problem_dim))


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
    problem_dim = len(ind.values())
    return np.abs(
        np.sum(i * ind[i] ** 4 + np.random.randn() for i in range(problem_dim))
    )


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
    problem_dim = len(ind.values())
    return 10 * problem_dim + np.sum(
        ind[i] ** 2 - 10 * np.cos(2 * np.pi * ind[i]) for i in range(problem_dim)
    )


def birastrigin(ind: Individual) -> float:
    """
    Lunacek's double-Rastrigin benchmark function.

    Lunacek, M., Whitley, D., & Sutton, A. (2008, September).
    The impact of global structure on search.
    In International Conference on Parallel Problem Solving from Nature
    (pp. 498-507). Springer, Berlin, Heidelberg.

    A double-funnel version of Rastrigin. This function isolates global structure as the main difference impacting
    problem difficulty on a well understood test case.

    Input domain: -5.12 <= x_i <= 5.12, i = 1,...,N
    Global minimum 0 at (x_i)_N = (µ_1)_N with µ_1 = 2.5
    The Propulate paper uses N = 30.

    Parameters
    ----------
    ind: : Individual
             Individual to be evaluated

    Returns
    -------
    float: function value
    """
    params = np.array(list(ind.values()))
    n = len(params)
    d = 1
    s = 1 - np.sqrt(1 / (2 * np.sqrt(n + 20) - 8.2))
    mu1 = 2.5
    mu2 = -np.sqrt((mu1**2 - d) / s)
    return min(
        np.sum((params - mu1) ** 2), d * n + s * np.sum((params - mu2) ** 2)
    ) + 10 * np.sum(1 - np.cos(2 * np.pi * (params - mu1)))


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
    problem_dim = len(ind.values())
    return (
        1
        + np.sum([ind[i] ** 2 for i in range(problem_dim)]) / 4000
        - np.prod([np.cos(ind[i] / np.sqrt(i + 1)) for i in range(problem_dim)])
    )


def schwefel(ind: Individual) -> float:
    """
    Schwefel 2.20 function: continuous, convex, separable, non-differentiable, non-multimodal

    This function has a second-best minimum far away from the global optimum.

    Input domain: -500 <= x_i <= 500, i = 1,...,N
    Global minimum 0 at (x_i)_N = (420.968746)_N
    The Propulate paper uses N = 10.

    Parameters
    ----------
    params: dict[str, float]
            function parameters

    Returns
    -------
    float: function value
    """
    v = 418.982887
    params = np.array(list(ind.values()))
    return np.abs(v * len(params) - np.sum(params * np.sin(np.sqrt(np.abs(params)))))


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
            limits[i] = (-600.0, 600.0)
    elif fname == "schwefel":
        function = schwefel
        limits = {}
        for i in range(problem_dimension):
            limits[i] = (-500.0, 500.0)
    elif fname == "sphere":
        function = sphere
        limits = {}
        for i in range(problem_dimension):
            limits[i] = (-5.12, 5.12)
    elif fname == "bisphere":
        function = bisphere
        limits = {}
        for i in range(problem_dimension):
            limits[i] = (-5.12, 5.12)
    elif fname == "birastrigin":
        function = birastrigin
        limits = {}
        for i in range(problem_dimension):
            limits[i] = (-5.12, 5.12)
    else:
        sys.exit("ERROR: Function undefined...exiting")

    return function, limits


def get_propagator(prop: str, limits: Dict) -> Propagator:
    propagator = None

    if prop == "default":
        propagator = get_default_propagator(
            pop_size=2 * MPI.COMM_WORLD.size,
            limits=limits,
            mate_prob=0.7,
            mut_prob=0.4,
            random_prob=0.1,
            rng=rng,
        )
    elif prop == "cmaBasic":
        propagator = CMAPropagator(
            BasicCMA(),
            limits,
            rng,
            exploration=args.exploration,
            select_worst_all_time=args.select_worst_all_time,
            pop_size=args.pop_size,
            pool_size=args.pool_size,
        )
    elif prop == "cmaActive":
        propagator = CMAPropagator(
            ActiveCMA(),
            limits,
            rng,
            exploration=args.exploration,
            select_worst_all_time=args.select_worst_all_time,
            pop_size=args.pop_size,
            pool_size=args.pool_size,
        )
    return propagator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark CMA-ES in Propulate")

    # parser.add_argument("--function", type=str, default="sphere", help="The function to optimize.")
    parser.add_argument(
        "--generation",
        type=int,
        default=100,
        help="The number of generations in the algorithm.",
    )
    parser.add_argument(
        "--pop_size",
        type=int,
        default=None,
        help="The size of the breeding population.",
    )
    parser.add_argument(
        "--num_islands",
        type=int,
        default=4,
        help="The number of isles for the Island model.",
    )
    parser.add_argument(
        "--migration_probability",
        type=float,
        default=0.9,
        help="The probability of migration for each generation.",
    )
    parser.add_argument(
        "--exploration",
        default=False,
        action="store_true",
        help="Whether to update the covariance matrix after each generation",
    )
    parser.add_argument(
        "--select_worst_all_time",
        default=False,
        action="store_true",
        help="Whether to always use the worst individuals of all time in the case of active cma.",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=2,
        help="The number of dimension in the search space.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./checkpoint_data/",
        help="The path for the checkpoint and logging data.",
    )
    parser.add_argument(
        "--pool_size", type=int, default=3, help="The pool size of the cma propagator."
    )
    parser.add_argument(
        "--propagator",
        type=str,
        choices=["default", "cmaBasic", "cmaActive"],
        default="default",
        help="default, cmaBasic or cmaActive",
    )
    parser.add_argument(
        "--pollination",
        default=False,
        action="store_true",
        help="Pollination activated or Migration",
    )

    args = parser.parse_args()

    functions = [
        "rosenbrock",
        "step",
        "quartic",
        "rastrigin",
        "griewank",
        "schwefel",
        "sphere",
        "bisphere",
        "birastrigin",
    ]

    # Set up migration topology.
    num_migrants = 1  # Set number of individuals migrating at once.

    for f in functions:
        for run in range(1, 3):
            print(MPI.COMM_WORLD.rank, f, run)
            rng = random.Random(
                MPI.COMM_WORLD.rank
            )  # Set up separate random number generator for evolutionary optimization process.
            if args.num_islands > 1:
                migration_topology = (
                    num_migrants
                    * np.ones(  # Set up fully connected migration topology.
                        (args.num_islands, args.num_islands), dtype=int
                    )
                )
                np.fill_diagonal(
                    migration_topology, 0
                )  # An island does not send migrants to itself.
            else:
                migration_topology = None
            migration_probability = (
                args.migration_probability if args.num_islands > 1 else 0
            )
            func, limits = get_function_search_space(
                f, args.dimension
            )  # Get callable function and search-space limits from function name.

            checkpoint_dir = f"benchmark/{args.checkpoint_path}_{f}_run{run}_dim{args.dimension}"
            parent_dir = os.path.dirname(checkpoint_dir)

            os.makedirs(parent_dir, exist_ok=True)

            islands = Islands(
                loss_fn=func,  # Function to optimize
                propagator=get_propagator(args.propagator, limits),  # Evolutionary operator
                rng=rng,  # Random number generator
                generations=args.generation,  # Number of generations
                num_islands=args.num_islands,  # Number of separate evolutionary islands
                migration_topology=migration_topology,  # Migration topology
                checkpoint_path=f"benchmark/{args.checkpoint_path}_{f}_run{run}_dim{args.dimension}",  # Path to potentially read checkpoints from and write new checkpoints to
                migration_probability=migration_probability,  # Migration probability
                emigration_propagator=SelectMin,  # Emigration propagator (how to select migrants)
                immigration_propagator=SelectMax,
                # Immigration propagator (only relevant for pollination, how to choose individuals to be replaced by immigrants)
                pollination=args.pollination,  # Pollination or real migration?
            )
            # Run actual optimization.
            islands.evolve(
                top_n=1,
                logging_interval=1,  # Logging interval used for print-outs.
                debug=2,  # Debug / verbosity level
            )
            del islands
            MPI.COMM_WORLD.barrier()
