#!/usr/bin/env python3
import random
import sys

import numpy as np
from mpi4py import MPI

from propulate import Propulator, Pollinator
from propulate.utils import get_default_propagator


def bukin_n6(params):
    """
    Bukin N.6 function: continuous, convex, non-separable, non-differentiable, multimodal

    Input domain: -15 <= x <= -5, -3 <= y <= 3
    Global minimum 0: (x, y) = (-10, 1)
    Params
    ------
    params : dict
             function parameters
    Returns
    -------
    float : function value
    """
    x = params["x"]
    y = params["y"]
    return 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)

def egg_crate(params):
    """
    Egg-crate: continuous, non-convex, separable, differentiable, multimodal
    
    Input domain: -5 <= x, y <= 5
    Global minimum -1 at (x, y) = (0, 0)
    Params
    ------
    params : dict
             function parameters
    Returns
    -------
    float : function value
    """
    x = params["x"]
    y = params["y"]
    return x**2 + y**2 + 25 * (np.sin(x) ** 2 + np.sin(y) ** 2)

def himmelblau(params):
    """
    Himmelblau: continuous, non-convex, non-separable, differentiable, multimodal
    
    Input domain: -6 <= x, y <= 6
    Global minimum 0 at (x, y) = (3, 2)
    Params
    ------
    params : dict
             function parameters
    Returns
    -------
    float : function value
    """
    x = params["x"]
    y = params["y"]
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

def keane(params):
    """
    Keane: continuous, non-convex, non-separable, differentiable, multimodal
    
    Input domain: -10 <= x, y <= 10
    Global minimum 0.6736675 at (x, y) = (1.3932491, 0) and (x, y) = (0, 1.3932491)
    Params
    ------
    params : dict
             function parameters
    Returns
    -------
    float : function value
    """
    x = params["x"]
    y = params["y"]
    return -np.sin(x - y) ** 2 * np.sin(x + y) ** 2 / np.sqrt(x**2 + y**2)

def leon(params):
    """
    Leon: continous, non-convex, non-separable, differentiable, non-multimodal, non-random, non-parametric
    
    Input domain: 0 <= x, y <= 10
    Global minimum 0 at (x, y) =(1, 1)
    Params
    ------
    params : dict
             function parameters
    Returns
    -------
    float : function value
    """
    x = params["x"]
    y = params["y"]
    return 100 * (y - x**3) ** 2 + (1 - x) ** 2

def rastrigin(params):
    """
    Rastrigin: continuous, non-convex, separable, differentiable, multimodal

    Input domain: -5.12 <= x, y <= 5.12
    Global minimum -20 at (x, y) = (0, 0)
    Params
    ------
    params : dict
             function parameters
    Returns
    -------
    float : function value
    """
    x = params["x"]
    y = params["y"]
    return x**2 - 10 * np.cos(2 * np.pi * x) + y**2 - 10 * np.cos(2 * np.pi * y)

def schwefel(params):
    """
    Schwefel 2.20: continuous, convex, separable, non-differentiable, non-multimodal

    Input domain: -100 <= x, y <= 100
    Global minimum 0 at (x, y) = (0, 0)
    Params
    ------
    params : dict
             function parameters
    Returns
    -------
    float : function value
    """
    x = params["x"]
    y = params["y"]
    return np.abs(x) + np.abs(y)

def sphere(params):
    """
    Sphere function: continuous, convex, separable, differentiable, unimodal

    Input domain: -5.12 <= x, y <= 5.12
    Global minimum 0 at (x, y) = (0, 0)
    Params
    ------
    params : dict
             function parameters
    Returns
    -------
    float : function value
    """
    x = params["x"]
    y = params["y"]
    return x**2 + y**2

def get_function_search_space(fname):
    """
    Get search space limits and function from function name.
    
    Params
    ------
    fname : str
            function name
    
    Returns
    -------
    callable : function
    dict: search space
    """
    if fname == "bukin":
        function = bukin_n6
        limits = {
            "x": (-15.0, -5.0),
            "y": (-3.0, 3.0),
        }
    elif fname == "eggcrate":
        function = egg_crate
        limits = {
            "x": (-5.0, 5.0),
            "y": (-5.0, 5.0),
        }
    elif fname == "himmelblau":
        function = himmelblau
        limits = {
            "x": (-6.0, 6.0),
            "y": (-6.0, 6.0),
        }
    elif fname == "keane":
        function = keane
        limits = {
            "x": (-10.0, 10.0),
            "y": (-10.0, 10.0),
        }
    elif fname == "leon":
        function = leon
        limits = {
            "x": (0.0, 10.0),
            "y": (0.0, 10.0),
        }
    elif fname == "rastrigin":
        function = rastrigin
        limits = {
            "x": (-5.12, 5.12),
            "y": (-5.12, 5.12),
        }
    elif fname == "schwefel":
        function = schwefel
        limits = {
            "x": (-100.0, 100.0),
            "y": (-100.0, 100.0),
        }
    elif fname == "sphere":
        function = sphere
        limits = {
            "x": (-5.12, 5.12),
            "y": (-5.12, 5.12),
        }
    else:
        sys.exit("ERROR: Function undefined...exiting")

    return function, limits


if __name__ == "__main__":
    fname = sys.argv[1]                                 # Get function name to optimize from command-line.
    generations = 10                                    # Set number of generations.
    pop_size = 2 * MPI.COMM_WORLD.size                  # Set size of breeding population.
    checkpoint_path = "./"                              # Path for possibly loading checkpoints from and writing new checkpoints to.
    DEBUG = 2                                           # Set verbosity / debug level.
    rng = random.Random(MPI.COMM_WORLD.rank)            # Set up separate random number generator for evolutionary optimization process.

    function, limits = get_function_search_space(fname) # Get callable function and search-space limits from function name.

    # Set up evolutionary operator.
    propagator = get_default_propagator(                # Get default evolutionary operator.
            pop_size=pop_size,                          # Breeding population size
            limits=limits,                              # Search-space limits
            mate_prob=0.7,                              # Crossover probability
            mut_prob=0.4,                               # Mutation probability
            random_prob=0.1,                            # Random-initialization probability
            rng=rng                                     # Random number generator    
        )

    # Set up propulator performing actual optimization.
    propulator = Propulator(
        function,                                       # Function to optimize
        propagator,                                     # Evolutionary operator
        comm=MPI.COMM_WORLD,                            # Communicator
        generations=generations,                        # Number of generations
        checkpoint_path=checkpoint_path,                # Path for checkpointing
        rng=rng,                                        # Random number generator
    )
    
    # Run actual optimization and print summary of results.
    propulator.propulate(logging_interval=1, DEBUG=DEBUG)
    propulator.summarize(top_n=2, DEBUG=DEBUG)
