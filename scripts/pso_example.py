#!/usr/bin/env python3
import random
import sys

import numpy as np
from mpi4py import MPI

from propulate import Islands
from propulate.propagators import Compose, InitUniform, SelectMin, SelectMax, Conditional, PSOPropagator


############
# SETTINGS #
############

fname = sys.argv[1]  # Get function to optimize from command-line.
NUM_GENERATIONS = 10  # Set number of generations.
POP_SIZE = 2 * MPI.COMM_WORLD.size  # Set size of breeding population.
num_migrants = 1


# BUKIN N.6
# continuous, convex, non-separable, non-differentiable, multimodal
# input domain: -15 <= x <= -5, -3 <= y <= 3
# global minimum 0 at (x, y) = (-10, 1)
def bukin_n6(params):
    x = params["x"]
    y = params["y"]
    return 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)


# EGG CRATE
# continuous, non-convex, separable, differentiable, multimodal
# input domain: -5 <= x, y <= 5
# global minimum -1 at (x, y) = (0, 0)
def egg_crate(params):
    x = params["x"]
    y = params["y"]
    return x**2 + y**2 + 25 * (np.sin(x) ** 2 + np.sin(y) ** 2)


# HIMMELBLAU
# continuous, non-convex, non-separable, differentiable, multimodal
# input domain: -6 <= x, y <= 6
# global minimum 0 at (x, y) = (3, 2)
def himmelblau(params):
    x = params["x"]
    y = params["y"]
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


# KEANE
# continuous, non-convex, non-separable, differentiable, multimodal
# input domain: -10 <= x, y <= 10
# global minimum 0.6736675 at (x, y) = (1.3932491, 0) and (x, y) = (0, 1.3932491)
def keane(params):
    x = params["x"]
    y = params["y"]
    return -np.sin(x - y) ** 2 * np.sin(x + y) ** 2 / np.sqrt(x**2 + y**2)


# LEON
# continous, non-convex, non-separable, differentiable, non-multimodal, non-random, non-parametric
# input domain: 0 <= x, y <= 10
# global minimum 0 at (x, y) =(1, 1)
def leon(params):
    x = params["x"]
    y = params["y"]
    return 100 * (y - x**3) ** 2 + (1 - x) ** 2


# RASTRIGIN
# continuous, non-convex, separable, differentiable, multimodal
# input domain: -5.12 <= x, y <= 5.12
# global minimum -20 at (x, y) = (0, 0)
def rastrigin(params):
    x = params["x"]
    y = params["y"]
    return x**2 - 10 * np.cos(2 * np.pi * x) + y**2 - 10 * np.cos(2 * np.pi * y)


# SCHWEFEL 2.20
# continuous, convex, separable, non-differentiable, non-multimodal
# input domain -100 <= x, y <= 100
# global minimum 0 at (x, y) = (0, 0)
def schwefel(params):
    x = params["x"]
    y = params["y"]
    return np.abs(x) + np.abs(y)


# SPHERE
# continuous, convex, separable, non-differentiable, non-multimodal
# input domain: -5.12 <= x, y <= 5.12
# global minimum 0 at (x, y) = (0, 0)
def sphere(params):
    x = params["x"]
    y = params["y"]
    return x**2 + y**2


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

if __name__ == "__main__":
    # migration_topology = num_migrants*np.ones((4, 4), dtype=int)
    # np.fill_diagonal(migration_topology, 0)

    rng = random.Random(MPI.COMM_WORLD.rank)

    propagator = Compose(
        [
            PSOPropagator(0, 0.7, 0.7, MPI.COMM_WORLD.rank, limits, rng),
            InitUniform(limits, parents=1, probability=0.1, rng=rng)
        ]
    )

    init = InitUniform(limits, rng=rng)
    propagator = Conditional(POP_SIZE, propagator, init)

    islands = Islands(function, propagator, rng, generations=NUM_GENERATIONS, checkpoint_path='./checkpoints/',
                      migration_probability=0)
    islands.evolve(top_n=1, logging_interval=1, DEBUG=2)
