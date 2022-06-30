#!/usr/bin/env python3
import random
import sys
from propulate import Islands, Propulator
from mpi4py import MPI
from propulate.utils import get_default_propagator
from propulate.propagators import SelectBest, SelectWorst, SelectUniform
import numpy as np

############
# SETTINGS #
############

#fname = sys.argv[1]  # Get function to optimize from command-line.
NUM_GENERATIONS = 1000  # Set number of generations.
POP_SIZE = 2 * MPI.COMM_WORLD.size  # Set size of breeding population.
num_migrants = 1

# Digalakis, J. G., & Margaritis, K. G. (2001). 
# On benchmarking functions for genetic algorithms. 
# International journal of computer mathematics, 77(4), 481-506.
#
# F1: SPHERE
# Use N_dim = 2.
# Limits: -5.12 <= x_i <= 5.12
def sphere(params):
    """Sphere benchmark function."""
    vec = np.array(list(params.values()))
    return np.sum(vec**2)

# F2: ROSENBROCK
# N_dim = 2
# Limits: -2.048 <= x_i <= 2.048
def rosenbrock(params):
    """Rosenbrock benchmark function."""
    vec = np.array(list(params.values()))
    return 100 * (vec[0]**2 - vec[1])**2 + (1 - vec[0])**2

# F3: STEP
# Use N_dim = 5.
# Limits: -5.12 <= x_i <= 5.12
def step(params):
    """Step benchmark function."""
    vec = np.array(list(params.values()))
    return np.sum(vec.astype(int))

# F4: QUARTIC
# Use N_dim = 30.
# Limits: -1.28 <= x_i <= 1.28
def quartic(params):
    """Quartic benchmark function."""
    vec = np.array(list(params.values()))
    idx = np.arange(1, len(vec)+1)
    gauss = np.random.normal(size = len(vec))
    return np.sum(idx * vec + gauss)

# F5: RASTRIGIN
# Use N_dim = 20.
# Limits: -5.12 <= x_i <= 5.12
def rastrigin(params):
    """Rastrigin benchmark function."""
    A = 10
    vec = np.array(list(params.values()))
    return 20*A + np.sum(vec**2 - 10 * np.cos(2 * np.pi * vec))

# F6: GRIEWANGK
# Use N_dim = 10.
# Limits: -600 <= x_i <= 600
def griewangk(params):
    """Griewangk benchmark function."""
    vec = np.array(list(params.values()))
    idx = np.arange(1, len(vec)+1)
    return 1 + np.sum(vec**2 / 4000) - np.prod(vec / np.sqrt(idx))

# F7: SCHWEFEL
# Use N_dim = 10.
# Limits: -500 <= x_i <= 500
def schwefel(params):
    """Schwefel benchmark function."""
    V = 4189.829101
    vec = np.array(list(params.values()))
    return 10 * V + np.sum(-vec * np.sin(np.sqrt(np.abs(vec))))

# Lunacek, M., Whitley, D., & Sutton, A. (2008, September). 
# The impact of global structure on search. 
# In International Conference on Parallel Problem Solving from Nature 
# (pp. 498-507). Springer, Berlin, Heidelberg.

# F8: DOUBLE-SPHERE
# Use N_dim = 30.
# Limits: -5.12 <= x_i <= 5.12
def bisphere(params):
    """Lunacek's double-sphere benchmark function."""
    vec = np.array(list(params.values()))
    N = len(vec)
    d = 1
    s = 1 - np.sqrt(1 / (2 * np.sqrt(N + 20) - 8.2))
    mu1 = 2.5
    mu2 = - np.sqrt( (mu1**2 - d) / s)
    return min(np.sum((vec - mu1)**2), d * N + s * np.sum((vec - mu2)**2) )

# F9: DOUBLE-RASTRIGIN
# Use N_dim = 30.
# Limits: -5.12 <= x <= 5.12 
def birastrigin(params):
    """Lunacek's double-Rastrigin benchmark function."""
    vec = np.array(list(params.values()))
    N = len(vec)
    d = 1
    s = 1 - np.sqrt(1 / (2 * np.sqrt(N + 20) - 8.2))
    mu1 = 2.5
    mu2 = - np.sqrt( (mu1**2 - d) / s)
    return min(np.sum((vec - mu1)**2), d * N + s * np.sum((vec - mu2)**2) ) + 10 * np.sum(1 - np.cos(2 * np.pi * (vec - mu1)))


def get_limits(fname):
    """Determine search-space limits of input benchmark function."""
    if fname == "sphere":
        function = sphere
        limits = {
            "a": (-5.12, 5.12),
            "b": (-5.12, 5.12),
        }
    
    elif fname == "rosenbrock":
        function = rosenbrock
        limits = {
            "a": (-2.048, 2.048),
            "b": (-2.048, 2.048),
        }
    
    elif fname == "step":
        function = step
        limits = {
            "a": (-1.28, 1.28),
            "b": (-1.28, 1.28),
            "c": (-1.28, 1.28),
            "d": (-1.28, 1.28),
            "e": (-1.28, 1.28),
            "f": (-1.28, 1.28),
            "g": (-1.28, 1.28),
            "h": (-1.28, 1.28),
            "i": (-1.28, 1.28),
            "j": (-1.28, 1.28),
            "k": (-1.28, 1.28),
            "l": (-1.28, 1.28),
            "m": (-1.28, 1.28),
            "n": (-1.28, 1.28),
            "o": (-1.28, 1.28),
            "p": (-1.28, 1.28),
            "q": (-1.28, 1.28),
            "r": (-1.28, 1.28),
            "s": (-1.28, 1.28),
            "t": (-1.28, 1.28),
            "u": (-1.28, 1.28),
            "v": (-1.28, 1.28),
            "w": (-1.28, 1.28),
            "x": (-1.28, 1.28),
            "y": (-1.28, 1.28),
            "z": (-1.28, 1.28),
            "A": (-1.28, 1.28),
            "B": (-1.28, 1.28),
            "C": (-1.28, 1.28),
            "D": (-1.28, 1.28)
        }
    
    elif fname == "quartic":
        function = quartic
        limits = {
            "a": (-5.12, 5.12),
            "b": (-5.12, 5.12),
            "c": (-5.12, 5.12),
            "d": (-5.12, 5.12),
            "e": (-5.12, 5.12),
            "f": (-5.12, 5.12),
            "g": (-5.12, 5.12),
            "h": (-5.12, 5.12),
            "i": (-5.12, 5.12),
            "j": (-5.12, 5.12),
            "k": (-5.12, 5.12),
            "l": (-5.12, 5.12),
            "m": (-5.12, 5.12),
            "n": (-5.12, 5.12),
            "o": (-5.12, 5.12),
            "p": (-5.12, 5.12),
            "q": (-5.12, 5.12),
            "r": (-5.12, 5.12),
            "s": (-5.12, 5.12),
            "t": (-5.12, 5.12),
            "u": (-5.12, 5.12),
            "v": (-5.12, 5.12),
            "w": (-5.12, 5.12),
            "x": (-5.12, 5.12),
            "y": (-5.12, 5.12),
            "z": (-5.12, 5.12),
            "A": (-5.12, 5.12),
            "B": (-5.12, 5.12),
            "C": (-5.12, 5.12),
            "D": (-5.12, 5.12)
        }
    
    elif fname == "bisphere":
        function = bisphere
        limits = {
            "a": (-5.12, 5.12),
            "b": (-5.12, 5.12),
            "c": (-5.12, 5.12),
            "d": (-5.12, 5.12),
            "e": (-5.12, 5.12),
            "f": (-5.12, 5.12),
            "g": (-5.12, 5.12),
            "h": (-5.12, 5.12),
            "i": (-5.12, 5.12),
            "j": (-5.12, 5.12),
            "k": (-5.12, 5.12),
            "l": (-5.12, 5.12),
            "m": (-5.12, 5.12),
            "n": (-5.12, 5.12),
            "o": (-5.12, 5.12),
            "p": (-5.12, 5.12),
            "q": (-5.12, 5.12),
            "r": (-5.12, 5.12),
            "s": (-5.12, 5.12),
            "t": (-5.12, 5.12),
            "u": (-5.12, 5.12),
            "v": (-5.12, 5.12),
            "w": (-5.12, 5.12),
            "x": (-5.12, 5.12),
            "y": (-5.12, 5.12),
            "z": (-5.12, 5.12),
            "A": (-5.12, 5.12),
            "B": (-5.12, 5.12),
            "C": (-5.12, 5.12),
            "D": (-5.12, 5.12)
        }
    
    elif fname == "birastrigin":
        function = birastrigin
        limits = {
            "a": (-5.12, 5.12),
            "b": (-5.12, 5.12),
            "c": (-5.12, 5.12),
            "d": (-5.12, 5.12),
            "e": (-5.12, 5.12),
            "f": (-5.12, 5.12),
            "g": (-5.12, 5.12),
            "h": (-5.12, 5.12),
            "i": (-5.12, 5.12),
            "j": (-5.12, 5.12),
            "k": (-5.12, 5.12),
            "l": (-5.12, 5.12),
            "m": (-5.12, 5.12),
            "n": (-5.12, 5.12),
            "o": (-5.12, 5.12),
            "p": (-5.12, 5.12),
            "q": (-5.12, 5.12),
            "r": (-5.12, 5.12),
            "s": (-5.12, 5.12),
            "t": (-5.12, 5.12),
            "u": (-5.12, 5.12),
            "v": (-5.12, 5.12),
            "w": (-5.12, 5.12),
            "x": (-5.12, 5.12),
            "y": (-5.12, 5.12),
            "z": (-5.12, 5.12),
            "A": (-5.12, 5.12),
            "B": (-5.12, 5.12),
            "C": (-5.12, 5.12),
            "D": (-5.12, 5.12)
        }
    
    elif fname == "rastrigin":
        function = rastrigin
        limits = {
            "a": (-5.12, 5.12),
            "b": (-5.12, 5.12),
            "c": (-5.12, 5.12),
            "d": (-5.12, 5.12),
            "e": (-5.12, 5.12),
            "f": (-5.12, 5.12),
            "g": (-5.12, 5.12),
            "h": (-5.12, 5.12),
            "i": (-5.12, 5.12),
            "j": (-5.12, 5.12),
            "k": (-5.12, 5.12),
            "l": (-5.12, 5.12),
            "m": (-5.12, 5.12),
            "n": (-5.12, 5.12),
            "o": (-5.12, 5.12),
            "p": (-5.12, 5.12),
            "q": (-5.12, 5.12),
            "r": (-5.12, 5.12),
            "s": (-5.12, 5.12),
            "t": (-5.12, 5.12)
        }
    
    elif fname == "griewangk":
        function = griewangk
        limits = {
            "a": (-600., 600.),
            "b": (-600., 600.),
            "c": (-600., 600.),
            "d": (-600., 600.),
            "e": (-600., 600.),
            "f": (-600., 600.),
            "g": (-600., 600.),
            "h": (-600., 600.),
            "i": (-600., 600.),
            "j": (-600., 600.)
        }
    
    elif fname == "schwefel":
        function = schwefel
        limits = {
            "a": (-5.12, 5.12),
            "b": (-5.12, 5.12),
            "c": (-5.12, 5.12),
            "d": (-5.12, 5.12),
            "e": (-5.12, 5.12)
            }
    else:
        sys.exit("ERROR: Function undefined...exiting")
    return function, limits


if __name__ == "__main__":
    #while True:
    functions = ["sphere", "rosenbrock", "step", "quartic", "rastrigin", "schwefel", "griewangk", "bisphere", "birastrigin"]
    for fname in functions:
        # migration_topology = num_migrants*np.ones((4, 4), dtype=int)
        # np.fill_diagonal(migration_topology, 0)
        function, limits = get_limits(fname)
        if MPI.COMM_WORLD.rank == 0:
            print(f"Minimize benchmark function {fname} with limits {limits} in {len(limits)} dimensions.")
        propagator = get_default_propagator(POP_SIZE, limits, 0.7, 0.4, 0.1)
        islands = Islands(
            function,
            propagator,
            generations=NUM_GENERATIONS,
            num_isles=2,
            isle_sizes=[19, 19, 19, 19],  # migration_topology=migration_topology,
            load_checkpoint="bla",  # pop_cpt.p",
            save_checkpoint="pop_cpt.p",
            migration_probability=0.9,
            emigration_propagator=SelectBest,
            immigration_propagator=SelectWorst,
            pollination=False,
        )
        islands.evolve(top_n=1, logging_interval=1, DEBUG=0)
        #break
