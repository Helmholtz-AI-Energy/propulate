#!/usr/bin/env python3

from mpi4py import MPI
from propulate import Propulator


comm = MPI.COMM_WORLD
num_generations = 1000
pop_size = comm.Get_size()

limits = {
        'x' : (-10., 10.),
        'y' : (-10., 10.),
        'z' : (-10., 10.),
        'u' : (-10., 10.),
        'v' : (-10., 10.),
        'w' : (-10., 10.),
        }

def loss(params):
    return sum([params[x]**2 for x in params])

propulator = Propulator(loss, limits, comm=comm, num_generations=num_generations)

propulator.propulate()
