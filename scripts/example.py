#!/usr/bin/env python3

from mpi4py import MPI
from propulate import Propulator


comm = MPI.COMM_WORLD
n_generations = 10
pop_size = comm.Get_size()

limits = {
        'x' : (-10., 10.),
        'y' : (-10., 10.),
        'z' : (-10., 10.),
        'u' : (-10., 10.),
        'v' : (-10., 10.),
        'w' : (-10., 10.),
        }

# TODO intra communicator for when running a single individual is parallelized
def loss(params):
    return sum([params[x]**2 for x in params])

propulator = Propulator(loss, comm=comm, generations=n_generations, pop_size=pop_size)

propulator.propulate()
