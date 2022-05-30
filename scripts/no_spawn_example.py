#!/usr/bin/env python3

from propulate import Propulator
from mpi4py import MPI
from propulate.utils import get_default_propagator

NUM_GENERATIONS = 10

limits = {
    "x": (-10.0, 10.0),
    "y": (-10.0, 10.0),
    "z": (-10.0, 10.0),
    "u": (-10.0, 10.0),
    "v": (-10.0, 10.0),
    "w": (-10.0, 10.0),
}


def loss(params):
    return sum([params[x] ** 2 for x in params])


coordinator_rank = MPI.COMM_WORLD.Get_size() - 1

if MPI.COMM_WORLD.Get_rank() != MPI.COMM_WORLD.Get_size() - 1:
    propagator = get_default_propagator(8, limits, 0.7, 0.4, 0.1)
    propulator = Propulator(
        loss, propagator, generations=NUM_GENERATIONS, coordinator_rank=coordinator_rank
    )
    propulator.propulate()

if MPI.COMM_WORLD.Get_rank() == coordinator_rank:
    import propulate

    coordinator = propulate.Coordinator(merge=False)
    coordinator._coordinate()
