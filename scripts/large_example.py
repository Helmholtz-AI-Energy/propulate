#!/usr/bin/env python
import numpy
from mpi4py import MPI

# NOTE this is a test to check the difference between message handling depending on its size

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = numpy.arange(30 * 10, dtype="f")
    comm.Send([data, MPI.FLOAT], dest=1, tag=77)
if rank == 1:
    data = numpy.arange(30 * 10, dtype="f")
    comm.Send([data, MPI.FLOAT], dest=0, tag=77)

if rank == 0:
    data = numpy.empty(30 * 10, dtype="f")
    comm.Recv([data, MPI.FLOAT], source=1, tag=77)
    print(data)

if rank == 1:
    data = numpy.empty(30 * 10, dtype="f")
    comm.Recv([data, MPI.FLOAT], source=0, tag=77)
    print(data)
