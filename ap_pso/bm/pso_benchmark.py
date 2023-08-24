#!/usr/bin/env python3
import random
import sys

from mpi4py import MPI

from ap_pso.propagators import PSOInitUniform, VelocityClampingPropagator, ConstrictionPropagator, PSOCompose, \
    BasicPSOPropagator, StatelessPSOPropagator, CanonicalPropagator
from propulate import Islands
from propulate.propagators import Conditional, InitUniform
from scripts.function_benchmark import get_function_search_space

############
# SETTINGS #
############

function_name = sys.argv[1]  # Get function to optimize from command-line.
NUM_GENERATIONS: int = int(sys.argv[2])  # Set number of generations.
POP_SIZE = 2 * MPI.COMM_WORLD.size  # Set size of breeding population.
PSO_TYPE = int(sys.argv[3]) # selects the propagator below
CHECKPOINT_PLACE = sys.argv[4]
num_migrants = 1

function, limits = get_function_search_space(function_name)

if __name__ == "__main__":
    # migration_topology = num_migrants*np.ones((4, 4), dtype=int)
    # np.fill_diagonal(migration_topology, 0)

    rng = random.Random(MPI.COMM_WORLD.rank)

    propagator = [
        VelocityClampingPropagator(0.7298, 1.49618, 1.49618, MPI.COMM_WORLD.rank, limits, rng, 0.6),
        ConstrictionPropagator(2.49618, 2.49618, MPI.COMM_WORLD.rank, limits, rng),
        BasicPSOPropagator(0.7298, 0.5, 0.5, MPI.COMM_WORLD.rank, limits, rng),
        CanonicalPropagator(2.49618, 2.49618, MPI.COMM_WORLD.rank, limits, rng)
    ][PSO_TYPE]

    init = PSOInitUniform(limits, rng=rng, rank=MPI.COMM_WORLD.rank)
    propagator = Conditional(POP_SIZE, propagator, init)

    islands = Islands(function, propagator, rng, generations=NUM_GENERATIONS, checkpoint_path=CHECKPOINT_PLACE,
                      migration_probability=0, pollination=False)
    islands.evolve(top_n=1, logging_interval=1)
    islands.propulator.paint_graphs(function_name)
