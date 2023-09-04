#!/usr/bin/env python3
import random
import sys

from mpi4py import MPI

from ap_pso.propagators import PSOInitUniform, VelocityClampingPropagator, ConstrictionPropagator, PSOCompose, \
    BasicPSOPropagator, StatelessPSOPropagator, CanonicalPropagator
from propulate import Islands
from propulate.propagators import Conditional
from function_benchmark import get_function_search_space

############
# SETTINGS #
############

function_name = sys.argv[1]  # Get function to optimize from command-line. Possible Options: See function_benchmark.py
NUM_GENERATIONS: int = int(sys.argv[2])  # Set number of generations.
POP_SIZE = 2 * MPI.COMM_WORLD.size  # Set size of breeding population.

function, limits = get_function_search_space(function_name)

if __name__ == "__main__":
    # migration_topology = num_migrants*np.ones((4, 4), dtype=int)
    # np.fill_diagonal(migration_topology, 0)

    rng = random.Random(MPI.COMM_WORLD.rank)

    pso_propagator = PSOCompose(
        [
            # VelocityClampingPropagator(0.7298, 1.49618, 1.49618, MPI.COMM_WORLD.rank, limits, rng, 0.6)
            # ConstrictionPropagator(2.49618, 2.49618, MPI.COMM_WORLD.rank, limits, rng)
            # BasicPSOPropagator(0.7298, 0.5, 0.5, MPI.COMM_WORLD.rank, limits, rng)
            CanonicalPropagator(2.49618, 2.49618, MPI.COMM_WORLD.rank, limits, rng)
            # StatelessPSOPropagator(0, 1.49618, 1.49618, MPI.COMM_WORLD.rank, limits, rng)
        ]
    )

    init = PSOInitUniform(limits, rng=rng, rank=MPI.COMM_WORLD.rank)
    propagator = Conditional(POP_SIZE, pso_propagator, init)

    islands = Islands(function, propagator, rng, generations=NUM_GENERATIONS, checkpoint_path='./checkpoints/',
                      migration_probability=0, pollination=False)
    islands.evolve(debug=0)