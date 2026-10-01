"""Minimum working example showing how to use Propulate."""

import random
from typing import Dict

from mpi4py import MPI

import propulate

# Set the communicator and the optimization parameters
comm = MPI.COMM_WORLD
rng = random.Random(MPI.COMM_WORLD.rank)
population_size = comm.size * 2
generations = 100
checkpoint = "./propulate_checkpoints"
propulate.utils.set_logger_config()


# Define the function to minimize and the search space, e.g., a 2D sphere function on (-5.12, 5.12)^2.
def loss_fn(params: Dict[str, float]) -> float:
    """Loss function to minimize."""
    return params["x"] ** 2 + params["y"] ** 2


limits = {"x": (-5.12, 5.12), "y": (-5.12, 5.12)}

# Initialize the propagator and propulator with default parameters.
propagator = propulate.utils.get_default_propagator(pop_size=population_size, limits=limits, rng=rng)
propulator = propulate.Propulator(
    loss_fn=loss_fn,
    propagator=propagator,
    rng=rng,
    island_comm=comm,
    generations=generations,
    checkpoint_path=checkpoint,
)

# Run optimization and get summary of results.
propulator.propulate()
