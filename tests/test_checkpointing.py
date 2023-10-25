import random
import tempfile
from copy import deepcopy

from propulate import Propulator
from propulate.utils import get_default_propagator, sphere


# TODO do with MPI
def test_checkpointing_propulator():
    rng = random.Random(42)  # Separate random number generator for optimization.
    limits = {
        "a": (-5.12, 5.12),
        "b": (-5.12, 5.12),
    }

    with tempfile.TemporaryDirectory() as checkpoint_directory:
        propagator = get_default_propagator(limits=limits, rng=rng)
        propulator = Propulator(
            loss_fn=sphere,
            propagator=propagator,
            generations=10,
            checkpoint_directory=checkpoint_directory,
            rng=rng,
        )

        propulator.propulate()

        population = deepcopy(propulator.population)

        del propulator

        propulator = Propulator(
            loss_fn=sphere,
            propagator=propagator,
            generations=20,
            checkpoint_directory=checkpoint_directory,
            rng=rng,
        )

        # TODO properly check the loaded from checkpoint  population is the same as the one retained from the first run
        assert population == propulator.population


def test_checkpointing_islands():
    raise
