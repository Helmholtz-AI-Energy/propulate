import logging
import random
import tempfile

from propulate import Propulator
from propulate.utils import get_default_propagator, set_logger_config
from propulate.utils.benchmark_functions import sphere


def test_propulator():
    """Test single worker using Propulator to optimize sphere."""
    rng = random.Random(42)  # Separate random number generator for optimization.
    limits = {
        "a": (-5.12, 5.12),
        "b": (-5.12, 5.12),
    }
    with tempfile.TemporaryDirectory() as checkpoint_path:
        set_logger_config(
            level=logging.INFO,
            log_file=checkpoint_path + "/propulate.log",
            log_to_stdout=True,
            log_rank=False,
            colors=True,
        )
        # Set up evolutionary operator.
        propagator = get_default_propagator(
            pop_size=4,
            limits=limits,
            crossover_prob=0.7,
            mutation_prob=9.0,
            random_init_prob=0.1,
            rng=rng,
        )

        # Set up propulator performing actual optimization.
        propulator = Propulator(
            loss_fn=sphere,
            propagator=propagator,
            rng=rng,
            generations=100,
            checkpoint_path=checkpoint_path,
        )

        # Run optimization and print summary of results.
        propulator.propulate()
        best = propulator.summarize()

        assert best[0][0].loss < 0.8
