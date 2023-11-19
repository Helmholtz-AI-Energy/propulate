import random
import tempfile
from operator import attrgetter
import logging
from copy import deepcopy
from mpi4py import MPI

import pytest

from propulate import Propulator
from propulate.utils import get_default_propagator, set_logger_config, sphere


@pytest.mark.mpi_skip
def test_Propulator():
    """
    Test single worker using Propulator to optimize sphere.
    """
    rng = random.Random(42)  # Separate random number generator for optimization.
    limits = {
        "a": (-5.12, 5.12),
        "b": (-5.12, 5.12),
    }
    with tempfile.TemporaryDirectory() as checkpoint_directory:
        set_logger_config(
            level=logging.INFO,
            log_file=checkpoint_directory + "/propulate.log",
            log_to_stdout=True,
            log_rank=False,
            colors=True,
        )
        # Set up evolutionary operator.
        propagator = get_default_propagator(  # Get default evolutionary operator.
            pop_size=4,  # Breeding pool size
            limits=limits,  # Search-space limits
            mate_prob=0.7,  # Crossover probability
            mut_prob=9.0,  # Mutation probability
            random_prob=0.1,  # Random-initialization probability
            rng=rng,  # Random number generator
        )

        # Set up propulator performing actual optimization.
        propulator = Propulator(
            loss_fn=sphere,
            propagator=propagator,
            generations=10,
            checkpoint_directory=checkpoint_directory,
            rng=rng,
        )

        # Run optimization and print summary of results.
        propulator.propulate()
        propulator.summarize()
        best = min(propulator.population, key=attrgetter("loss"))

        assert best.loss < 0.8


@pytest.mark.mpi_skip
def test_Propulator_checkpointing():
    rng = random.Random(42)  # Separate random number generator for optimization.
    limits = {
        "a": (-5.12, 5.12),
        "b": (-5.12, 5.12),
    }

    with tempfile.TemporaryDirectory() as checkpoint_directory:
        propagator = get_default_propagator(  # Get default evolutionary operator.
            pop_size=4,  # Breeding pool size
            limits=limits,  # Search-space limits
            mate_prob=0.7,  # Crossover probability
            mut_prob=9.0,  # Mutation probability
            random_prob=0.1,  # Random-initialization probability
            rng=rng,  # Random number generator
        )
        propulator = Propulator(
            loss_fn=sphere,
            propagator=propagator,
            generations=10,
            checkpoint_directory=checkpoint_directory,
            rng=rng,
        )

        propulator.propulate()

        population = deepcopy(propulator.population)
        generations = propulator.generations
        poplist = [None] * generations
        assert len(population) == generations
        for ind in population:
            poplist[ind.generation] = ind

        del propulator

        propulator = Propulator(
            loss_fn=sphere,
            propagator=propagator,
            generations=20,
            checkpoint_directory=checkpoint_directory,
            rng=rng,
        )

        assert len(propulator.population) == generations
        newpoplist = [None] * generations
        for ind in propulator.population:
            newpoplist[ind.generation] = ind

        for ind1, ind2 in zip(poplist, newpoplist):
            for key in ind1:
                assert ind1[key] == pytest.approx(ind2[key])
            assert ind1.loss == pytest.approx(ind2.loss)
            assert ind1.generation == ind2.generation
            assert ind1.rank == ind2.rank
            assert ind1.island == ind2.island
            assert ind1.active == ind2.active


@pytest.mark.mpi(min_size=4)
def test_Propulator_parallel_checkpointing(tmp_path):
    from h5py import File

    with File(tmp_path, "w", driver="mpio", comm=MPI.COMM_WORLD) as f:
        assert f
        assert f.driver == "mpio"
    # rng = random.Random(
    #     42 + MPI.COMM_WORLD.Get_rank()
    # )  # Separate random number generator for optimization.
    # limits = {
    #     "a": (-5.12, 5.12),
    #     "b": (-5.12, 5.12),
    # }
    # checkpoint_directory = tmp_path
    # print(checkpoint_directory)

    # propagator = get_default_propagator(  # Get default evolutionary operator.
    #     pop_size=4,  # Breeding pool size
    #     limits=limits,  # Search-space limits
    #     mate_prob=0.7,  # Crossover probability
    #     mut_prob=9.0,  # Mutation probability
    #     random_prob=0.1,  # Random-initialization probability
    #     rng=rng,  # Random number generator
    # )

    # # import h5py

    # # with h5py.File(
    # #     checkpoint_directory / "test.hdf5", "a", driver="mpio", comm=MPI.COMM_WORLD
    # # ) as f:
    # #     print(f)
    # #     raise

    # propulator = Propulator(
    #     loss_fn=sphere,
    #     propagator=propagator,
    #     generations=10,
    #     checkpoint_directory=checkpoint_directory,
    #     rng=rng,
    # )

    # propulator.propulate()

    # population = deepcopy(propulator.population)
    # generations = propulator.generations
    # poplist = [None] * generations
    # assert len(population) == generations
    # for ind in population:
    #     poplist[ind.generation] = ind

    # del propulator

    # propulator = Propulator(
    #     loss_fn=sphere,
    #     propagator=propagator,
    #     generations=20,
    #     checkpoint_directory=checkpoint_directory,
    #     rng=rng,
    # )

    # assert len(propulator.population) == generations
    # newpoplist = [None] * generations
    # for ind in propulator.population:
    #     newpoplist[ind.generation] = ind

    # for ind1, ind2 in zip(poplist, newpoplist):
    #     for key in ind1:
    #         assert ind1[key] == pytest.approx(ind2[key])
    #     assert ind1.loss == pytest.approx(ind2.loss)
    #     assert ind1.generation == ind2.generation
    #     assert ind1.rank == ind2.rank
    #     assert ind1.island == ind2.island
    #     assert ind1.active == ind2.active


# @pytest.mark.mpi_skip
# def test_checkpointing_propulator_midevaluation():
#     # TODO test if loading a checkpoint with an unfinished evaluation works correctly
#     # i.e. the not yet evaluated individual should be present but without result
#     # not sure how yet, would have to send a signal to kill the process without destroying the temp files and the restarting
#     raise
