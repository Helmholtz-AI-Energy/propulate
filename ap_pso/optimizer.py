from pathlib import Path
from random import Random
from typing import Callable, Optional

import numpy as np
from mpi4py import MPI

from particle import Particle
from propagators import Propagator
from propulate.propulate import Propulator, PolliPropulator
from propulate.propulate.propagators import SelectMin, SelectMax


class Optimizer:
    """
    This class is the executor of the algorithm and holds the main part of the API.

    To use the algorith, you create an Instance of this class and call its optimize (or evolve, for convenience)
    function. Then everything is done by this framework.
    """

    def __init__(self,
                 loss_fn: Callable,
                 propagator: Propagator,
                 rng: Random,
                 generations: int = 0,
                 num_swarms: int = 1,
                 workers_per_swarm: list[int] = None,
                 migration_topology: np.ndarray = None,
                 migration_probability: float = 0.0,
                 emigration_propagator: Propagator = SelectMin,
                 immigration_propagator: Propagator = SelectMax,
                 pollination: bool = False,
                 checkpoint_path: Path = Path('./')
                 ):
        """
        Constructor of Islands() class.

        Parameters
        ----------
        loss_fn :                   callable
            loss function to be minimized
        propagator :                propagators.Propagator
            propagator to apply for optimization
        generations :               int
            number of optimization iterations
        num_swarms :                int
            number of separate, equally sized swarms (differences +-1 possible due to load balancing)
        workers_per_swarm :         list[int]
            list with numbers of workers for each swarm (heterogeneous case)
        migration_topology :        numpy.ndarray
            2D matrix where each entry (i,j) specifies how many individuals are sent by isle i to isle j (int: absolute
            number, float: relative fraction of population)
        migration_probability :     float
            probability of migration after each generation
        emigration_propagator :     propagators.Propagator
            emigration propagator, i.e., how to choose individuals for emigration that are sent to destination island.
            Should be some kind of selection operator.
        immigration_propagator :    propagators.Propagator
            immigration propagator, i.e., how to choose individuals on target isle to be replaced by immigrants.
            Should be some kind of selection operator.
        pollination :               bool
            If True, copies of emigrants are sent, otherwise, emigrants are removed from original isle.
        checkpoint_path :           pathlib.Path
            Path where checkpoints are loaded from and stored.
        """

        if num_swarms < 1:
            raise ValueError(f"Invalid number of Swarms: {num_swarms}")
        assert migration_topology.shape == (num_swarms, num_swarms)
        assert len(workers_per_swarm) == num_swarms
        assert all((x > 0 for x in workers_per_swarm))
        assert 0.0 <= migration_probability <= 1.0

        self.loss_fn = loss_fn
        self.propagator = propagator
        self.generations = generations

        # Set up MPI stuff
        self.mpi_size = MPI.COMM_WORLD.size
        self.mpi_rank = MPI.COMM_WORLD.rank

        # FIXME: Print log message/welcome screen!

        if workers_per_swarm is None:
            # Homogeneous case
            av_wps = self.mpi_size // num_swarms
            leftover = self.mpi_size % num_swarms
            workers_per_swarm = [av_wps + 1] * leftover
            workers_per_swarm += [av_wps] * (num_swarms - leftover)

        # If this doesn't fit, someone did bad things.
        given_workers: int = np.sum(workers_per_swarm)

        if given_workers != self.mpi_size:
            # Erroneous inhomogeneous case
            raise ValueError(f"Given total number of workers ({given_workers}) does not match MPI.COMM_WORLD.size, "
                             f"which is the number of available workers ({self.mpi_size}).")
        # In correct inhomogeneous case, we don't have to do anything, as we got our worker distribution given as
        # method parameter.

        # Set up communicators:
        intra_color = np.concatenate([idx * np.ones(el, dtype=int) for idx, el in enumerate(
            workers_per_swarm)]).reshape(-1)  # color because of MPI's parameter naming TODO: Get rid of magic numbers!

        _, u_indices = np.unique(intra_color, return_index=True)
        inter_color = np.zeros(self.mpi_size)
        # FIXME: Print log message!
        inter_color[u_indices] = 1  # TODO: Get rid of this magic number!

        intra_color = intra_color[self.mpi_rank]
        inter_color = inter_color[self.mpi_rank]

        self.comm_intra = MPI.COMM_WORLD.Split(intra_color, self.mpi_rank)
        self.comm_inter = MPI.COMM_WORLD.Split(inter_color, self.mpi_rank)  # TODO: What exactly is this?

        self.swarm_index: Optional[int] = None
        if self.comm_intra.rank == 0:  # We're root of our intra-communicator
            self.swarm_idx = self.comm_inter.rank  # And now we set the index of our swarm to our rank in
            # inter-communicator
        swarm_idx = self.comm_intra.bcast(self.swarm_idx)

        if migration_topology is None:
            migration_topology = np.ones((num_swarms, num_swarms), dtype=int)
            np.fill_diagonal(migration_topology, 0)

            # FIXME: Print log message!

        self.migration_topology = migration_topology
        self.migration_probability = migration_probability / self.comm_intra.size

        # FIXME: Print log messages!

        MPI.COMM_WORLD.barrier()

        self.propulator: Propulator

        if pollination:
            self.propulator = Propulator(
                self.loss_fn,
                self.propagator,
                swarm_idx,
                self.comm_intra,
                self.generations,
                checkpoint_path,
                self.migration_topology,
                self.comm_inter,
                self.migration_probability,
                emigration_propagator,
                u_indices,
                workers_per_swarm,
                rng
            )
        else:
            self.propulator = PolliPropulator(
                self.loss_fn,
                self.propagator,
                swarm_idx,
                self.comm_intra,
                self.generations,
                checkpoint_path,
                self.migration_topology,
                self.comm_inter,
                self.migration_probability,
                emigration_propagator,
                immigration_propagator,
                u_indices,
                workers_per_swarm,
                rng
            )

        # TODO: Outfactor all printing and stuff that is not CLEARLY debug stuff or error messages. See FixMe's

    def optimize(self, top_n: int = 3, out_file="summary.png", logging_interval: int = 10, debug: int = 1) -> \
            Optional[Particle]:
        """
        This method runs the PSO algorithm on this swarm.

        Parameters
        ----------
        top_n :            int
                number of best results to report
        out_file :         str
                What's o' ever - what is this for a parameter?
        logging_interval : int
                Number of generations to generate some logging output
        debug :            bool
                Debug verbosity level
        """

        self.propulator.propulate(logging_interval, debug)
        if debug > -1:
            best: Particle = self.propulator.summarize(top_n, out_file=out_file, DEBUG=debug)
            return best
        else:
            return None

    def evolve(self, top_n: int = 3, out_file="summary.png", logging_interval: int = 10, debug: int = 1) -> \
            Optional[Particle]:
        """
        This is a wrapper for the optimize method in order to ensure compatibility to Propulate.

        Parameters
        ----------
        top_n :            int
            number of best results to report
        out_file :         str
            What's o' ever - what is this for a parameter?
        logging_interval : int
            Number of generations to generate some logging output
        debug :            bool
            Debug verbosity level
        """
        return self.optimize(top_n, out_file, logging_interval, debug)
