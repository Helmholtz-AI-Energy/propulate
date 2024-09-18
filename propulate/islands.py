import logging
import platform
import random
from pathlib import Path
from typing import Callable, Generator, Optional, Type, Union

import numpy as np
from mpi4py import MPI

from .migrator import Migrator
from .pollinator import Pollinator
from .propagators import Propagator, SelectMax, SelectMin
from .propulator import Propulator
from .surrogate import Surrogate

log = logging.getLogger(__name__)  # Get logger instance.


class Islands:
    """
    Wrapper class for Propulate optimization runs with multiple separate evolutionary islands.

    Propulate employs an island model, which combines independent evolution of self-contained subpopulations with
    intermittent exchange of selected individuals. To coordinate the search globally, each island occasionally delegates
    migrants to be included in the target islands' populations. With worse performing islands typically receiving
    candidates from better performing ones, islands communicate genetic information competitively, thus increasing
    diversity among the subpopulations compared to panmictic models.

    Attributes
    ----------
    propulator : propulate.Propulator
        The Propulator instance performing the actual optimization.

    Methods
    -------
    propulate()
        Run Propulate optimization.
    """

    def __init__(
        self,
        loss_fn: Union[Callable, Generator[float, None, None]],
        propagator: Propagator,
        rng: random.Random,
        generations: int = 0,
        num_islands: int = 1,
        island_sizes: Optional[np.ndarray] = None,
        migration_topology: Optional[np.ndarray] = None,
        migration_probability: float = 0.9,
        emigration_propagator: Type[Propagator] = SelectMin,
        immigration_propagator: Type[Propagator] = SelectMax,
        pollination: bool = True,
        checkpoint_path: Union[str, Path] = Path("./"),
        ranks_per_worker: int = 1,
        surrogate_factory: Optional[Callable[[], Surrogate]] = None,
    ) -> None:
        """
        Initialize an island model with the given parameters.

        Parameters
        ----------
        loss_fn : Union[Callable, Generator[float, None, None]]
            The loss function to be minimized.
        propagator : propulate.propagators.Propagator
            The propagator, i.e., evolutionary operator, to apply for breeding.
        rng : random.Random
             The separate random number generator for the Propulate optimization.
        generations : int, optional
            The number of generations. Default is 0.
        num_islands : int, optional
            The number of separate equally sized evolutionary islands. Ignored if ``island_sizes`` is not ``None``.
            Differences of +-1 are possible due to load balancing. Default is 1.
        island_sizes : numpy.ndarray[int], optional
            An array with numbers of workers for each island (heterogeneous case).
        migration_topology : numpy.ndarray[int], optional
            A two-dimensional matrix where entry (i,j) specifies how many individuals are sent by island i to island j
        migration_probability : float, optional
            The probability of migration after each generation.
        emigration_propagator : Type[propulate.propagators.Propagator], optional
            The emigration propagator, i.e., how to choose individuals for emigration that are sent to the destination
            island. Should be some kind of selection operator. Default is ``SelectMin``.
        immigration_propagator : Type[propulate.propagators.Propagator], optional
            The immigration propagator, i.e., how to choose individuals on the target island to be replaced by the
            immigrants. Should be some kind of selection operator. Default is ``SelectMax``.
        pollination : bool, optional
            If True, copies of emigrants are sent; otherwise, emigrants are removed from the original island.
            Default is True.
        checkpoint_path : pathlib.Path | str, optional
            The path where checkpoints are loaded from and stored to. Default is the current working directory.
        ranks_per_worker : int, optional
            The number of ranks per worker. Default is 1.
        surrogate_factory : Callable[[], propulate.surrogate.Surrogate], optional
           Function that returns a new instance of a ``Surrogate`` model.
           Only used when ``loss_fn`` is a generator function.

        Raises
        ------
        ValueError
            If the overall number of ranks is not evenly divisible by the requested number of ranks per worker.
            If the specified number of islands is smaller than 1.
            If the number of workers in the custom worker distribution does not equal overall number of ranks.
            If a custom migration topology has the wrong shape.
            If the migration probability is not within [0, 1].
        """
        # Set up full world communicator.
        full_world_rank, full_world_size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size

        if full_world_rank == 0:
            print(
                "#################################################\n"
                "# PROPULATE: Parallel Propagator of Populations #\n"
                "#################################################\n"
            )
            if "Windows" not in platform.system():
                print(
                    "        ⠀⠀⠀⠈⠉⠛⢷⣦⡀⠀⣀⣠⣤⠤⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n"
                    "⠀        ⠀⠀⠀⠀⠀⣀⣻⣿⣿⣿⣋⣀⡀⠀⠀⢀⣠⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n"
                    "⠀        ⠀⠀⣠⠾⠛⠛⢻⣿⣿⣿⠟⠛⠛⠓⠢⠀⠀⠉⢿⣿⣆⣀⣠⣤⣀⣀⠀⠀⠀\n"
                    "⠀        ⠀⠘⠁⠀⠀⣰⡿⠛⠿⠿⣧⡀⠀⠀⢀⣤⣤⣤⣼⣿⣿⣿⡿⠟⠋⠉⠉⠀⠀\n"
                    "⠀        ⠀⠀⠀⠀⠠⠋⠀⠀⠀⠀⠘⣷⡀⠀⠀⠀⠀⠹⣿⣿⣿⠟⠻⢶⣄⠀⠀⠀⠀\n"
                    "⠀⠀        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣧⠀⠀⠀⠀⢠⡿⠁⠀⠀⠀⠀⠈⠀⠀⠀⠀\n"
                    "⠀⠀        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿⡄⠀⠀⢠⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n"
                    "⠀⠀        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⣾⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n"
                    "⠀        ⣤⣤⣤⣤⣤⣤⡤⠄⠀⠀⣀⡀⢸⡇⢠⣤⣁⣀⠀⠀⠠⢤⣤⣤⣤⣤⣤⣤⠀\n"
                    "⠀⠀⠀⠀⠀        ⠀⣀⣤⣶⣾⣿⣿⣷⣤⣤⣾⣿⣿⣿⣿⣷⣶⣤⣀⠀⠀⠀⠀⠀⠀\n"
                    "        ⠀⠀⠀⣠⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⣄⠀⠀⠀\n"
                    "⠀        ⠀⠼⠿⣿⣿⠿⠛⠉⠉⠉⠙⠛⠿⣿⣿⠿⠛⠛⠛⠛⠿⢿⣿⣿⠿⠿⠇⠀⠀\n"
                    "⠀        ⢶⣤⣀⣀⣠⣴⠶⠛⠋⠙⠻⣦⣄⣀⣀⣠⣤⣴⠶⠶⣦⣄⣀⣀⣠⣤⣤⡶⠀\n"
                    "        ⠀⠀⠈⠉⠉⠉⠀⠀⠀⠀⠀⠀⠀⠉⠉⠉⠉⠀⠀⠀⠀⠀⠉⠉⠉⠉⠀⠀⠀⠀\n"
                )

        # Split the full world communicator into (multi rank) workers.
        if full_world_size % ranks_per_worker != 0:
            raise ValueError(
                f"The number of overall ranks, i.e., {full_world_size}, should be evenly divisible by "
                f"the number of ranks per worker, i.e., {ranks_per_worker}."
            )
        worker_idx = full_world_rank // ranks_per_worker  # This is the same for full world ranks belonging to the same worker.
        if ranks_per_worker > 1:
            # Create new communicators by splitting MPI.COMM_WORLD into group of sub-communicators based on
            # input values `color` and `key`. `color` determines to which new communicator each processes will belong.
            # `key` determines the ordering (rank) within each new communicator.
            worker_sub_comm = MPI.COMM_WORLD.Split(color=worker_idx, key=full_world_rank)

        else:
            worker_sub_comm = MPI.COMM_SELF

        # Create the Propulate world communicator, consisting of rank 0 of each worker's sub communicator.
        worker_root_ranks = [rank for rank in list(range(full_world_size)) if rank % ranks_per_worker == 0]
        propulate_world_group = MPI.COMM_WORLD.group.Incl(worker_root_ranks)
        propulate_world_comm = MPI.COMM_WORLD.Create_group(propulate_world_group)

        # Make sure that the Propulate world communicator is only defined on rank 0 of each worker's sub communicator.
        # Only those ranks are involved in the actual Propulate optimization logic and need to know about the related
        # variables and settings.
        if full_world_rank in worker_root_ranks:
            propulate_world_rank, propulate_world_size = (
                propulate_world_comm.rank,
                propulate_world_comm.size,
            )
        else:
            propulate_world_comm = None

        # Set up Propulate optimization logic only on ranks that are part of the Propulate world communicator.
        if propulate_world_comm is not None:
            # Homogeneous case with equal island sizes (differences of +-1 possible due to load balancing).
            if island_sizes is None:
                if num_islands < 1:
                    raise ValueError(f"Invalid number of evolutionary islands, needs to be >= 1 but was {num_islands}.")
                base_size = propulate_world_size // num_islands  # Base number of workers of each island
                remainder = propulate_world_size % num_islands  # Number of remaining workers to be distributed
                island_sizes = base_size * np.ones(num_islands, dtype=int)
                island_sizes[:remainder] += 1  # Distribute remaining workers equally for balanced load.

            # Heterogeneous case with user-defined island sizes.
            if np.sum(island_sizes) != propulate_world_size:
                raise ValueError(
                    f"There should be COMM_WORLD.size = {propulate_world_size} workers "
                    f"but only {np.sum(island_sizes)} were specified."
                )
            num_islands = island_sizes.size  # Determine number of islands.

            #  Set up intra-island communicator for communication within each island.
            island_colors = np.concatenate([idx * np.ones(el, dtype=int) for idx, el in enumerate(island_sizes)]).ravel()
            island_idx = island_colors[propulate_world_rank]  # Determine island index (which is also each rank's intra color).
            island_key = propulate_world_rank

            # Determine displacements as positions of unique elements, where # unique elements equals number of islands.
            _, island_displs = np.unique(island_colors, return_index=True)

            if full_world_rank == 0:
                log.info(
                    f"Worker distribution {island_colors} with island counts "
                    f"{island_sizes} and island displacements {island_displs}."
                )

            island_comm = propulate_world_comm.Split(color=island_idx, key=island_key)

            # Set up migration topology.
            if migration_topology is None:
                migration_topology = np.ones((num_islands, num_islands), dtype=int)
                np.fill_diagonal(migration_topology, 0)  # No island self-talk.
                if full_world_rank == 0:
                    log.info("NOTE: No migration topology given, using fully connected top-1 topology.")

            if full_world_rank == 0:
                log.info(f"Migration topology {migration_topology} has shape {migration_topology.shape}.")

            if migration_topology.shape != (num_islands, num_islands):
                raise ValueError(
                    f"Migration topology must be a quadratic matrix of size "
                    f"{island_displs.size} x {island_displs.size} but has shape {migration_topology.shape}."
                )

            if migration_probability > 1.0:
                raise ValueError(f"Migration probability must be in [0, 1] but was set to {migration_probability}.")
            migration_prob_rank = migration_probability / island_comm.size

            if full_world_rank == 0:
                log.info(
                    f"NOTE: Island migration probability {migration_probability} "
                    f"results in per-rank migration probability {migration_prob_rank}.\n"
                    "Starting parallel optimization process."
                )
        else:
            migration_prob_rank = None  # type: ignore
            island_displs = None  # type: ignore
            island_idx = None  # type: ignore
            island_comm = None  # type: ignore
            emigration_propagator = None  # type: ignore
            immigration_propagator = None  # type: ignore

        MPI.COMM_WORLD.barrier()
        # Set up one Propulator for each island.
        if pollination is False:
            if full_world_rank == 0:
                log.info("Use island model with real migration.")
            self.propulator: Propulator = Migrator(
                loss_fn=loss_fn,
                propagator=propagator,
                rng=rng,
                island_idx=island_idx,
                island_comm=island_comm,
                propulate_comm=propulate_world_comm,
                worker_sub_comm=worker_sub_comm,
                generations=generations,
                checkpoint_path=checkpoint_path,
                migration_topology=migration_topology,
                migration_prob=migration_prob_rank,
                emigration_propagator=emigration_propagator,
                island_displs=island_displs,
                island_counts=island_sizes,
                surrogate_factory=surrogate_factory,
            )
        else:
            if full_world_rank == 0:
                log.info("Use island model with pollination.")
            self.propulator = Pollinator(
                loss_fn=loss_fn,
                propagator=propagator,
                rng=rng,
                island_idx=island_idx,
                island_comm=island_comm,
                propulate_comm=propulate_world_comm,
                worker_sub_comm=worker_sub_comm,
                generations=generations,
                checkpoint_path=checkpoint_path,
                migration_topology=migration_topology,
                migration_prob=migration_prob_rank,
                emigration_propagator=emigration_propagator,
                immigration_propagator=immigration_propagator,
                island_displs=island_displs,
                island_counts=island_sizes,
                surrogate_factory=surrogate_factory,
            )

    def propulate(self, logging_interval: int = 10, debug: int = 1) -> None:
        """
        Run Propulate optimization.

        Parameters
        ----------
        logging_interval : int
            The logging interval.
        debug : int
            The debug level.
        """
        self.propulator.propulate(logging_interval, debug)
