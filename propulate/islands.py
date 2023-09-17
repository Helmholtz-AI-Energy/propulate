import random
from pathlib import Path
from typing import Callable, Union, List, Type
import logging

from mpi4py import MPI
import numpy as np

from .propagators import Propagator, SelectMin, SelectMax
from .migrator import Migrator
from .pollinator import Pollinator
from .population import Individual


log = logging.getLogger(__name__)  # Get logger instance.


class Islands:
    """
    Wrapper class for Propulate optimization runs with multiple separate evolutionary islands.

    Propulate employs an island model, which combines independent evolution of self-contained subpopulations with
    intermittent exchange of selected individuals. To coordinate the search globally, each island occasionally delegates
    migrants to be included in the target islands' populations. With worse performing islands typically receiving
    candidates from better performing ones, islands communicate genetic information competitively, thus increasing
    diversity among the subpopulations compared to panmictic models.
    """

    def __init__(
        self,
        loss_fn: Callable,
        propagator: Propagator,
        rng: random.Random,
        generations: int = 0,
        num_islands: int = 1,
        island_sizes: np.ndarray = None,
        migration_topology: np.ndarray = None,
        migration_probability: float = 0.0,
        emigration_propagator: Type[Propagator] = SelectMin,
        immigration_propagator: Type[Propagator] = SelectMax,
        pollination: bool = True,
        checkpoint_path: Union[str, Path] = Path("./"),
    ) -> None:
        """
        Initialize island model with given parameters.

        Parameters
        ----------
        loss_fn: Callable
                 loss function to be minimized
        propagator: propulate.propagators.Propagator
                    propagator to apply for breeding
        rng: random.Random
             random number generator
        generations: int
                     number of generations
        num_islands: int
                     number of separate, equally sized evolutionary islands (ignored if ``island_sizes`` is not None)
                     (differences +-1 possible due to load balancing)
        island_sizes: numpy.ndarray
                      array with numbers of workers for each island (heterogeneous case)
        migration_topology: numpy.ndarray
                            2D matrix where entry (i,j) specifies how many individuals are sent
                            by island i to island j
                            (int: absolute number, float: relative fraction of population)
        migration_probability: float
                               probability of migration after each generation
        emigration_propagator: type[propulate.propagators.Propagator]
                               emigration propagator, i.e., how to choose individuals for emigration
                               that are sent to destination island.
                               Should be some kind of selection operator.
        immigration_propagator: type[propulate.propagators.Propagator]
                                immigration propagator, i.e., how to choose individuals on target island
                                to be replaced by immigrants.
                                Should be some kind of selection operator.
        pollination: bool
                     If True, copies of emigrants are sent, otherwise, emigrants are removed from
                     original island.
        checkpoint_path: Union[Path, str]
                         Path where checkpoints are loaded from and stored.

        Raises
        ------
        ValueError
            If specified number of islands is smaller than 1.
        ValueError
            If number of workers in custom worker distribution does not equal overall number of processors.
        ValueError
            If custom migration topology has the wrong shape.
        ValueError
            If migration probability is not within [0, 1].
        """
        # Set up communicator.
        rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size

        if rank == 0:
            print(
                "#################################################\n"
                "# PROPULATE: Parallel Propagator of Populations #\n"
                "#################################################\n\n"
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

        # Homogeneous case with equal island sizes (differences of +-1 possible due to load balancing).
        if island_sizes is None:
            if num_islands < 1:
                raise ValueError(
                    f"Invalid number of evolutionary islands, needs to be >= 1 but was {num_islands}."
                )
            base_size = size // num_islands  # Base number of workers of each island
            remainder = (
                size % num_islands
            )  # Number of remaining workers to be distributed
            island_sizes = base_size * np.ones(num_islands, dtype=int)
            island_sizes[
                :remainder
            ] += 1  # Distribute remaining workers equally for balanced load.

        # Heterogeneous case with user-defined island sizes.
        if np.sum(island_sizes) != size:
            raise ValueError(
                f"There should be COMM_WORLD.size = {size} workers but only {np.sum(island_sizes)} were specified."
            )
        num_islands = island_sizes.size  # Determine number of islands.

        #  Set up intra-island communicator for communication within each island.
        intra_color = np.concatenate(
            [idx * np.ones(el, dtype=int) for idx, el in enumerate(island_sizes)]
        ).ravel()
        island_idx = intra_color[
            rank
        ]  # Determine island index (which is also each rank's intra color).
        intra_key = rank

        # Determine displacements as positions of unique elements, where # unique elements equals number of islands.
        _, island_displs = np.unique(intra_color, return_index=True)

        if rank == 0:
            log.info(
                f"Worker distribution {intra_color} with island counts "
                f"{island_sizes} and island displacements {island_displs}."
            )

        # Create new communicators by splitting MPI.COMM_WORLD into group of sub-communicators based on
        # input values `color` and `key`. `color` determines to which new communicator each processes will belong.
        # `key` determines the ordering (rank) within each new communicator.
        comm_intra = MPI.COMM_WORLD.Split(color=island_idx, key=intra_key)

        # Set up migration topology.
        if migration_topology is None:
            migration_topology = np.ones((num_islands, num_islands), dtype=int)
            np.fill_diagonal(migration_topology, 0)  # No island self-talk.
            if rank == 0:
                log.info(
                    "NOTE: No migration topology given, using fully connected top-1 topology."
                )

        if rank == 0:
            log.info(
                f"Migration topology {migration_topology} has shape {migration_topology.shape}."
            )

        if migration_topology.shape != (num_islands, num_islands):
            raise ValueError(
                f"Migration topology must be a quadratic matrix of size "
                f"{island_displs.size} x {island_displs.size} but has shape {migration_topology.shape}."
            )

        if migration_probability > 1.0:
            raise ValueError(
                f"Migration probability must be in [0, 1] but was set to {migration_probability}."
            )
        migration_prob_rank = migration_probability / comm_intra.size

        if rank == 0:
            log.info(
                f"NOTE: Island migration probability {migration_probability} "
                f"results in per-rank migration probability {migration_prob_rank}.\n"
                "Starting parallel optimization process."
            )

        MPI.COMM_WORLD.barrier()
        # Set up one Propulator for each island.
        if pollination is False:
            if rank == 0:
                log.info("Use island model with real migration.")
            self.propulator = Migrator(
                loss_fn=loss_fn,
                propagator=propagator,
                island_idx=island_idx,
                comm=comm_intra,
                generations=generations,
                checkpoint_path=checkpoint_path,
                migration_topology=migration_topology,
                migration_prob=migration_prob_rank,
                emigration_propagator=emigration_propagator,
                island_displs=island_displs,
                island_counts=island_sizes,
                rng=rng,
            )
        else:
            if rank == 0:
                log.info("Use island model with pollination.")
            self.propulator = Pollinator(
                loss_fn=loss_fn,
                propagator=propagator,
                island_idx=island_idx,
                comm=comm_intra,
                generations=generations,
                checkpoint_path=checkpoint_path,
                migration_topology=migration_topology,
                migration_prob=migration_prob_rank,
                emigration_propagator=emigration_propagator,
                immigration_propagator=immigration_propagator,
                island_displs=island_displs,
                island_counts=island_sizes,
                rng=rng,
            )

    def _run(
        self, top_n: int = 3, logging_interval: int = 10, debug: int = 1
    ) -> List[Union[List[Individual], Individual]]:
        """
        Run Propulate optimization.

        Parameters
        ----------
        top_n: int
               number of best results to report
        logging_interval: int
                          logging interval
        debug: int
               verbosity/debug level

        Returns
        -------
        list[list[Individual] | Individual]
            top-n best individuals on each island
        """
        self.propulator.propulate(logging_interval, debug)
        return self.propulator.summarize(top_n, debug)

    def evolve(
        self, top_n: int = 3, logging_interval: int = 10, debug: int = 1
    ) -> List[Union[List[Individual], Individual]]:
        """
        Run Propulate optimization.

        Parameters
        ----------
        top_n: int
               number of best results to report
        logging_interval: int
                          logging interval
        debug: int
               verbosity/debug level

        Returns
        -------
        list[list[Individual] | Individual]
            top-n best individuals on each island
        """
        return self._run(top_n, logging_interval, debug)
