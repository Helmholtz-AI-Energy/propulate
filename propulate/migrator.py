import copy
import logging
import random
from pathlib import Path
from typing import Callable, Generator, List, Optional, Type, Union

import h5py
import numpy as np
from mpi4py import MPI

from ._globals import MIGRATION_TAG, SYNCHRONIZATION_TAG
from .population import Individual
from .propagators import Propagator, SelectMin
from .propulator import Propulator
from .surrogate import Surrogate

log = logging.getLogger(__name__)


class Migrator(Propulator):
    """
    Parallel propagator of populations using an island model with real migration.

    Individuals can only exist on one evolutionary island at a time, i.e., they are removed
    (i.e. deactivated for breeding) from the sending island upon emigration.

    Methods
    -------
    propulate()
        Run asynchronous evolutionary optimization routine with actual migration.

    Notes
    -----
    The ``Migrator`` class inherits all methods and attributes from the ``Propulator`` class.

    See Also
    --------
    :class:`Propulator` : The parent class.
    """

    def __init__(
        self,
        loss_fn: Union[Callable, Generator[float, None, None]],
        propagator: Propagator,
        rng: random.Random,
        generations: int,
        island_idx: int = 0,
        island_comm: MPI.Comm = MPI.COMM_WORLD,
        propulate_comm: MPI.Comm = MPI.COMM_WORLD,
        worker_sub_comm: MPI.Comm = MPI.COMM_SELF,
        checkpoint_path: Union[str, Path] = Path("./"),
        migration_topology: Optional[np.ndarray] = None,
        migration_prob: float = 0.0,
        emigration_propagator: Type[Propagator] = SelectMin,
        immigration_propagator: Optional[Type[Propagator]] = None,
        island_displs: Optional[np.ndarray] = None,
        island_sizes: Optional[np.ndarray] = None,
        surrogate_factory: Optional[Callable[[], Surrogate]] = None,
    ) -> None:
        """
        Initialize ``Migrator`` with given parameters.

        Parameters
        ----------
        loss_fn : Union[Callable, Generator[float, None, None]]
            The loss function to be minimized.
        propagator: propulate.propagators.Propagator
            The propagator to apply for breeding.
        rng : random.Random
            The separate random number generator for the Propulate optimization.
        island_idx: int, optional
            The island's index. Default is 0.
        island_comm: MPI.Comm, optional
            The intra-island communicator for communication within that island. Default is ``MPI.COMM_WORLD``.
        propulate_comm : MPI.Comm, optional
            The Propulate world communicator, consisting of rank 0 of each worker's sub communicator.
            Default is ``MPI.COMM_WORLD``.
        worker_sub_comm : MPI.Comm, optional
            The sub communicator for each (multi rank) worker. Default is ``MPI.COMM_SELF``.
        generations : int, optional
            The number of generations to run. Default is -1, i.e., run into wall-clock time limit.
        checkpoint_path : pathlib.Path | str, optional
            The path where the checkpoints are loaded from and stored. Default is current working directory.
        migration_topology : numpy.ndarray, optional
            The migration topology, i.e., a 2D matrix where entry (i,j) specifies how many individuals are sent by
            island i to island j.
        migration_prob : float, optional
            The per-worker migration probability. Default is 0.0.
        emigration_propagator : Type[propulate.propagators.Propagator]
            The emigration propagator, i.e., how to choose individuals for emigration that are sent to the destination
            island. Should be some kind of selection operator. Default is ``SelectMin``.
        immigration_propagator : Type[propulate.propagators.Propagator], optional
            The immigration propagator, i.e., how to choose individuals to be replaced by immigrants on a target island.
            Should be some kind of selection operator. Default is ``None``.
        island_displs : numpy.ndarray, optional
            An array with ``propulate_comm`` rank of each island's worker 0. Element i specifies the rank of worker 0 on
            island with index i in the Propulate communicator.
        island_sizes : numpy.ndarray, optional
            An array with the number of workers per island. Element i specifies the number of workers on island i.
        surrogate_factory : Callable[[], propulate.surrogate.Surrogate], optional
           Function that returns a new instance of a ``Surrogate`` model.
           Only used when ``loss_fn`` is a generator function.
        """
        super().__init__(
            loss_fn=loss_fn,
            propagator=propagator,
            rng=rng,
            generations=generations,
            island_idx=island_idx,
            island_comm=island_comm,
            propulate_comm=propulate_comm,
            worker_sub_comm=worker_sub_comm,
            checkpoint_path=checkpoint_path,
            migration_topology=migration_topology,
            migration_prob=migration_prob,
            emigration_propagator=emigration_propagator,
            immigration_propagator=immigration_propagator,
            island_displs=island_displs,
            island_sizes=island_sizes,
            surrogate_factory=surrogate_factory,
        )
        log.debug("Migrator set up.")

    def _send_emigrants(self, hdf5_checkpoint: h5py.File) -> None:
        """Perform migration, i.e. island sends individuals out to other islands."""
        log_string = f"Island {self.island_idx} Worker {self.island_comm.rank} Generation {self.generation}: EMIGRATION\n"
        # Determine relevant line of migration topology.
        assert self.migration_topology is not None
        to_migrate = self.migration_topology[self.island_idx, :]
        num_emigrants = np.sum(to_migrate, dtype=int).item()  # Determine overall number of emigrants to be sent out.
        # Each individual is emigrated by exactly one worker of the island, the one responsible for it.
        eligible_emigrants = [
            ind for ind in self.population.values() if ind.active > 0 and self._responsible_rank(ind) == self.island_comm.rank
        ]

        # Only perform migration if overall number of emigrants to be sent
        # out is smaller than current number of eligible potential emigrants.
        if num_emigrants > len(eligible_emigrants):
            log.debug(
                f"Island {self.island_idx} worker {self.island_comm.rank} generation {self.generation}: \n"
                f"Population size {len(eligible_emigrants)} too small "
                f"to select {num_emigrants} migrants."
            )
            return

        # Select all migrants to be sent out in this migration step.
        emigrator = self.emigration_propagator(num_emigrants)  # Set up emigration propagator.
        all_emigrants = emigrator(eligible_emigrants)  # Choose `offspring` eligible emigrants.
        assert isinstance(all_emigrants, list)
        self.rng.shuffle(all_emigrants)
        # Loop through relevant part of migration topology.
        offspring_sent = 0
        for target_island, offspring in enumerate(to_migrate):
            if offspring == 0:
                continue
            # Determine self.propulate_comm ranks of workers on target island.
            assert self.island_displs is not None
            assert self.island_sizes is not None
            displ = self.island_displs[target_island]
            count = self.island_sizes[target_island]
            dest_island_workers = np.arange(displ, displ + count)

            # Worker sends *different* individuals to each target island.
            emigrants = all_emigrants[offspring_sent : offspring_sent + offspring]  # Choose `offspring` eligible emigrants.
            offspring_sent += offspring
            log_string += f"Chose {len(emigrants)} emigrant(s): {emigrants}\n"

            # Deactivate emigrants on the sending island (true migration): in the own population, in the checkpoint, and
            # on the other workers of this island. The checkpoint activates them on the target island before it
            # deactivates them here, so that a run interrupted in between leaves an extra copy rather than losing the
            # individual. The responsible worker is the only one that writes an individual's bit of its own island;
            # the target bit it writes here is later cleared by the target island's responsible worker, after the
            # emigrant has arrived there, i.e., causally after this write.
            for emigrant in emigrants:
                self.population[(emigrant.island, emigrant.island_rank, emigrant.generation)].active -= 1
                active_on_island = hdf5_checkpoint[f"{emigrant.island}"][f"{emigrant.island_rank}"]["active_on_island"]
                active_on_island[emigrant.generation, target_island] = 1
                active_on_island[emigrant.generation, self.island_idx] = 0
            departing = copy.deepcopy(emigrants)
            for ind in departing:
                # Informational only, every worker determines the responsible one via ``_responsible_rank``.
                ind.migrator_island_rank = ind.generation % self.island_sizes[target_island]
            for r in range(self.island_comm.size):  # Send emigrants to other intra-island workers for deactivation.
                if r == self.island_comm.rank:
                    continue  # No self-talk.
                self._send_message(self.island_comm, departing, r, SYNCHRONIZATION_TAG)
                log_string += f"Sent {len(departing)} individual(s) {departing} to intra-island worker {r} to deactivate.\n"
            for r in dest_island_workers:  # Send emigrants to all workers on the target island.
                self._send_message(self.propulate_comm, departing, r, MIGRATION_TAG)
                log_string += (
                    f"Sent {len(departing)} individual(s) to worker {r - self.island_displs[target_island]} "
                    + f"on target island {target_island}.\n"
                )

        n_active = len(self._get_active_individuals())
        log_string += f"After emigration: {n_active}/{len(self.population)} active.\n"
        log.debug(log_string)

    def _receive_immigrants(self) -> None:
        """Check for and possibly receive immigrants sent by other islands."""
        log_string = f"Island {self.island_idx} Worker {self.island_comm.rank} Generation {self.generation}: IMMIGRATION\n"
        probe_migrants = True
        while probe_migrants:
            stat = MPI.Status()
            probe_migrants = self.propulate_comm.iprobe(source=MPI.ANY_SOURCE, tag=MIGRATION_TAG, status=stat)
            log_string += f"Immigrant(s) to receive?...{probe_migrants}\n"
            if probe_migrants:
                immigrants = self.propulate_comm.recv(source=stat.Get_source(), tag=MIGRATION_TAG)
                self._received[MIGRATION_TAG][stat.Get_source()] += 1
                log_string += f"Received {len(immigrants)} immigrant(s) from global worker {stat.Get_source()}: {immigrants}\n"
                self._process_immigrants(immigrants)

        n_active = len(self._get_active_individuals())
        log_string += f"After immigration: {n_active}/{len(self.population)} active.\n"
        log.debug(log_string)

    def _deactivate_emigrants(self) -> None:
        """Check for and possibly receive emigrants from other intra-island workers to be deactivated."""
        log_string = f"Island {self.island_idx} Worker {self.island_comm.rank} Generation {self.generation}: DEACTIVATION\n"
        probe_sync = True
        while probe_sync:
            stat = MPI.Status()
            probe_sync = self.island_comm.iprobe(source=MPI.ANY_SOURCE, tag=SYNCHRONIZATION_TAG, status=stat)
            log_string += f"Emigrants from others to be deactivated to be received?...{probe_sync}\n"
            if probe_sync:
                emigrants = self.island_comm.recv(source=stat.Get_source(), tag=SYNCHRONIZATION_TAG)
                self._received[SYNCHRONIZATION_TAG][stat.Get_source()] += 1
                log_string += f"Got {len(emigrants)} emigrant(s) {emigrants} from worker {stat.Get_source()} to be deactivated.\n"
                self._process_emigrants(emigrants)

        n_active = len(self._get_active_individuals())
        log_string += f"After synchronization: {n_active}/{len(self.population)} active.\n"
        log.debug(log_string)

    # With real migration, an individual's ``active`` counter on a worker is the sum of the events the worker has
    # processed for it: +1 when it is bred on this island, +1 when it immigrates, -1 when it emigrates. Each event is
    # applied exactly once, so the counter is the same on all workers of the island once all messages are processed, no
    # matter in which order they arrived (it may transiently be 2 or -1 in between). The individual is active on the
    # island if and only if the counter is 1.

    def _process_bred_individual(self, ind: Individual) -> None:
        """Apply the event "``ind`` was bred and evaluated on this island by another worker" to the local population."""
        entry, _ = self._upsert_individual(ind)
        entry.active += 1

    def _process_immigrants(self, immigrants: List[Individual]) -> None:
        """Apply the event "``immigrants`` migrated to this island" to the local population."""
        for immigrant in immigrants:
            entry, _ = self._upsert_individual(immigrant)
            entry.active += 1

    def _process_emigrants(self, emigrants: List[Individual]) -> None:
        """Apply the event "``emigrants`` were sent away from this island by another worker" to the local population."""
        for emigrant in emigrants:
            entry, _ = self._upsert_individual(emigrant)
            entry.active -= 1

    def _migration_step(self, hdf5_checkpoint: h5py.File) -> None:
        """Perform the migration part of a generation."""
        if self.migration_prob <= 0.0:
            return
        # Emigration: Island sends individuals out. Happens on per-worker basis with certain probability.
        # NOTE skip emigration at the end; mostly to make checks easier
        if self.generation < self.generations - 1 and self.rng.random() < self.migration_prob:
            self._send_emigrants(hdf5_checkpoint)
        # Immigration: Check for incoming individuals from other islands.
        self._receive_immigrants()
        # Emigration: Check for emigrants from other intra-island workers to be deactivated.
        self._deactivate_emigrants()

    def _drain_migration_messages(self, hdf5_checkpoint: h5py.File) -> None:
        """Receive all outstanding immigrants and deactivation notices."""
        self._drain_messages(self.propulate_comm, MIGRATION_TAG, self._receive_immigrants)
        self._drain_messages(self.island_comm, SYNCHRONIZATION_TAG, self._deactivate_emigrants)
