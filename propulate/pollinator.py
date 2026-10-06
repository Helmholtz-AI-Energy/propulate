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
from .propagators import Propagator, SelectMax, SelectMin
from .propulator import Propulator
from .surrogate import Surrogate

log = logging.getLogger(__name__)


class Pollinator(Propulator):
    """
    Parallel propagator of populations with pollination.

    Individuals can actively exist on multiple evolutionary islands at a time, i.e., copies of emigrants are sent out
    and emigrating individuals are not deactivated on sending island for breeding. Instead, immigrants replace
    individuals on the target island according to an immigration policy set by the immigration propagator.

    Attributes
    ----------
    immigration_propagator : Type[propulate.propagators.Propagator]
        The immigration propagator, i.e., how to choose individuals to be replaced by immigrants on target island.

    Methods
    -------
    propulate()
        Run asynchronous evolutionary optimization routine with pollination.

    Notes
    -----
    The ``Pollinator`` class inherits all methods and attributes from the ``Propulator`` class.

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
        checkpoint_path: Union[Path, str] = Path("./"),
        migration_topology: Optional[np.ndarray] = None,
        migration_prob: float = 0.0,
        emigration_propagator: Type[Propagator] = SelectMin,
        immigration_propagator: Type[Propagator] = SelectMax,
        island_displs: Optional[np.ndarray] = None,
        island_sizes: Optional[np.ndarray] = None,
        surrogate_factory: Optional[Callable[[], Surrogate]] = None,
    ) -> None:
        """
        Initialize ``Pollinator`` with given parameters.

        Parameters
        ----------
        loss_fn : Union[Callable, Generator[float, None, None]]
            The loss function to be minimized.
        propagator : propulate.propagators.Propagator
            The propagator to apply for breeding.
        rng : random.Random
            The separate random number generator for the Propulate optimization.
        island_idx : int, optional
            The island index. Default is 0.
        island_comm : MPI.Comm, optional
            The intra-island communicator for communication within this island. Default is ``MPI.COMM_WORLD``.
        propulate_comm : MPI.Comm, optional
            The Propulate world communicator, consisting of rank 0 of each worker's sub communicator.
            Default is ``MPI.COMM_WORLD``.
        worker_sub_comm : MPI.Comm, optional
            The sub communicator for each (multi rank) worker. Default is ``MPI.COMM_SELF``.
        generations : int, optional
            The number of generations to run. Default is -1, i.e., run into wall-clock time limit.
        checkpoint_path : pathlib.Path, optional
            The path where checkpoints are loaded from and stored. Default is current working directory.
        migration_topology : numpy.ndarray, optional
            The 2D matrix where entry (i,j) specifies how many individuals are sent by island i to island j.
        migration_prob : float, optional
            The per-worker migration probability. Default is 0.0.
        emigration_propagator : Type[propulate.propagators.Propagator], optional
            The emigration propagator, i.e., how to choose individuals for emigration that are sent to the destination
            island. Should be some kind of selection operator. Default is ``SelectMin``.
        immigration_propagator : Type[propulate.propagators.Propagator], optional
            The immigration propagator, i.e., how to choose individuals to be replaced by immigrants on a target island.
            Should be some kind of selection operator. Default is ``SelectMax``.
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
        log.debug("Pollinator set up.")

    def _send_emigrants(self) -> None:
        """Perform migration, i.e. island sends individuals out to other islands."""
        log_string = f"Island {self.island_idx} Worker {self.island_comm.rank} Generation {self.generation}: EMIGRATION\n"
        # Determine relevant line of migration topology.
        assert self.migration_topology is not None
        to_migrate = self.migration_topology[self.island_idx, :]
        num_emigrants = np.amax(to_migrate)  # Determine maximum number of emigrants to be sent out at once.
        # For pollination, an emigration-responsible worker is not necessary as emigrating individuals are not
        # deactivated and copies are allowed.
        # All active individuals are eligible emigrants.
        eligible_emigrants = self._get_active_individuals()

        # Only perform migration if maximum number of emigrants to be sent out at once is smaller than current number
        # of eligible emigrants.
        if num_emigrants > len(eligible_emigrants):
            log.debug(
                f"Island {self.island_idx} Worker {self.island_comm.rank} Generation {self.generation}: "
                f"Population size {len(eligible_emigrants)} too small "
                f"to select {num_emigrants} migrants."
            )
            return

        # Loop through relevant part of migration topology.
        for target_island, offspring in enumerate(to_migrate):
            if offspring == 0:
                continue
            # Determine ranks of workers on target island in the Propulate world communicator.
            assert self.island_displs is not None
            assert self.island_sizes is not None
            displ = self.island_displs[target_island]
            count = self.island_sizes[target_island]
            dest_island = np.arange(displ, displ + count)

            # Worker in principle sends *different* individuals to each target island,
            # even though copies are allowed for pollination.
            emigrator = self.emigration_propagator(offspring)  # Set up emigration propagator.
            emigrants = emigrator(eligible_emigrants)  # Choose `offspring` eligible emigrants.
            assert isinstance(emigrants, list)
            log_string += f"Chose {len(emigrants)} emigrant(s): {emigrants}\n"

            # For pollination, do not deactivate emigrants on sending island!
            # Send emigrants to all workers on target island but only responsible worker will
            # choose individuals to be replaced by immigrants and tell other workers on target
            # island about them.
            departing = copy.deepcopy(emigrants)
            for ind in departing:
                # Informational only, every worker determines the responsible one via ``_responsible_rank``.
                ind.migrator_island_rank = ind.generation % self.island_sizes[target_island]
            for r in dest_island:  # Loop through Propulate world destination ranks.
                self._send_message(self.propulate_comm, departing, r, MIGRATION_TAG)
                log_string += (
                    f"Sent {len(departing)} individual(s) to worker {r - self.island_displs[target_island]} "
                    f"on target island {target_island}.\n"
                )

        num_active = len(self._get_active_individuals())
        log_string += f"After emigration: {num_active}/{len(self.population)} active.\n"
        log.debug(log_string)

    def _receive_immigrants(self, hdf5_checkpoint: h5py.File) -> None:
        """Check for and possibly receive immigrants send by other islands."""
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
                self._process_immigrants(immigrants, hdf5_checkpoint)

        num_active = len(self._get_active_individuals())
        log_string += f"After immigration: {num_active}/{len(self.population)} active."
        log.debug(log_string)

    def _deactivate_replaced_individuals(self) -> None:
        """Check for and receive individuals from other intra-island workers to be deactivated due to immigration."""
        log_string = f"Island {self.island_idx} Worker {self.island_comm.rank} Generation {self.generation}: REPLACEMENT\n"
        probe_sync = True
        while probe_sync:
            stat = MPI.Status()
            probe_sync = self.island_comm.iprobe(source=MPI.ANY_SOURCE, tag=SYNCHRONIZATION_TAG, status=stat)
            log_string += f"Individual(s) to replace...{probe_sync}\n"
            if probe_sync:
                to_replace = self.island_comm.recv(source=stat.Get_source(), tag=SYNCHRONIZATION_TAG)
                self._received[SYNCHRONIZATION_TAG][stat.Get_source()] += 1
                log_string += f"Got {len(to_replace)} replaced individual(s) {to_replace} from worker {stat.Get_source()} to be deactivated.\n"
                self._process_replaced_individuals(to_replace)

        num_active = len(self._get_active_individuals())
        log_string += f"After synchronization: {num_active}/{len(self.population)} active.\n"
        log.debug(log_string)

    # With pollination, an individual is active on an island (``active == 1``) from the moment it is bred there or the
    # first copy of it arrives until it is replaced (``active == 0``). Breeding and arrival of copies are idempotent
    # and replacement is final, so that all workers of the island end up with the same state regardless of the order in
    # which the corresponding messages arrive. The responsible worker replaces one of its individuals for each immigrant
    # that was new to the island, so the number of active individuals on an island equals the number bred there.

    def _process_immigrants(self, immigrants: List[Individual], hdf5_checkpoint: h5py.File) -> None:
        """Apply the event "copies of ``immigrants`` were pollinated to this island" to the local population."""
        replace_num = 0
        for immigrant in immigrants:
            entry, created = self._upsert_individual(immigrant)
            if not created:
                # Already known on this island (native, earlier copy, or already replaced): nothing to do.
                continue
            entry.active = 1
            if self._responsible_rank(entry) == self.island_comm.rank:
                # Only the responsible worker writes an individual's activity bit on this island, so that this write
                # cannot race with the deactivation it may perform below.
                hdf5_checkpoint[f"{entry.island}"][f"{entry.island_rank}"]["active_on_island"][
                    entry.generation, self.island_idx
                ] = 1
                replace_num += 1

        if replace_num == 0:
            return
        # Responsible worker only, so that different intra-island workers cannot choose the same individual
        # independently for replacement and thus deactivation.
        log.debug(
            f"Island {self.island_idx} Worker {self.island_comm.rank} Generation {self.generation}: "
            f"Responsible for choosing {replace_num} individual(s) to be replaced by immigrants."
        )
        eligible_for_replacement = [
            ind for ind in self.population.values() if ind.active > 0 and self._responsible_rank(ind) == self.island_comm.rank
        ]
        assert self.immigration_propagator is not None
        immigrator = self.immigration_propagator(replace_num)  # Set up immigration propagator.
        to_replace = immigrator(eligible_for_replacement)  # Choose individuals to be replaced by immigrants.
        assert isinstance(to_replace, list)
        for r in range(self.island_comm.size):  # Send individuals to be replaced to other intra-island workers.
            if r == self.island_comm.rank:
                continue  # No self-talk.
            self._send_message(self.island_comm, to_replace, r, SYNCHRONIZATION_TAG)
        self._process_replaced_individuals(to_replace)
        for individual in to_replace:
            hdf5_checkpoint[f"{individual.island}"][f"{individual.island_rank}"]["active_on_island"][
                individual.generation, self.island_idx
            ] = 0

    def _process_replaced_individuals(self, replaced: List[Individual]) -> None:
        """Apply the event "``replaced`` were replaced by immigrants on this island" to the local population."""
        for individual in replaced:
            entry, _ = self._upsert_individual(individual)
            entry.active = 0

    def _migration_step(self, hdf5_checkpoint: h5py.File) -> None:
        """Perform the pollination part of a generation."""
        if self.migration_prob <= 0.0:
            return
        # Emigration: Island sends individuals out. Happens on per-worker basis with certain probability.
        if self.rng.random() < self.migration_prob:
            self._send_emigrants()
        # Immigration: Island checks for incoming individuals from other islands.
        self._receive_immigrants(hdf5_checkpoint)
        # Immigration: Check for individuals replaced by other intra-island workers to be deactivated.
        self._deactivate_replaced_individuals()

    def _drain_migration_messages(self, hdf5_checkpoint: h5py.File) -> None:
        """Receive all outstanding immigrants and the replacement notices they trigger."""
        self._drain_messages(self.propulate_comm, MIGRATION_TAG, lambda: self._receive_immigrants(hdf5_checkpoint))
        self._drain_messages(self.island_comm, SYNCHRONIZATION_TAG, self._deactivate_replaced_individuals)
