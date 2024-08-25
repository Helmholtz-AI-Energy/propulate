import copy
import logging
import random
import time
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

    Attributes
    ----------
    emigrated : List[propulate.population.Individual]
        A list of emigrated individuals to be deactivated on the sending island.

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
        island_idx: int = 0,
        island_comm: MPI.Comm = MPI.COMM_WORLD,
        propulate_comm: MPI.Comm = MPI.COMM_WORLD,
        worker_sub_comm: MPI.Comm = MPI.COMM_SELF,
        generations: int = -1,
        checkpoint_path: Union[str, Path] = Path("./"),
        migration_topology: Optional[np.ndarray] = None,
        migration_prob: float = 0.0,
        emigration_propagator: Type[Propagator] = SelectMin,
        island_displs: Optional[np.ndarray] = None,
        island_counts: Optional[np.ndarray] = None,
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
        island_displs : numpy.ndarray, optional
            An array with ``propulate_comm`` rank of each island's worker 0. Element i specifies the rank of worker 0 on
            island with index i in the Propulate communicator.
        island_counts : numpy.ndarray, optional
            An array with the number of workers per island. Element i specifies the number of workers on island i.
        surrogate_factory : Callable[[], propulate.surrogate.Surrogate], optional
           Function that returns a new instance of a ``Surrogate`` model.
           Only used when ``loss_fn`` is a generator function.
        """
        super().__init__(
            loss_fn,
            propagator,
            rng,
            island_idx,
            island_comm,
            propulate_comm,
            worker_sub_comm,
            generations,
            checkpoint_path,
            migration_topology,
            migration_prob,
            emigration_propagator,
            island_displs,
            island_counts,
            surrogate_factory,
        )
        # Set class attributes.
        self.emigrated: List[Individual] = []  # Emigrated individuals to be deactivated on sending island

    def _send_emigrants(self, hdf5_checkpoint: h5py.File) -> None:
        """Perform migration, i.e. island sends individuals out to other islands."""
        log_string = f"Island {self.island_idx} Worker {self.island_comm.rank} Generation {self.generation}: EMIGRATION\n"
        # Determine relevant line of migration topology.
        assert self.migration_topology is not None
        to_migrate = self.migration_topology[self.island_idx, :]
        num_emigrants = np.sum(to_migrate, dtype=int).item()  # Determine overall number of emigrants to be sent out.
        eligible_emigrants = [ind for ind in self.population.values() if ind.active and ind.current == self.island_comm.rank]

        # Only perform migration if overall number of emigrants to be sent
        # out is smaller than current number of eligible emigrants.
        if num_emigrants <= len(eligible_emigrants):
            # Select all migrants to be sent out in this migration step.
            emigrator = self.emigration_propagator(num_emigrants)  # Set up emigration propagator.
            all_emigrants = emigrator(eligible_emigrants)  # Choose `offspring` eligible emigrants.
            assert isinstance(all_emigrants, list)
            self.rng.shuffle(all_emigrants)
            # Loop through relevant part of migration topology.
            offsprings_sent = 0
            for target_island, offspring in enumerate(to_migrate):
                if offspring == 0:
                    continue
                # Determine self.propulate_comm ranks of workers on target island.
                assert self.island_displs is not None
                assert self.island_counts is not None
                displ = self.island_displs[target_island]
                count = self.island_counts[target_island]
                dest_island = np.arange(displ, displ + count)

                # Worker sends *different* individuals to each target island.
                emigrants = all_emigrants[offsprings_sent : offsprings_sent + offspring]  # Choose `offspring` eligible emigrants.
                offsprings_sent += offspring
                log_string += f"Chose {len(emigrants)} emigrant(s): {emigrants}\n"

                # Deactivate emigrants on sending island (true migration).
                for r in range(self.island_comm.size):  # Send emigrants to other intra-island workers for deactivation.
                    if r == self.island_comm.rank:
                        continue  # No self-talk.
                    self.island_comm.send(copy.deepcopy(emigrants), dest=r, tag=SYNCHRONIZATION_TAG)
                    log_string += f"Sent {len(emigrants)} individual(s) {emigrants} to intra-island worker {r} to deactivate.\n"

                # Send emigrants to target island.
                departing = copy.deepcopy(emigrants)
                for ind in departing:
                    hdf5_checkpoint[f"{ind.island}"][f"{ind.island_rank}"]["active_on_island"][ind.generation, self.island_idx] = (
                        False
                    )
                # Determine new responsible worker on target island.
                for ind in departing:
                    ind.current = self.rng.randrange(0, count)
                for r in dest_island:  # Loop over self.propulate_comm destination ranks.
                    self.propulate_comm.send(copy.deepcopy(departing), dest=r, tag=MIGRATION_TAG)
                    log_string += (
                        f"Sent {len(departing)} individual(s) to worker {r - self.island_displs[target_island]} "
                        + f"on target island {target_island}.\n"
                    )

                # Deactivate emigrants for sending worker.
                for emigrant in emigrants:
                    assert isinstance(emigrant, Individual)
                    # Look for emigrant to deactivate in original population list.
                    to_deactivate = [
                        key
                        for key, ind in self.population.items()
                        if ind == emigrant and ind.migration_steps == emigrant.migration_steps
                    ]
                    assert len(to_deactivate) == 1  # There should be exactly one!
                    _, n_active_before = self._get_active_individuals()
                    self.population[to_deactivate[0]].active = False  # Deactivate emigrant in population.
                    _, n_active_after = self._get_active_individuals()
                    log_string += (
                        f"Deactivated own emigrant {self.population[to_deactivate[0]]}. "
                        + f"Active before/after: {n_active_before}/{n_active_after}\n"
                    )
            _, n_active = self._get_active_individuals()
            log_string += f"After emigration: {n_active}/{len(self.population)} active.\n"

            log.debug(log_string)

        else:
            log.debug(
                f"Island {self.island_idx} worker {self.island_comm.rank} generation {self.generation}: \n"
                f"Population size {len(eligible_emigrants)} too small "
                f"to select {num_emigrants} migrants."
            )

    def _receive_immigrants(self, hdf5_checkpoint: h5py.File) -> None:
        """
        Check for and possibly receive immigrants send by other islands.

        Raises
        ------
        RuntimeError
            If identical immigrant is already active on target island for real migration.
        """
        log_string = f"Island {self.island_idx} Worker {self.island_comm.rank} Generation {self.generation}: IMMIGRATION\n"
        probe_migrants = True
        while probe_migrants:
            stat = MPI.Status()
            probe_migrants = self.propulate_comm.iprobe(source=MPI.ANY_SOURCE, tag=MIGRATION_TAG, status=stat)
            log_string += f"Immigrant(s) to receive?...{probe_migrants}\n"
            if probe_migrants:
                immigrants = self.propulate_comm.recv(source=stat.Get_source(), tag=MIGRATION_TAG)
                log_string += f"Received {len(immigrants)} immigrant(s) from global worker {stat.Get_source()}: {immigrants}\n"
                for immigrant in immigrants:
                    immigrant.migration_steps += 1
                    assert immigrant.active is True
                    catastrophic_failure = (
                        len(
                            [
                                ind
                                for ind in self.population.values()
                                if ind == immigrant
                                and immigrant.migration_steps == ind.migration_steps
                                and immigrant.current == ind.current
                            ]
                        )
                        > 0
                    )
                    if catastrophic_failure:
                        raise RuntimeError(
                            log_string + f"Identical immigrant {immigrant} already active on target  island {self.island_idx}."
                        )
                    hdf5_checkpoint[f"{immigrant.island}"][f"{immigrant.island_rank}"]["active_on_island"][immigrant.generation] = (
                        True
                    )
                    self.population[immigrant.island, immigrant.rank, immigrant.generation] = copy.deepcopy(
                        immigrant
                    )  # add immigrant to population
                    log_string += f"Added immigrant {immigrant} to population.\n"

                    # NOTE Do not remove obsolete individuals from population upon immigration
                    # as they should be deactivated in the next step anyway.

        _, n_active = self._get_active_individuals()
        log_string += f"After immigration: {n_active}/{len(self.population)} active.\n"

        log.debug(log_string)

    def _check_emigrants_to_deactivate(self) -> bool:
        """
        Redundant safety check for existence of emigrants that could not be deactivated in population.

        Returns
        -------
        bool
            True if emigrants to be deactivated exist in population, False if not.
        """
        check = False
        # Loop over emigrants still to be deactivated.
        for idx, emigrant in enumerate(self.emigrated):
            existing_ind = [
                ind for ind in self.population.values() if ind == emigrant and ind.migration_steps == emigrant.migration_steps
            ]
            if len(existing_ind) > 0:
                check = True
                # Check equivalence of actual traits, i.e., (hyper-)parameter values.
                compare_traits = True
                for key in emigrant.keys():
                    if existing_ind[0][key] == emigrant[key]:
                        continue
                    else:
                        compare_traits = False
                        break

                log.info(
                    f"Island {self.island_idx} Worker {self.island_comm.rank} Generation {self.generation}:\n"
                    f"Currently in emigrated: {emigrant}\n"
                    f"Island {self.island_idx} Worker {self.island_comm.rank} Generation {self.generation}: "
                    f"Currently in population: {existing_ind}\nEquivalence check: {existing_ind[0] == emigrant} "
                    f"{compare_traits} {existing_ind[0].loss == self.emigrated[idx].loss} "
                    f"{existing_ind[0].active == emigrant.active} {existing_ind[0].current == emigrant.current} "
                    f"{existing_ind[0].island == emigrant.island} "
                    f"{existing_ind[0].migration_steps == emigrant.migration_steps}"
                )
                break

        return check

    def _deactivate_emigrants(self) -> None:
        """Check for and possibly receive emigrants from other intra-island workers to be deactivated."""
        log_string = f"Island {self.island_idx} Worker {self.island_comm.rank} Generation {self.generation}: DEACTIVATION\n"
        probe_sync = True
        while probe_sync:
            stat = MPI.Status()
            probe_sync = self.island_comm.iprobe(source=MPI.ANY_SOURCE, tag=SYNCHRONIZATION_TAG, status=stat)
            log_string += f"Emigrants from others to be deactivated to be received?...{probe_sync}\n"
            if probe_sync:
                # Receive new emigrants.
                new_emigrants = self.island_comm.recv(source=stat.Get_source(), tag=SYNCHRONIZATION_TAG)
                # Add new emigrants to list of emigrants to be deactivated.
                self.emigrated = self.emigrated + copy.deepcopy(new_emigrants)
                log_string += (
                    f"Got {len(new_emigrants)} new emigrant(s) {new_emigrants} "
                    + f"from worker {stat.Get_source()} to be deactivated.\n"
                    + f"Overall {len(self.emigrated)} individuals to deactivate: {self.emigrated}\n"
                )
            emigrated_copy = copy.deepcopy(self.emigrated)
            for emigrant in emigrated_copy:
                assert emigrant.active is True
                to_deactivate = [
                    idx
                    for idx, ind in self.population.items()
                    if ind == emigrant and ind.migration_steps == emigrant.migration_steps
                ]
                if len(to_deactivate) == 0:
                    log_string += f"Individual {emigrant} to deactivate not yet received.\n"
                    continue
                assert len(to_deactivate) == 1
                self.population[to_deactivate[0]].active = False
                # NOTE emigrated is a list, population is a dict
                to_remove = [
                    idx
                    for idx, ind in enumerate(self.emigrated)
                    if ind == emigrant and ind.migration_steps == emigrant.migration_steps
                ]
                assert len(to_remove) == 1
                self.emigrated.pop(to_remove[0])
                log_string += (
                    f"Deactivated {self.population[to_deactivate[0]]}.\n" + f"{len(self.emigrated)} individuals in emigrated.\n"
                )
        _, n_active = self._get_active_individuals()
        log_string += (
            "After synchronization: "
            + f"{n_active}/{len(self.population)} active.\n"
            + f"{len(self.emigrated)} individuals in emigrated.\n"
        )
        log.debug(log_string)

    def propulate(self, logging_interval: int = 10, debug: int = 1) -> None:
        """
        Execute evolutionary algorithm using island model with real migration in parallel.

        Parameters
        ----------
        logging_interval : int, optional
            The logging interval. Default is 10.
        debug : int
            The debug level; 0 - silent; 1 - moderate, 2 - noisy (debug mode). Default is 1.

        Raises
        ------
        ValueError
            If any individuals are left that should have been deactivated before (only for debug > 0).
        """
        # TODO setting start_time in function that is overwritten is probably not great
        self.start_time = time.time_ns()
        if self.worker_sub_comm != MPI.COMM_SELF:
            self.generation = self.worker_sub_comm.bcast(self.generation, root=0)
        if self.propulate_comm is None:
            while self.generations <= -1 or self.generation < self.generations:
                # Breed and evaluate individual.
                # TODO this should be refactored, the subworkers don't need the logfile
                # TODO this needs to be addressed before merge, since multirank workers should fail with this
                # self._evaluate_individual(None)
                self.generation += 1
            return

        if self.island_comm.rank == 0:
            log.info(f"Island {self.island_idx} has {self.island_comm.size} workers with {self.worker_sub_comm.size} ranks each.")

        migration = True if self.migration_prob > 0 else False
        self.propulate_comm.barrier()

        # Loop over generations.
        # TODO this should probably be refactored, checkpointing can probably be handled in one place
        # TODO does not work for multi rank workers
        with h5py.File(self.checkpoint_path, "a", driver="mpio", comm=self.propulate_comm) as f:
            while self.generation < self.generations:
                if self.generation % int(logging_interval) == 0:
                    log.info(f"Island {self.island_idx} Worker {self.island_comm.rank}: In generation {self.generation}...")

                # Breed and evaluate individual.
                self._evaluate_individual(f)

                # Check for and possibly receive incoming individuals from other intra-island workers.
                self._receive_intra_island_individuals()

                # Migration.
                if migration:
                    # Emigration: Island sends individuals out.
                    # Happens on per-worker basis with certain probability.
                    if self.rng.random() < self.migration_prob:
                        self._send_emigrants(f)

                    # Immigration: Check for incoming individuals from other islands.
                    self._receive_immigrants(f)

                    # Emigration: Check for emigrants from other intra-island workers to be deactivated.
                    self._deactivate_emigrants()
                    if debug == 2:
                        check = self._check_emigrants_to_deactivate()
                        assert check is False
                self.generation += 1  # Go to next generation.

        # Having completed all generations, the workers have to wait for each other.
        # Once all workers are done, they should check for incoming messages once again
        # so that each of them holds the complete final population and the found optimum
        # irrespective of the order they finished.

        self.propulate_comm.barrier()
        if self.propulate_comm.rank == 0:
            log.info("OPTIMIZATION DONE.")
            log.info("NEXT: Final checks for incoming messages...")
        self.propulate_comm.barrier()

        # Final check for incoming individuals evaluated by other intra-island workers.
        self._receive_intra_island_individuals()
        self.propulate_comm.barrier()

        if migration:
            # Final check for incoming individuals from other islands.
            with h5py.File(self.checkpoint_path, "a", driver="mpio", comm=self.propulate_comm) as f:
                self._receive_immigrants(f)
            self.propulate_comm.barrier()

            # Emigration: Final check for emigrants from other intra-island workers to be deactivated.
            self._deactivate_emigrants()

            if debug > 0:
                check = self._check_emigrants_to_deactivate()
                assert check is False
                self.propulate_comm.barrier()
                if len(self.emigrated) > 0:
                    log.info(
                        f"Island {self.island_idx} Worker {self.island_comm.rank} Generation {self.generation}: "
                        f"Finally {len(self.emigrated)} individual(s) in emigrated: {self.emigrated}:\n"
                        f"{self.population}"
                    )
                    self._deactivate_emigrants()
                    if self._check_emigrants_to_deactivate():
                        raise ValueError("There should not be any individuals left that need to be deactivated.")
