import copy
import random
import logging
from pathlib import Path
from typing import Callable, Union, Type

import numpy as np
from mpi4py import MPI

from .propagators import Propagator, SelectMin
from .propulator import Propulator
from ._globals import MIGRATION_TAG, SYNCHRONIZATION_TAG


log = logging.getLogger(__name__)


class Migrator(Propulator):
    """
    Parallel propagator of populations using an island model with real migration.

    Individuals can only exist on one evolutionary island at a time, i.e., they are removed
    (i.e. deactivated for breeding) from the sending island upon emigration.
    """

    def __init__(
        self,
        loss_fn: Callable,
        propagator: Propagator,
        island_idx: int = 0,
        comm: MPI.Comm = MPI.COMM_WORLD,
        generations: int = -1,
        checkpoint_path: Union[str, Path] = Path("./"),
        migration_topology: np.ndarray = None,
        migration_prob: float = 0.0,
        emigration_propagator: Type[Propagator] = SelectMin,
        island_displs: np.ndarray = None,
        island_counts: np.ndarray = None,
        rng: random.Random = None,
    ) -> None:
        """
        Initialize ``Migrator`` with given parameters.

        Parameters
        ----------
        loss_fn: Callable
                 loss function to be minimized
        propagator: propulate.propagators.Propagator
                    propagator to apply for breeding
        island_idx: int
                    index of island
        comm: MPI.Comm
              intra-island communicator
        generations: int
                     number of generations to run
        checkpoint_path: Union[Path, str]
                         Path where checkpoints are loaded from and stored.
        migration_topology: numpy.ndarray
                            2D matrix where entry (i,j) specifies how many
                            individuals are sent by island i to island j
        migration_prob: float
                        per-worker migration probability
        emigration_propagator: type[propulate.propagators.Propagator]
                               emigration propagator, i.e., how to choose individuals
                               for emigration that are sent to destination island.
                               Should be some kind of selection operator.
        island_displs: numpy.ndarray
                    array with MPI.COMM_WORLD rank of each island's worker 0
                    Element i specifies MPI.COMM_WORLD rank of worker 0 on island with index i.
        island_counts: numpy.ndarray
                       array with number of workers per island
                       Element i specifies number of workers on island with index i.
        rng: random.Random
             random number generator
        """
        super().__init__(
            loss_fn,
            propagator,
            island_idx,
            comm,
            generations,
            checkpoint_path,
            migration_topology,
            migration_prob,
            emigration_propagator,
            island_displs,
            island_counts,
            rng,
        )
        # Set class attributes.
        self.emigrated = []  # emigrated individuals to be deactivated on sending island

    def _send_emigrants(self) -> None:
        """
        Perform migration, i.e. island sends individuals out to other islands.
        """
        log_string = f"Island {self.island_idx} Worker {self.comm.rank} Generation {self.generation}: EMIGRATION\n"
        # Determine relevant line of migration topology.
        to_migrate = self.migration_topology[self.island_idx, :]
        num_emigrants = np.sum(
            to_migrate, dtype=int
        ).item()  # Determine overall number of emigrants to be sent out.
        eligible_emigrants = [
            ind
            for ind in self.population
            if ind.active and ind.current == self.comm.rank
        ]

        # Only perform migration if overall number of emigrants to be sent
        # out is smaller than current number of eligible emigrants.
        if num_emigrants <= len(eligible_emigrants):
            # Select all migrants to be sent out in this migration step.
            emigrator = self.emigration_propagator(
                num_emigrants
            )  # Set up emigration propagator.
            all_emigrants = emigrator(
                eligible_emigrants
            )  # Choose `offspring` eligible emigrants.
            self.rng.shuffle(all_emigrants)
            # Loop through relevant part of migration topology.
            offsprings_sent = 0
            for target_island, offspring in enumerate(to_migrate):
                if offspring == 0:
                    continue
                # Determine MPI.COMM_WORLD ranks of workers on target island.
                displ = self.island_displs[target_island]
                count = self.island_counts[target_island]
                dest_island = np.arange(displ, displ + count)

                # Worker sends *different* individuals to each target island.
                emigrants = all_emigrants[
                    offsprings_sent : offsprings_sent + offspring
                ]  # Choose `offspring` eligible emigrants.
                offsprings_sent += offspring
                log_string += f"Chose {len(emigrants)} emigrant(s): {emigrants}\n"

                # Deactivate emigrants on sending island (true migration).
                for r in range(
                    self.comm.size
                ):  # Send emigrants to other intra-island workers for deactivation.
                    if r == self.comm.rank:
                        continue  # No self-talk.
                    self.comm.send(
                        copy.deepcopy(emigrants), dest=r, tag=SYNCHRONIZATION_TAG
                    )
                    log_string += (
                        f"Sent {len(emigrants)} individual(s) {emigrants} to "
                        f"intra-island worker {r} to deactivate.\n"
                    )

                # Send emigrants to target island.
                departing = copy.deepcopy(emigrants)
                # Determine new responsible worker on target island.
                for ind in departing:
                    ind.current = self.rng.randrange(0, count)
                for r in dest_island:  # Loop over MPI.COMM_WORLD destination ranks.
                    MPI.COMM_WORLD.send(
                        copy.deepcopy(departing), dest=r, tag=MIGRATION_TAG
                    )
                    log_string += (
                        f"Sent {len(departing)} individual(s) to worker {r-self.island_displs[target_island]} "
                        + f"on target island {target_island}.\n"
                    )

                # Deactivate emigrants for sending worker.
                for emigrant in emigrants:
                    # Look for emigrant to deactivate in original population list.
                    to_deactivate = [
                        idx
                        for idx, ind in enumerate(self.population)
                        if ind == emigrant
                        and ind.migration_steps == emigrant.migration_steps
                    ]
                    assert len(to_deactivate) == 1  # There should be exactly one!
                    _, n_active_before = self._get_active_individuals()
                    self.population[
                        to_deactivate[0]
                    ].active = False  # Deactivate emigrant in population.
                    _, n_active_after = self._get_active_individuals()
                    log_string += (
                        f"Deactivated own emigrant {self.population[to_deactivate[0]]}. "
                        + f"Active before/after: {n_active_before}/{n_active_after}\n"
                    )
            _, n_active = self._get_active_individuals()
            log_string += (
                f"After emigration: {n_active}/{len(self.population)} active.\n"
            )

            log.debug(log_string)

        else:
            log.debug(
                f"Island {self.island_idx} worker {self.comm.rank} generation {self.generation}: \n"
                f"Population size {len(eligible_emigrants)} too small "
                f"to select {num_emigrants} migrants."
            )

    def _receive_immigrants(self) -> None:
        """
        Check for and possibly receive immigrants send by other islands.

        Raises
        ------
        RuntimeError
            If identical immigrant is already active on target island for real migration.
        """
        log_string = f"Island {self.island_idx} Worker {self.comm.rank} Generation {self.generation}: IMMIGRATION\n"
        probe_migrants = True
        while probe_migrants:
            stat = MPI.Status()
            probe_migrants = MPI.COMM_WORLD.iprobe(
                source=MPI.ANY_SOURCE, tag=MIGRATION_TAG, status=stat
            )
            log_string += f"Immigrant(s) to receive?...{probe_migrants}\n"
            if probe_migrants:
                immigrants = MPI.COMM_WORLD.recv(
                    source=stat.Get_source(), tag=MIGRATION_TAG
                )
                log_string += (
                    f"Received {len(immigrants)} immigrant(s) from global "
                    f"worker {stat.Get_source()}: {immigrants}\n"
                )
                for immigrant in immigrants:
                    immigrant.migration_steps += 1
                    assert immigrant.active is True
                    catastrophic_failure = (
                        len(
                            [
                                ind
                                for ind in self.population
                                if ind == immigrant
                                and immigrant.migration_steps == ind.migration_steps
                                and immigrant.current == ind.current
                            ]
                        )
                        > 0
                    )
                    if catastrophic_failure:
                        raise RuntimeError(
                            log_string
                            + f"Identical immigrant {immigrant} already active on target  island {self.island_idx}."
                        )
                    self.population.append(
                        copy.deepcopy(immigrant)
                    )  # Append immigrant to population.
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
                ind
                for ind in self.population
                if ind == emigrant and ind.migration_steps == emigrant.migration_steps
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
                    f"Island {self.island_idx} Worker {self.comm.rank} Generation {self.generation}:\n"
                    f"Currently in emigrated: {emigrant}\n"
                    f"Island {self.island_idx} Worker {self.comm.rank} Generation {self.generation}: "
                    f"Currently in population: {existing_ind}\nEquivalence check: {existing_ind[0] == emigrant} "
                    f"{compare_traits} {existing_ind[0].loss == self.emigrated[idx].loss} "
                    f"{existing_ind[0].active == emigrant.active} {existing_ind[0].current == emigrant.current} "
                    f"{existing_ind[0].island == emigrant.island} "
                    f"{existing_ind[0].migration_steps == emigrant.migration_steps}"
                )
                break

        return check

    def _deactivate_emigrants(self) -> None:
        """
        Check for and possibly receive emigrants from other intra-island workers to be deactivated.
        """
        log_string = f"Island {self.island_idx} Worker {self.comm.rank} Generation {self.generation}: DEACTIVATION\n"
        probe_sync = True
        while probe_sync:
            stat = MPI.Status()
            probe_sync = self.comm.iprobe(
                source=MPI.ANY_SOURCE, tag=SYNCHRONIZATION_TAG, status=stat
            )
            log_string += f"Emigrants from others to be deactivated to be received?...{probe_sync}\n"
            if probe_sync:
                # Receive new emigrants.
                new_emigrants = self.comm.recv(
                    source=stat.Get_source(), tag=SYNCHRONIZATION_TAG
                )
                # Add new emigrants to list of emigrants to be deactivated.
                self.emigrated = self.emigrated + copy.deepcopy(new_emigrants)
                log_string += (
                    f"Got {len(new_emigrants)} new emigrant(s) {new_emigrants} "
                    + f"from worker {stat.Get_source()} to be deactivated.\n"
                    + f"Overall {len(self.emigrated)} individuals to deactivate: {self.emigrated}\n"
                )
            # TODO In while loop or not?
            emigrated_copy = copy.deepcopy(self.emigrated)
            for emigrant in emigrated_copy:
                assert emigrant.active is True
                to_deactivate = [
                    idx
                    for idx, ind in enumerate(self.population)
                    if ind == emigrant
                    and ind.migration_steps == emigrant.migration_steps
                ]
                if len(to_deactivate) == 0:
                    log_string += (
                        f"Individual {emigrant} to deactivate not yet received.\n"
                    )
                    continue
                assert len(to_deactivate) == 1
                self.population[to_deactivate[0]].active = False
                to_remove = [
                    idx
                    for idx, ind in enumerate(self.emigrated)
                    if ind == emigrant
                    and ind.migration_steps == emigrant.migration_steps
                ]
                assert len(to_remove) == 1
                self.emigrated.pop(to_remove[0])
                log_string += (
                    f"Deactivated {self.population[to_deactivate[0]]}.\n"
                    + f"{len(self.emigrated)} individuals in emigrated.\n"
                )
        _, n_active = self._get_active_individuals()
        log_string += (
            "After synchronization: "
            + f"{n_active}/{len(self.population)} active.\n"
            + f"{len(self.emigrated)} individuals in emigrated.\n"
        )
        log.debug(log_string)

    def _work(self, logging_interval: int, debug: int):
        """
        Execute evolutionary algorithm using island model with real migration in parallel.

        Parameters
        ----------
        logging_interval: int
                          logging interval
        debug: int
               verbosity/debug level; 0 - silent; 1 - moderate, 2 - noisy (debug mode)

        Raises
        ------
        ValueError
            If any individuals are left that should have been deactivated before (only for debug > 0).
        """

        if self.comm.rank == 0:
            log.info(f"Island {self.island_idx} has {self.comm.size} workers.")

        dump = True if self.comm.rank == 0 else False
        migration = True if self.migration_prob > 0 else False
        MPI.COMM_WORLD.barrier()

        # Loop over generations.
        while self.generations <= -1 or self.generation < self.generations:
            if self.generation % int(logging_interval) == 0:
                log.info(
                    f"Island {self.island_idx} Worker {self.comm.rank}: In generation {self.generation}..."
                )

            # Breed and evaluate individual.
            self._evaluate_individual()

            # Check for and possibly receive incoming individuals from other intra-island workers.
            self._receive_intra_island_individuals()

            # Migration.
            if migration:
                # Emigration: Island sends individuals out.
                # Happens on per-worker basis with certain probability.
                if self.rng.random() < self.migration_prob:
                    self._send_emigrants()

                # Immigration: Check for incoming individuals from other islands.
                self._receive_immigrants()

                # Emigration: Check for emigrants from other intra-island workers to be deactivated.
                self._deactivate_emigrants()
                if debug == 2:
                    check = self._check_emigrants_to_deactivate()
                    assert check is False

            if dump:  # Dump checkpoint.
                self._dump_checkpoint()

            dump = (
                self._determine_worker_dumping_next()
            )  # Determine worker dumping checkpoint in the next generation.
            self.generation += 1  # Go to next generation.

        # Having completed all generations, the workers have to wait for each other.
        # Once all workers are done, they should check for incoming messages once again
        # so that each of them holds the complete final population and the found optimum
        # irrespective of the order they finished.

        MPI.COMM_WORLD.barrier()
        if MPI.COMM_WORLD.rank == 0:
            log.info("OPTIMIZATION DONE.")
            log.info("NEXT: Final checks for incoming messages...")
        MPI.COMM_WORLD.barrier()

        # Final check for incoming individuals evaluated by other intra-island workers.
        self._receive_intra_island_individuals()
        MPI.COMM_WORLD.barrier()

        if migration:
            # Final check for incoming individuals from other islands.
            self._receive_immigrants()
            MPI.COMM_WORLD.barrier()

            # Emigration: Final check for emigrants from other intra-island workers to be deactivated.
            self._deactivate_emigrants()

            if debug > 0:
                check = self._check_emigrants_to_deactivate()
                assert check is False
                MPI.COMM_WORLD.barrier()
                if len(self.emigrated) > 0:
                    log.info(
                        f"Island {self.island_idx} Worker {self.comm.rank} Generation {self.generation}: "
                        f"Finally {len(self.emigrated)} individual(s) in emigrated: {self.emigrated}:\n"
                        f"{self.population}"
                    )
                    self._deactivate_emigrants()
                    if self._check_emigrants_to_deactivate():
                        raise ValueError(
                            "There should not be any individuals left that need to be deactivated."
                        )

            MPI.COMM_WORLD.barrier()

        # Final checkpointing on rank 0.
        if self.comm.rank == 0:
            self._dump_final_checkpoint()  # Dump checkpoint.
        MPI.COMM_WORLD.barrier()
        _ = self._determine_worker_dumping_next()
        MPI.COMM_WORLD.barrier()
