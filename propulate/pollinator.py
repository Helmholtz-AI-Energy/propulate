import copy
import logging
import random
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple, Type, Union

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
    replaced : List[propulate.population.Individual]
        The individuals to be replaced by the immigrants.

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
        island_idx: int = 0,
        island_comm: MPI.Comm = MPI.COMM_WORLD,
        propulate_comm: MPI.Comm = MPI.COMM_WORLD,
        worker_sub_comm: MPI.Comm = MPI.COMM_SELF,
        generations: int = 0,
        checkpoint_path: Union[Path, str] = Path("./"),
        migration_topology: Optional[np.ndarray] = None,
        migration_prob: float = 0.0,
        emigration_propagator: Type[Propagator] = SelectMin,
        immigration_propagator: Type[Propagator] = SelectMax,
        island_displs: Optional[np.ndarray] = None,
        island_counts: Optional[np.ndarray] = None,
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
        self.immigration_propagator = immigration_propagator  # Immigration propagator
        self.replaced: List[Individual] = []  # Individuals to be replaced by immigrants

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
        eligible_emigrants, _ = self._get_active_individuals()

        # Only perform migration if maximum number of emigrants to be sent out at once is smaller than current number
        # of eligible emigrants.
        if num_emigrants <= len(eligible_emigrants):
            # Loop through relevant part of migration topology.
            for target_island, offspring in enumerate(to_migrate):
                if offspring == 0:
                    continue
                # Determine ranks of workers on target island in the Propulate world communicator.
                assert self.island_displs is not None
                assert self.island_counts is not None
                displ = self.island_displs[target_island]
                count = self.island_counts[target_island]
                dest_island = np.arange(displ, displ + count)

                # Worker in principle sends *different* individuals to each target island,
                # even though copies are allowed for pollination.
                emigrator = self.emigration_propagator(offspring)  # Set up emigration propagator.
                emigrants = emigrator(eligible_emigrants)  # Choose `offspring` eligible emigrants.
                log_string += f"Chose {len(emigrants)} emigrant(s): {emigrants}\n"

                # For pollination, do not deactivate emigrants on sending island!
                # Send emigrants to all workers on target island but only responsible worker will
                # choose individuals to be replaced by immigrants and tell other workers on target
                # island about them.
                departing = copy.deepcopy(emigrants)
                # Determine new responsible worker on target island.
                for ind in departing:
                    assert isinstance(ind, Individual)
                    ind.current = self.rng.randrange(0, count)
                    ind.migration_history += f"-{target_island}"
                    log_string += f"{ind} with migration history {ind.migration_history}\n"
                for r in dest_island:  # Loop through Propulate world destination ranks.
                    self.propulate_comm.send(copy.deepcopy(departing), dest=r, tag=MIGRATION_TAG)
                    log_string += (
                        f"Sent {len(departing)} individual(s) to worker {r - self.island_displs[target_island]} "
                        f"on target island {target_island}.\n"
                    )

            _, num_active = self._get_active_individuals()
            log_string += f"After emigration: {num_active}/{len(self.population)} active.\n"
            log.debug(log_string)

        else:
            log.debug(
                f"Island {self.island_idx} Worker {self.island_comm.rank} Generation {self.generation}: "
                f"Population size {len(eligible_emigrants)} too small "
                f"to select {num_emigrants} migrants."
            )

    def _receive_immigrants(self) -> None:
        """Check for and possibly receive immigrants send by other islands."""
        replace_num = 0
        log_string = f"Island {self.island_idx} Worker {self.island_comm.rank} Generation {self.generation}: IMMIGRATION\n"
        probe_migrants = True
        while probe_migrants:
            stat = MPI.Status()
            probe_migrants = self.propulate_comm.iprobe(source=MPI.ANY_SOURCE, tag=MIGRATION_TAG, status=stat)
            log_string += f"Immigrant(s) to receive?...{probe_migrants}\n"
            if probe_migrants:
                immigrants = self.propulate_comm.recv(source=stat.Get_source(), tag=MIGRATION_TAG)
                log_string += f"Received {len(immigrants)} immigrant(s) from global worker {stat.Get_source()}: {immigrants}\n"

                # Add immigrants to own population.
                for immigrant in immigrants:
                    immigrant.migration_steps += 1
                    assert immigrant.active is True
                    self.population.append(copy.deepcopy(immigrant))  # Append immigrant to population.

                    replace_num = 0
                    if self.island_comm.rank == immigrant.current:
                        replace_num += 1
                    log_string += f"Responsible for choosing {replace_num} individual(s) to be replaced by immigrants.\n"

                # Check whether rank equals responsible worker's rank so different intra-island workers
                # cannot choose the same individual independently for replacement and thus deactivation.
                if replace_num > 0:
                    # From current population, choose `replace_num` individuals to be replaced.
                    eligible_for_replacement = [
                        ind for ind in self.population if ind.active and ind.current == self.island_comm.rank
                    ]

                    immigrator = self.immigration_propagator(replace_num)  # Set up immigration propagator.
                    to_replace = immigrator(eligible_for_replacement)  # Choose individual to be replaced by immigrant.

                    # Send individuals to be replaced to other intra-island workers for deactivation.
                    for r in range(self.island_comm.size):
                        if r == self.island_comm.rank:
                            continue  # No self-talk.
                        self.island_comm.send(copy.deepcopy(to_replace), dest=r, tag=SYNCHRONIZATION_TAG)
                        log_string += (
                            f"Sent {len(to_replace)} individual(s) {to_replace} to intra-island worker {r} for replacement.\n"
                        )

                    # Deactivate individuals to be replaced in own population.
                    for individual in to_replace:
                        assert isinstance(individual, Individual)
                        assert individual.active is True
                        individual.active = False

        _, num_active = self._get_active_individuals()
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
                # Receive new individuals.
                to_replace = self.island_comm.recv(source=stat.Get_source(), tag=SYNCHRONIZATION_TAG)
                # Add new emigrants to list of emigrants to be deactivated.
                self.replaced = self.replaced + copy.deepcopy(to_replace)
                log_string += (
                    f"Got {len(to_replace)} new replaced individual(s) {to_replace} "
                    f"from worker {stat.Get_source()} to be deactivated.\n"
                    f"Overall {len(self.replaced)} individuals to deactivate: {self.replaced}\n"
                )
        replaced_copy = copy.deepcopy(self.replaced)
        for individual in replaced_copy:
            assert individual.active is True
            to_deactivate = [
                idx
                for idx, ind in enumerate(self.population)
                if ind == individual and ind.migration_steps == individual.migration_steps
            ]
            if len(to_deactivate) == 0:
                log_string += f"Individual {individual} to deactivate not yet received.\n"
                continue
            # NOTE As copies are allowed, len(to_deactivate) can be greater than 1.
            # However, only one of the copies should be replaced / deactivated.
            _, num_active_before = self._get_active_individuals()
            self.population[to_deactivate[0]].active = False
            self.replaced.remove(individual)
            _, num_active_after = self._get_active_individuals()
            log_string += (
                f"Before deactivation: {num_active_before}/{len(self.population)} active.\n"
                f"Deactivated {self.population[to_deactivate[0]]}.\n"
                f"{len(self.replaced)} individuals in replaced.\n"
                f"After deactivation: {num_active_after}/{len(self.population)} active.\n"
            )
        _, num_active = self._get_active_individuals()
        log_string += (
            f"After synchronization: {num_active}/{len(self.population)} active.\n{len(self.replaced)} individuals in replaced.\n"
        )
        log.debug(log_string)

    def _check_for_duplicates(self, active: bool, debug: int = 1) -> Tuple[List[List[Union[Individual, int]]], List[Individual]]:
        """
        Check for duplicates in current population.

        For pollination, duplicates are allowed as emigrants are sent as copies and not deactivated on sending island.

        Parameters
        ----------
        active : bool
            Whether to consider active individuals (True) or all individuals (False).
        debug : int, optional
            The debug level; 0 - silent; 1 - moderate, 2 - noisy (debug mode). Default is 1.

        Returns
        -------
        List[List[propulate.population.Individual | int]]
            The individuals and their occurrences.
        List[propulate.population.Individual]
            All unique individuals in the population.
        """
        if active:
            population, _ = self._get_active_individuals()
        else:
            population = self.population
        unique_inds: List[Individual] = []
        occurrences: List[List[Union[Individual, int]]] = []
        for individual in population:
            considered = False
            for ind in unique_inds:
                # As copies of individuals are allowed for pollination,
                # check for equivalence of traits and loss only when
                # determining unique individuals. To do so, use
                # self.equals(other) member function of Individual()
                # class instead of `==` operator.
                if individual.equals(ind):
                    considered = True
                    break
            if not considered:
                num_copies = population.count(individual)
                log.debug(
                    f"Island {self.island_idx} Worker {self.island_comm.rank} Generation {self.generation}: "
                    f"{individual} occurs {num_copies} time(s)."
                )
                unique_inds.append(individual)
                occurrences.append([individual, num_copies])
        return occurrences, unique_inds

    def propulate(self, logging_interval: int = 10, debug: int = 1) -> None:
        """
        Execute evolutionary algorithm using island model with pollination in parallel.

        Parameters
        ----------
        logging_interval : int, optional
            The logging interval. Default is 10.
        debug : int, optional
            The debug level; 0 - silent; 1 - moderate, 2 - noisy (debug mode). Default is 1.
        """
        if self.worker_sub_comm != MPI.COMM_SELF:
            self.generation = self.worker_sub_comm.bcast(self.generation, root=0)
        if self.propulate_comm is None:
            while self.generations <= -1 or self.generation < self.generations:
                # Breed and evaluate individual.
                self._evaluate_individual()
                self.generation += 1
            return
        if self.island_comm.rank == 0:
            log.info(f"Island {self.island_idx} has {self.island_comm.size} workers.")

        migration = True if self.migration_prob > 0 else False
        self.propulate_comm.barrier()

        # Loop over generations.
        # TODO this should probably be refactored, checkpointing can probably be handled in one place
        with h5py.File(self.checkpoint_path, "a", driver="mpio", comm=self.propulate_comm) as f:
            while self.generation < self.generations:
                if self.generation % int(logging_interval) == 0:
                    log.info(f"Island {self.island_idx} Worker {self.island_comm.rank}: In generation {self.generation}...")

                # Breed and evaluate individual.
                self._evaluate_individual(f)

                # Check for and possibly receive incoming individuals from other intra-island workers.
                self._receive_intra_island_individuals()

                if migration:
                    # Emigration: Island sends individuals out.
                    # Happens on per-worker basis with certain probability.
                    if self.rng.random() < self.migration_prob:
                        self._send_emigrants()

                    # Immigration: Island checks for incoming individuals from other islands.
                    self._receive_immigrants()

                    # Immigration: Check for individuals replaced by other intra-island workers to be deactivated.
                    self._deactivate_replaced_individuals()

                self.generation += 1  # Go to next generation.

        # Having completed all generations, the workers have to wait for each other.
        # Once all workers are done, they should check for incoming messages once again
        # so that each of them holds the complete final population and the found optimum
        # irrespective of the order they finished.

        self.propulate_comm.barrier()
        if self.propulate_comm.rank == 0:
            log.info("OPTIMIZATION DONE.\nNEXT: Final checks for incoming messages...")
        self.propulate_comm.barrier()

        # Final check for incoming individuals evaluated by other intra-island workers.
        self._receive_intra_island_individuals()
        self.propulate_comm.barrier()

        if migration:
            # Final check for incoming individuals from other islands.
            self._receive_immigrants()
            self.propulate_comm.barrier()

            # Immigration: Final check for individuals replaced by other intra-island workers to be deactivated.
            self._deactivate_replaced_individuals()
            self.propulate_comm.barrier()

            if len(self.replaced) > 0:
                log.info(
                    f"Island {self.island_idx} Worker {self.island_comm.rank} Generation {self.generation}: "
                    f"Finally {len(self.replaced)} individual(s) in replaced: {self.replaced}:\n{self.population}"
                )
                self._deactivate_replaced_individuals()
