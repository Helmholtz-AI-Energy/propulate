import copy
import random
import os
import pickle
import time
from typing import Callable, Union, Tuple, List
from operator import attrgetter
from pathlib import Path

import deepdiff
import numpy as np
from mpi4py import MPI

from ._globals import DUMP_TAG, INDIVIDUAL_TAG, MIGRATION_TAG, SYNCHRONIZATION_TAG
from .propagators import Propagator, SelectMin, SelectMax
from .population import Individual


class Pollinator:
    """
    Parallel propagator of populations with pollination.

    Individuals can actively exist on multiple evolutionary islands at a time, i.e.,
    copies of emigrants are sent out and emigrating individuals are not deactivated on
    sending island for breeding. Instead, immigrants replace individuals on the target
    island according to an immigration policy set by the immigration propagator.
    """

    def __init__(
        self,
        loss_fn: Callable,
        propagator: Propagator,
        island_idx: int = 0,
        comm: MPI.Comm = MPI.COMM_WORLD,
        generations: int = 0,
        checkpoint_path: Union[Path, str] = Path("./"),
        migration_topology: np.ndarray = None,
        migration_prob: float = 0.0,
        emigration_propagator: Propagator = SelectMin,
        immigration_propagator: Propagator = SelectMax,
        island_displs: np.ndarray = None,
        island_counts: np.ndarray = None,
        rng: random.Random = None
    ) -> None:
        """
        Initialize Pollinator with given parameters.

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
        checkpoint_path: Path
                         path where checkpoints are loaded from and stored.
        migration_topology: numpy.ndarray
                            2D matrix where entry (i,j) specifies how many
                            individuals are sent by island i to island j
        migration_prob: float
                        per-worker migration probability
        emigration_propagator: propulate.propagators.Propagator
                               emigration propagator, i.e., how to choose individuals
                               for emigration that are sent to destination island.
                               Should be some kind of selection operator.
        immigration_propagator: propulate.propagators.Propagator
                                immigration propagator, i.e., how to choose individuals
                                to be replaced by immigrants on target island.
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
        # Set class attributes.
        self.loss_fn = loss_fn  # callable loss function
        self.propagator = propagator  # evolutionary propagator
        if generations == 0:  # If number of iterations requested == 0.
            if MPI.COMM_WORLD.rank == 0:
                print("Requested number of generations is zero...[RETURN]")
            return
        self.generations = generations  # number of generations, i.e., number of evaluations per individual
        self.generation = 0  # current generation not yet evaluated
        self.island_idx = island_idx  # island index
        self.comm = comm  # intra-island communicator
        self.checkpoint_path = Path(checkpoint_path)  # checkpoint path
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.migration_prob = migration_prob  # per-rank migration probability
        self.migration_topology = migration_topology  # migration topology
        self.island_displs = island_displs  # MPI.COMM_WORLD rank of each island's worker 0
        self.island_counts = island_counts  # number of workers on each island
        self.emigration_propagator = emigration_propagator  # emigration propagator
        self.immigration_propagator = immigration_propagator  # immigration propagator
        self.replaced = []  # individuals to be replaced by immigrants
        self.rng = rng

        # Load initial population of evaluated individuals from checkpoint if exists.
        load_ckpt_file = self.checkpoint_path / f"island_{self.island_idx}_ckpt.pkl"
        if not os.path.isfile(load_ckpt_file):  # If not exists, check for backup file.
            load_ckpt_file = load_ckpt_file.with_suffix(".bkp")

        if os.path.isfile(load_ckpt_file):
            with open(load_ckpt_file, "rb") as f:
                try:
                    self.population = pickle.load(f)
                    self.generation = max([x.generation for x in self.population if x.rank == self.comm.rank]) + 1
                    if self.comm.rank == 0:
                        print(
                            "NOTE: Valid checkpoint file found. "
                            f"Resuming from generation {self.generation} of loaded population..."
                        )
                except OSError:
                    self.population = []
                    if self.comm.rank == 0:
                        print(
                            "NOTE: No valid checkpoint file found. "
                            "Initializing population randomly..."
                        )
        else:
            self.population = []
            if self.comm.rank == 0:
                print(
                    "NOTE: No valid checkpoint file given. "
                    "Initializing population randomly..."
                )

    def propulate(
            self,
            logging_interval: int = 10,
            debug: int = 1
    ) -> None:
        """
        Run evolutionary optimization.

        Parameters
        ----------
        logging_interval : int
                           Print each worker's progress every
                           `logging_interval`th generation.
        debug: int
               verbosity/debug level; 0 - silent; 1 - moderate, 2 - noisy (debug mode)
        """
        self._work(logging_interval, debug)

    def _get_active_individuals(self) -> Tuple[List[Individual], int]:
        """
        Get active individuals in current population list.

        Returns
        -------
        list[propulate.population.Individual]: currently active individuals in population
        int: number of currently active individuals
        """
        active_pop = [ind for ind in self.population if ind.active]

        return active_pop, len(active_pop)

    def _breed(self) -> Individual:
        """
        Apply propagator to current population of active
        individuals to breed new individual.

        Returns
        -------
        propulate.population.Individual: newly bred individual
        """
        active_pop, _ = self._get_active_individuals()
        ind = self.propagator(active_pop)  # Breed new individual from active population.
        ind.generation = self.generation  # Set generation.
        ind.rank = self.comm.rank  # Set worker rank.
        ind.active = True  # If True, individual is active for breeding.
        ind.island = self.island_idx  # Set birth island.
        ind.current = self.comm.rank  # Set worker responsible for migration.
        ind.migration_steps = 0  # Set number of migration steps performed.
        ind.migration_history = str(self.island_idx)
        return ind  # Return new individual.

    def _evaluate_individual(
            self,
            debug: int
    ) -> None:
        """
        Breed and evaluate individual.

        Parameters
        ----------
        debug: int
               verbosity/debug level; 0 - silent; 1 - moderate, 2 - noisy (debug mode)
        """
        ind = self._breed()  # Breed new individual.
        start_time = time.time()  # Start evaluation timer.
        ind.loss = self.loss_fn(ind)  # Evaluate loss.
        ind.evaltime = time.time()  # Stop evaluation timer.
        ind.evalperiod = ind.evaltime - start_time  # Calculate evaluation time.
        self.population.append(ind)  # Add evaluated individual to worker-local population.
        if debug == 2:
            print(
                f"Island {self.island_idx} Worker {self.comm.rank} Generation {self.generation}: BREEDING\n"
                f"Bred and evaluated individual {ind}.\n"
            )

        # Tell other workers in own island about results to synchronize their populations.
        for r in range(self.comm.size):  # Loop over ranks in intra-island communicator.
            if r == self.comm.rank:
                continue  # No self-talk.
            self.comm.send(copy.deepcopy(ind), dest=r, tag=INDIVIDUAL_TAG)

    def _receive_intra_island_individuals(
            self,
            debug: int
    ) -> None:
        """
        Check for and possibly receive incoming individuals
        evaluated by other workers within own island.

        Parameters
        ----------
        debug: int
               verbosity level; 0 - silent; 1 - moderate, 2 - noisy (debug mode)
        """
        log_string = f"Island {self.island_idx} Worker {self.comm.rank} Generation {self.generation}: " \
                     f"INTRA-ISLAND SYNCHRONIZATION\n"
        probe_ind = True
        while probe_ind:
            stat = MPI.Status()  # Retrieve status of reception operation, including source and tag.
            probe_ind = self.comm.iprobe(source=MPI.ANY_SOURCE, tag=INDIVIDUAL_TAG, status=stat)
            # If True, continue checking for incoming messages. Tells whether message corresponding
            # to filters passed is waiting for reception via a flag that it sets.
            # If no such message has arrived yet, it returns False.
            log_string += f"Incoming individual to receive?...{probe_ind}\n"
            if probe_ind:
                # Receive individual and add it to own population.
                ind_temp = self.comm.recv(source=stat.Get_source(), tag=INDIVIDUAL_TAG)
                self.population.append(copy.deepcopy(ind_temp))
                log_string += f"Added individual {ind_temp} from worker {stat.Get_source()} to own population.\n"
        _, n_active = self._get_active_individuals()
        log_string += (
            f"After probing within island: {n_active}/{len(self.population)} active.\n"
        )
        if debug == 2:
            print(log_string)

    def _send_emigrants(
            self,
            debug: int
    ) -> None:
        """
        Perform migration, i.e. island sends individuals out to other islands.

        Parameters
        ----------
        debug: int
               verbosity/debug level; 0 - silent; 1 - moderate, 2 - noisy (debug mode)
        """
        log_string = f"Island {self.island_idx} Worker {self.comm.rank} Generation {self.generation}: EMIGRATION\n"
        # Determine relevant line of migration topology.
        to_migrate = self.migration_topology[self.island_idx, :]
        num_emigrants = np.amax(to_migrate)  # Determine maximum number of emigrants to be sent out at once.
        # NOTE For pollination, emigration-responsible worker not necessary as emigrating individuals are
        # not deactivated and copies are allowed.
        # All active individuals are eligible emigrants.
        eligible_emigrants, _ = self._get_active_individuals()

        # Only perform migration if maximum number of emigrants to be sent
        # out at once is smaller than current number of eligible emigrants.
        if num_emigrants <= len(eligible_emigrants):
            # Loop through relevant part of migration topology.
            for target_island, offspring in enumerate(to_migrate):
                if offspring == 0:
                    continue
                # Determine MPI.COMM_WORLD ranks of workers on target island.
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
                    ind.current = self.rng.randrange(0, count)
                    ind.migration_history += f"-{target_island}"
                    ind.timestamp = time.time()
                    if debug == 2:
                        log_string += f"{ind} with migration history {ind.migration_history}\n"
                for r in dest_island:  # Loop through MPI.COMM_WORLD destination ranks.
                    MPI.COMM_WORLD.send(copy.deepcopy(departing), dest=r, tag=MIGRATION_TAG)
                    log_string += (
                        f"Sent {len(departing)} individual(s) to worker {r-self.island_displs[target_island]} "
                        f"on target island {target_island}.\n"
                    )

            _, num_active = self._get_active_individuals()
            log_string += (
                f"After emigration: {num_active}/{len(self.population)} active.\n"
            )
            if debug == 2:
                print(log_string)

        else:
            if debug == 2:
                print(
                    f"Island {self.island_idx} Worker {self.comm.rank} Generation {self.generation}: "
                    f"Population size {len(eligible_emigrants)} too small "
                    f"to select {num_emigrants} migrants."
                )

    def _receive_immigrants(
            self,
            debug: int
    ) -> None:
        """
        Check for and possibly receive immigrants send by other islands.

        Parameters
        ----------
        debug: int
               verbosity level; 0 - silent; 1 - moderate, 2 - noisy (debug mode)
        """
        log_string = f"Island {self.island_idx} Worker {self.comm.rank} Generation {self.generation}: IMMIGRATION\n"
        probe_migrants = True
        while probe_migrants:
            stat = MPI.Status()
            probe_migrants = MPI.COMM_WORLD.iprobe(source=MPI.ANY_SOURCE, tag=MIGRATION_TAG, status=stat)
            log_string += f"Immigrant(s) to receive?...{probe_migrants}\n"
            if probe_migrants:
                immigrants = MPI.COMM_WORLD.recv(source=stat.Get_source(), tag=MIGRATION_TAG)
                log_string += (
                    f"Received {len(immigrants)} immigrant(s) from global "
                    f"worker {stat.Get_source()}: {immigrants}\n"
                )

                # Add immigrants to own population.
                for immigrant in immigrants:
                    immigrant.migration_steps += 1
                    assert immigrant.active is True
                    self.population.append(copy.deepcopy(immigrant))  # Append immigrant to population.

                    replace_num = 0
                    if self.comm.rank == immigrant.current:
                        replace_num += 1
                    log_string += (
                        f"Responsible for choosing {replace_num} individual(s) "
                        f"to be replaced by immigrants.\n"
                    )

                # Check whether rank equals responsible worker's rank so different intra-island workers
                # cannot choose the same individual independently for replacement and thus deactivation.
                if replace_num > 0:
                    # From current population, choose `replace_num` individuals to be replaced.
                    # eligible_for_replacement = [ind for ind in self.population[:-len(immigrants)] if ind.active \
                    eligible_for_replacement = [
                        ind
                        for ind in self.population
                        if ind.active and ind.current == self.comm.rank
                    ]

                    immigrator = self.immigration_propagator(replace_num)  # Set up immigration propagator.
                    to_replace = immigrator(eligible_for_replacement)  # Choose individual to be replaced by immigrant.

                    # Send individuals to be replaced to other intra-island workers for deactivation.
                    for r in range(self.comm.size):
                        if r == self.comm.rank:
                            continue  # No self-talk.
                        self.comm.send(
                            copy.deepcopy(to_replace), dest=r, tag=SYNCHRONIZATION_TAG
                        )
                        log_string += (
                            f"Sent {len(to_replace)} individual(s) {to_replace} to "
                            f"intra-island worker {r} for replacement.\n"
                        )

                    # Deactivate individuals to be replaced in own population.
                    for individual in to_replace:
                        assert individual.active is True
                        individual.active = False

        _, num_active = self._get_active_individuals()
        log_string += (
            f"After immigration: {num_active}/{len(self.population)} active.\n"
        )
        if debug == 2:
            print(log_string)

    def _deactivate_replaced_individuals(
            self,
            debug: int
    ) -> None:
        """
        Check for and possibly receive individuals from other intra-island workers to be deactivated
        because of immigration.

        Parameters
        ----------
        debug: int
               verbosity/debug level; 0 - silent; 1 - moderate, 2 - noisy (debug mode)
        """
        log_string = f"Island {self.island_idx} Worker {self.comm.rank} Generation {self.generation}: REPLACEMENT\n"
        probe_sync = True
        while probe_sync:
            stat = MPI.Status()
            probe_sync = self.comm.iprobe(
                source=MPI.ANY_SOURCE, tag=SYNCHRONIZATION_TAG, status=stat
            )
            log_string += f"Individual(s) to replace...{probe_sync}\n"
            if probe_sync:
                # Receive new individuals.
                to_replace = self.comm.recv(
                    source=stat.Get_source(), tag=SYNCHRONIZATION_TAG
                )
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
                if ind == individual
                and ind.migration_steps == individual.migration_steps
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
            f"After synchronization: {num_active}/{len(self.population)} active.\n"
            f"{len(self.replaced)} individuals in replaced.\n"
        )
        if debug == 2:
            print(log_string)

    def _get_unique_individuals(self) -> List[Individual]:
        """
        Get unique individuals in terms of traits and loss in current population.

        Returns
        -------
        list[propulate.population.Individual]: unique individuals
        """
        unique_inds = []
        for individual in self.population:
            considered = False
            for ind in unique_inds:
                # Check for equivalence of traits only when
                # determining unique individuals. To do so, use
                # self.equals(other) member function of Individual()
                # class instead of `==` operator.
                if individual.equals(ind):
                    considered = True
                    break
            if not considered:
                unique_inds.append(individual)
        return unique_inds

    def _check_for_duplicates(
            self,
            active: bool,
            debug: int
    ) -> Tuple[List[List[Union[Individual, int]]], List[Individual]]:
        """
        Check for duplicates in current population.

        For pollination, duplicates are allowed as emigrants are sent as copies
        and not deactivated on sending island.

        Parameters
        ----------
        active: bool
                Whether to consider active individuals (True) or all individuals (False)
        debug: int
               verbosity/debug level; 0 - silent; 1 - moderate, 2 - noisy (debug mode)

        Returns
        -------
        list[list[propulate.population.Individual | int]]: individuals and their occurrences
        list[propulate.population.Individual]: unique individuals in population
        """
        if active:
            population, _ = self._get_active_individuals()
        else:
            population = self.population
        unique_inds = []
        occurrences = []
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
                if debug == 2:
                    print(
                        f"Island {self.island_idx} Worker {self.comm.rank} Generation {self.generation}: "
                        f"{individual} occurs {num_copies} time(s)."
                    )
                unique_inds.append(individual)
                occurrences.append([individual, num_copies])
        return occurrences, unique_inds

    def _check_intra_island_synchronization(
            self,
            populations: List[List[Individual]]
    ) -> bool:
        """
        Check synchronization of populations of workers within one island.

        Parameters
        ----------
        populations: list[list[propulate.population.Individual]]
                     list of islands' sorted population lists

        Returns
        -------
        bool: True if populations are synchronized, False if not.
        """
        synchronized = True
        for idx, population in enumerate(populations):
            difference = deepdiff.DeepDiff(population, populations[0], ignore_order=True)
            if len(difference) == 0:
                continue
            print(
                f"Island {self.island_idx} Worker {self.comm.rank}: Population not synchronized:\n"
                f"{difference}\n"
            )
            synchronized = False
        return synchronized

    def _work(
            self,
            logging_interval: int,
            debug: int):
        """
        Execute evolutionary algorithm in parallel.

        Parameters
        ----------
        logging_interval: int
                          logging interval
        debug: int
               verbosity/debug level; 0 - silent; 1 - moderate, 2 - noisy (debug mode)
        """
        if self.comm.rank == 0:
            print(f"Island {self.island_idx} has {self.comm.size} workers.")

        dump = True if self.comm.rank == 0 else False
        migration = True if self.migration_prob > 0 else False
        MPI.COMM_WORLD.barrier()

        # Loop over generations.
        while self.generations <= -1 or self.generation < self.generations:

            if debug == 1 and self.generation % int(logging_interval) == 0:
                print(f"Island {self.island_idx} Worker {self.comm.rank}: In generation {self.generation}...")

            # Breed and evaluate individual.
            self._evaluate_individual(debug)

            # Check for and possibly receive incoming individuals from other intra-island workers.
            self._receive_intra_island_individuals(debug)

            if migration:
                # Emigration: Island sends individuals out.
                # Happens on per-worker basis with certain probability.
                if self.rng.random() < self.migration_prob:
                    self._send_emigrants(debug)

                # Immigration: Island checks for incoming individuals from other islands.
                self._receive_immigrants(debug)

                # Immigration: Check for individuals replaced by other intra-island workers to be deactivated.
                self._deactivate_replaced_individuals(debug)

            if dump:  # Dump checkpoint.
                if debug == 2:
                    print(
                        f"Island {self.island_idx} Worker {self.comm.rank} Generation {self.generation}: "
                          f"Dumping checkpoint..."
                    )
                save_ckpt_file = self.checkpoint_path / f"island_{self.island_idx}_ckpt.pkl"
                if os.path.isfile(save_ckpt_file):
                    try:
                        os.replace(save_ckpt_file, save_ckpt_file.with_suffix(".bkp"))
                    except OSError as e:
                        print(e)
                with open(save_ckpt_file, "wb") as f:
                    pickle.dump(self.population, f)

                dest = self.comm.rank + 1 if self.comm.rank + 1 < self.comm.size else 0
                self.comm.send(copy.deepcopy(dump), dest=dest, tag=DUMP_TAG)
                dump = False

            stat = MPI.Status()
            probe_dump = self.comm.iprobe(source=MPI.ANY_SOURCE, tag=DUMP_TAG, status=stat)
            if probe_dump:
                dump = self.comm.recv(source=stat.Get_source(), tag=DUMP_TAG)
                if debug == 2:
                    print(
                        f"Island {self.island_idx} Worker {self.comm.rank} Generation {self.generation}: "
                        f"Going to dump next: {dump}. Before: Worker {stat.Get_source()}"
                    )

            # Go to next generation.
            self.generation += 1

        # Having completed all generations, the workers have to wait for each other.
        # Once all workers are done, they should check for incoming messages once again
        # so that each of them holds the complete final population and the found optimum
        # irrespective of the order they finished.

        MPI.COMM_WORLD.barrier()
        if MPI.COMM_WORLD.rank == 0:
            print("OPTIMIZATION DONE.\nNEXT: Final checks for incoming messages...")
        MPI.COMM_WORLD.barrier()

        # Final check for incoming individuals evaluated by other intra-island workers.
        self._receive_intra_island_individuals(debug)
        MPI.COMM_WORLD.barrier()

        if migration:
            # Final check for incoming individuals from other islands.
            self._receive_immigrants(debug)
            MPI.COMM_WORLD.barrier()

            # Immigration: Final check for individuals replaced by other intra-island workers to be deactivated.
            self._deactivate_replaced_individuals(debug)
            MPI.COMM_WORLD.barrier()

            if debug > 0:
                if len(self.replaced) > 0:
                    print(
                        f"Island {self.island_idx} Worker {self.comm.rank} Generation {self.generation}: "
                        f"Finally {len(self.replaced)} individual(s) in replaced: {self.replaced}:\n"
                        f"{self.population}"
                    )
                    self._deactivate_replaced_individuals(debug)
                MPI.COMM_WORLD.barrier()

        # Final checkpointing on rank 0.
        save_ckpt_file = self.checkpoint_path / f"island_{self.island_idx}_ckpt.pkl"
        if self.comm.rank == 0:  # Dump checkpoint.
            if os.path.isfile(save_ckpt_file):
                try:
                    os.replace(save_ckpt_file, save_ckpt_file.with_suffix(".bkp"))
                except OSError as e:
                    print(e)
                with open(save_ckpt_file, "wb") as f:
                    pickle.dump(self.population, f)

        MPI.COMM_WORLD.barrier()
        stat = MPI.Status()
        probe_dump = self.comm.iprobe(source=MPI.ANY_SOURCE, tag=DUMP_TAG, status=stat)
        if probe_dump:
            _ = self.comm.recv(source=stat.Get_source(), tag=DUMP_TAG)
        MPI.COMM_WORLD.barrier()

    def summarize(
            self,
            top_n: int = 1,
            debug: int = 1
    ) -> List[Union[List[Individual], Individual]]:
        """
        Get top-n results from propulate optimization.

        Parameters
        ----------
        top_n: int
               number of best results to report
        debug: int
               verbosity/debug level; 0 - silent; 1 - moderate, 2 - noisy (debug mode)

        Returns
        -------
        list[list[Individual] | Individual]]: top-n best individuals on each island
        """
        active_pop, num_active = self._get_active_individuals()
        assert (np.all(np.array(self.comm.allgather(num_active), dtype=int) == num_active))
        if self.island_counts is not None:
            num_active = int(MPI.COMM_WORLD.allreduce(num_active / self.island_counts[self.island_idx]))

        MPI.COMM_WORLD.barrier()
        if MPI.COMM_WORLD.rank == 0:
            print("\n###########")
            print("# SUMMARY #")
            print("###########\n")
            print(
                f"Number of currently active individuals is {num_active}. "
                f"\nExpected overall number of evaluations is {self.generations*MPI.COMM_WORLD.size}."
            )
        # Only double-check number of occurrences of each individual for debug level 2.
        if debug == 2:
            populations = self.comm.gather(self.population, root=0)
            occurrences, _ = self._check_for_duplicates(True, debug)
            if self.comm.rank == 0:
                if self._check_intra_island_synchronization(populations):
                    print(f"Island {self.island_idx}: Populations among workers synchronized.")
                else:
                    print(f"Island {self.island_idx}: Populations among workers not synchronized:\n{populations}")
                print(
                    f"Island {self.island_idx}: {len(active_pop)}/{len(self.population)} "
                    f"individuals active ({len(occurrences)} unique)."
                )
        MPI.COMM_WORLD.barrier()
        if debug == 0:
            best = min(self.population, key=attrgetter("loss"))
            if self.comm.rank == 0:
                print(f"Top result on island {self.island_idx}: {best}\n")
        else:
            unique_pop = self._get_unique_individuals()
            unique_pop.sort(key=lambda x: x.loss)
            best = unique_pop[:top_n]
            if self.comm.rank == 0:
                res_str = f"Top {top_n} result(s) on island {self.island_idx}:\n"
                for i in range(top_n):
                    res_str += f"({i+1}): {unique_pop[i]}\n"
                print(res_str)
        return MPI.COMM_WORLD.allgather(best)
