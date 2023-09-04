import copy
import logging
import os
import pickle
import random
import time
from operator import attrgetter
from pathlib import Path
from typing import Callable, Union, List, Tuple, Type

import deepdiff
import numpy as np
from mpi4py import MPI

from .propagators import Propagator, SelectMin
from .population import Individual
from ._globals import DUMP_TAG, INDIVIDUAL_TAG


log = logging.getLogger(__name__)  # Get logger instance.


class Propulator:
    """
    Parallel propagator of populations.

    This class provides Propulate's basic asynchronous population-based optimization routine (without an island model).
    At the same time, it serves as a base class of ``Migrator`` and ``Pollinator``, which implement an asynchronous
    island model on top of the asynchronous base optimizer with real migration and pollination, respectively.
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
        Initialize Propulator with given parameters.

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
        # Set class attributes.
        self.loss_fn = loss_fn  # callable loss function
        self.propagator = propagator  # evolutionary propagator
        if generations == 0:  # If number of iterations requested == 0.
            if MPI.COMM_WORLD.rank == 0:
                log.info("Requested number of generations is zero...[RETURN]")
            return
        self.generations = (
            generations  # number of generations (evaluations per individual)
        )
        self.generation = 0  # current generation not yet evaluated
        self.island_idx = island_idx  # island index
        self.comm = comm  # intra-island communicator
        self.checkpoint_path = Path(checkpoint_path)  # checkpoint path
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.migration_prob = migration_prob  # per-rank migration probability
        self.migration_topology = migration_topology  # migration topology
        self.island_displs = (
            island_displs  # MPI.COMM_WORLD rank of each island's worker 0
        )
        self.island_counts = island_counts  # number of workers on each island
        self.emigration_propagator = emigration_propagator  # emigration propagator
        self.rng = rng

        # Load initial population of evaluated individuals from checkpoint if exists.
        load_ckpt_file = self.checkpoint_path / f"island_{self.island_idx}_ckpt.pkl"
        if not os.path.isfile(load_ckpt_file):  # If not exists, check for backup file.
            load_ckpt_file = load_ckpt_file.with_suffix(".bkp")

        if os.path.isfile(load_ckpt_file):
            with open(load_ckpt_file, "rb") as f:
                try:
                    self.population = pickle.load(f)
                    self.generation = (
                        max(
                            [
                                x.generation
                                for x in self.population
                                if x.rank == self.comm.rank
                            ]
                        )
                        + 1
                    )
                    if self.comm.rank == 0:
                        log.info(
                            "Valid checkpoint file found. "
                            f"Resuming from generation {self.generation} of loaded population..."
                        )
                except OSError:
                    self.population = []
                    if self.comm.rank == 0:
                        log.info(
                            "No valid checkpoint file. Initializing population randomly..."
                        )
        else:
            self.population = []
            if self.comm.rank == 0:
                log.info(
                    "No valid checkpoint file given. Initializing population randomly..."
                )

    def propulate(self, logging_interval: int = 10, debug: int = 1) -> None:
        """
        Run asynchronous evolutionary optimization routine.

        Parameters
        ----------
        logging_interval: int
                          Print each worker's progress every ``logging_interval`` th generation.
        debug: int
               verbosity/debug level; 0 - silent; 1 - moderate, 2 - noisy (debug mode)
        """
        self._work(logging_interval, debug)

    def _get_active_individuals(self) -> Tuple[List[Individual], int]:
        """
        Get active individuals in current population list.

        Returns
        -------
        list[propulate.population.Individual]
            currently active individuals in population
        int
            number of currently active individuals
        """
        active_pop = [ind for ind in self.population if ind.active]

        return active_pop, len(active_pop)

    def _breed(self) -> Individual:
        """
        Apply propagator to current population of active individuals to breed new individual.

        Returns
        -------
        propulate.population.Individual
            newly bred individual
        """
        active_pop, _ = self._get_active_individuals()
        ind = self.propagator(
            active_pop
        )  # Breed new individual from active population.
        ind.generation = self.generation  # Set generation.
        ind.rank = self.comm.rank  # Set worker rank.
        ind.active = True  # If True, individual is active for breeding.
        ind.island = self.island_idx  # Set birth island.
        ind.current = self.comm.rank  # Set worker responsible for migration.
        ind.migration_steps = 0  # Set number of migration steps performed.
        ind.migration_history = str(self.island_idx)

        return ind  # Return new individual.

    def _evaluate_individual(self) -> None:
        """
        Breed and evaluate individual.
        """
        ind = self._breed()  # Breed new individual.
        start_time = time.time()  # Start evaluation timer.
        ind.loss = self.loss_fn(ind)  # Evaluate its loss.
        ind.evaltime = time.time()  # Stop evaluation timer.
        ind.evalperiod = ind.evaltime - start_time  # Calculate evaluation duration.
        self.population.append(
            ind
        )  # Add evaluated individual to worker-local population.
        log.debug(
            f"Island {self.island_idx} Worker {self.comm.rank} Generation {self.generation}: BREEDING\n"
            f"Bred and evaluated individual {ind}.\n"
        )

        # Tell other workers in own island about results to synchronize their populations.
        for r in range(self.comm.size):  # Loop over ranks in intra-island communicator.
            if r == self.comm.rank:
                continue  # No self-talk.
            self.comm.send(copy.deepcopy(ind), dest=r, tag=INDIVIDUAL_TAG)

    def _receive_intra_island_individuals(self) -> None:
        """
        Check for and possibly receive incoming individuals
        evaluated by other workers within own island.
        """
        log_string = (
            f"Island {self.island_idx} Worker {self.comm.rank} Generation {self.generation}: "
            f"INTRA-ISLAND SYNCHRONIZATION\n"
        )
        probe_ind = True
        while probe_ind:
            stat = (
                MPI.Status()
            )  # Retrieve status of reception operation, including source and tag.
            probe_ind = self.comm.iprobe(
                source=MPI.ANY_SOURCE, tag=INDIVIDUAL_TAG, status=stat
            )
            # If True, continue checking for incoming messages. Tells whether message corresponding
            # to filters passed is waiting for reception via a flag that it sets.
            # If no such message has arrived yet, it returns False.
            log_string += f"Incoming individual to receive?...{probe_ind}\n"
            if probe_ind:
                # Receive individual and add it to own population.
                ind_temp = self.comm.recv(source=stat.Get_source(), tag=INDIVIDUAL_TAG)
                self.population.append(
                    ind_temp
                )  # Add received individual to own worker-local population.
                log_string += f"Added individual {ind_temp} from W{stat.Get_source()} to own population.\n"
        _, n_active = self._get_active_individuals()
        log_string += (
            f"After probing within island: {n_active}/{len(self.population)} active.\n"
        )
        log.debug(log_string)

    def _send_emigrants(self) -> None:
        """
        Perform migration, i.e. island sends individuals out to other islands.

        Raises
        ------
        NotImplementedError
            Not implemented in ``Propulator`` base class. Exact migration and pollination behavior is defined in the
            ``Migrator`` and ``Pollinator`` classes, respectively.
        """
        raise NotImplementedError

    def _receive_immigrants(self) -> None:
        """
        Check for and possibly receive immigrants send by other islands.

        Raises
        ------
        NotImplementedError
            Not implemented in ``Propulator`` base class. Exact migration and pollination behavior is defined in the
            ``Migrator`` and ``Pollinator`` classes, respectively.
        """
        raise NotImplementedError

    def _get_unique_individuals(self) -> List[Individual]:
        """
        Get unique individuals in terms of traits and loss in current population.

        Returns
        -------
        list[propulate.population.Individual]
            unique individuals
        """
        unique_inds = []
        for individual in self.population:
            considered = False
            for ind in unique_inds:
                # Check for equivalence of traits only when determining unique individuals. To do so, use
                # self.equals(other) member function of Individual() class instead of `==` operator.
                if individual.equals(ind):
                    considered = True
                    break
            if not considered:
                unique_inds.append(individual)
        return unique_inds

    def _check_intra_island_synchronization(
        self, populations: List[List[Individual]]
    ) -> bool:
        """
        Check synchronization of populations of workers within one island.

        Parameters
        ----------
        populations: list[list[propulate.population.Individual]]
                     list of islands' sorted population lists

        Returns
        -------
        bool
            True if populations are synchronized, False if not.
        """
        synchronized = True
        for population in populations:
            difference = deepdiff.DeepDiff(
                population, populations[0], ignore_order=True
            )
            if len(difference) == 0:
                continue
            log.info(
                f"Island {self.island_idx} Worker {self.comm.rank}: Population not synchronized:\n"
                f"{difference}\n"
            )
            synchronized = False
        return synchronized

    def _work(self, logging_interval: int, debug: int):
        """
        Execute evolutionary algorithm in parallel.

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

            if dump:  # Dump checkpoint.
                self._dump_checkpoint()

            dump = (
                self._determine_worker_dumping_next()
            )  # Determine worker dumping checkpoint in the next generation.

            # Go to next generation.
            self.generation += 1

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

        # Final checkpointing on rank 0.
        if self.comm.rank == 0:
            self._dump_final_checkpoint()  # Dump checkpoint.
        MPI.COMM_WORLD.barrier()
        _ = self._determine_worker_dumping_next()
        MPI.COMM_WORLD.barrier()

    def _dump_checkpoint(self):
        """
        Dump checkpoint to file.
        """
        log.debug(
            f"Island {self.island_idx} Worker {self.comm.rank} Generation {self.generation}: "
            f"Dumping checkpoint..."
        )
        save_ckpt_file = self.checkpoint_path / f"island_{self.island_idx}_ckpt.pkl"
        if os.path.isfile(save_ckpt_file):
            try:
                os.replace(save_ckpt_file, save_ckpt_file.with_suffix(".bkp"))
            except OSError as e:
                log.warning(e)
        with open(save_ckpt_file, "wb") as f:
            pickle.dump(self.population, f)

        dest = self.comm.rank + 1 if self.comm.rank + 1 < self.comm.size else 0
        self.comm.send(True, dest=dest, tag=DUMP_TAG)

    def _determine_worker_dumping_next(self):
        """
        Determine the worker who dumps the checkpoint in the next generation.
        """
        dump = False
        stat = MPI.Status()
        probe_dump = self.comm.iprobe(source=MPI.ANY_SOURCE, tag=DUMP_TAG, status=stat)
        if probe_dump:
            dump = self.comm.recv(source=stat.Get_source(), tag=DUMP_TAG)
            log.debug(
                f"Island {self.island_idx} Worker {self.comm.rank} Generation {self.generation}: "
                f"Going to dump next: {dump}. Before: Worker {stat.Get_source()}"
            )
        return dump

    def _dump_final_checkpoint(self):
        """
        Dump final checkpoint.
        """
        save_ckpt_file = self.checkpoint_path / f"island_{self.island_idx}_ckpt.pkl"
        if os.path.isfile(save_ckpt_file):
            try:
                os.replace(save_ckpt_file, save_ckpt_file.with_suffix(".bkp"))
            except OSError as e:
                log.warning(e)
            with open(save_ckpt_file, "wb") as f:
                pickle.dump(self.population, f)

    def _check_for_duplicates(
        self, active: bool, debug: int
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
        list[list[propulate.population.Individual | int]]
            individuals and their occurrences
        list[propulate.population.Individual]
            unique individuals in population
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
                if individual == ind:
                    considered = True
                    break
            if not considered:
                num_copies = population.count(individual)
                log.debug(
                    f"Island {self.island_idx} Worker {self.comm.rank} Generation {self.generation}: "
                    f"{individual} occurs {num_copies} time(s)."
                )
                unique_inds.append(individual)
                occurrences.append([individual, num_copies])
        return occurrences, unique_inds

    def summarize(
        self, top_n: int = 1, debug: int = 1
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
        list[list[Individual] | Individual]
            top-n best individuals on each island
        """
        active_pop, num_active = self._get_active_individuals()
        assert np.all(
            np.array(self.comm.allgather(num_active), dtype=int) == num_active
        )
        if self.island_counts is not None:
            num_active = int(
                MPI.COMM_WORLD.allreduce(
                    num_active / self.island_counts[self.island_idx]
                )
            )

        MPI.COMM_WORLD.barrier()
        if MPI.COMM_WORLD.rank == 0:
            log.info("###########")
            log.info("# SUMMARY #")
            log.info("###########")
            log.info(f"Number of currently active individuals is {num_active}. ")
            log.info(
                f"Expected overall number of evaluations is {self.generations*MPI.COMM_WORLD.size}."
            )
        # Only double-check number of occurrences of each individual for DEBUG level 2.
        if debug == 2:
            populations = self.comm.gather(self.population, root=0)
            occurrences, _ = self._check_for_duplicates(True, debug)
            if self.comm.rank == 0:
                if self._check_intra_island_synchronization(populations):
                    log.info(
                        f"Island {self.island_idx}: Populations among workers synchronized."
                    )
                else:
                    log.info(
                        f"Island {self.island_idx}: Populations among workers not synchronized:\n{populations}"
                    )
                log.info(
                    f"Island {self.island_idx}: {len(active_pop)}/{len(self.population)} "
                    f"individuals active ({len(occurrences)} unique)"
                )
        MPI.COMM_WORLD.barrier()
        if debug == 0:
            best = min(self.population, key=attrgetter("loss"))
            if self.comm.rank == 0:
                log.info(f"Top result on island {self.island_idx}: {best}\n")
        else:
            unique_pop = self._get_unique_individuals()
            unique_pop.sort(key=lambda x: x.loss)
            best = unique_pop[:top_n]
            if self.comm.rank == 0:
                res_str = f"Top {top_n} result(s) on island {self.island_idx}:\n"
                for i in range(top_n):
                    res_str += f"({i+1}): {unique_pop[i]}\n"
                log.info(res_str)
        return MPI.COMM_WORLD.allgather(best)
