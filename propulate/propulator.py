import copy
import logging
import os
import random
import time
from operator import attrgetter
from pathlib import Path
from typing import Callable, Union, List, Tuple, Type

import deepdiff
import numpy as np
from mpi4py import MPI
import h5py

from ._globals import INDIVIDUAL_TAG
from .population import Individual
from .propagators import Propagator, SelectMin

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
        generations: int = 100,
        checkpoint_directory: Union[str, Path] = Path("./"),
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
        checkpoint_directory: Union[Path, str]
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
        self.generations = generations  # number of generations (evaluations per rank)
        self.generation = 0  # current generation not yet evaluated
        self.island_idx = island_idx  # island index
        self.comm = comm  # intra-island communicator
        self.checkpoint_path = Path(checkpoint_directory)  # checkpoint path
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.migration_prob = migration_prob  # per-rank migration probability
        self.migration_topology = migration_topology  # migration topology
        self.island_displs = (
            island_displs  # MPI.COMM_WORLD rank of each island's worker 0
        )
        self.island_counts = island_counts  # number of workers on each island
        self.emigration_propagator = emigration_propagator  # emigration propagator
        self.rng = rng

        # Load initial population of individuals from checkpoint if exists.
        self.checkpoint_path = self.checkpoint_path / "ckpt.hdf5"

        self.population = []
        if os.path.isfile(self.checkpoint_path):
            self.load_checkpoint(self.checkpoint_path)
            if self.comm.rank == 0:
                log.info(
                    "Valid checkpoint file found. "
                    f"Resuming from generation {self.generation} of loaded population..."
                )
        else:
            if self.comm.rank == 0:
                log.info(
                    "No valid checkpoint file given. Initializing population randomly..."
                )
        # consistency check and ensure enough space is allocated
        self.set_up_checkpoint(self.checkpoint_path)

    def load_checkpoint(self, ckpt_file):
        """
        Load checkpoint from HDF5 file.
        Since this is only a read, all workers can do this in read-only mode without the mpio driver.

        Parameters
        ----------

        ckpt_file: str
                   Path to the file to load
        """
        # TODO what happens if the compute setup is different when loading the checkpoint i.e. different number of workers?
        # TODO load the migrated individuals from the other islands checkpoints
        # NOTE each individual is only stored once at the position given by its origin island and worker, the modifications have to be put in the checkpoint file during migration  TODO test if this works as intended reliably
        # TODO get the started but not yet completed ones from the difference in start time and evaltime
        with h5py.File(self.checkpoint_path, "r", driver=None) as f:
            group = f[f"{self.island_idx}"]
            for rank in range(self.comm.size):
                for ckpt_idx in range(len(group[f"{rank}"])):
                    if group[f"{rank}"]["current"][ckpt_idx] == self.island_idx:
                        ind = Individual(
                            group[f"{rank}"]["x"][ckpt_idx, 0],
                            self.propagator.limits,
                        )
                        ind.current = group[f"{rank}"]["current"][ckpt_idx]
                        # TODO velocity loading
                        # if len(group[f"{rank}"].shape) > 1:
                        #     ind.velocity = group[f"{rank}"]["x"][ckpt_idx, 1]
                        ind.loss = group[f"{rank}"]["loss"][ckpt_idx]
                        ind.startime = group[f"{rank}"]["starttime"][ckpt_idx]
                        ind.evaltime = group[f"{rank}"]["evaltime"][ckpt_idx]
                        ind.evalperiod = group[f"{rank}"]["evalperiod"][ckpt_idx]
                        ind.generation = ckpt_idx
                        self.population.append(ind)

    def set_up_checkpoint(self, checkpoint_path):
        limit_dim = 0
        for key in self.propagator.limits:
            if isinstance(self.propagator.limits[key][0], str):
                limit_dim += len(self.propagator.limits[key])
            else:
                limit_dim += 1

        num_islands = 1
        if self.island_counts is not None:
            num_islands = len(self.island_counts)

        with h5py.File(
            self.checkpoint_path, "a", driver="mpio", comm=MPI.COMM_WORLD
        ) as f:
            # limits
            for key in self.propagator.limits:
                if key not in f.attrs:
                    f.attrs[key] = str(self.propagator.limits[key])
                else:
                    if not str(self.propagator.limits[key]) == f.attrs[key]:
                        raise RuntimeError("Limits inconsistent with checkpoint")

            # TODO resize dataset if necessary

            # population
            for i in range(num_islands):
                f.require_group(f"{i}")
                for worker_idx in range(self.comm.Get_size()):
                    f[f"{i}"].require_group(f"{worker_idx}")
                    # TODO conditional space for velocity
                    f[f"{i}"][f"{worker_idx}"].require_dataset(
                        "x", (self.generations, 2, limit_dim), dtype=np.float32
                    )
                    f[f"{i}"][f"{worker_idx}"].require_dataset(
                        "loss", (self.generations,), np.float32
                    )
                    f[f"{i}"][f"{worker_idx}"].require_dataset(
                        "active", (self.generations,), np.bool_
                    )
                    f[f"{i}"][f"{worker_idx}"].require_dataset(
                        "current", (self.generations,), np.int16
                    )
                    f[f"{i}"][f"{worker_idx}"].require_dataset(
                        "migration_steps", (self.generations,), np.int32
                    )
                    f[f"{i}"][f"{worker_idx}"].require_dataset(
                        "starttime", (self.generations,), np.float32
                    )
                    f[f"{i}"][f"{worker_idx}"].require_dataset(
                        "evaltime", (self.generations,), np.float32
                    )
                    f[f"{i}"][f"{worker_idx}"].require_dataset(
                        "evalperiod", (self.generations,), np.float32
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
        list[propulate.individual.Individual]
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
        propulate.individual.Individual
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

    def _evaluate_individual(self, hdf5_checkpoint) -> None:
        """
        Breed and evaluate individual.
        """
        ind = self._breed()  # Breed new individual.
        start_time = time.time()  # Start evaluation timer.
        ind.starttime = start_time
        ckpt_idx = ind.generation

        group = hdf5_checkpoint[f"{self.island_idx}"][f"{self.comm.Get_rank()}"]
        # save candidate
        group["x"][ckpt_idx, 0, :] = ind.position[:]
        if ind.velocity is not None:
            group["x"][ckpt_idx, 1, :] = ind.velocity[:]
        group["starttime"][ckpt_idx] = start_time

        ind.loss = self.loss_fn(ind)  # Evaluate its loss.
        ind.evaltime = time.time()  # Stop evaluation timer.
        ind.evalperiod = ind.evaltime - start_time  # Calculate evaluation duration.

        # save result for candidate
        group["loss"][ckpt_idx] = ind.loss
        group["evaltime"][ckpt_idx] = ind.evaltime
        group["evalperiod"][ckpt_idx] = ind.evalperiod

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
        list[propulate.individual.Individual]
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
        populations: list[list[propulate.individual.Individual]]
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
        Execute optimization algorithm in parallel.

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

        MPI.COMM_WORLD.barrier()

        # Loop over generations.
        with h5py.File(
            self.checkpoint_path, "a", driver="mpio", comm=MPI.COMM_WORLD
        ) as f:
            while self.generation < self.generations:
                if self.generation % int(logging_interval) == 0:
                    log.info(
                        f"Island {self.island_idx} Worker {self.comm.rank}: In generation {self.generation}..."
                    )

                # Breed and evaluate individual.
                self._evaluate_individual(f)

                # Check for and possibly receive incoming individuals from other intra-island workers.
                self._receive_intra_island_individuals()

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
        list[list[propulate.individual.Individual | int]]
            individuals and their occurrences
        list[propulate.individual.Individual]
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
