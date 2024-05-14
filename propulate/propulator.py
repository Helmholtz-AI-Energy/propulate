import copy
import inspect
import logging
import os
import random
import time
from operator import attrgetter
from pathlib import Path
from typing import Callable, Final, Generator, List, Optional, Tuple, Type, Union

import deepdiff
import h5py
import numpy as np
from mpi4py import MPI

from ._globals import INDIVIDUAL_TAG
from .population import Individual
from .propagators import BasicPSO, Conditional, Propagator, SelectMin
from .surrogate import Surrogate

log = logging.getLogger(__name__)  # Get logger instance.
SURROGATE_KEY: Final[str] = "_s"  # Key for ``Surrogate`` data in ``Individual``


class Propulator:
    """
    Parallel propagator of populations.

    This class provides Propulate's basic asynchronous population-based optimization routine (without an island model).
    At the same time, it serves as a base class of ``Migrator`` and ``Pollinator``, which implement an asynchronous
    island model on top of the asynchronous base optimizer with real migration and pollination, respectively.

    Attributes
    ----------
    checkpoint_path : str | pathlib.Path
        The path where checkpoints are loaded from and stored to.
    emigration_propagator : Type[propulate.Propagator]
        The emigration propagator, i.e., how to choose individuals for emigration that are sent to the destination
        island. Should be some kind of selection operator.
    generation : int
        The current generation.
    generations : int
        The overall number of generations.
    island_comm : MPI.Comm
        The intra-island communicator for communication within the island.
    island_counts : numpy.ndarray
        The numbers of workers for each island.
    island_displs : numpy.ndarray
        The island displacements.
    island_idx : int
        The island's index.
    loss_fn : Union[Callable, Generator[float, None, None]]
        The loss function to be minimized.
    migration_prob : float
        The migration probability.
    migration_topology : np.ndarray
        The migration topology.
    population : List[propulate.population.Individual]
        The population list of individuals on that island.
    propagator : propulate.Propagator
        The evolutionary operator.
    propulate_comm : MPI.Comm
        The Propulate world communicator, consisting of rank 0 of each worker's sub communicator.
    rng : random.Random
        The separate random number generator for the Propulate optimization.
    surrogate : propulate.surrogate.Surrogate, optional
        The local surrogate model.
    worker_sub_comm : MPI.Comm
        The worker's internal communicator for parallelized evaluation of single individuals.

    Methods
    -------
    propulate()
        Run asynchronous evolutionary optimization routine without migration or pollination.
    summarize()
        Get top-n results from Propulate optimization.
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
        Initialize Propulator with given parameters.

        Parameters
        ----------
        loss_fn : Union[Callable, Generator[float, None, None]]
            The loss function to be minimized.
        propagator : propulate.propagators.Propagator
            The propagator to apply for breeding.
        rng : random.Random
            The separate random number generator for the Propulate optimization.
        island_idx : int, optional
            The island's index. Default is 0.
        island_comm : MPI.Comm, optional
            The intra-island communicator. Default is ``MPI.COMM_WORLD``.
        propulate_comm : MPI.Comm, optional
            The Propulate world communicator, consisting of rank 0 of each worker's sub communicator.
            Default is ``MPI.COMM_WORLD``.
        worker_sub_comm : MPI.Comm, optional
            The sub communicator for each (multi rank) worker. Default is ``MPI.COMM_SELF``.
        generations : int, optional
            The number of generations to run. Default is -1, i.e., run into wall-clock time limit.
        checkpoint_path : pathlib.Path | str, optional
            The path where checkpoints are loaded from and stored. Default is current working directory.
        migration_topology : numpy.ndarray, optional
            The migration topology, i.e., a 2D matrix where entry (i,j) specifies how many individuals are sent by
            island i to island j.
        migration_prob : float, optional
            The per-worker migration probability. Default is 0.0.
        emigration_propagator : Type[propulate.propagators.Propagator], optional
            The emigration propagator, i.e., how to choose individuals for emigration that are sent to destination
            island. Should be some kind of selection operator. Default is ``SelectMin``.
        island_displs : numpy.ndarray, optional
            An array with ``propulate_comm`` rank of each island's worker 0. Element i specifies the rank of worker 0 on
            island with index i in the Propulate communicator.
        island_counts : numpy.ndarray, optional
            An array with the number of workers per island. Element i specifies the number of workers on island with
            index i.
        surrogate_factory : Callable[[], propulate.surrogate.Surrogate], optional
           Function that returns a new instance of a ``Surrogate`` model.
           Only used when ``loss_fn`` is a generator function.
        """
        # Set class attributes.
        self.loss_fn = loss_fn  # Callable loss function
        self.propagator = propagator  # Evolutionary propagator
        if generations == 0:  # If number of iterations requested == 0.
            if MPI.COMM_WORLD.rank == 0:
                log.info("Requested number of generations is zero...[RETURN]")
            return
        self.generations = generations  # Number of generations (evaluations per individual)
        self.generation = 0  # Current generation not yet evaluated
        self.island_idx = island_idx  # Island index
        self.island_comm = island_comm  # Intra-island communicator
        self.propulate_comm = propulate_comm  # Propulate world communicator
        self.worker_sub_comm = worker_sub_comm  # Sub communicator for each (multi rank) worker

        # Always initialize the ``Surrogate`` as the class attribute has to be set for ``None`` checks later.
        self.surrogate = None if surrogate_factory is None else surrogate_factory()

        if self.propulate_comm is None:  # Exit early for sub-worker only ranks.
            # These ranks are not used for anything aside from the calculation of the user-defined loss function.
            return
        self.checkpoint_path = Path(checkpoint_path)  # Checkpoint path
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.migration_prob = migration_prob  # Per-rank migration probability
        self.migration_topology = migration_topology  # Migration topology
        self.island_displs = island_displs  # Propulate world rank of each island's worker
        self.island_counts = island_counts  # Number of workers on each island
        self.emigration_propagator = emigration_propagator  # Emigration propagator
        self.rng = rng  # Generator for inter-island communication

        self.intra_requests: list[MPI.Request] = []  # Keep track of intra-island send requests.
        self.intra_buffers: list[Individual] = []  # Send buffers for intra-island communication

        # Load initial population of evaluated individuals from checkpoint if exists.
        self.checkpoint_path = self.checkpoint_path / "ckpt.hdf5"

        self.population = []
        # consistency check and ensure enough space is allocated
        if os.path.isfile(self.checkpoint_path):
            self.load_checkpoint()
            if self.propulate_comm.rank == 0:
                log.info("Valid checkpoint file found. " f"Resuming from generation {self.generation} of loaded population...")
                # TODO it says resuming from generation 0, so something is not right
                # TODO also each worker might be on a different generation so this message probably does not make all of the sense
        else:
            if self.propulate_comm.rank == 0:
                log.info("No valid checkpoint file given. Initializing population randomly...")
        self.set_up_checkpoint()

    def load_checkpoint(self):
        """Load checkpoint from HDF5 file. Since this is only a read, all workers can do this in read-only mode without the mpio driver."""
        # TODO check that the island and worker setup is the same as in the checkpoint
        # NOTE each individual is only stored once at the position given by its origin island and worker, the modifications have to be put in the checkpoint file during migration  TODO test if this works as intended reliably
        # TODO get the started but not yet completed ones from the difference in start time and evaltime

        with h5py.File(self.checkpoint_path, "r", driver=None) as f:
            islandgroup = f[f"{self.island_idx}"]

            # NOTE check limits are consistent
            limitsgroup = f["limits"]
            for key in self.propagator.limits:
                if set(limitsgroup.attrs.keys()) != set(self.propagator.limits):
                    raise RuntimeError("Limits inconsistent with checkpoint")
            # TODO check island sizes are consistent

            self.generation = int(f["generations"][self.propulate_comm.Get_rank()])
            if islandgroup[f"{self.island_comm.Get_rank()}"]["evalperiod"][self.generation] > 0.0:
                self.generation += 1
            # NOTE load individuals, since they might have migrated, every worker has to check each dataset
            for rank in range(self.propulate_comm.size):
                # for generation in range(len(islandgroup[f"{rank}"])):
                for generation in range(f["generations"][rank] + 1):
                    if islandgroup[f"{rank}"]["current"][generation] == self.island_idx:
                        ind = Individual(
                            islandgroup[f"{rank}"]["x"][generation, 0],
                            self.propagator.limits,
                        )
                        ind.rank = rank
                        ind.island = self.island_idx
                        ind.current = islandgroup[f"{rank}"]["current"][generation]
                        # TODO velocity loading
                        # if len(group[f"{rank}"].shape) > 1:
                        #     ind.velocity = islandgroup[f"{rank}"]["x"][generation, 1]
                        ind.loss = islandgroup[f"{rank}"]["loss"][generation]
                        ind.startime = islandgroup[f"{rank}"]["starttime"][generation]
                        ind.evaltime = islandgroup[f"{rank}"]["evaltime"][generation]
                        ind.evalperiod = islandgroup[f"{rank}"]["evalperiod"][generation]
                        ind.generation = generation
                        self.population.append(ind)
                        if ind.loss is None:
                            # TODO resume evaluation
                            raise

    def set_up_checkpoint(self):
        """Initialize checkpoint file or check consistenct with an existing one."""
        limit_dim = 0
        for key in self.propagator.limits:
            if isinstance(self.propagator.limits[key][0], str):
                limit_dim += len(self.propagator.limits[key])
            else:
                limit_dim += 1

        num_islands = 1
        if self.island_counts is not None:
            num_islands = len(self.island_counts)

        with h5py.File(self.checkpoint_path, "a", driver="mpio", comm=self.propulate_comm) as f:
            # limits
            limitsgroup = f.require_group("limits")
            for key in self.propagator.limits:
                limitsgroup.attrs[key] = str(self.propagator.limits[key])

            xdim = 1
            # TODO clean this up when reorganizing propagators
            if isinstance(self.propagator, BasicPSO) or (
                isinstance(self.propagator, Conditional) and isinstance(self.propagator.true_prop, BasicPSO)
            ):
                xdim = 2

            oldgenerations = self.generations
            if "0" in f:
                oldgenerations = f["0"]["0"]["x"].shape[0]
            # Store per worker what generation they are at, since islands can be different sizes, it's flat
            f.require_dataset(
                "generations",
                (self.propulate_comm.Get_size(),),
                dtype=np.int32,
                data=np.zeros((self.propulate_comm.Get_size(),), dtype=np.int32),
            )

            # population
            for i in range(num_islands):
                f.require_group(f"{i}")
                for worker_idx in range(self.propulate_comm.Get_size()):
                    group = f[f"{i}"].require_group(f"{worker_idx}")
                    if oldgenerations < self.generations:
                        group["x"].resize(self.generations, axis=0)
                        group["loss"].resize(self.generations, axis=0)
                        group["active"].resize(self.generations, axis=0)
                        group["current"].resize(self.generations, axis=0)
                        group["migration_steps"].resize(self.generations, axis=0)
                        group["starttime"].resize(self.generations, axis=0)
                        group["evaltime"].resize(self.generations, axis=0)
                        group["evalperiod"].resize(self.generations, axis=0)

                    group.require_dataset(
                        "x",
                        (self.generations, xdim, limit_dim),
                        dtype=np.float32,
                        chunks=True,
                        maxshape=(None, xdim, limit_dim),
                    )
                    group.require_dataset(
                        "loss",
                        (self.generations,),
                        np.float32,
                        chunks=True,
                        maxshape=(None,),
                    )
                    group.require_dataset(
                        "active",
                        (self.generations,),
                        np.bool_,
                        chunks=True,
                        maxshape=(None,),
                    )
                    group.require_dataset(
                        "current",
                        (self.generations,),
                        np.int16,
                        chunks=True,
                        maxshape=(None,),
                    )
                    group.require_dataset(
                        "migration_steps",
                        (self.generations,),
                        np.int32,
                        chunks=True,
                        maxshape=(None,),
                    )
                    group.require_dataset(
                        "starttime",
                        (self.generations,),
                        np.uint64,
                        chunks=True,
                        maxshape=(None,),
                    )
                    group.require_dataset(
                        "evaltime",
                        (self.generations,),
                        np.uint64,
                        chunks=True,
                        maxshape=(None,),
                    )
                    group.require_dataset(
                        "evalperiod",
                        (self.generations,),
                        np.uint64,
                        chunks=True,
                        maxshape=(None,),
                        data=-1 * np.ones((self.generations,)),
                    )

    def propulate(self, logging_interval: int = 10, debug: int = 1) -> None:
        """
        Run asynchronous evolutionary optimization routine.

        Parameters
        ----------
        logging_interval : int, optional
            Print each worker's progress every ``logging_interval``-th generation. Default is 10.
        debug : int, optional
            The debug level; 0 - silent; 1 - moderate, 2 - noisy (debug mode). Default is 1.
        """
        # TODO note the propulating times in the checkpoint file for when the run is interrupted
        self.start_time = time.time_ns()
        self._work(logging_interval, debug)

    def _get_active_individuals(self) -> Tuple[List[Individual], int]:
        """
        Get active individuals in current population list.

        Returns
        -------
        List[propulate.population.Individual]
            All active individuals in the current population.
        int
            The number of currently active individuals.
        """
        active_pop = [ind for ind in self.population if ind.active]
        return active_pop, len(active_pop)

    def _breed(self) -> Individual:
        """
        Apply propagator to current population of active individuals to breed new individual.

        Returns
        -------
        propulate.population.Individual
            The newly bred individual.
        """
        if (
            self.propulate_comm is not None
        ):  # Only processes in the Propulate world communicator, consisting of rank 0 of each worker's sub
            # communicator, are involved in the actual optimization routine.
            active_pop, _ = self._get_active_individuals()
            ind = self.propagator(active_pop)  # Breed new individual from active population.
            assert isinstance(ind, Individual)
            ind.generation = self.generation  # Set generation.
            ind.rank = self.island_comm.rank  # Set worker rank.
            ind.active = True  # If True, individual is active for breeding.
            ind.island = self.island_idx  # Set birth island.
            ind.current = self.island_comm.rank  # Set worker responsible for migration.
            ind.migration_steps = 0  # Set number of migration steps performed.
            ind.migration_history = str(self.island_idx)
        else:  # The other processes do not breed themselves.
            ind = None

        if self.worker_sub_comm != MPI.COMM_SELF:  # Broadcast newly bred individual to all internal ranks of a worker from rank 0,
            # which is also part of the Propulate comm.
            ind = self.worker_sub_comm.bcast(obj=ind, root=0)

        assert isinstance(ind, Individual)
        return ind  # Return new individual.

    def _evaluate_individual(self, hdf5_checkpoint) -> None:
        """Breed and evaluate individual."""
        ind = self._breed()  # Breed new individual.
        start_time = time.time_ns() - self.start_time  # Start evaluation timer.
        ind.starttime = start_time
        ckpt_idx = ind.generation
        hdf5_checkpoint["generations"][self.propulate_comm.Get_rank()] = ind.generation

        group = hdf5_checkpoint[f"{self.island_idx}"][f"{self.propulate_comm.Get_rank()}"]
        # save candidate
        group["x"][ckpt_idx, 0, :] = ind.position[:]
        if ind.velocity is not None:
            group["x"][ckpt_idx, 1, :] = ind.velocity[:]
        group["starttime"][ckpt_idx] = start_time

        ind.loss = self.loss_fn(ind)  # Evaluate its loss.
        ind.evaltime = time.time_ns() - self.start_time  # Stop evaluation timer.
        ind.evalperiod = ind.evaltime - start_time  # Calculate evaluation duration.

        # save result for candidate
        group["loss"][ckpt_idx] = ind.loss
        group["evaltime"][ckpt_idx] = ind.evaltime
        group["evalperiod"][ckpt_idx] = ind.evalperiod
        # Signal start of run to surrogate model.
        if self.surrogate is not None:
            self.surrogate.start_run(ind)

        # Check if ``loss_fn`` is generator, prerequisite for surrogate model.
        if inspect.isgeneratorfunction(self.loss_fn):

            def loss_gen(individual: Individual) -> Generator[float, None, None]:
                if self.worker_sub_comm != MPI.COMM_SELF:
                    # NOTE mypy complains here. no idea why
                    for x in self.loss_fn(individual, self.worker_sub_comm):  # type: ignore
                        yield x
                else:
                    # NOTE mypy complains here. no idea why
                    for x in self.loss_fn(individual):  # type: ignore
                        yield x

            last = float("inf")
            for idx, last in enumerate(loss_gen(ind)):
                log.debug(
                    f"Island {self.island_idx} Worker {self.island_comm.rank} Generation {self.generation} -- Individual loss iteration {idx}: Value {last}"
                )
                if self.surrogate is not None:
                    if self.surrogate.cancel(last):  # Check cancel for each yield.
                        log.debug(
                            f"Island {self.island_idx} Worker {self.island_comm.rank} Generation {self.generation}: PRUNING\n{ind}"
                        )
                        break
            ind.loss = float(last)  # Set final loss as individual's loss.
        else:
            # Define local ``loss_fn`` for parallelized evaluation.
            def loss_fn(individual: Individual) -> float:
                if self.worker_sub_comm != MPI.COMM_SELF:
                    # NOTE this is not a generator, but mypy thinks it is
                    return self.loss_fn(individual, self.worker_sub_comm)  # type: ignore
                else:
                    # NOTE this is not a generator, but mypy thinks it is
                    return self.loss_fn(individual)  # type: ignore

            ind.loss = float(loss_fn(ind))  # Evaluate its loss.

        # Add final value to surrogate.
        if self.surrogate is not None:
            self.surrogate.update(ind.loss)
        if self.propulate_comm is None:
            return
        ind.evaltime = time.time()  # Stop evaluation timer.
        ind.evalperiod = ind.evaltime - start_time  # Calculate evaluation duration.
        self.population.append(ind)  # Add evaluated individual to worker-local population.
        log.debug(
            f"Island {self.island_idx} Worker {self.island_comm.rank} Generation {self.generation}: BREEDING\n"
            f"Bred and evaluated individual {ind}."
        )

        if self.surrogate is not None:
            # Add surrogate model data to individual for synchronization.
            ind[SURROGATE_KEY] = self.surrogate.data()

        # Tell other workers in own island about results to synchronize their populations.
        for r in range(self.island_comm.size):  # Loop over ranks in intra-island communicator.
            if r == self.island_comm.rank:
                continue  # No self-talk.
            self.intra_buffers.append(copy.deepcopy(ind))
            self.intra_requests.append(self.island_comm.isend(self.intra_buffers[-1], dest=r, tag=INDIVIDUAL_TAG))

        if self.surrogate is not None:
            # Remove data from individual again as ``__eq__`` fails otherwise.
            del ind[SURROGATE_KEY]

    def _receive_intra_island_individuals(self) -> None:
        """Check for and possibly receive incoming individuals evaluated by other workers within own island."""
        log_string = (
            f"Island {self.island_idx} Worker {self.island_comm.rank} Generation {self.generation}: INTRA-ISLAND SYNCHRONIZATION\n"
        )
        probe_ind = True
        while probe_ind:
            stat = MPI.Status()  # Retrieve status of reception operation, including source and tag.
            probe_ind = self.island_comm.iprobe(source=MPI.ANY_SOURCE, tag=INDIVIDUAL_TAG, status=stat)
            # If True, continue checking for incoming messages. Tells whether message corresponding
            # to filters passed is waiting for reception via a flag that it sets.
            # If no such message has arrived yet, it returns False.
            log_string += f"Incoming individual to receive?...{probe_ind}\n"
            if probe_ind:
                # Receive individual and add it to own population.
                ind_temp = self.island_comm.recv(source=stat.Get_source(), tag=INDIVIDUAL_TAG)

                # Only merge if surrogate model is used.
                if SURROGATE_KEY in ind_temp and self.surrogate is not None:
                    self.surrogate.merge(ind_temp[SURROGATE_KEY])
                # Remove data from individual again as ``__eq__`` fails otherwise.
                if SURROGATE_KEY in ind_temp:
                    del ind_temp[SURROGATE_KEY]

                self.population.append(ind_temp)  # Add received individual to own worker-local population.

                log_string += f"Added individual {ind_temp} from W{stat.Get_source()} to own population.\n"
        _, n_active = self._get_active_individuals()
        log_string += f"After probing within island: {n_active}/{len(self.population)} active."
        log.debug(log_string)

    def _send_emigrants(self) -> None:
        """
        Perform migration, i.e., island sends individuals out to other islands.

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
        List[propulate.population.Individual]
            All unique individuals in the current population.
        """
        unique_inds: List[Individual] = []
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

    def _check_intra_island_synchronization(self, populations: List[List[Individual]]) -> bool:
        """
        Check synchronization of populations of workers within one island.

        Parameters
        ----------
        populations : List[List[propulate.population.Individual]]
            A list of all islands' sorted population lists.

        Returns
        -------
        bool
            True if populations are synchronized, False if not.
        """
        synchronized = True
        for population in populations:
            difference = deepdiff.DeepDiff(population, populations[0], ignore_order=True)
            if len(difference) == 0:
                continue
            log.info(f"Island {self.island_idx} Worker {self.island_comm.rank}: Population not synchronized:\n{difference}")
            synchronized = False
        return synchronized

    def propulate(self, logging_interval: int = 10, debug: int = -1) -> None:
        """
        Execute evolutionary algorithm in parallel.

        Parameters
        ----------
        logging_interval : int, optional
            Print each worker's progress every ``logging_interval``-th generation. Default is 10.
        debug : int, optional
            The debug level; 0 - silent; 1 - moderate, 2 - noisy (debug mode). Default is 1.

        Raises
        ------
        ValueError
            If any individuals are left that should have been deactivated before (only for debug > 0).

        """
        if self.worker_sub_comm != MPI.COMM_SELF:
            self.generation = self.worker_sub_comm.bcast(self.generation, root=0)
        if self.propulate_comm is None:
            while self.generation < self.generations:
                # Breed and evaluate individual.
                # TODO this should be refactored, the subworkers don't need the logfile
                self._evaluate_individual()
                self.generation += 1
            return

        if self.island_comm.rank == 0:
            log.info(f"Island {self.island_idx} has {self.island_comm.size} workers.")

        # Loop over generations.
        with h5py.File(self.checkpoint_path, "a", driver="mpio", comm=MPI.COMM_WORLD) as f:
            while self.generation < self.generations:
                if self.generation % int(logging_interval) == 0:
                    log.info(f"Island {self.island_idx} Worker {self.island_comm.rank}: In generation {self.generation}...")

                # Breed and evaluate individual.
                log.debug(f"breeding and evaluating {self.generation}")
                self._evaluate_individual(f)

                # Check for and possibly receive incoming individuals from other intra-island workers.
                log.debug(f"receive intra island {self.generation}")
                self._receive_intra_island_individuals()

                # Clean up requests and buffers.
                self._intra_send_cleanup()
                # Go to next generation.
                self.generation += 1

        # Having completed all generations, the workers have to wait for each other.
        # Once all workers are done, they should check for incoming messages once again
        # so that each of them holds the complete final population and the found optimum
        # irrespective of the order they finished.
        log.debug("barrier")

        log.debug(f"Island {self.island_idx} Worker {self.island_comm.rank}: Waiting on final synchronization barrier...")
        self.propulate_comm.barrier()
        if self.propulate_comm.rank == 0:
            log.info("OPTIMIZATION DONE.\nNEXT: Final checks for incoming messages...")

        # Final check for incoming individuals evaluated by other intra-island workers.
        log.debug("final sync")
        self._receive_intra_island_individuals()
        self.propulate_comm.barrier()

    def _intra_send_cleanup(self) -> None:
        """Delete all send buffers that have been sent."""
        # Test for requests to complete.
        completed = MPI.Request.Testsome(self.intra_requests)
        # Remove requests and buffers of complete send operations.
        self.intra_requests = [r for i, r in enumerate(self.intra_requests) if i not in completed]
        self.intra_buffers = [b for i, b in enumerate(self.intra_buffers) if i not in completed]

    def _check_for_duplicates(self, active: bool, debug: int = 1) -> Tuple[List[List[Union[Individual, int]]], List[Individual]]:
        """
        Check for duplicates in current population.

        For pollination, duplicates are allowed as emigrants are sent as copies
        and not deactivated on sending island.

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
            The unique individuals in the population.
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
                if individual == ind:
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

    def summarize(self, top_n: int = 1, debug: int = 1) -> Union[List[Union[List[Individual], Individual]], None]:
        """
        Get top-n results from Propulate optimization.

        Parameters
        ----------
        top_n : int, optional
            The number of best results to report. Default is 1.
        debug : int, optional
            The debug level; 0 - silent; 1 - moderate, 2 - noisy (debug mode). Default is 1.

        Returns
        -------
        List[List[propulate.population.Individual] | propulate.population.Individual]
            The top-n best individuals on each island.
        """
        if self.propulate_comm is None:
            return None
        active_pop, num_active = self._get_active_individuals()
        assert np.all(np.array(self.island_comm.allgather(num_active), dtype=int) == num_active)
        if self.island_counts is not None:
            num_active = int(self.propulate_comm.allreduce(num_active / self.island_counts[self.island_idx]))

        self.propulate_comm.barrier()
        if self.propulate_comm.rank == 0:
            log.info(
                "###########\n# SUMMARY #\n###########\n"
                f"Number of currently active individuals is {num_active}.\n"
                f"Expected overall number of evaluations is {self.generations * self.propulate_comm.size}."
            )
        # Only double-check number of occurrences of each individual for DEBUG level 2.
        if debug == 2:
            populations = self.island_comm.gather(self.population, root=0)
            occurrences, _ = self._check_for_duplicates(True, debug)
            if self.island_comm.rank == 0:
                if self._check_intra_island_synchronization(populations):
                    log.info(f"Island {self.island_idx}: Populations among workers synchronized.")
                else:
                    log.info(f"Island {self.island_idx}: Populations among workers not synchronized:\n{populations}")
                log.info(
                    f"Island {self.island_idx}: {len(active_pop)}/{len(self.population)} "
                    f"individuals active ({len(occurrences)} unique)"
                )
        self.propulate_comm.barrier()
        if debug == 0:
            best = min(self.population, key=attrgetter("loss"))
            if self.island_comm.rank == 0:
                log.info(f"Top result on island {self.island_idx}: {best}")
        else:
            unique_pop = self._get_unique_individuals()
            unique_pop.sort(key=lambda x: x.loss)
            best = unique_pop[:top_n]
            if self.island_comm.rank == 0:
                res_str = f"Top {top_n} result(s) on island {self.island_idx}:\n"
                for i in range(top_n):
                    res_str += f"({i + 1}): {unique_pop[i]}\n"
                log.info(res_str)
        return self.propulate_comm.allgather(best)
