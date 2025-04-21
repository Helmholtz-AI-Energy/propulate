import copy
import inspect
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Final, Generator, List, Optional, Tuple, Type, Union

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
        immigration_propagator : Type[propulate.propagators.Propagator], optional
            The immigration propagator, i.e., how to choose individuals to be replaced by immigrants on a target island.
            Should be some kind of selection operator. Default is ``None``.
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
        if island_counts is None:
            self.island_counts = np.array([self.island_comm.Get_size()])
        else:
            self.island_counts = island_counts  # Number of workers on each island
        self.emigration_propagator = emigration_propagator  # Emigration propagator
        self.immigration_propagator = immigration_propagator  # Immigration propagator
        self.rng = rng  # Generator for inter-island communication

        self.intra_requests: list[MPI.Request] = []  # Keep track of intra-island send requests.
        self.intra_buffers: list[Individual] = []  # Send buffers for intra-island communication
        self.inter_requests: list[MPI.Request] = []  # Keep track of inter-island send requests.
        self.inter_buffers: list[list[Individual]] = []  # Send buffers for inter-island communication

        # Load initial population of evaluated individuals from checkpoint if exists.
        self.checkpoint_path = self.checkpoint_path / "ckpt.hdf5"

        self.population: dict[Tuple[int, int, int], Individual] = {}
        # consistency check and ensure enough space is allocated
        if os.path.isfile(self.checkpoint_path):
            self.load_checkpoint()
            if self.propulate_comm.rank == 0:
                log.info("Checkpoint loaded. " f"Resuming from generation {self.generation} of loaded population...")
                # TODO it says resuming from generation 0, so something is not right
                # TODO also each worker might be on a different generation so this message probably does not make all of the sense
        else:
            if self.propulate_comm.rank == 0:
                log.info("No valid checkpoint file given. Initializing population randomly...")
        self.set_up_checkpoint()

    def load_checkpoint(self) -> None:
        """Load checkpoint from HDF5 file. Since this is only a read, all workers can do this in read-only mode without the mpio driver."""
        # TODO check that the island and worker setup is the same as in the checkpoint
        # NOTE each individual is only stored once at the position given by its origin island and worker, the modifications have to be put in the checkpoint file during migration  TODO test if this works as intended reliably
        # TODO get the started but not yet completed ones from the difference in start time and evaltime
        # TODO only load an incomplete one, if you're then going to evaluate it
        log.info(f"Loading checkpoint from {self.checkpoint_path}.")

        with h5py.File(self.checkpoint_path, "r", driver=None) as f:
            # NOTE check limits are consistent
            limitsgroup = f["limits"]
            if set(limitsgroup.attrs.keys()) != set(self.propagator.limits):
                raise RuntimeError("Limits inconsistent with checkpoint")
            # TODO check island sizes are consistent

            # NOTE generation is the index of individuals whose evaluation has begun
            self.generation = int(f["generations"][self.propulate_comm.Get_rank()])

            # NOTE load individuals, since they might have migrated, every worker has to check each dataset
            num_islands = len(self.island_counts)
            for i in range(num_islands):
                islandgroup = f[f"{i}"]
                for rank in range(self.island_counts[i]):
                    for generation in range(f["generations"][rank] + 1):
                        if islandgroup[f"{rank}"]["active_on_island"][generation][self.island_idx]:
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
                            # ind.startime = islandgroup[f"{rank}"]["starttime"][generation]
                            ind.evaltime = islandgroup[f"{rank}"]["evaltime"][generation]
                            ind.evalperiod = islandgroup[f"{rank}"]["evalperiod"][generation]
                            ind.generation = generation
                            ind.island_rank = rank
                            self.population[(i, rank, generation)] = ind

    def set_up_checkpoint(self) -> None:
        """Initialize checkpoint file or check consistenct with an existing one."""
        log.info(f"Initializing checkpoint in {self.checkpoint_path}")
        limit_dim = 0
        for key in self.propagator.limits:
            if isinstance(self.propagator.limits[key][0], str):
                limit_dim += len(self.propagator.limits[key])
            else:
                limit_dim += 1

        num_islands = len(self.island_counts)

        with h5py.File(self.checkpoint_path, "a", driver="mpio", comm=self.propulate_comm) as f:
            # limits
            limitsgroup = f.require_group("limits")
            for key in self.propagator.limits:
                limitsgroup.attrs[key] = str(self.propagator.limits[key])

            xdim = 1
            # TODO clean this up when reorganizing propagators
            # TODO store velocity in its own dataset?
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
                for worker_idx in range(self.island_counts[i]):
                    group = f[f"{i}"].require_group(f"{worker_idx}")
                    if oldgenerations < self.generations:
                        group["x"].resize(self.generations, axis=0)
                        group["loss"].resize(self.generations, axis=0)
                        group["current"].resize(self.generations, axis=0)
                        group["starttime"].resize(self.generations, axis=0)
                        group["evaltime"].resize(self.generations, axis=0)
                        group["evalperiod"].resize(self.generations, axis=0)
                        group["active_on_island"].resize(self.generations, axis=0)
                        if xdim == 2:
                            group["x"].resize(xdim, axis=1)

                    group.require_dataset(
                        "x",
                        (self.generations, xdim, limit_dim),
                        dtype=np.float32,
                        chunks=True,
                        maxshape=(None, 2, limit_dim),
                        data=np.full(
                            (self.generations, xdim, limit_dim),
                            np.nan,
                            dtype=np.float32,
                        ),
                        fillvalue=np.nan,
                    )
                    group.require_dataset(
                        "loss",
                        (self.generations,),
                        np.float32,
                        data=np.array([np.nan] * self.generations),
                        chunks=True,
                        maxshape=(None,),
                        fillvalue=np.nan,
                    )
                    group.require_dataset(
                        "current",
                        (self.generations,),
                        np.int16,
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
                    group.require_dataset(
                        "active_on_island",
                        (self.generations, num_islands),
                        dtype=bool,
                        chunks=True,
                        maxshape=(None, None),
                        data=np.zeros((self.generations, num_islands), dtype=bool),
                    )

    def _get_active_individuals(self) -> List[Individual]:
        """
        Get active individuals in current population list.

        Returns
        -------
        List[propulate.population.Individual]
            All active individuals in the current population.
        int
            The number of currently active individuals.
        """
        active_pop = [ind for ind in self.population.values() if ind.active]
        return active_pop

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
            active_pop = self._get_active_individuals()
            ind = self.propagator(active_pop)  # Breed new individual from active population.
            assert isinstance(ind, Individual)
            ind.generation = self.generation  # Set generation.
            ind.rank = self.island_comm.rank  # Set worker rank.
            ind.active = True  # If True, individual is active for breeding.
            ind.island = self.island_idx  # Set birth island.
            ind.current = self.island_comm.rank  # Set worker responsible for migration.
            ind.migration_history = str(self.island_idx)
        else:  # The other processes do not breed themselves.
            ind = None

        if self.worker_sub_comm != MPI.COMM_SELF:  # Broadcast newly bred individual to all internal ranks of a worker from rank 0,
            # which is also part of the Propulate comm.
            ind = self.worker_sub_comm.bcast(obj=ind, root=0)

        assert isinstance(ind, Individual)
        return ind  # Return new individual.

    def _evaluate_individual(self, ind: Individual) -> None:
        """Evaluate individual."""
        # Signal start of run to surrogate model.
        if self.surrogate is not None:
            self.surrogate.start_run(ind)

        # TODO check whenever the loss_fn is called, that it returns not NaN, infinities are allowed
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
                        assert ind.loss == float("inf")
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
        self.population[
            (self.island_idx, self.island_comm.rank, ind.generation)
        ] = ind  # Add evaluated individual to worker-local population.
        log.debug(
            f"Island {self.island_idx} Worker {self.island_comm.rank} Generation {self.generation}: BREEDING\n"
            f"Bred and evaluated individual {ind}."
        )

    def _sub_rank_evaluate_individual(self) -> None:
        """Receive parameters from worker rank 0 and evaluate."""
        pass

    def _send_intra_island_individuals(self, ind: Individual) -> None:
        """Send evaluated individual to other workers within own island."""
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

                self.population[
                    (ind_temp.island, ind_temp.rank, ind_temp.generation)
                ] = ind_temp  # Add received individual to own worker-local population.

                log_string += f"Added individual {ind_temp} from W{stat.Get_source()} to own population.\n"
        n_active = len(self._get_active_individuals())
        log_string += f"After probing within island: {n_active}/{len(self.population)} active."
        log.debug(log_string)

    def _send_emigrants(self, *args: Any, **kwargs: Any) -> None:
        """
        Perform migration, i.e., island sends individuals out to other islands.

        Raises
        ------
        NotImplementedError
            Not implemented in ``Propulator`` base class. Exact migration and pollination behavior is defined in the
            ``Migrator`` and ``Pollinator`` classes, respectively.
        """
        raise NotImplementedError

    def _receive_immigrants(self, *args: Any, **kwargs: Any) -> None:
        """
        Check for and possibly receive immigrants send by other islands.

        Raises
        ------
        NotImplementedError
            Not implemented in ``Propulator`` base class. Exact migration and pollination behavior is defined in the
            ``Migrator`` and ``Pollinator`` classes, respectively.
        """
        raise NotImplementedError

    def _pre_eval_checkpoint(self, ind: Individual, f: h5py.File) -> None:
        """
        Write a bred but not yet evaluated individual to the checkpoint file.

        Parameters
        ----------
        ind : Individual
            Individual to be stored.
        f : h5py.File
            Open and initialized hdf5 file to be written to.
        """
        ind.island_rank = self.island_comm.Get_rank()
        start_time = time.time_ns() - self.start_time  # Start evaluation timer.
        ind.start_time = start_time
        ckpt_idx = ind.generation
        f["generations"][self.propulate_comm.Get_rank()] = ind.generation

        group = f[f"{self.island_idx}"][f"{self.island_comm.Get_rank()}"]
        # save candidate
        group["x"][ckpt_idx, 0, :] = ind.position[:]
        if ind.velocity is not None:
            group["x"][ckpt_idx, 1, :] = ind.velocity[:]
        group["starttime"][ckpt_idx] = start_time
        group["current"][ckpt_idx] = ind.current

    def _post_eval_checkpoint(self, ind: Individual, f: h5py.File) -> None:
        """
        Update an evaluated individual previously written to the checkpoint file.

        Parameters
        ----------
        ind : Individual
            Individual to be stored.
        f : h5py.File
            Open and initialized hdf5 file to be written to.
        """
        ckpt_idx = ind.generation
        group = f[f"{self.island_idx}"][f"{self.island_comm.Get_rank()}"]
        ind.evaltime = time.time_ns() - self.start_time  # Stop evaluation timer.
        ind.evalperiod = ind.evaltime - ind.start_time  # Calculate evaluation duration.
        # save result for candidate
        group["evaltime"][ckpt_idx] = ind.evaltime
        group["evalperiod"][ckpt_idx] = ind.evalperiod
        group["active_on_island"][ckpt_idx, self.island_idx] = True
        # TODO fix evalperiod for resumed from checkpoint individuals
        # TODO somehow store migration history, maybe just as islands_visited

        group["loss"][ckpt_idx] = ind.loss

    def _get_unique_individuals(self) -> List[Individual]:
        """
        Get unique individuals in terms of traits and loss in current population.

        Returns
        -------
        List[propulate.population.Individual]
            All unique individuals in the current population.
        """
        unique_inds: List[Individual] = []
        for individual in self.population.values():
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

    def propulate(self, logging_interval: int = 10) -> None:
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
        # TODO refactor timing
        self.start_time = time.time_ns()
        if self.worker_sub_comm != MPI.COMM_SELF:
            self.generation = self.worker_sub_comm.bcast(self.generation, root=0)
        # NOTE if true, this rank is a worker sub rank, so it does not breeding and checkpointing, just evaluation
        if self.propulate_comm is None:
            while self.generation < self.generations:
                # Breed and evaluate individual.
                # TODO this should be refactored, the subworkers don't need the logfile
                # TODO this needs to be addressed before merge, since multirank workers should fail with this
                self._sub_rank_evaluate_individual()
                self.generation += 1
            return

        if self.island_comm.rank == 0:
            log.info(f"Island {self.island_idx} has {self.island_comm.size} workers.")

        # Loop over generations.
        # NOTE there is an implicit barrier when closing the file in parallel mode
        with h5py.File(self.checkpoint_path, "a", driver="mpio", comm=self.propulate_comm) as f:
            # NOTE check if there is an individual still to evaluate
            # TODO for now we write a loss to the checkpoint only if the evaluation has completed
            # TODO how do we save the surrogate model?
            current_idx = (self.island_idx, self.island_comm.rank, self.generation)
            if current_idx in self.population:
                print(self.population)
                print(f["0"]["0"]["loss"][:])
                ind = self.population[current_idx]
                if np.isnan(ind.loss):
                    log.info(f"Continuing evaluation of individual {current_idx} loaded from checkpoint.")
                    self._evaluate_individual(ind)
                    self._post_eval_checkpoint(ind, f)
                    self._send_intra_island_individuals(ind)
                self.generation += 1

            while self.generation < self.generations:
                if self.generation % int(logging_interval) == 0:
                    log.info(f"Island {self.island_idx} Worker {self.island_comm.rank}: In generation {self.generation}...")

                # Breed and evaluate individual.
                ind = self._breed()  # Breed new individual.
                # NOTE write started individual to file
                self._pre_eval_checkpoint(ind, f)
                # NOTE start evaluation
                self._evaluate_individual(ind)
                # NOTE update finished individual in checkpoint
                self._post_eval_checkpoint(ind, f)
                self._send_intra_island_individuals(ind)
                # Check for and possibly receive incoming individuals from other intra-island workers.
                self._receive_intra_island_individuals()
                # Clean up requests and buffers.
                self._intra_send_cleanup()
                # Go to next generation.
                self.generation += 1

        log.info(f"Island {self.island_idx} Worker {self.island_comm.rank}: OPTIMIZATION DONE!")

    def _intra_send_cleanup(self) -> None:
        """Delete all send buffers that have been sent."""
        # Test for requests to complete.
        completed = MPI.Request.Testsome(self.intra_requests)
        # Remove requests and buffers of complete send operations.
        assert len(self.intra_requests) == len(self.intra_buffers)
        self.intra_requests = [r for i, r in enumerate(self.intra_requests) if i not in completed]
        self.intra_buffers = [b for i, b in enumerate(self.intra_buffers) if i not in completed]

    def _inter_send_cleanup(self) -> None:
        """Delete all send buffers that have been sent."""
        # Test for requests to complete.
        completed = MPI.Request.Testsome(self.inter_requests)
        # Remove requests and buffers of complete send operations.
        assert len(self.inter_requests) == len(self.inter_buffers)
        self.inter_requests = [r for i, r in enumerate(self.inter_requests) if i not in completed]
        self.inter_buffers = [b for i, b in enumerate(self.inter_buffers) if i not in completed]
