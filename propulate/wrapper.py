import os
import pickle
from operator import attrgetter
from mpi4py import MPI
import numpy as np

from .population import Individual
from .propulator import Propulator, PolliPropulator
from .propagators import SelectBest, SelectUniform, SelectWorst


class Islands:
    """
    Wrapper class for propulate optimization runs with multiple separate evolutionary isles.
    """

    def __init__(
        self,
        loss_fn,
        propagator,
        rng, 
        generations=0,
        num_isles=1,
        isle_sizes=None,
        migration_topology=None,
        migration_probability=0.0,
        emigration_propagator=SelectBest,
        immigration_propagator=SelectWorst,
        pollination=False,
        load_checkpoint="pop_cpt.p",
        save_checkpoint="pop_cpt.p",
    ):
        """
        Constructor of Islands() class.

        Parameters
        ----------
        loss_fn : callable
                  loss function to be minimized
        propagator : propulate.propagators.Propagator
                     propagator to apply for breeding
        generations : int
                      number of generations
        num_isles : int
                    number of separate, equally sized evolutionary isles (ignored if `isle_sizes` is not None)
                    (differences +-1 possible due to load balancing)
        isle_sizes : list
                     list with numbers of workers for each evolutionary isle (heterogeneous case)
        migration_topology : numpy array
                             2D matrix where entry (i,j) specifies how many individuals are sent
                             by isle i to isle j
                             (int: absolute number, float: relative fraction of population)
        migration_probability : float
                                probability of migration after each generation
        emigration_propagator : propulate.propagators.Propagator
                                emigration propagator, i.e., how to choose individuals for emigration
                                that are sent to destination island.
                                Should be some kind of selection operator.
        immigration_propagator : propulate.propagators.Propagator
                                 immigration propagator, i.e., how to choose individuals on target isle
                                 to be replaced by immigrants.
                                 Should be some kind of selection operator.
        pollination : bool
                      If True, copies of emigrants are sent, otherwise, emigrants are removed from
                      original isle.
        load_checkpoint : str
                          checkpoint file to resume optimization from
        save_checkpoint : str
                          checkpoint file to write checkpoints to
        """
        # Set attributes.
        self.loss_fn = loss_fn  # callable loss function
        self.propagator = propagator  # propagator
        self.generations = int(
            generations
        )  # number of generations, i.e., evaluations per individual

        # Set up communicators.
        size = MPI.COMM_WORLD.size
        rank = MPI.COMM_WORLD.rank

        if rank == 0:
            print("#################################################")
            print("# PROPULATE: parallel PROpagator of POPULations #")
            print("#################################################\n")

        # Homogeneous case with equal isle sizes (differences of +-1 possible due to load balancing).
        if isle_sizes is None:
            if num_isles < 1:
                raise ValueError(
                    f"Invalid number of evolutionary isles, needs to be >= 1, but was {num_isles}."
                )
            num_isles = int(num_isles)  # number of separate isles
            base_size = int(size // num_isles)  # base size of each isle
            remainder = int(
                size % num_isles
            )  # remaining workers to be distributed equally for load balancing

            isle_sizes = []
            for i in range(num_isles):
                if i < remainder:
                    temp = isle_sizes.append(base_size + 1)
                else:
                    isle_sizes.append(base_size)

        isle_sizes = np.array(isle_sizes)

        # Heterogeneous case with user-defined isle sizes.
        if np.sum(isle_sizes) != size:
            raise ValueError(
                f"There should be MPI.COMM_WORLD.size (i.e., {size}) workers but only {np.sum(isle_sizes)} were specified."
            )

        # intra-isle communicator (for communication within each separate isle)
        Intra_color = []
        for idx, el in enumerate(isle_sizes):
            Intra_color.append(idx * np.ones(el, dtype=int))
        Intra_color = np.concatenate(Intra_color).ravel()
        intra_color = Intra_color[rank]
        intra_key = rank

        # inter-isle communicator (for communication between different isles)
        # Determine unique elements, where # unique elements equals number of isles.
        _, unique_ind = np.unique(Intra_color, return_index=True)
        num_isles = (
            unique_ind.size
        )  # Determine number of isles as number of unique elements.
        Inter_color = np.zeros(size)  # Initialize inter color with only zeros.
        if rank == 0:
            print(
                f"Island sizes {Intra_color} with counts {isle_sizes} and start displacements {unique_ind}."
            )
        Inter_color[unique_ind] = 1
        inter_color = Inter_color[rank]
        inter_key = rank

        # Create new communicators by "splitting" MPI.COMM_WORLD into group of sub-communicators based on
        # input values `color` and `key`. The original communicator does not go away but a new communicator
        # is created on each process. `color` determines to which new communicator each processes will belong.
        # All processes which pass in the same value for `color` are assigned to the same communicator. `key`
        # determines the ordering (rank) within each new communicator. The process which passes in the smallest
        # value for `key` will be rank 0 and so on.
        comm_intra = MPI.COMM_WORLD.Split(color=intra_color, key=intra_key)
        comm_inter = MPI.COMM_WORLD.Split(color=inter_color, key=inter_key)

        # Determine isle index and broadcast to all ranks in intra-isle communicator.
        if comm_intra.rank == 0:
            isle_idx = comm_inter.rank
        else:
            isle_idx = None
        isle_idx = comm_intra.bcast(isle_idx, root=0)

        if migration_topology is None:
            migration_topology = np.ones((num_isles, num_isles), dtype=int)
            np.fill_diagonal(migration_topology, 0)
            if rank == 0:
                print(
                    "NOTE: No migration topology given, using fully connected top-1 topology..."
                )

        if rank == 0:
            print(
                f"Migration topology {migration_topology} has shape {migration_topology.shape}."
            )

        if migration_topology.shape != (num_isles, num_isles):
            raise ValueError(
                f"Migration topology must be a quadratic matrix of size "
                f"{unique.size} x {unique.size} but has shape {migration_topology.shape}."
            )

        if migration_probability > 1.0:
            raise ValueError(
                f"Migration probability must be in [0, 1] but was set to {migration_probability}."
            )
        migration_prob = float(migration_probability) / comm_intra.size

        if rank == 0:
            print(
                f"NOTE: Isle migration probability of {migration_probability} "
                f"results in per-rank migration probability of {migration_prob}."
            )
        load_rank_cpt = "isle_" + str(isle_idx) + "_" + load_checkpoint
        save_rank_cpt = "isle_" + str(isle_idx) + "_" + save_checkpoint

        self.emigration_propagator = emigration_propagator

        if rank == 0:
            print("Starting parallel optimization process...")
        MPI.COMM_WORLD.barrier()
        # Set up Propulator objects, one for each isle.
        if pollination == False:
            if MPI.COMM_WORLD.rank == 0:
                print("No pollination.")
            self.propulator = Propulator(
                loss_fn,
                propagator,
                comm=comm_intra,
                generations=generations,
                isle_idx=isle_idx,
                load_checkpoint=load_rank_cpt,
                save_checkpoint=save_rank_cpt,
                comm_inter=comm_inter,
                migration_topology=migration_topology,
                migration_prob=migration_prob,
                emigration_propagator=emigration_propagator,
                unique_ind=unique_ind,
                unique_counts=isle_sizes,
                rng=rng,
            )
        elif pollination == True:
            if MPI.COMM_WORLD.rank == 0:
                print("Pollination.")
            self.propulator = PolliPropulator(
                loss_fn,
                propagator,
                comm=comm_intra,
                generations=generations,
                isle_idx=isle_idx,
                load_checkpoint=load_rank_cpt,
                save_checkpoint=save_rank_cpt,
                comm_inter=comm_inter,
                migration_topology=migration_topology,
                migration_prob=migration_prob,
                emigration_propagator=emigration_propagator,
                immigration_propagator=immigration_propagator,
                unique_ind=unique_ind,
                unique_counts=isle_sizes,
                rng=rng,
            )

    def _run(self, top_n, out_file, logging_interval, DEBUG):
        """
        Run propulate optimization.

        Parameters
        ----------
        top_n : int
                number of best results to report
        """
        self.propulator.propulate(logging_interval, DEBUG)
        if DEBUG > -1:
            best = self.propulator.summarize(top_n, out_file=out_file, DEBUG=DEBUG)
            return best
        else:
            return None

    def evolve(self, top_n=3, out_file="summary.png", logging_interval=10, DEBUG=1):
        """
        Run propulate optimization.

        Parameters
        ----------
        top_n : int
                number of best results to report
        """
        best = self._run(top_n, out_file, logging_interval, DEBUG)
        return best
