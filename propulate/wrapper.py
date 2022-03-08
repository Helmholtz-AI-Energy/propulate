import os
import pickle
from operator import attrgetter
from mpi4py import MPI
import numpy as np

from ._globals import INDIVIDUAL_TAG, LOSS_REPORT_TAG, INIT_TAG, POPULATION_TAG, DUMP_TAG
from .population import Individual
from .propulator import Propulator

class Islands():
    """
    Wrapper class for propulate optimization runs with multiple separate evolutionary isles.
    """
    def __init__(self, loss_fn, propagator, generations=0, 
                 num_isles=1, isle_sizes=None, migration_topology=None,
                 migration_probability=0.1, emigration_policy="best", immigration_policy="worst", pollination=False,
                 load_checkpoint = "pop_cpt.p", save_checkpoint="pop_cpt.p", seed=None):
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
        isle_sizes : array
                     array with sizes of different evolutionary isles (heterogeneous case)
        migration_topology : array
                             2D matrix where entry (i,j) specifies how many individuals are sent
                             by isle i to isle j 
                             (int: absolute number, float: relative fraction of population)
        migration_probability : float
                                probability of migration after each generation
        emigration_policy : str
                            emigration policy, i.e., how to choose individuals for emigration
                            that are sent to destination island

        pollination : bool
                      If True, copies of emigrants are sent, otherwise, emigrants are removed from
                      original isle.

        immigration_policy : str
                             immigration policy, i.e., how to replace individuals on destination isle
                             with immigrating individuals

        load_checkpoint : str
                          checkpoint file to resume optimization from
        save_checkpoint : str
                          checkpoint file to write checkpoints to
        seed : int
               base seed for random number generator
        """
        # Set attributes.
        self.loss_fn = loss_fn              # callable loss function
        self.propagator = propagator        # propagator
        if generations < -1: raise ValueError("Invalid number of generations, needs to be > -1, but was {}.".format(generations))
        self.generations = int(generations) # number of generations, i.e., evaluations per individual

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
                raise ValueError("Invalid number of evolutionary isles, needs to be >= 1, but was {}".format(num_isles))
            num_isles = int(num_isles) # number of separate isles
            base_size = size // num_isles
            remainder = size % num_isles

            isle_sizes = []
            for i in range(num_isles):
                if i < remainder: temp = i* np.ones(base_size+1, dtype=int)
                else: temp = i* np.ones(base_size, dtype=int)
                isle_sizes.append(temp)
            isle_sizes = np.concatenate(isle_sizes).ravel()
        
        # Heterogeneous case with user-defined isle sizes.
        if isle_sizes.size != size: 
            raise ValueError(
                    "`isle_sizes` should have MPI.COMM_WORLD.size (i.e., {}) elements but has only {}.".format(size, isle_sizes.size)
                    )
        # intra-isle communicator (for communication within each separate isle)
        intra_color = isle_sizes[rank]
        intra_key   = rank

        # inter-isle communicator (for communication between different isles)
        # Determine unique elements, where # unique elements equals number of isles.
        _, unique_ind, unique_counts = np.unique(isle_sizes, return_index=True, return_counts=True) 
        num_isles = unique_ind.size     # Determine number of isles as number of unique elements.
        inter_color = np.zeros(size)    # Initialize inter color with only zeros.
        if rank==0: print("Island sizes {} with counts {} and start displacements {}.".format(isle_sizes, unique_counts, unique_ind))
        inter_color[unique_ind] = 1 
        inter_color = inter_color[rank]
        inter_key  = rank
        
        # Create new communicators by "splitting" MPI.COMM_WORLD into group of sub-communicators based on
        # input values `color` and `key`. The original communicator does not go away but a new communicator
        # is created on each process. `color` determines to which new communicator each processes will belong.
        # All processes which pass in the same value for `color` are assigned to the same communicator. `key`
        # determines the ordering (rank) within each new communicator. The process which passes in the smallest
        # value for `key` will be rank 0 and so on.
        comm_intra  = MPI.COMM_WORLD.Split(color=intra_color, key=intra_key)
        comm_inter  = MPI.COMM_WORLD.Split(color=inter_color, key=inter_key)

        # Determine isle index and broadcast to all ranks in intra-isle communicator.
        if comm_intra.rank == 0: isle_idx = comm_inter.rank
        else: isle_idx = None
        isle_idx = comm_intra.bcast(isle_idx, root=0)

        if migration_topology is None:
            migration_topology = np.ones((num_isles, num_isles), dtype=int)
            np.fill_diagonal(migration_topology, 0)
            if rank == 0: print("NOTE: No migration topology given, using fully connected top-1 topology...")
        
        if rank == 0: print("Migration topology {} has shape {}.".format(migration_topology, migration_topology.shape))

        if migration_topology.shape != (num_isles, num_isles):
            raise ValueError(
                    "Migration topology must be a quadratic matrix of size {} x {} but has shape {}.".format(unique.size, 
                        unique.size, migration_topology.shape)
                    )

        if migration_probability > 1.: 
            raise ValueError("Migration probability must be in [0, 1] but was set to {}.".format(migration_probability))
        migration_prob = float(migration_probability) / comm_intra.size

        if rank==0: 
            print("NOTE: Isle migration probability of {} results in per-rank migration probability of {}.".format(migration_probability, 
                                                                                                                   migration_prob))
        load_rank_cpt = "isle_" + str(isle_idx) + "_" + load_checkpoint
        save_rank_cpt = "isle_" + str(isle_idx) + "_" + save_checkpoint

        if rank == 0: print("Starting parallel optimization process...")
        # Set up Propulator objects, one for each isle.
        self.propulator = Propulator(loss_fn, propagator, comm=comm_intra, generations=generations, isle_idx=isle_idx,
                                     load_checkpoint=load_rank_cpt, save_checkpoint=save_rank_cpt, 
                                     seed=9, comm_migrate=comm_inter, migration_topology=migration_topology,
                                     migration_prob=migration_prob, emigration_policy=emigration_policy, 
                                     immigration_policy=immigration_policy, pollination=pollination,
                                     unique_ind=unique_ind, unique_counts=unique_counts)



    def _run(self, top_n=3, logging_interval=10):        
        """
        Run propulate optimization.

        Parameters
        ----------
        top_n : int
                number of best results to report
        """
        self.propulator.propulate(logging_interval)
        self.propulator.summarize(top_n)


    def evolve(self, top_n=3, logging_interval=10):
        """
        Run propulate optimization.

        Parameters
        ----------
        top_n : int
                number of best results to report
        """
        self._run(top_n, logging_interval)
