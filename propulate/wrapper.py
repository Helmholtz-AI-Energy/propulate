import os
import pickle
from operator import attrgetter
from mpi4py import MPI

from ._globals import INDIVIDUAL_TAG, LOSS_REPORT_TAG, INIT_TAG, POPULATION_TAG, DUMP_TAG
from .population import Individual
from .propulator import Propulator

class Islands():
    """
    Wrapper class for propulate optimization runs with multiple separate evolutionary islands.
    """
    def __init__(self, loss_fn, propagator, num_islands=1, generations=0, load_checkpoint = "pop_cpt.p", save_checkpoint="pop_cpt.p", seed=9, migrator = None):
        """
        Constructor of Islands() class.

        Parameters
        ----------
        loss_fn : callable
                  loss function to be minimized
        propagator : propulate.propagators.Propagator
                     propagator to apply for breeding
        num_islands : int
                      number of separate evolutionary islands to consider
        generations : int
                      number of generations to run
        load_checkpoint : str
                          checkpoint file to resume optimization from
        save_checkpoint : str
                          checkpoint file to write checkpoints to
        seed : int
               base seed for random number generator
        migrator : 
                   migration strategy to use for migration between separate evolutionary islands
        """
        # Set attributes.
        self.loss_fn = loss_fn              # Set callable loss function to use.
        self.propagator = propagator        # Set propagator to use.
        self.num_islands = int(num_islands) # Set number of separate evolutionary islands.
        if self.num_islands < 1:
            raise ValueError("Invalid number of evolutionary islands, needs to be >= 1, but was {}".format(self.num_islands))
        self.generations = int(generations) # Set number of generations, i.e., number of evaluation per individual.
        if self.generations < -1:
            raise ValueError("Invalid number of generations, needs to be > -1, but was {}".format(self.generations))

        # Set up communicators.
        size = MPI.COMM_WORLD.size
        rank = MPI.COMM_WORLD.rank
        if size % self.num_islands != 0:
            raise ValueError("Number of requested MPI tasks {} is not evenly divisible by requested number of islands {}.".format(self.size, self.num_islands))
        
        isle_size = size // self.num_islands

        # intra-island communicator (for communication within each separate island)
        intra_color = rank // isle_size
        intra_key   = rank % isle_size
        comm_intra  = MPI.COMM_WORLD.Split(color=intra_color, key=intra_key)

        # inter-island communicator (for communication between different islands)
        inter_color = int(rank % isle_size == 0)
        inter_key   = rank // isle_size
        comm_inter  = MPI.COMM_WORLD.Split(color=inter_color, key=inter_key)
    
        load_rank_cpt = str(comm_inter.rank) + "_" + load_checkpoint
        save_rank_cpt = str(comm_inter.rank) + "_" + save_checkpoint
        
        # Set up Propulator objects, one for each island.
        self.propulator = Propulator(loss_fn, propagator, comm=comm_intra, generations=generations, 
                                     load_checkpoint=load_rank_cpt, save_checkpoint=save_rank_cpt, 
                                     seed=9, migrator=None, comm_migrate=comm_inter)


    def _run(self, top_n):        
        """
        Run propulate optimization.

        Parameters
        ----------
        top_n : int
                number of best results to report
        """
        self.propulator.propulate()
        self.propulator.summarize(top_n)


    def evolve(self, top_n):
        """
        Run propulate optimization.

        Parameters
        ----------
        top_n : int
                number of best results to report
        """
        self._run(top_n)
