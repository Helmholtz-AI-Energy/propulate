import os
import pickle
import random
import numpy as np
from operator import attrgetter
from mpi4py import MPI

from ._globals import INDIVIDUAL_TAG, LOSS_REPORT_TAG, INIT_TAG, POPULATION_TAG, DUMP_TAG, EMIGRATION_TAG, IMMIGRATION_TAG
from .population import Individual
from .propagators import SelectBest, SelectUniform

DEBUG = False

class Propulator():
    """
    Parallel propagator of populations.
    """
    def __init__(self, loss_fn, propagator, isle_idx, comm=MPI.COMM_WORLD, generations=0,
                 load_checkpoint = "pop_cpt.p", save_checkpoint="pop_cpt.p", 
                 seed=9, migration_topology=None, comm_migrate=MPI.COMM_WORLD,
                 migration_prob=None, emigration_policy=None, 
                 pollination=True, immigration_policy=None,
                 unique_ind=None, unique_counts=None):
        """
        Constructor of Propulator class.

        Parameters
        ----------
        loss_fn : callable
                  loss function to be minimized
        propagator : propulate.propagators.Propagator
                     propagator to apply for breeding
        comm : MPI communicator
               intra-isle communicator
        generations : int
                      number of generations to run
        isle_idx : int
                   isle index
        load_checkpoint : str
                          checkpoint file to resume optimization from
        save_checkpoint : str
                          checkpoint file to write checkpoints to
        seed : int
               base seed for random number generator
        comm_migrate : MPI communicator
                       inter-isle communicator for migration
        """
        # Set class attributes.
        self.loss_fn = loss_fn                      # callable loss function
        self.propagator = propagator                # propagator
        self.generations = int(generations)         # number of generations, i.e., number of evaluation per individual
        self.isle_idx = int(isle_idx)               # isle index
        self.comm = comm                            # intra-isle communicator for actual evolutionary optimization within isle
        self.comm_migrate = comm_migrate            # inter-isle communicator for migration between isles
        self.load_checkpoint = str(load_checkpoint) # path to checkpoint file to be read
        self.save_checkpoint = str(save_checkpoint) # path to checkpoint file to be written
        self.migration_prob = float(migration_prob) # per-rank migration probability
        self.migration_topology = migration_topology# migration topology
        self.unique_ind = unique_ind
        self.unique_counts = unique_counts
        if emigration_policy == "best": self.emigration_propagator = SelectBest
        elif emigration_policy == "random": self.emigration_propagator = SelectUniform
        else: raise ValueError("Invalid emigration policy {}, should be ``random'' or ``best''.".format(emigration_policy))
        # Load initial population of evaluated individuals from checkpoint if exists.
        if not os.path.isfile(self.load_checkpoint): # If not exists, check for backup file.
            self.load_checkpoint = self.load_checkpoint+'.bkp'

        if os.path.isfile(self.load_checkpoint):
            with open(self.load_checkpoint, 'rb') as f:
                try:
                    self.population = pickle.load(f)
                    self.best = min(self.population, key=attrgetter('loss'))
                    if self.comm.rank == 0: 
                        print("NOTE: Valid checkpoint file found. Resuming from loaded population...")
                except Exception:
                    self.population = []
                    self.best = None
                    if self.comm.rank == 0:
                        print("NOTE: No valid checkpoint file found. Initializing population randomly...")
        else:
            self.population=[]
            self.best = None
            if self.comm.rank == 0: 
                print("NOTE: No valid checkpoint file given. Initializing population randomly...")


    def propulate(self):
        """
        Run actual evolutionary optimization.""
        """
        self._work()


    def _breed(self, generation):
        """
        Apply propagator to current population to breed new individual.

        Parameters
        ----------
        generation : int
                     generation of newly bred individual

        Returns
        -------
        ind : propulate.population.Individual
              newly bred individual
        """
        ind = None                             # Initialize new individual.
        ind = self.propagator(self.population) # Breed new individual from current population.
        ind.generation = generation            # Set generation as individual's attribute.
        ind.rank = self.comm.rank              # Set worker rank as individual's attribute.

        return ind                             # Return newly bred individual.


    def _work(self):
        """
        Execute evolutionary algorithm in parallel.
        """
        generation = 0        # Start from generation 0.
        rank = self.comm.rank # Determine individual worker's MPI rank.

        if rank == 0: print("Isle {} has a population of {} individuals.".format(self.isle_idx, self.comm.size)) 
        
        dump = True if rank == 0 else False

        if self.generations == 0: # If number of iterations requested == 0.
            if MPI.COMM_WORLD.rank == 0: print("Number of requested generations is zero...[RETURN]")
            return

        while self.generations == -1 or generation < self.generations:
            print("Isle {}: Worker {} in generation {}...".format(self.isle_idx, rank, generation)) 
            ind = self._breed(generation)   # Breed new individual to evaluate.
            ind.loss = self.loss_fn(ind)    # Evaluate individual's loss.
            self.population.append(ind)     # Append own result to own history list.

            # Tell the world about your great results.
            # Use immediate send for asynchronous communication.
            for r in range(self.comm.size):
                if r == rank: continue
                self.comm.isend(ind, dest=r, tag=LOSS_REPORT_TAG)
            
            # Check for incoming messages.
            probe = True
            while probe:
                if DEBUG: print("Isle {}: Worker {} checking for incoming messages...".format(self.isle_idx, rank))
                stat = MPI.Status()
                probe = self.comm.iprobe(source=MPI.ANY_SOURCE, tag=LOSS_REPORT_TAG, status=stat)
                if stat.Get_source() >=0:
                    if DEBUG: print("Isle {}: Worker {} incoming message from worker {}...".format(self.isle_idx, rank, stat.Get_source()))
                if probe:
                    req = self.comm.irecv(source=stat.Get_source(), tag=LOSS_REPORT_TAG)
                    ind_temp = req.wait()
                    if DEBUG: print("Isle {}: Worker {} received message from worker {}.".format(self.isle_idx, rank, stat.Get_source()))
                    self.population.append(ind_temp)

            if DEBUG: print("Isle {}: Worker {} checking for incoming messages...[DONE]".format(self.isle_idx, rank))
            # Determine current best as individual with minimum loss.
            self.best = min(self.population, key=attrgetter('loss'))

            if dump: # Dump checkpoint.
                #if DEBUG: 
                print("Isle {}: Worker {} dumping checkpoint...".format(self.isle_idx, rank))
                if os.path.isfile(self.save_checkpoint):
                    try: os.replace(self.save_checkpoint, self.save_checkpoint+".bkp")
                    except Exception as e: print(e)
                with open(self.save_checkpoint, 'wb') as f:
                    pickle.dump((self.population), f)
                
                dest = rank+1 if rank+1 < self.comm.size else 0
                self.comm.isend(dump, dest=dest, tag=DUMP_TAG)
                dump = False
            
            stat = MPI.Status()
            probe = self.comm.iprobe(source=MPI.ANY_SOURCE, tag=DUMP_TAG, status=stat)
            if probe:
                req_dump = self.comm.irecv(source=stat.Get_source(), tag=DUMP_TAG)
                dump = req_dump.wait()
                if DEBUG: print("Isle", self.isle_idx, "Worker", rank, "is going to dump next:", dump, ". Before: worker", stat.Get_source())
           
            # Emigration: Isle sends individuals out.
            if random.random() < self.migration_prob: 
                print("Isle {}: Worker {} wants to send migrants.".format(self.isle_idx, rank))
                to_migrate = self.migration_topology[self.isle_idx,:]
                print("Isle {}: Migration topology {}:".format(self.isle_idx, to_migrate))
                for target_isle, offspring in enumerate(to_migrate):
                    if offspring == 0: continue
                    print("Isle {}: Worker {} about to send {} individual(s) to isle {}.".format(self.isle_idx, rank, offspring, target_isle))
                    displ = self.unique_ind[target_isle]
                    count = self.unique_counts[target_isle]
                    print("{} worker(s) on target isle {}, COMM_WORLD displ is {}.".format(count, target_isle, displ))
                    dest = np.arange(displ,displ+count)
                    print("MPI.COMM_WORLD dest ranks: {}".format(dest))
                    emigrator = self.emigration_propagator(offspring)
                    emigrants = emigrator(self.population)
                    print("Emigrating individual(s): {}".format(emigrants))

                for r in range(self.comm.size):
                    if r == rank: continue
                    self.comm.isend(ind, dest=r, tag=LOSS_REPORT_TAG)
                
                # Check for incoming messages.
                probe = True
                while probe:
                    if DEBUG: print("Isle {}: Worker {} checking for incoming messages...".format(self.isle_idx, rank))
                    stat = MPI.Status()
                    probe = self.comm.iprobe(source=MPI.ANY_SOURCE, tag=LOSS_REPORT_TAG, status=stat)
                    if stat.Get_source() >=0:
                        if DEBUG: print("Isle {}: Worker {} incoming message from worker {}...".format(self.isle_idx, rank, stat.Get_source()))
                    if probe:
                        req = self.comm.irecv(source=stat.Get_source(), tag=LOSS_REPORT_TAG)
                        ind_temp = req.wait()
                        if DEBUG: print("Isle {}: Worker {} received message from worker {}.".format(self.isle_idx, rank, stat.Get_source()))
                        self.population.append(ind_temp)
            # Immigration: Isle checks for immigrants, i.e., incoming individuals from other islands.

            generation += 1
        
        # Having completed all generations, the workers have to wait for each other.
        # Once all workers are done, they should check for incoming messages once again
        # so that each of them holds the complete final population and the found optimum
        # irrespective of the order they finished.

        self.comm.barrier()

        # Final check for incoming messages.
        probe = True
        while probe:
            stat = MPI.Status()
            # TODO check whether MPI.ANY_SOURCE or stat.MPI_SOURCE should be used here.
            probe = self.comm.iprobe(source=MPI.ANY_SOURCE, tag=LOSS_REPORT_TAG, status=stat)
            if probe:
                req = self.comm.irecv(source=stat.Get_source(), tag=LOSS_REPORT_TAG)
                ind_temp = req.wait()
                self.population.append(ind_temp)
        
        # Determine final best as individual with minimum loss.
        self.best = min(self.population, key=attrgetter('loss'))
   
        # Final checkpointing on rank 0.
        if rank == 0: # Dump checkpoint.
            if os.path.isfile(self.save_checkpoint):
                os.replace(self.save_checkpoint, self.save_checkpoint+".bkp")
                with open(self.save_checkpoint, 'wb') as f:
                    pickle.dump((self.population), f)

        MPI.COMM_WORLD.barrier()


    def summarize(self, top_n=1, out_file='summary.png'):
        """
        Get top-n results from propulate optimization.

        Parameters
        ----------
        top_n : int
                number of best results to report
        out_file : string
                   path to results plot (rank-specific loss vs. generation)
        """
        top_n = int(top_n)
        total = self.comm_migrate.reduce(len(self.population), root=0)
        if MPI.COMM_WORLD.rank == 0:
            print("\n###########")
            print("# SUMMARY #")
            print("###########\n")
            print("{} individuals have been evaluated overall.".format(total))
        MPI.COMM_WORLD.barrier()
        if self.comm.rank == 0:
            print("{} individuals have been evaluated on isle {}.".format(len(self.population), self.isle_idx))
            self.population.sort(key=lambda x: x.loss)
            print("Top {} result(s) on isle {}:".format(top_n, self.isle_idx))
            for i in range(top_n):
                print("({}): {} with loss {}".format(i+1, self.population[i], self.population[i].loss))
            import matplotlib.pyplot as plt
            xs = [x.generation for x in self.population]
            ys = [x.loss for x in self.population]
            zs = [x.rank for x in self.population]

            fig, ax = plt.subplots()
            scatter = ax.scatter(xs, ys, c=zs)
            plt.xlabel("Generation")
            plt.ylabel("Loss")
            legend = ax.legend(*scatter.legend_elements(), title="Rank") 
            plt.savefig(out_file)
