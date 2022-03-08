import os
import pickle
import random
import numpy as np
from operator import attrgetter
from mpi4py import MPI

from ._globals import INDIVIDUAL_TAG, LOSS_REPORT_TAG, INIT_TAG, POPULATION_TAG, DUMP_TAG, MIGRATION_TAG, POLLINATION_TAG
from .population import Individual
from .propagators import SelectBest, SelectWorst, SelectUniform

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
        else: raise ValueError("Invalid emigration policy {}, should be ``best'' or ``random''.".format(emigration_policy))

        if immigration_policy == "worst": self.immigration_propagator = SelectWorst
        elif immigration_policy == "random": self.immigration_propagator = SelectUniform
        else: raise ValueError("Invalid immigration policy {}, should be ``worst'' or ``random''.".format(emigration_policy))
        self.pollination = pollination
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

        self.graveyard=[]

    def propulate(self, logging_interval=10):
        """
        Run actual evolutionary optimization.""
        """
        self._work(logging_interval)


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
        # Emigrated individuals should not be used for breeding.
        active_population = [ind for ind in self.population if ind.emigrated == False]
        if DEBUG: 
            print("Isle {}: {} / {} individuals are active for breeding.".format(isle_idx, len(active_population),len(self.population)))
        ind = self.propagator(active_population) # Breed new individual from current active population.
        ind.generation = generation              # Set generation as individual's attribute.
        ind.rank = self.comm.rank                # Set worker rank as individual's attribute.
        ind.emigrated = False                    # Set emigrated flag. If True, individual is not active for breeding.
        ind.isle = self.isle_idx                 # Set birth island.

        return ind                             # Return newly bred individual.


    def _work(self, logging_interval=10):
        """
        Execute evolutionary algorithm in parallel.
        """
        generation = 0        # Start from generation 0.
        rank = self.comm.rank # Determine individual worker's MPI rank.

        if rank == 0: print("Isle {} has a population of {} individuals.".format(self.isle_idx, self.comm.size)) 
        
        dump = True if rank == 0 else False

        if self.generations == 0: # If number of iterations requested == 0.
            if MPI.COMM_WORLD.rank == 0: print("Requested number of generations is zero...[RETURN]")
            return

        # Loop over generations.
        while self.generations == -1 or generation < self.generations:

            if generation % int(logging_interval) ==0: 
                print("Isle {}: Worker {} in generation {}...".format(self.isle_idx, rank, generation)) 
            
            # Evaluate individual.
            ind = self._breed(generation)   # Breed new individual.
            ind.loss = self.loss_fn(ind)    # Evaluate individual's loss.
            self.population.append(ind)     # Append evaluated individual to worker-specific population list.

            # Tell the world about your great results to keep worker-specific populations synchronous within each isle.
            for r in range(self.comm.size):                         # Loop over ranks in intra-isle communicator.
                if r == rank: continue                              # No self-talk.
                self.comm.isend(ind, dest=r, tag=LOSS_REPORT_TAG)   # Use immediate send for asynchronous communication.
            
            # Check for incoming individuals from other workers within isle.
            probe = True # If True, continue checking for incoming messages.
            while probe:
                if DEBUG: print("Isle {}: Worker {} checking for incoming individuals...".format(self.isle_idx, rank))
                stat = MPI.Status()
                probe = self.comm.iprobe(source=MPI.ANY_SOURCE, tag=LOSS_REPORT_TAG, status=stat)
                if DEBUG:
                    if stat.Get_source() >=0:
                        print("Isle {}: Worker {} incoming individual from worker {}...".format(self.isle_idx, rank, stat.Get_source()))
                if probe:
                    req = self.comm.irecv(source=stat.Get_source(), tag=LOSS_REPORT_TAG)
                    ind_temp = req.wait()
                    if DEBUG: print("Isle {}: Worker {} received individual from worker {}.".format(self.isle_idx, rank, stat.Get_source()))
                    self.population.append(ind_temp) # Append received individual to worker-specific population list.
            if DEBUG: print("Isle {}: Worker {} checking for incoming individuals...[DONE]".format(self.isle_idx, rank))

            # Determine current best as individual with minimum loss.
            self.best = min(self.population, key=attrgetter('loss'))
           
            # Emigration: Isle sends individuals out.
            # Emigration happens on per-worker basis with certain probability.
            if random.random() < self.migration_prob: 
                if DEBUG: print("Isle {}: Worker {} has per-rank migration probability of {}.".format(self.isle_idx, rank, self.migration_prob))
                if DEBUG: print("Isle {}: Worker {} wants to send migrants.".format(self.isle_idx, rank))
                # Determine relevant line of migration topology.
                to_migrate = self.migration_topology[self.isle_idx,:] 
                if DEBUG: print("Isle {}: Migration topology {}:".format(self.isle_idx, to_migrate))
                for target_isle, offspring in enumerate(to_migrate):
                    if offspring == 0: continue
                    if DEBUG: print("Isle {}: Worker {} about to send {} individual(s) to isle {}.".format(self.isle_idx, rank, 
                                                                                                           offspring, target_isle))
                    displ = self.unique_ind[target_isle]
                    count = self.unique_counts[target_isle]
                    if DEBUG: print("{} worker(s) on target isle {}, COMM_WORLD displ is {}.".format(count, target_isle, displ))
                    dest = np.arange(displ,displ+count)
                    if DEBUG: print("MPI.COMM_WORLD dest ranks: {}".format(dest))
                    emigrator = self.emigration_propagator(offspring)
                    emigrants = emigrator(self.population)
                    print("{} emigrant(s): {}".format(len(emigrants), emigrants))
                    
                    # For actual migration (no pollination), deactivate emigrants on original isle for breeding.
                    # All workers must be considered!
                    if self.pollination == False: 
                        # Send emigrants to other intra-isle workers so they can deactivate them.
                        for r in range(self.comm.size):
                            if r == rank: continue
                            self.comm.isend(emigrants, dest=r, tag=POLLINATION_TAG)

                        # Deactivate emigrants in population of sending worker.
                        for ind in emigrants:
                            ind.emigrated = True
                            #self.population.remove(ind)
                        

                    for r in dest:
                        if offspring == 0: continue
                        print("Isle {}: Worker {} sending {} individual(s) to {} workers on target isle {}.".format(self.isle_idx, rank, offspring, 
                                                                                                                count, target_isle))
                        MPI.COMM_WORLD.isend(emigrants, dest=r, tag=MIGRATION_TAG)
                
            # Immigration: Isle checks for incoming individuals from other islands.
            probe_migrants = True
            while probe_migrants:
                stat = MPI.Status()
                probe_migrants = MPI.COMM_WORLD.iprobe(source=MPI.ANY_SOURCE, tag=MIGRATION_TAG, status=stat)
                if DEBUG: print("Isle {}: Worker {} checking for immigrants...probe_migrants: {}".format(self.isle_idx, rank, probe_migrants))
                if probe_migrants:
                    req = MPI.COMM_WORLD.irecv(source=stat.Get_source(), tag=MIGRATION_TAG)
                    immigrants = req.wait()
                    num_migr = len(immigrants)
                    print("Isle {}: Worker {} received {} immigrant(s) from global worker {}: {}".format(self.isle_idx, rank, num_migr, 
                                                                                                         stat.Get_source(), immigrants))
                    for ind in immigrants:
                        ind.generation = generation
                    #immigrator = self.immigration_propagator(num_migr)
                    #replace_inds = immigrator(self.population)
                    #print("Isle {} Worker {}: Immigrants replace the following {} individual(s): {}".format(self.isle_idx, rank,
                    #                                                                                        len(replace_inds), replace_inds))
                    #if DEBUG: print("Isle {} Worker {}: Population size before removal is {}.".format(self.isle_idx, rank, len(self.population)))
                    #for ind in replace_inds:
                    #    self.population.remove(ind)
                    #if DEBUG: print("Isle {} Worker {}: Population size after removal is {}.".format(self.isle_idx, rank, len(self.population)))
                    self.population = self.population + immigrants # Append immigrants to population.
                    #if DEBUG: print("Isle {} Worker {}: Population size after immigration is {}.".format(self.isle_idx, rank, len(self.population)))
            # Check for emigrants from other intra-isle workers to be removed from population.
            if self.pollination == False:
                probe_poll = True
                while probe_poll:
                    if DEBUG: print("Isle {}: Worker {} checking for emigrants from other workers to be removed...".format(self.isle_idx, rank))
                    stat = MPI.Status()
                    probe_poll = self.comm.iprobe(source=MPI.ANY_SOURCE, tag=POLLINATION_TAG, status=stat)
                    if probe_poll:
                        req = self.comm.irecv(source=stat.Get_source(), tag=POLLINATION_TAG)
                        emigrants_temp = req.wait()
                        if DEBUG: 
                            print("Isle {}: Worker {} received message from worker {}.".format(self.isle_idx, rank, stat.Get_source()))
                            print("Isle {}: Worker {} population size before removal {}.".format(self.isle_idx, rank, len(self.population)))
                        for ind in emigrants_temp:
                            try: 
                                #self.population.remove(ind)
                                i = self.population.index(ind)
                                print(i)
                                self.population[i].emigrated = True
                            except Exception as e:
                                print(e)
                                continue
                        print("Isle {}: Worker {} removed {} emigrants {} sent by worker {} from population.".format(self.isle_idx, rank, 
                                                                                                                     len(emigrants_temp),
                                                                                                                     emigrants_temp,
                                                                                                                     stat.Get_source()))
                        if DEBUG: print("Isle {}: Worker {} population size after removal {}.".format(self.isle_idx, rank, len(self.population)))

            if dump: # Dump checkpoint.
                if DEBUG: print("Isle {}: Worker {} dumping checkpoint...".format(self.isle_idx, rank))
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
            
            # Go to next generation.
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

        self.comm.barrier()
        # Final check for incoming individuals from other islands.
        probe_migrants = True
        while probe_migrants:
            stat = MPI.Status()
            probe_migrants = MPI.COMM_WORLD.iprobe(source=MPI.ANY_SOURCE, tag=MIGRATION_TAG, status=stat)
            if DEBUG: print("Isle {}: Worker {} checking for immigrants...probe_migrants: {}".format(self.isle_idx, rank, probe_migrants))
            if probe_migrants:
                req = MPI.COMM_WORLD.irecv(source=stat.Get_source(), tag=MIGRATION_TAG)
                immigrants = req.wait()
                num_migr = len(immigrants)
                print("Isle {}: Worker {} received {} immigrant(s) from global worker {}: {}".format(self.isle_idx, rank, num_migr, 
                                                                                                         stat.Get_source(), immigrants))
                for ind in immigrants:
                    ind.generation = generation
                    ind.emigrated = False
                    #immigrator = self.immigration_propagator(num_migr)
                    #replace_inds = immigrator(self.population)
                    #print("Isle {} Worker {}: Immigrants replace the following {} individual(s): {}".format(self.isle_idx, rank,
                    #for ind in replace_inds:
                    #    self.population.remove(ind)
                self.population = self.population + immigrants # Append immigrants to population.

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
        active_population = [ind for ind in self.population if ind.emigrated == False]
        total = self.comm_migrate.reduce(len(active_population), root=0)
        if MPI.COMM_WORLD.rank == 0:
            print("\n###########")
            print("# SUMMARY #")
            print("###########\n")
            print("{} individuals have been evaluated overall.".format(total))
        MPI.COMM_WORLD.barrier()
        if self.comm.rank == 0:
            active_population = [ind for ind in self.population if ind.emigrated == False]
            print("{} individuals have been evaluated on isle {}.".format(len(active_population), self.isle_idx))
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
