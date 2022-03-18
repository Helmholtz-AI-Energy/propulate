import os
import pickle
import random
import numpy as np
import copy
from operator import attrgetter
from mpi4py import MPI

from ._globals import INDIVIDUAL_TAG, INIT_TAG, POPULATION_TAG, DUMP_TAG, MIGRATION_TAG, SYNCHRONIZATION_TAG
from .population import Individual
from .propagators import SelectBest, SelectWorst, SelectUniform

DEBUG = False

class Propulator():
    """
    Parallel propagator of populations with real migration.

    Individuals can only exist on one evolutionary island at a time, i.e., they are removed 
    (i.e. deactivated for breeding) from the sending island upon emigration.
    """
    def __init__(self, loss_fn, propagator, isle_idx, comm=MPI.COMM_WORLD, generations=0,
                 load_checkpoint = "pop_cpt.p", save_checkpoint="pop_cpt.p", 
                 migration_topology=None, comm_inter=MPI.COMM_WORLD,
                 migration_prob=None, emigration_propagator=None,
                 unique_ind=None, unique_counts=None, seed=None):
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
        comm_inter : MPI communicator
                     inter-isle communicator for migration
        """
        # Set class attributes.
        self.loss_fn = loss_fn                      # callable loss function
        self.propagator = propagator                # propagator
        if generations == 0: # If number of iterations requested == 0.
            if MPI.COMM_WORLD.rank == 0: 
                print("Requested number of generations is zero...[RETURN]")
            return
        self.generations = int(generations)         # number of generations, i.e., number of evaluation per individual
        self.isle_idx = int(isle_idx)               # isle index
        self.comm = comm                            # intra-isle communicator for actual evolutionary optimization within isle
        self.comm_inter = comm_inter                # inter-isle communicator for migration between isles
        self.load_checkpoint = str(load_checkpoint) # path to checkpoint file to be read
        self.save_checkpoint = str(save_checkpoint) # path to checkpoint file to be written
        self.migration_prob = float(migration_prob) # per-rank migration probability
        self.migration_topology = migration_topology# migration topology
        self.unique_ind = unique_ind
        self.unique_counts = unique_counts
        self.emigration_propagator = emigration_propagator
        self.emigrated = []

        # Load initial population of evaluated individuals from checkpoint if exists.
        if not os.path.isfile(self.load_checkpoint): # If not exists, check for backup file.
            self.load_checkpoint = self.load_checkpoint+".bkp"

        if os.path.isfile(self.load_checkpoint):
            with open(self.load_checkpoint, "rb") as f:
                try:
                    self.population = pickle.load(f)
                    self.best = min(self.population, key=attrgetter("loss"))
                    if self.comm.rank == 0: 
                        print("NOTE: Valid checkpoint file found. " \
                              "Resuming from loaded population...")
                except Exception:
                    self.population = []
                    self.best = None
                    if self.comm.rank == 0:
                        print("NOTE: No valid checkpoint file found. " \
                              "Initializing population randomly...")
        else:
            self.population=[]
            self.best = None
            if self.comm.rank == 0: 
                print("NOTE: No valid checkpoint file given. " \
                      "Initializing population randomly...")


    def propulate(self, logging_interval=10):
        """
        Run actual evolutionary optimization.""
        """
        self._work(logging_interval)


    def _breed(self, generation):
        """
        Apply propagator to current population of active individuals to breed new individual.

        Parameters
        ----------
        generation : int
                     generation of newly bred individual

        Returns
        -------
        ind : propulate.population.Individual
              newly bred individual
        """
        active_pop = [ind for ind in self.population if ind.active == True]
        if DEBUG: 
            print(f"I{self.isle_idx}: {len(active_pop)}/{len(self.population)} " \
                   "individuals active for breeding.")
        ind = self.propagator(active_pop)   # Breed new individual from current active population only.
        ind.generation = generation         # Set generation.
        ind.rank = self.comm.rank           # Set worker rank.
        ind.active = True                   # If True, individual is active for breeding.
        ind.isle = self.isle_idx            # Set birth island.
        ind.current = self.comm.rank        # Set worker responsible for migration.
        ind.migration_steps = 0             # Set number of migration steps performed
        return ind                          # Return new individual.
            

    def _receive_intra_isle_individuals(self, generation, DEBUG=False):
        """
        Check for and possibly receive incoming individuals from other intra-isle workers.
        """
        probe_ind = True 
        while probe_ind:
            stat = MPI.Status() # Retrieve status of reception operation, containing source, tag, and error.
            probe_ind = self.comm.iprobe(source=MPI.ANY_SOURCE, tag=INDIVIDUAL_TAG, status=stat)
            # If True, continue checking for incoming messages. 
            # Tells whether message corresponding to filters passed is waiting for reception or not, via a flag that it sets. 
            # If no such message has arrived yet, it does not wait but sets the flag to false and returns.
            if DEBUG: 
                print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                      f"Checking for incoming individuals...{probe_ind}")
            if probe_ind:
                # MPI Receive with immediate return; does not block until message is received.
                # To know if message has been received, use MPI wait or MPI test on MPI request filled.
                req_ind = self.comm.irecv(source=stat.Get_source(), tag=INDIVIDUAL_TAG)
                ind_temp = req_ind.wait()           # Wait for non-blocking operation to complete.
                if DEBUG: 
                    print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                          f"Received individual from W{stat.Get_source()}.")
                self.population.append(ind_temp)    # Append received individual to worker-specific population list.
        active_pop = [ind for ind in self.population if ind.active == True]
        print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
              f"After probing within isle: {len(active_pop)}/{len(self.population)} active: ")# \
              #f"\n{self.population}")


    def _send_emigrants(self, generation, DEBUG=False):
        """
        Perform migration, i.e. isle sends individuals out to other islands.
        """
        # Determine relevant line of migration topology.
        to_migrate = self.migration_topology[self.isle_idx,:] 
        num_emigrants = np.sum(to_migrate)
        eligible_emigrants = [ind for ind in self.population if ind.active == True and ind.current == self.comm.rank]
                
        if num_emigrants <= len(eligible_emigrants):
            # Loop through relevant part of migration topology.
            for target_isle, offspring in enumerate(to_migrate):
                if offspring == 0: continue
                eligible_emigrants = [ind for ind in self.population if ind.active == True and ind.current == self.comm.rank]
                displ = self.unique_ind[target_isle]
                count = self.unique_counts[target_isle]
                dest_isle = np.arange(displ, displ+count)
                
                # Worker in principle sends *different* individuals to each target isle.
                emigrator = self.emigration_propagator(offspring) 
                emigrants = emigrator(eligible_emigrants) # Choose `offspring` emigrants from eligible emigrants.
                print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                      f"Chose {len(emigrants)} emigrant(s): {emigrants}")
                    
                # Deactivate emigrants in population of sending isle, i.e., for all workers on original isle, for breeding.        
                for r in range(self.comm.size): # Send emigrants to other intra-isle workers so they can deactivate them.
                    if r == self.comm.rank: continue      # No self-talk.
                    self.comm.isend(copy.deepcopy(emigrants), dest=r, tag=SYNCHRONIZATION_TAG)
                    print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                          f"Sent {len(emigrants)} individual(s) to intra-isle W{r} for deactivation.")

                # Send emigrants to all workers of target island.
                departing = copy.deepcopy(emigrants)
                for ind in departing:
                    print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                          f"W{ind.current} responsible on sending I{self.isle_idx}.")
                    ind.current = random.randrange(0, count)
                    print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                          f"W{ind.current} responsible on target I{target_isle}.")
                for r in dest_isle:
                    if offspring == 0: continue
                    MPI.COMM_WORLD.isend(departing, dest=r, tag=MIGRATION_TAG)
                    print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                          f"Sent {len(departing)} individual(s) to W{r-self.unique_ind[target_isle]} " \
                          f"on target I{target_isle}.")

                # Deactive emigrants for sending worker.
                for emigrant in emigrants: 
                    to_deactivate = [ind for ind in self.population if ind == emigrant and ind.migration_steps == emigrant.migration_steps]
                    #to_deactivate = [ind for ind in self.population if ind == emigrant and ind.migration_steps == emigrant.migration_steps]
                    assert len(to_deactivate) == 1
                    to_deactivate[0].active = False
                    print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                          f"Deactivated own emigrant {to_deactivate[0]}.")

            active_pop = [ind for ind in self.population if ind.active == True]
            print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                  f"After emigration: {len(active_pop)}/{len(self.population)} active:" )#\
                  #f"\n{self.population}")
                        
        else: 
            print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                  f"Population size {len(eligible_emigrants)} too small " \
                  f"to select {num_emigrants} migrants.")
       

    def _receive_immigrants(self, generation, DEBUG=False):
        """
        Check for and possibly receive immigrants send by other islands.
        """
        probe_migrants = True
        while probe_migrants:
            stat = MPI.Status()
            probe_migrants = MPI.COMM_WORLD.iprobe(source=MPI.ANY_SOURCE, tag=MIGRATION_TAG, status=stat)
            if DEBUG: 
                print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                      f"Checking for immigrants...: {probe_migrants}")
            if probe_migrants:
                req_immigr = MPI.COMM_WORLD.irecv(source=stat.Get_source(), tag=MIGRATION_TAG)
                immigrants = req_immigr.wait()
                print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                      f"Received {len(immigrants)} immigrant(s) from global " \
                      f"W{stat.Get_source()}: {immigrants}")
                for immigrant in immigrants: 
                    immigrant.migration_steps += 1
                    assert immigrant.active == True
                    immigrant_obsolete = len([ind for ind in self.population if ind == immigrant and immigrant.migration_steps < ind.migration_steps]) > 0
                    #existing_obsolete = [ind for ind in self.population if ind == immigrant and immigrant.migration_steps > ind.migration_steps]
                    catastrophic_failure = len([ind for ind in self.population if ind == immigrant and immigrant.migration_steps == ind.migration_steps]) > 0
                    if catastrophic_failure:
                        raise RuntimeError(f"Identical immigrant {immigrant} already " \
                                           f"active on target island I{self.isle_idx}.")
                    if immigrant_obsolete:
                        immigrant.active = False
                    # NOTE Do not remove obsolete individuals from population upon immigration
                    # as they should be deactivated in the next step nevertheless.
                    #for ind in existing_obsolete:
                    #    ind.active = False
                        #print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                        #      f"{immigrant} already in population...: {e}")
                        #continue
                    self.population.append(immigrant) # Append immigrant to population.

        active_pop = [ind for ind in self.population if ind.active == True]
        print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: After immigration: " \
              f"{len(active_pop)}/{len(self.population)} active:")#\n{self.population}")

    def _deactivate_emigrants(self, generation, DEBUG=False):
        """
        Check for and possibly receive emigrants from other intra-isle workers to be deactivated.
        """
        probe_sync = True
        while probe_sync:
            if DEBUG: 
                print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                      f"Checking for emigrants from others to be deactivated...")
            stat = MPI.Status()
            probe_sync = self.comm.iprobe(source=MPI.ANY_SOURCE, tag=SYNCHRONIZATION_TAG, status=stat)
            if probe_sync:
                req_sync = self.comm.irecv(source=stat.Get_source(), tag=SYNCHRONIZATION_TAG)
                new_emigrants = req_sync.wait()
                self.emigrated = self.emigrated + new_emigrants
                if DEBUG: 
                    print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: Emigrated {self.emigrated}")
                print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                      f"Got {len(new_emigrants)} new emigrant(s) {new_emigrants} " \
                      f"from W{stat.Get_source()} to be deactivated.")
                emigrated_copy = copy.deepcopy(self.emigrated)
                for emigrant in emigrated_copy:
                    assert emigrant.active == True
                    #to_deactivate = [idx for idx, ind in enumerate(self.population) if ind == emigrant and ind.migration_steps <= emigrant.migration_steps]
                    to_deactivate = [ind for ind in self.population if ind == emigrant and ind.migration_steps == emigrant.migration_steps]
                    if len(to_deactivate) == 0: 
                        print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                              f"Individual {emigrant} to deactivate not yet received " \
                               "or already deactivated before.")
                        continue
                    #assert len(to_deactivate) == 1
                    for ind in to_deactivate:
                        ind.active = False
                        print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                              f"Deactivated {ind}.")
                    #self.population[to_deactivate[0]].active = False
                    self.emigrated.remove(emigrant)

        active_pop = [ind for ind in self.population if ind.active == True]
        print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: After synchronization: "\
              f"{len(active_pop)}/{len(self.population)} active:")#\n{self.population}")

        if DEBUG: 
            print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                  f"{len(self.emigrated)} individuals in emigrated.")


    def _work(self, logging_interval=10, DEBUG=False):
        """
        Execute evolutionary algorithm in parallel.
        """
        generation = 0        # Start from generation 0.

        if self.comm.rank == 0: 
            print(f"I{self.isle_idx} has {self.comm.size} workers.") 
        
        dump = True if self.comm.rank == 0 else False
        migration = True if self.migration_prob > 0 else False
        MPI.COMM_WORLD.barrier()

        # Loop over generations.
        while self.generations == -1 or generation < self.generations: 

            if DEBUG and generation % int(logging_interval) == 0: 
                print(f"I{self.isle_idx} W{self.comm.rank}: In generation {generation}...") 
            
            # Evaluate individual.
            ind = self._breed(generation)   # Breed new individual.
            ind.loss = self.loss_fn(ind)    # Evaluate individual's loss.
            self.population.append(ind)     # Append evaluated individual to worker-specific population list.

            # Tell other intra-isle workers about results to keep worker-specific populations synchronous.
            for r in range(self.comm.size):                                     # Loop over ranks in intra-isle communicator.
                if r == self.comm.rank: continue                                          # No self-talk.
                self.comm.isend(copy.deepcopy(ind), dest=r, tag=INDIVIDUAL_TAG) # Immediate send for asynchronous communication.
            
            # Check for and possibly receive incoming individuals from other intra-isle workers.
            self._receive_intra_isle_individuals(generation, DEBUG)
            
            if migration:
                # Emigration: Isle sends individuals out.
                # Happens on per-worker basis with certain probability.
                if random.random() < self.migration_prob: 
                    self._send_emigrants(generation, DEBUG)

                # Immigration: Isle checks for incoming individuals from other islands.
                self._receive_immigrants(generation, DEBUG)

                # Emigration: Check for emigrants from other intra-isle workers to be deactivated.
                self._deactivate_emigrants(generation, DEBUG)
            
            if dump: # Dump checkpoint.
                if DEBUG: 
                    print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: Dumping checkpoint...")
                if os.path.isfile(self.save_checkpoint):
                    try: os.replace(self.save_checkpoint, self.save_checkpoint+".bkp")
                    except Exception as e: print(e)
                with open(self.save_checkpoint, 'wb') as f:
                    pickle.dump((self.population), f)
                
                dest = self.comm.rank+1 if self.comm.rank+1 < self.comm.size else 0
                self.comm.isend(dump, dest=dest, tag=DUMP_TAG)
                dump = False
            
            stat = MPI.Status()
            probe_dump = self.comm.iprobe(source=MPI.ANY_SOURCE, tag=DUMP_TAG, status=stat)
            if probe_dump:
                req_dump = self.comm.irecv(source=stat.Get_source(), tag=DUMP_TAG)
                dump = req_dump.wait()
                if DEBUG: 
                    print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: "\
                          f"Going to dump next: {dump}. Before: W{stat.Get_source()}")
            
            # Go to next generation.
            generation += 1
        
        # Having completed all generations, the workers have to wait for each other.
        # Once all workers are done, they should check for incoming messages once again
        # so that each of them holds the complete final population and the found optimum
        # irrespective of the order they finished.

        MPI.COMM_WORLD.barrier()
        if MPI.COMM_WORLD.rank == 0: 
            print("OPTIMIZATION DONE.\nNEXT: Final checks for incoming messages...")
        MPI.COMM_WORLD.barrier()

        # Final check for incoming individuals evaluated by other intra-isle workers.
        self._receive_intra_isle_individuals(generation, DEBUG)
        active_pop = [ind for ind in self.population if ind.active == True]
        print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: After final probe_ind: " \
              f"{len(active_pop)}/{len(self.population)} active:")#\n{self.population}")
   
        MPI.COMM_WORLD.barrier()

        if migration:
            # Final check for incoming individuals from other islands.
            self._receive_immigrants(generation, DEBUG)
            active_pop = [ind for ind in self.population if ind.active == True]
            print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: After immigration: " \
                  f"{len(active_pop)}/{len(self.population)} active:")#\n{self.population}")
            MPI.COMM_WORLD.barrier()

            # Emigration: Final check for emigrants from other intra-isle workers to be deactivated.
            self._deactivate_emigrants(generation, DEBUG)
            #active_pop = [ind for ind in self.population if ind.active == True]
            MPI.COMM_WORLD.barrier()

            print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                    f"Currently {len(self.emigrated)} individuals in emigrated: {self.emigrated}")
            self._deactivate_emigrants(generation, DEBUG)
            print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                    f"Finally {len(self.emigrated)} individuals in emigrated: {self.emigrated}.")
            #emigrated_copy = copy.deepcopy(self.emigrated)
            #for emigrant in emigrated_copy:
            #    to_deactivate = [idx for idx, ind in enumerate(self.population) if ind == emigrant]
            #    if len(to_deactivate) == 0: 
            #        print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
            #              f"Individual {emigrant} to deactivate not yet received or already deactivated before.")
            #        continue
            #    assert len(to_deactivate) == 1
            #    self.population[to_deactivate[0]].active = False
            #    print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
            #          f"Deactivated individual {self.population[to_deactivate[0]]}.")
            #    self.emigrated.remove(emigrant)
            MPI.COMM_WORLD.barrier()
        
        # Final checkpointing on rank 0.
        if self.comm.rank == 0: # Dump checkpoint.
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
        active_pop = [ind for ind in self.population if ind.active == True]
        total = self.comm_inter.reduce(len(active_pop), root=0)
        MPI.COMM_WORLD.barrier()
        if MPI.COMM_WORLD.rank == 0:
            print("\n###########")
            print("# SUMMARY #")
            print("###########\n")
            print(f"Number of currently active individuals is {total}. "\
                  f"\nExpected overall number of evaluations is {self.generations*MPI.COMM_WORLD.size}.")
        MPI.COMM_WORLD.barrier()
        print(f"I{self.isle_idx} W{self.comm.rank}: " \
              f"{len(active_pop)}/{len(self.population)} individuals active:")#\n{self.population}")
        MPI.COMM_WORLD.barrier()
        if self.comm.rank == 0:
            #print("{} individuals have been evaluated on isle {}.".format(len(active_pop), self.isle_idx))
            active_pop.sort(key=lambda x: x.loss)
            self.population.sort(key=lambda x: x.loss)
            res_str=f"Top {top_n} result(s) on isle {self.isle_idx}:\n"
            for i in range(top_n):
                res_str += f"({i+1}): {self.population[i]}\n"
            print(res_str)
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

########################################################################################################################################

class PolliPropulator():
    """
    Parallel propagator of populations with pollination.
    """
    def __init__(self, loss_fn, propagator, isle_idx, comm=MPI.COMM_WORLD, generations=0,
                 load_checkpoint = "pop_cpt.p", save_checkpoint="pop_cpt.p", 
                 seed=9, migration_topology=None, comm_inter=MPI.COMM_WORLD,
                 migration_prob=None, emigration_propagator=None,
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
        comm_inter : MPI communicator
                     inter-isle communicator for migration
        """
        # Set class attributes.
        self.loss_fn = loss_fn                      # callable loss function
        self.propagator = propagator                # propagator
        self.generations = int(generations)         # number of generations, i.e., number of evaluation per individual
        self.isle_idx = int(isle_idx)               # isle index
        self.comm = comm                            # intra-isle communicator for actual evolutionary optimization within isle
        self.comm_inter = comm_inter                # inter-isle communicator for migration between isles
        self.load_checkpoint = str(load_checkpoint) # path to checkpoint file to be read
        self.save_checkpoint = str(save_checkpoint) # path to checkpoint file to be written
        self.migration_prob = float(migration_prob) # per-rank migration probability
        self.migration_topology = migration_topology# migration topology
        self.unique_ind = unique_ind
        self.unique_counts = unique_counts
        self.emigration_propagator = emigration_propagator

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
        # Do not use emigrated individuals for breeding.
        active_pop = [ind for ind in self.population if ind.active == True]
        if DEBUG: 
            print("Isle {}: {}/{} individuals active for breeding.".format(isle_idx, len(active_pop), len(self.population)))
        ind = self.propagator(active_pop)   # Breed new individual from current active population.
        ind.generation = generation         # Set generation as individual's attribute.
        ind.rank = self.comm.rank           # Set worker rank as individual's attribute.
        ind.active = True                   # Set active flag. If True, individual is active for breeding.
        ind.isle = self.isle_idx            # Set birth island.
        ind.current = self.comm.rank        # Set worker responsible for migration.
        return ind                          # Return newly bred individual.


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

            # Tell the other workers in your isle about your great results to keep the worker-specific populations synchronous.
            for r in range(self.comm.size):                         # Loop over ranks in intra-isle communicator.
                if r == rank: continue                              # No self-talk.
                self.comm.isend(copy.deepcopy(ind), dest=r, tag=INDIVIDUAL_TAG)   # Use immediate send for asynchronous communication.
            
            # Check for incoming individuals from other intra-isle workers.
            probe = True # If True, continue checking for incoming messages.
            while probe:
                if DEBUG: print("Isle {}: Worker {} checking for incoming individuals...".format(self.isle_idx, rank))
                stat = MPI.Status() # Retrieve status of reception operation, containing source, tag, and error.
                probe = self.comm.iprobe(source=MPI.ANY_SOURCE, tag=INDIVIDUAL_TAG, status=stat)
                # Tells whether message corresponding to filters passed is waiting for reception or not, via a flag that it sets. 
                # If no such message has arrived yet, it does not wait but sets the flag to false and returns.
                if probe:
                    # MPI Receive with immediate return; it does not block until the message is received.
                    # To know if message has been received, use MPI wait or MPI test on MPI request filled.
                    req = self.comm.irecv(source=stat.Get_source(), tag=INDIVIDUAL_TAG)
                    ind_temp = req.wait() # Wait for non-blocking operation to complete.
                    if DEBUG: print("Isle {}: Worker {} received individual from worker {}.".format(self.isle_idx, rank, stat.Get_source()))
                    self.population.append(ind_temp) # Append received individual to worker-specific population list.
            if DEBUG: print("Isle {}: Worker {} checking for incoming individuals...[DONE]".format(self.isle_idx, rank))

            # Emigration: Isle sends individuals out.
            # Emigration happens on per-worker basis with certain probability.

            if random.random() < self.migration_prob: 
                if DEBUG: print("Isle {}: Worker {} has per-rank migration probability of {}.".format(self.isle_idx, rank, self.migration_prob))
                if DEBUG: print("Isle {}: Worker {} wants to send migrants.".format(self.isle_idx, rank))
                # Determine relevant line of migration topology.
                to_migrate = self.migration_topology[self.isle_idx,:] 
                if DEBUG: print("Isle {}: Migration topology {}:".format(self.isle_idx, to_migrate))

                # Loop through relevant part of migration topology.
                for target_isle, offspring in enumerate(to_migrate):
                    if offspring == 0: continue
                    if DEBUG: print("Isle {}: Worker {} about to send {} individual(s) to isle {}.".format(self.isle_idx, rank, 
                                                                                                           offspring, target_isle))
                    displ = self.unique_ind[target_isle]
                    count = self.unique_counts[target_isle]
                    if DEBUG: print("{} worker(s) on target isle {}, COMM_WORLD displ is {}.".format(count, target_isle, displ))
                    dest = np.arange(displ, displ+count)
                    if DEBUG: print("MPI.COMM_WORLD dest ranks: {}".format(dest))
                    emigrator = self.emigration_propagator(offspring)
                    active_population = [ind for ind in self.population if ind.active == True]
                    emigrants = emigrator(active_population)
                    print("Isle {}: Worker {} chose {} emigrant(s): {}".format(self.isle_idx, rank, len(emigrants), emigrants))
                    
                    # For actual migration (no pollination), deactivate emigrants on original isle for breeding.
                    # All workers must be considered!
                    # Send emigrants to other intra-isle workers so they can deactivate them.
                    for r in range(self.comm.size):
                        if r == rank: continue
                        self.comm.isend(copy.deepcopy(emigrants), dest=r, tag=SYNCHRONIZATION_TAG)

                    # Deactivate emigrants in population of sending worker.
                    # Worker sends *different* individuals to each isle.
                    for ind in emigrants:
                        ind.active = False
                        #self.population.remove(ind)    

                    for r in dest:
                        if offspring == 0: continue
                        MPI.COMM_WORLD.isend(copy.deepcopy(emigrants), dest=r, tag=MIGRATION_TAG)
                        print("Isle {}: Worker {} sent {} individual(s) to global worker {} on target isle {}.".format(self.isle_idx, rank, 
                                                                                                                       offspring, r, target_isle))
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
                        #assert ind.active == True #ind.generation = generation
                        ind.active = True
                    self.population = self.population + immigrants # Append immigrants to population.
                    #immigrator = self.immigration_propagator(num_migr)
                    #active_population = [ind for ind in self.population if ind.active == True]
                    #replace_inds = immigrator(active_population)
                    #print("Isle {} Worker {}: Immigrants replace {} individual(s): {}".format(self.isle_idx, rank,
                    #                                                                          len(replace_inds), replace_inds))
                    #for ind in replace_inds: ind.active = False # Replace worst individuals with immigrants.

            # Emigration: Check for emigrants from other intra-isle workers to be deactivated.
            probe_sync = True
            while probe_sync:
                if DEBUG: print("Isle {}: Worker {} checking for emigrants from other workers to be deactivated...".format(self.isle_idx, rank))
                stat = MPI.Status()
                probe_sync = self.comm.iprobe(source=MPI.ANY_SOURCE, tag=SYNCHRONIZATION_TAG, status=stat)
                if probe_sync:
                    req = self.comm.irecv(source=stat.Get_source(), tag=SYNCHRONIZATION_TAG)
                    emigrants_temp = req.wait()
                    print(f"Isle {self.isle_idx}: Worker {rank} received emigrant(s) from worker {stat.Get_source()} to be deactivated.")
                    for ind in emigrants_temp: 
                        try: 
                            i = self.population.index(ind)  
                            self.population[self.population.index(ind)].active = False 

                        except ValueError:
                            continue
                    print("Isle {}: Worker {} deactivated {} emigrant(s) {} sent by worker {} from population.".format(self.isle_idx, rank, 
                                                                                                                         len(emigrants_temp),
                                                                                                                         emigrants_temp,
                                                                                                                         stat.Get_source()))
            
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
            probe_dump = self.comm.iprobe(source=MPI.ANY_SOURCE, tag=DUMP_TAG, status=stat)
            if probe_dump:
                req_dump = self.comm.irecv(source=stat.Get_source(), tag=DUMP_TAG)
                dump = req_dump.wait()
                if DEBUG: print("Isle", self.isle_idx, "Worker", rank, "is going to dump next:", dump, ". Before: worker", stat.Get_source())
            
            # Go to next generation.
            generation += 1
        
        # Having completed all generations, the workers have to wait for each other.
        # Once all workers are done, they should check for incoming messages once again
        # so that each of them holds the complete final population and the found optimum
        # irrespective of the order they finished.

        MPI.COMM_WORLD.barrier()
        if MPI.COMM_WORLD.rank == 0: 
            print("OPTIMIZATION DONE.")
            print("NEXT: Final checks for incoming messages...")
        MPI.COMM_WORLD.barrier()

        # Final check for incoming individuals from intra-isle workers.
        probe = True
        while probe:
            stat = MPI.Status()
            probe = self.comm.iprobe(source=MPI.ANY_SOURCE, tag=INDIVIDUAL_TAG, status=stat)
            if probe:
                req = self.comm.irecv(source=stat.Get_source(), tag=INDIVIDUAL_TAG)
                ind_temp = req.wait()
                self.population.append(ind_temp)
   
        MPI.COMM_WORLD.barrier()

        # Final check for incoming individuals from other islands.
        probe_migrants = True
        while probe_migrants:
            stat = MPI.Status()
            probe_migrants = MPI.COMM_WORLD.iprobe(source=MPI.ANY_SOURCE, tag=MIGRATION_TAG, status=stat)
            #if DEBUG: 
            print(f"Isle {self.isle_idx}: Worker {rank} final check for immigrants: {probe_migrants}")
            if probe_migrants:
                req = MPI.COMM_WORLD.irecv(source=stat.Get_source(), tag=MIGRATION_TAG)
                immigrants = req.wait()
                num_migr = len(immigrants)
                print("Isle {}: Worker {} received {} immigrant(s) from global worker {}: {}".format(self.isle_idx, rank, num_migr, 
                                                                                                         stat.Get_source(), immigrants))
                for ind in immigrants: assert ind.active == True
                self.population = self.population + immigrants # Append immigrants to population.
        
        MPI.COMM_WORLD.barrier()

        # Emigration: Check for emigrants from other intra-isle workers to be deactivated.
        probe_sync = True
        while probe_sync:
            if DEBUG: print("Isle {}: Worker {} checking for emigrants from other workers to be deactivated...".format(self.isle_idx, rank))
            stat = MPI.Status()
            probe_sync = self.comm.iprobe(source=MPI.ANY_SOURCE, tag=SYNCHRONIZATION_TAG, status=stat)
            if probe_sync:
                req = self.comm.irecv(source=stat.Get_source(), tag=SYNCHRONIZATION_TAG)
                emigrants_temp = req.wait()
                print(f"Isle {self.isle_idx}: Worker {rank} received emigrant(s) from worker {stat.Get_source()} to be deactivated.")
                for ind in emigrants_temp: 
                    try: 
                        i = self.population.index(ind)  
                        self.population[self.population.index(ind)].active = False 

                    except Exception as e:
                        print(e)
                        continue
                print(f"Isle {self.isle_idx}: Worker {rank} deactivated {len(emigrants_temp)} emigrant(s) {emigrants_temp} sent by worker {stat.Get_source()} from population.")

        MPI.COMM_WORLD.barrier()
        
        # Final checkpointing on rank 0.
        if rank == 0: # Dump checkpoint.
            if os.path.isfile(self.save_checkpoint):
                os.replace(self.save_checkpoint, self.save_checkpoint+".bkp")
                with open(self.save_checkpoint, 'wb') as f:
                    pickle.dump((self.population), f)

        self.comm.barrier()


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
        active_pop = [ind for ind in self.population if ind.active == True]
        total = self.comm_inter.reduce(len(active_pop), root=0)
        if MPI.COMM_WORLD.rank == 0:
            print("\n###########")
            print("# SUMMARY #")
            print("###########\n")
            #print("{} individuals are currently active.".format(total))
        print(f"Isle {self.isle_idx} Worker {self.comm.rank}: {len(active_pop)}/{len(self.population)} individuals active.")
        MPI.COMM_WORLD.barrier()
        if self.comm.rank == 0:
            #print("{} individuals have been evaluated on isle {}.".format(len(active_pop), self.isle_idx))
            active_pop.sort(key=lambda x: x.loss)
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
