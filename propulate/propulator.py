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
        isle_idx : int
                   index of isle
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
        migration_topology : numpy array
                             2D matrix where entry (i,j) specifies how many 
                             individuals are sent by isle i to isle j
        comm_inter : MPI communicator
                     inter-isle communicator for migration
        migration_prob : float
                         per-worker migration probability
        emigration_propagator : propulate.propagators.Propagator
                                emigration propagator, i.e., how to choose individuals 
                                for emigration that are sent to destination island.
                                Should be some kind of selection operator.
        unique_ind : numpy array
                     array with MPI.COMM_WORLD rank of each isle's worker 0
                     Element i specifies MPI.COMM_WORLD rank of worker 0 on isle with index i.
        unique_counts : numpy array
                        array with number of workers per isle
                        Element i specifies number of workers on isle with index i.
        seed : int
               base seed for random number generator
        """
        # Set class attributes.
        self.loss_fn = loss_fn                              # callable loss function
        self.propagator = propagator                        # evolutionary propagator
        if generations == 0:                                # If number of iterations requested == 0.
            if MPI.COMM_WORLD.rank == 0: 
                print("Requested number of generations is zero...[RETURN]")
            return
        self.generations = int(generations)                 # number of generations, i.e., number of evaluation per individual
        self.isle_idx = int(isle_idx)                       # isle index
        self.comm = comm                                    # intra-isle communicator
        self.comm_inter = comm_inter                        # inter-isle communicator
        self.load_checkpoint = str(load_checkpoint)         # path to checkpoint file to be read
        self.save_checkpoint = str(save_checkpoint)         # path to checkpoint file to be written
        self.migration_prob = float(migration_prob)         # per-rank migration probability
        self.migration_topology = migration_topology        # migration topology
        self.unique_ind = unique_ind                        # MPI.COMM_WORLD rank of each isle's worker 0
        self.unique_counts = unique_counts                  # number of workers on each isle
        self.emigration_propagator = emigration_propagator  # emigration propagator
        self.emigrated = []                                 # emigrated individuals to be deactivated on sending isle

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

        Parameters
        ----------

        logging_interval : int
                           Print each worker's progress every
                           `logging_interval`th generation.
        """
        self._work(logging_interval)


    def _breed(self, generation):
        """
        Apply propagator to current population of active 
        individuals to breed new individual.

        Parameters
        ----------
        generation : int
                     generation of new individual

        Returns
        -------
        ind : propulate.population.Individual
              newly bred individual
        """
        active_pop = [ind for ind in self.population if ind.active]
        ind = self.propagator(active_pop)   # Breed new individual from active population.
        ind.generation = generation         # Set generation.
        ind.rank = self.comm.rank           # Set worker rank.
        ind.active = True                   # If True, individual is active for breeding.
        ind.isle = self.isle_idx            # Set birth island.
        ind.current = self.comm.rank        # Set worker responsible for migration.
        ind.migration_steps = 0             # Set number of migration steps performed.
        return ind                          # Return new individual.
            

    def _evaluate_individual(self, generation):
        """
        Breed and evaluate individual.

        Parameters
        ----------
        generation : int
                     generation of new individual
        """
        ind = self._breed(generation)   # Breed new individual.
        ind.loss = self.loss_fn(ind)    # Evaluate its loss.
        self.population.append(ind)     # Add evaluated individual to own worker-local population.

        # Tell other workers in own isle about results to synchronize their populations.
        # Use immediate send for asynchronous communication.
        for r in range(self.comm.size): # Loop over ranks in intra-isle communicator.
            if r == self.comm.rank: 
                continue                # No self-talk.
            self.comm.isend(ind, dest=r, tag=INDIVIDUAL_TAG) 


    def _receive_intra_isle_individuals(self, generation, DEBUG=False):
        """
        Check for and possibly receive incoming individuals 
        evaluated by other workers within own isle.

        Parameters
        ----------
        generation : int
                     generation of new individual
        DEBUG : bool
                flag for additional debug prints
        """
        probe_ind = True 
        while probe_ind:
            stat = MPI.Status() # Retrieve status of reception operation, including source, tag, and error.
            probe_ind = self.comm.iprobe(source=MPI.ANY_SOURCE, tag=INDIVIDUAL_TAG, status=stat)
            # If True, continue checking for incoming messages. 
            # Tells whether message corresponding to filters passed is waiting for reception via a flag that it sets. 
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
                self.population.append(ind_temp)    # Add received individual to own worker-local population.
        active_pop = [ind for ind in self.population if ind.active]
        print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
              f"After probing within isle: {len(active_pop)}/{len(self.population)} active.")


    def _send_emigrants(self, generation, DEBUG=False):
        """
        Perform migration, i.e. isle sends individuals out to other islands.

        Parameters
        ----------
        generation : int
                     generation of new individual
        DEBUG : bool
                flag for additional debug prints
        """
        # Determine relevant line of migration topology.
        to_migrate = self.migration_topology[self.isle_idx,:] 
        num_emigrants = np.sum(to_migrate) # Determine overall number of emigrants to be sent out.
        eligible_emigrants = [ind for ind in self.population if ind.active \
                              and ind.current == self.comm.rank]
                
        # Only perform migration if overall number of emigrants to be sent
        # out is smaller than current number of eligible emigrants.
        if num_emigrants <= len(eligible_emigrants): 
            # Loop through relevant part of migration topology.
            for target_isle, offspring in enumerate(to_migrate):
                if offspring == 0: continue
                eligible_emigrants = [ind for ind in self.population if ind.active and ind.current == self.comm.rank]
                # Determine MPI.COMM_WORLD ranks of workers on target isle.
                displ = self.unique_ind[target_isle]
                count = self.unique_counts[target_isle]
                dest_isle = np.arange(displ, displ+count)
                
                # Worker sends *different* individuals to each target isle.
                emigrator = self.emigration_propagator(offspring) # Set up emigration propagator.
                emigrants = emigrator(eligible_emigrants)         # Choose `offspring` eligible emigrants.
                print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                      f"Chose {len(emigrants)} emigrant(s): {emigrants}")
                    
                # Deactivate emigrants on sending isle for breeding (true migration).
                for r in range(self.comm.size): # Send emigrants to other intra-isle workers for deactivation.
                    if r == self.comm.rank: 
                        continue # No self-talk.
                    self.comm.isend(emigrants, dest=r, tag=SYNCHRONIZATION_TAG)
                    print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                          f"Sent {len(emigrants)} individual(s) to intra-isle W{r} to deactivate.")

                # Send emigrants to target island.
                departing = copy.deepcopy(emigrants)
                # Determine new responsible worker on target isle.
                for ind in departing:
                    ind.current = random.randrange(0, count)
                for r in dest_isle: # Loop through MPI.COMM_WORLD destination ranks.
                    MPI.COMM_WORLD.isend(departing, dest=r, tag=MIGRATION_TAG)
                    print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                          f"Sent {len(departing)} individual(s) to W{r-self.unique_ind[target_isle]} " \
                          f"on target I{target_isle}.")

                # Deactivate emigrants for sending worker.
                for emigrant in emigrants: 
                    to_deactivate = [ind for ind in self.population if ind == emigrant and ind.migration_steps == emigrant.migration_steps]
                    assert len(to_deactivate) == 1
                    to_deactivate[0].active = False
                    print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                          f"Deactivated own emigrant {to_deactivate[0]}.")

            active_pop = [ind for ind in self.population if ind.active]
            print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                  f"After emigration: {len(active_pop)}/{len(self.population)} active.")
                        
        else: 
            print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                  f"Population size {len(eligible_emigrants)} too small " \
                  f"to select {num_emigrants} migrants.")
       

    def _receive_immigrants(self, generation, DEBUG=False):
        """
        Check for and possibly receive immigrants send by other islands.

        Parameters
        ----------
        generation : int
                     generation of new individual
        DEBUG : bool
                flag for additional debug prints
        """
        probe_migrants = True
        while probe_migrants:
            stat = MPI.Status()
            probe_migrants = MPI.COMM_WORLD.iprobe(source=MPI.ANY_SOURCE, tag=MIGRATION_TAG, status=stat)
            if DEBUG: 
                print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                      f"Checking for immigrants...{probe_migrants}")
            if probe_migrants:
                req_immigr = MPI.COMM_WORLD.irecv(source=stat.Get_source(), tag=MIGRATION_TAG)
                immigrants = req_immigr.wait()
                print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                      f"Received {len(immigrants)} immigrant(s) from global " \
                      f"W{stat.Get_source()}: {immigrants}")
                for immigrant in immigrants: 
                    immigrant.migration_steps += 1
                    assert immigrant.active == True
                    catastrophic_failure = len([ind for ind in self.population if ind == immigrant and \
                                               immigrant.migration_steps == ind.migration_steps]) > 0
                    if catastrophic_failure:
                        raise RuntimeError(f"Identical immigrant {immigrant} already " \
                                           f"active on target I{self.isle_idx}.")
                    immigrant_obsolete = len([ind for ind in self.population if ind == immigrant and \
                                             immigrant.migration_steps < ind.migration_steps]) > 0
                    if immigrant_obsolete:
                        immigrant.active = False
                    self.population.append(immigrant) # Append immigrant to population.

                    # NOTE Do not remove obsolete individuals from population upon immigration
                    # as they should be deactivated in the next step anyways.

        active_pop = [ind for ind in self.population if ind.active]
        print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: After immigration: " \
              f"{len(active_pop)}/{len(self.population)} active.")


    def _check_emigrants_to_deactivate(self, generation):
        """
        Redundant safety check for existence of emigrants that could not be deactived in population.

        Parameters
        ----------
        generation : int
                     generation of new individual
        """
        check = False
        # Loop over emigrants still to be deactivated.
        for idx, emigrant in enumerate(self.emigrated):
            existing_ind = [ind for ind in self.population if ind == emigrant and ind.migration_steps == emigrant.migration_steps]
            if len(existing_ind) > 0: 
                check = True
                break
        if check:
            # Check equivalence of actual traits, i.e., hyperparameter values.
            compare_traits = True
            for key in emigrant.keys():
                if existing_ind[0][key] == emigrant[key]:
                    continue
                else:
                    compare_traits = False
                    break

            print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: Currently in emigrated: {emigrant}\n" \
                  f"I{self.isle_idx} W{self.comm.rank} G{generation}: Currently in population: {existing_ind}" \
                  f"Equivalence check: ", existing_ind[0] == emigrant, compare_traits, \
                  existing_ind[0].loss == self.emigrated[idx].loss, \
                  existing_ind[0].active == emigrant.active, \
                  existing_ind[0].current == emigrant.current, \
                  existing_ind[0].isle == emigrant.isle, \
                  existing_ind[0].migration_steps == emigrant.migration_steps)

        return check


    def _deactivate_emigrants(self, generation, DEBUG=False):
        """
        Check for and possibly receive emigrants from other intra-isle workers to be deactivated.

        Parameters
        ----------
        generation : int
                     generation of new individual
        DEBUG : bool
                flag for additional debug prints
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
                # Receive new emigrants.
                new_emigrants = req_sync.wait()
                # Add new emigrants to list of emigrants to be deactivated.
                self.emigrated = self.emigrated + new_emigrants 
                print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                      f"Got {len(new_emigrants)} new emigrant(s) {new_emigrants} " \
                      f"from W{stat.Get_source()} to be deactivated.\n" \
                      f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                      f"Overall {len(self.emigrated)} individuals to deactivate: {self.emigrated}")
        emigrated_copy = copy.deepcopy(self.emigrated)
        for emigrant in emigrated_copy:
            assert emigrant.active == True
            to_deactivate = [ind for ind in self.population if ind == emigrant and ind.migration_steps == emigrant.migration_steps]
            if len(to_deactivate) == 0: 
                print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                    f"Individual {emigrant} to deactivate not yet received.")
                continue
            assert len(to_deactivate) == 1
            active_pop = [ind for ind in self.population if ind.active]
            active_before = len(active_pop)
            for ind in to_deactivate:
                ind.active = False
            self.emigrated.remove(emigrant)
            active_pop = [ind for ind in self.population if ind.active]
            active_after = len(active_pop)
            print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: Before deactivation: " \
                  f"{active_before}/{len(self.population)} active.\n" \
                  f"I{self.isle_idx} W{self.comm.rank} G{generation}: Deactivated {ind}.\n" \
                  f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                  f"{len(self.emigrated)} individuals in emigrated.\n" \
                  f"I{self.isle_idx} W{self.comm.rank} G{generation}: After deactivation: " \
                  f"{active_after}/{len(self.population)} active.")
        active_pop = [ind for ind in self.population if ind.active]
        print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: After synchronization: "\
              f"{len(active_pop)}/{len(self.population)} active.\n" \
              f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
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
           
            # Breed and evaluate individual.
            self._evaluate_individual(generation)

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
                check = self._check_emigrants_to_deactivate(generation)
                assert check == False

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

        generation -= 1
        
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
        MPI.COMM_WORLD.barrier()

        if migration:
            # Final check for incoming individuals from other islands.
            self._receive_immigrants(generation, DEBUG)
            MPI.COMM_WORLD.barrier()

            # Emigration: Final check for emigrants from other intra-isle workers to be deactivated.
            self._deactivate_emigrants(generation, DEBUG)
            check = self._check_emigrants_to_deactivate(generation)
            assert check == False
            MPI.COMM_WORLD.barrier()
            if len(self.emigrated) > 0:
                print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                    f"Finally {len(self.emigrated)} individual(s) in emigrated: {self.emigrated}:\n" \
                    f"{self.population}"  )
                self._deactivate_emigrants(generation, DEBUG)
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
        active_pop = [ind for ind in self.population if ind.active]
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


class PolliPropulator():
    """
    Parallel propagator of populations with pollination.

    Individuals can actively exist on multiple evolutionary islands at a time, i.e., 
    copies of emigrants are sent out and emigrating individuals are not deactivated on
    sending island for breeding. Instead, immigrants replace individuals on the target 
    island according to an immigration policy set by the immigration propagator.
    """
    def __init__(self, loss_fn, propagator, isle_idx, comm=MPI.COMM_WORLD, generations=0,
                 load_checkpoint = "pop_cpt.p", save_checkpoint="pop_cpt.p", 
                 migration_topology=None, comm_inter=MPI.COMM_WORLD,
                 migration_prob=None, emigration_propagator=None, immigration_propagator=None,
                 unique_ind=None, unique_counts=None, seed=None):
        """
        Constructor of Propulator class.

        Parameters
        ----------
        loss_fn : callable
                  loss function to be minimized
        propagator : propulate.propagators.Propagator
                     propagator to apply for breeding
        isle_idx : int
                   index of isle
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
        migration_topology : numpy array
                             2D matrix where entry (i,j) specifies how many 
                             individuals are sent by isle i to isle j
        comm_inter : MPI communicator
                     inter-isle communicator for migration
        migration_prob : float
                         per-worker migration probability
        emigration_propagator : propulate.propagators.Propagator
                                emigration propagator, i.e., how to choose individuals 
                                for emigration that are sent to destination island.
                                Should be some kind of selection operator.
        immigration_propagator : propulate.propagators.Propagator
                                 immigration propagator, i.e., how to choose individuals 
                                 to be replaced by immigrants on target island.
                                 Should be some kind of selection operator.
        unique_ind : numpy array
                     array with MPI.COMM_WORLD rank of each isle's worker 0
                     Element i specifies MPI.COMM_WORLD rank of worker 0 on isle with index i.
        unique_counts : numpy array
                        array with number of workers per isle
                        Element i specifies number of workers on isle with index i.
        seed : int
               base seed for random number generator
        """
        # Set class attributes.
        self.loss_fn = loss_fn                              # callable loss function
        self.propagator = propagator                        # evolutionary propagator
        if generations == 0:                                # If number of iterations requested == 0.
            if MPI.COMM_WORLD.rank == 0: 
                print("Requested number of generations is zero...[RETURN]")
            return
        self.generations = int(generations)                 # number of generations, i.e., number of evaluation per individual
        self.isle_idx = int(isle_idx)                       # isle index
        self.comm = comm                                    # intra-isle communicator
        self.comm_inter = comm_inter                        # inter-isle communicator
        self.load_checkpoint = str(load_checkpoint)         # path to checkpoint file to be read
        self.save_checkpoint = str(save_checkpoint)         # path to checkpoint file to be written
        self.migration_prob = float(migration_prob)         # per-rank migration probability
        self.migration_topology = migration_topology        # migration topology
        self.unique_ind = unique_ind                        # MPI.COMM_WORLD rank of each isle's worker 0
        self.unique_counts = unique_counts                  # number of workers on each isle
        self.emigration_propagator = emigration_propagator  # emigration propagator
        self.immigration_propagator = immigration_propagator# immigration propagator
        self.to_replace = []                                # individuals to be replaced by immigrants

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

        Parameters
        ----------

        logging_interval : int
                           Print each worker's progress every
                           `logging_interval`th generation.
        """
        self._work(logging_interval)


    def _breed(self, generation):
        """
        Apply propagator to current population of active 
        individuals to breed new individual.

        Parameters
        ----------
        generation : int
                     generation of new individual

        Returns
        -------
        ind : propulate.population.Individual
              newly bred individual
        """
        active_pop = [ind for ind in self.population if ind.active]
        ind = self.propagator(active_pop)   # Breed new individual from active population.
        ind.generation = generation         # Set generation.
        ind.rank = self.comm.rank           # Set worker rank.
        ind.active = True                   # If True, individual is active for breeding.
        ind.isle = self.isle_idx            # Set birth island.
        ind.current = self.comm.rank        # Set worker responsible for migration.
        ind.migration_steps = 0             # Set number of migration steps performed.
        return ind                          # Return new individual.
            

    def _evaluate_individual(self, generation):
        """
        Breed and evaluate individual.

        Parameters
        ----------
        generation : int
                     generation of new individual
        """
        ind = self._breed(generation)   # Breed new individual.
        ind.loss = self.loss_fn(ind)    # Evaluate its loss.
        self.population.append(ind)     # Add evaluated individual to own worker-local population.

        # Tell other workers in own isle about results to synchronize their populations.
        # Use immediate send for asynchronous communication.
        for r in range(self.comm.size): # Loop over ranks in intra-isle communicator.
            if r == self.comm.rank: 
                continue                # No self-talk.
            self.comm.isend(ind, dest=r, tag=INDIVIDUAL_TAG) 


    def _receive_intra_isle_individuals(self, generation, DEBUG=False):
        """
        Check for and possibly receive incoming individuals 
        evaluated by other workers within own isle.

        Parameters
        ----------
        generation : int
                     generation of new individual
        DEBUG : bool
                flag for additional debug prints
        """
        probe_ind = True 
        while probe_ind:
            stat = MPI.Status() # Retrieve status of reception operation, including source, tag, and error.
            probe_ind = self.comm.iprobe(source=MPI.ANY_SOURCE, tag=INDIVIDUAL_TAG, status=stat)
            # If True, continue checking for incoming messages. 
            # Tells whether message corresponding to filters passed is waiting for reception via a flag that it sets. 
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
                self.population.append(ind_temp)    # Add received individual to own worker-local population.
        active_pop = [ind for ind in self.population if ind.active]
        print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
              f"After probing within isle: {len(active_pop)}/{len(self.population)} active.")


    def _send_emigrants(self, generation, DEBUG=False):
        """
        Perform migration, i.e. isle sends individuals out to other islands.

        Parameters
        ----------
        generation : int
                     generation of new individual
        DEBUG : bool
                flag for additional debug prints
        """
        # Determine relevant line of migration topology.
        to_migrate = self.migration_topology[self.isle_idx,:] 
        num_emigrants = np.amax(to_migrate) # Determine maximum number of emigrants to be sent out at once.
        # NOTE Do we need emigration-responsible worker for pollination or can all workers in principle
        # send out any individual? `eligible_emigrants` or `active_pop` here?
        eligible_emigrants = [ind for ind in self.population if ind.active \
                              and ind.current == self.comm.rank]
                
        # Only perform migration if maximum number of emigrants to be sent
        # out is smaller than current number of eligible emigrants.
        if num_emigrants <= len(eligible_emigrants): 
            # Loop through relevant part of migration topology.
            for target_isle, offspring in enumerate(to_migrate):
                if offspring == 0: continue
                eligible_emigrants = [ind for ind in self.population if ind.active and ind.current == self.comm.rank] # See NOTE above!
                # Determine MPI.COMM_WORLD ranks of workers on target isle.
                displ = self.unique_ind[target_isle]
                count = self.unique_counts[target_isle]
                dest_isle = np.arange(displ, displ+count)
                
                # Worker in principle sends *different* individuals to each target isle,
                # even though copies are allowed for pollination.
                emigrator = self.emigration_propagator(offspring) # Set up emigration propagator.
                emigrants = emigrator(eligible_emigrants)         # Choose `offspring` eligible emigrants.
                print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                      f"Chose {len(emigrants)} emigrant(s): {emigrants}")
                    
                # NOTE Emigrants do not need to be deactivated on sending isle for pollination!
                # Send emigrants to all workers on target isle but only responsible worker will
                # choose individuals to be replaced by immigrants and tell other workers on target
                # isle about them.
                departing = copy.deepcopy(emigrants)
                # Determine new responsible worker on target isle.
                for ind in departing:
                    ind.current = random.randrange(0, count)
                for r in dest_isle: # Loop through MPI.COMM_WORLD destination ranks.
                    MPI.COMM_WORLD.isend(departing, dest=r, tag=MIGRATION_TAG)
                    print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                          f"Sent {len(departing)} individual(s) to W{r-self.unique_ind[target_isle]} " \
                          f"on target I{target_isle}.")

            active_pop = [ind for ind in self.population if ind.active]
            print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                  f"After emigration: {len(active_pop)}/{len(self.population)} active.")
                        
        else: 
            print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                  f"Population size {len(eligible_emigrants)} too small " \
                  f"to select {num_emigrants} migrants.")
       

    def _receive_immigrants(self, generation, DEBUG=False):
        """
        Check for and possibly receive immigrants send by other islands.

        Parameters
        ----------
        generation : int
                     generation of new individual
        DEBUG : bool
                flag for additional debug prints
        """
        probe_migrants = True
        while probe_migrants:
            stat = MPI.Status()
            probe_migrants = MPI.COMM_WORLD.iprobe(source=MPI.ANY_SOURCE, tag=MIGRATION_TAG, status=stat)
            if DEBUG: 
                print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                      f"Checking for immigrants...{probe_migrants}")
            if probe_migrants:
                req_immigr = MPI.COMM_WORLD.irecv(source=stat.Get_source(), tag=MIGRATION_TAG)
                immigrants = req_immigr.wait()
                print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                      f"Received {len(immigrants)} immigrant(s) from global " \
                      f"W{stat.Get_source()}: {immigrants}")
                
                # Add immigrants to own population.
                for immigrant in immigrants: 
                    immigrant.migration_steps += 1
                    assert immigrant.active == True
                    catastrophic_failure = len([ind for ind in self.population if ind == immigrant and \
                                               immigrant.migration_steps == ind.migration_steps]) > 0
                    if catastrophic_failure:
                        raise RuntimeError(f"Identical immigrant {immigrant} already " \
                                           f"active on target I{self.isle_idx}.")
                    immigrant_obsolete = len([ind for ind in self.population if ind == immigrant and \
                                             immigrant.migration_steps < ind.migration_steps]) > 0
                    if immigrant_obsolete:
                        immigrant.active = False
                    self.population.append(immigrant) # Append immigrant to population.
                    
                    if self.comm.rank == immigrant.current:
                        print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                              f"Responsible for choosing individuals to be replaced " \
                              f"by immigrants.")
                        # NOTE With or without check for ind.current == self.comm.rank?
                        eligible_for_replacement = [ind for ind in self.population if ind.active and ind.current == self.comm.rank]
                        immigrator = self.immigration_propagator(offspring=1)   # Set up immigration propagator.
                        to_be_replaced = emigrator(eligible_for_replacement)    # Choose individual to be replaced by immigrant.
                        self.to_replace = self.to_replace + to_be_replaced
                        print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                        f"Chose individual {to_be_replaced[0]} to be replaced by {immigrant}.")

                    if len(self.to_replace) > 0:
                    for r in range(self.comm.size): # Send emigrants to other intra-isle workers for deactivation.
        #            if r == self.comm.rank: 
        #                continue # No self-talk.
        #            self.comm.isend(emigrants, dest=r, tag=SYNCHRONIZATION_TAG)
        #            print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
        #                  f"Sent {len(emigrants)} individual(s) to intra-isle W{r} to deactivate.")
                    # NOTE Do not remove obsolete individuals from population upon immigration
                    # as they should be deactivated in the next step anyways.

        active_pop = [ind for ind in self.population if ind.active]
        print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: After immigration: " \
              f"{len(active_pop)}/{len(self.population)} active.")
                
        # Deactivate emigrants on sending isle for breeding (true migration).
                # Deactivate emigrants for sending worker.
                #for emigrant in emigrants: 
                #    to_deactivate = [ind for ind in self.population if ind == emigrant and ind.migration_steps == emigrant.migration_steps]
                #    assert len(to_deactivate) == 1
                #    to_deactivate[0].active = False
                #    print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                #          f"Deactivated own emigrant {to_deactivate[0]}.")




    def _check_emigrants_to_deactivate(self, generation):
        """
        Redundant safety check for existence of emigrants that could not be deactived in population.

        Parameters
        ----------
        generation : int
                     generation of new individual
        """
        check = False
        # Loop over emigrants still to be deactivated.
        for idx, emigrant in enumerate(self.emigrated):
            existing_ind = [ind for ind in self.population if ind == emigrant and ind.migration_steps == emigrant.migration_steps]
            if len(existing_ind) > 0: 
                check = True
                break
        if check:
            # Check equivalence of actual traits, i.e., hyperparameter values.
            compare_traits = True
            for key in emigrant.keys():
                if existing_ind[0][key] == emigrant[key]:
                    continue
                else:
                    compare_traits = False
                    break

            print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: Currently in emigrated: {emigrant}\n" \
                  f"I{self.isle_idx} W{self.comm.rank} G{generation}: Currently in population: {existing_ind}" \
                  f"Equivalence check: ", existing_ind[0] == emigrant, compare_traits, \
                  existing_ind[0].loss == self.emigrated[idx].loss, \
                  existing_ind[0].active == emigrant.active, \
                  existing_ind[0].current == emigrant.current, \
                  existing_ind[0].isle == emigrant.isle, \
                  existing_ind[0].migration_steps == emigrant.migration_steps)

        return check


    def _deactivate_emigrants(self, generation, DEBUG=False):
        """
        Check for and possibly receive emigrants from other intra-isle workers to be deactivated.

        Parameters
        ----------
        generation : int
                     generation of new individual
        DEBUG : bool
                flag for additional debug prints
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
                # Receive new emigrants.
                new_emigrants = req_sync.wait()
                # Add new emigrants to list of emigrants to be deactivated.
                self.emigrated = self.emigrated + new_emigrants 
                print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                      f"Got {len(new_emigrants)} new emigrant(s) {new_emigrants} " \
                      f"from W{stat.Get_source()} to be deactivated.\n" \
                      f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                      f"Overall {len(self.emigrated)} individuals to deactivate: {self.emigrated}")
        emigrated_copy = copy.deepcopy(self.emigrated)
        for emigrant in emigrated_copy:
            assert emigrant.active == True
            to_deactivate = [ind for ind in self.population if ind == emigrant and ind.migration_steps == emigrant.migration_steps]
            if len(to_deactivate) == 0: 
                print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                    f"Individual {emigrant} to deactivate not yet received.")
                continue
            assert len(to_deactivate) == 1
            active_pop = [ind for ind in self.population if ind.active]
            active_before = len(active_pop)
            for ind in to_deactivate:
                ind.active = False
            self.emigrated.remove(emigrant)
            active_pop = [ind for ind in self.population if ind.active]
            active_after = len(active_pop)
            print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: Before deactivation: " \
                  f"{active_before}/{len(self.population)} active.\n" \
                  f"I{self.isle_idx} W{self.comm.rank} G{generation}: Deactivated {ind}.\n" \
                  f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                  f"{len(self.emigrated)} individuals in emigrated.\n" \
                  f"I{self.isle_idx} W{self.comm.rank} G{generation}: After deactivation: " \
                  f"{active_after}/{len(self.population)} active.")
        active_pop = [ind for ind in self.population if ind.active]
        print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: After synchronization: "\
              f"{len(active_pop)}/{len(self.population)} active.\n" \
              f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
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
           
            # Breed and evaluate individual.
            self._evaluate_individual(generation)

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
                check = self._check_emigrants_to_deactivate(generation)
                assert check == False

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

        generation -= 1
        
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
        MPI.COMM_WORLD.barrier()

        if migration:
            # Final check for incoming individuals from other islands.
            self._receive_immigrants(generation, DEBUG)
            MPI.COMM_WORLD.barrier()

            # Emigration: Final check for emigrants from other intra-isle workers to be deactivated.
            self._deactivate_emigrants(generation, DEBUG)
            check = self._check_emigrants_to_deactivate(generation)
            assert check == False
            MPI.COMM_WORLD.barrier()
            if len(self.emigrated) > 0:
                print(f"I{self.isle_idx} W{self.comm.rank} G{generation}: " \
                    f"Finally {len(self.emigrated)} individual(s) in emigrated: {self.emigrated}:\n" \
                    f"{self.population}"  )
                self._deactivate_emigrants(generation, DEBUG)
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
        active_pop = [ind for ind in self.population if ind.active]
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
