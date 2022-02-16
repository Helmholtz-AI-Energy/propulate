import os
import pickle
from operator import attrgetter
from mpi4py import MPI

from ._globals import INDIVIDUAL_TAG, LOSS_REPORT_TAG, INIT_TAG, POPULATION_TAG, DUMP_TAG
from .population import Individual

class Propulator():
    """
    """
    def __init__(self, loss_fn, propagator, comm=MPI.COMM_WORLD, generations=0, load_checkpoint = "pop_cpt.p", save_checkpoint="pop_cpt.p", seed=9):
        self.loss_fn = loss_fn              # Set callable loss function to use.
        self.propagator = propagator        # Set propagator to use.
        self.generations = int(generations) # Set number of generations, i.e., number of evaluation per individual.
        if self.generations < -1:
            raise ValueError("Invalid number of generations, needs to be larger than -1, but was {}".format(self.generations))
        self.comm = comm
        self.load_checkpoint = str(load_checkpoint)
        self.save_checkpoint = str(save_checkpoint)

        # Load initial population of evaluated individuals from checkpoint if exists.
        # TODO checks and error messages here.
        # TODO if usual checkpoint file does not exist, check for checkpoint.bkp
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
            if os.path.isfile(self.load_checkpoint+'.bkp'):
                with open(self.load_checkpoint+'bkp', 'rb') as f:
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
        self._work()


    def _breed(self, generation):
        """
        Apply propagator to current population to breed new individual.
        """
        ind = None                             # Initialize new individual.
        ind = self.propagator(self.population) # Breed new individual from current population.
        ind.generation = generation            # Set generation as individual's attribute.
        ind.rank = self.comm.rank              # Set worker rank as individual's attribute.

        return ind                             # Return newly bred individual.


    # NOTE individual level checkpointing is left to the user
    def _work(self):
        """
        Execute evolutionary algorithm in parallel.
        """
        generation = 0        # Start from generation 0.
        rank = self.comm.rank # Determine individual worker's MPI rank.
        
        dump = True if rank == 0 else False

        if self.generations == 0: # If number of iterations requested == 0.
            if rank == 0: print("Number of requested generations is zero...[RETURN]")
            return


        while self.generations == -1 or generation < self.generations:
            print("Worker", rank, "in generation", generation, "...") 
            ind = self._breed(generation) # Breed new individual to evaluate.
            ind.loss = self.loss_fn(ind)  # Evaluate individual's loss.
            self.population.append(ind)  # Append own result to own history list.

            # Tell the world about your great results.
            # Use immediate send for asynchronous communication.
            for r in range(self.comm.size):
                if r == rank: continue
                self.comm.isend(ind, dest=r, tag=LOSS_REPORT_TAG)
            
            # Check for incoming messages.
            probe = True
            while probe:
                print("Worker", rank, "checking for incoming messages...")
                stat = MPI.Status()
                probe = self.comm.iprobe(source=MPI.ANY_SOURCE, tag=LOSS_REPORT_TAG, status=stat)
                if stat.Get_source() >=0:
                    print("Worker", rank, "incoming message from worker", stat.Get_source(), "...")
                if probe:
                    req = self.comm.irecv(source=stat.Get_source(), tag=LOSS_REPORT_TAG)
                    ind_temp = req.wait()
                    print("Worker", rank, "received message from worker", stat.Get_source())
                    self.population.append(ind_temp)

            print("Worker", self.comm.rank, "checking for incoming messages...[DONE]")
            # Determine current best as individual with minimum loss.
            self.best = min(self.population, key=attrgetter('loss'))

            if dump: # Dump checkpoint.
                print("Worker", rank, "dumping checkpoint...")
                if os.path.isfile(self.save_checkpoint):
                    os.replace(self.save_checkpoint, self.save_checkpoint+".bkp")
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
                print("Worker", rank, "is going to dump next:", dump, ". Before: worker", stat.Get_source())

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


    def summarize(self, n=1, out_file='summary.png'):
        n = int(n)
        if self.comm.rank == 0:
            print("#################################################")
            print("# PROPULATE: parallel PROpagator of POPULations #")
            print("#################################################\n")
            print(len(self.population), "individuals have been evaluated in total.")
            if n<=1: 
                print("Minimum", self.best.loss, "found at", self.best, ".")
            else:
                self.population.sort(key=lambda x: x.loss)
                print("Top", n, "results are:")#, self.population[:n])
                for i in range(n):
                    print("(", i+1,"):", self.population[i], "with loss", self.population[i].loss)
            import matplotlib.pyplot as plt
            xs = [x.generation for x in self.population]
            ys = [x.loss for x in self.population]
            zs = [x.rank for x in self.population]

            fig, ax = plt.subplots()
            scatter = ax.scatter(xs, ys, c=zs)
            plt.xlabel("generation")
            plt.ylabel("loss")
            legend = ax.legend(*scatter.legend_elements(), title="rank") 
            plt.savefig(out_file)
