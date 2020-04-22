import os
import random
import pickle
import threading

from mpi4py import MPI

from .population import Individual


INDIVIDUAL_TAG = 1
LOSS_REPORT_TAG = 2

COORDINATOR_RANK = 0


class Propulator():
    def __init__(self, loss_fn, propagator, fallback_propagator, comm=None, num_generations=0, checkpoint_file=None):
        self.loss_fn = loss_fn
        self.propagator = propagator
        self.fallback_propagator = fallback_propagator
        if fallback_propagator.parents != 0 or fallback_propagator.offspring != 1:
            raise ValueError("Fallback propagator has create 1 offspring from 0 parents")
        self.generations = int(num_generations)
        if self.generations < -1:
            raise ValueError("Invalid number of generations, needs to be larger than -1, but was {}".format(self.generations))
        self.comm = comm if comm is not None else MPI.COMM_WORLD
        self.best = float('inf')

        self.running = [None] * self.comm.Get_size()
        self.population = []
        self.retired = []

        self.checkpoint_file = str(checkpoint_file)

    def propulate(self, resume=False):
        self.load_checkpoint = resume
        if self.comm.Get_rank() == COORDINATOR_RANK:
            thread = threading.Thread(target=self._coordinate, name="coord_thread")
            thread.start()
        self._work()
        if self.comm.Get_rank() == COORDINATOR_RANK:
            thread.join()

    # NOTE individual level checkpointing is left to the user
    def _work(self):
        generation = 0
        rank = self.comm.Get_rank()

        while self.generations == -1 or generation < self.generations:
            individual = self.comm.recv(source=COORDINATOR_RANK, tag=INDIVIDUAL_TAG)

            loss = self.loss_fn(individual)
            # NOTE report loss to coordinator
            message = (loss, generation,)
            req = self.comm.isend(message, dest=COORDINATOR_RANK, tag=LOSS_REPORT_TAG)

            generation += 1

        req.wait()

    def _breed(self, generation, rank):
        ind = None

        try:
            ind = self.propagator(self.population)
            if ind.loss is not None:
                raise ValueError("No propagator applied, individual already evaluated")
        except ValueError:
            ind = self.fallback_propagator()

        ind.generation = generation
        ind.rank = rank

        return ind

    # TODO different algorithms
    def _coordinate(self):
        if self.checkpoint_file is not None:
            if os.path.isfile(self.checkpoint_file) and self.load_checkpoint:
                with open(self.checkpoint_file, 'rb') as f:
                    self.population, self.running = pickle.load(f)

        size = self.comm.Get_size()
        for i in range(size):
            individual = self._breed(0, i)
            self.running[i] = individual

        if self.generations == 0:
            return

        for i in range(size):
            self.comm.isend(self.running[i], dest=i, tag=INDIVIDUAL_TAG)

        self.terminated_ranks = 0
        while self.terminated_ranks < self.comm.Get_size():
            status = MPI.Status()
            message = self.comm.recv(source=MPI.ANY_SOURCE, tag=LOSS_REPORT_TAG, status=status)
            source = status.source

            loss, generation = message
            if loss < self.best:
                self.best = loss

            self.running[source].loss = loss
            self.population.append(self.running[source])

            if generation == self.generations - 1:
                self.running[source] = None
                self.terminated_ranks += 1
            else:
                self.running[source] = self._breed(generation + 1, source)
                self.comm.isend(self.running[source], dest=source, tag=INDIVIDUAL_TAG)


    def summarize(self, out_file=None):
        if self.comm.Get_rank() == COORDINATOR_RANK:
            import matplotlib.pyplot as plt
            xs = [i.generation for i in self.population]
            ys = [i.loss for i in self.population]
            zs = [i.rank for i in self.population]

            print("Best loss: ", self.best)
            fig, ax = plt.subplots()
            scatter = ax.scatter(xs,ys, c=zs)
            plt.xlabel("generation")
            plt.ylabel("loss")
            legend = ax.legend(*scatter.legend_elements(), title="rank")
            if out_file is None:
                plt.show()
            else:
                plt.savefig(outfile)
