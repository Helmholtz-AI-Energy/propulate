import os
import random
import pickle
import threading

from mpi4py import MPI

from .population import Individual


COORD_REQUEST_TAG = 1
COORD_REPLY_TAG = 2

IND_REQUEST_SUBTAG = 1
LOSS_REPORT_SUBTAG = 2

coordinator_rank = 0


class Propulator():
    def __init__(self, loss_fn, propagator, fallback_propagator, comm=None, num_generations=0, checkpoint_file=None):
        self.loss_fn = loss_fn
        self.propagator = propagator
        self.fallback_propagator = fallback_propagator
        if fallback_propagator.parents != 0 or fallback_propagator.offspring != 1:
            raise ValueError("Fallback propagator has create 1 offspring from 0 parents")
        self.generations = num_generations
        self.comm = comm
        if comm is None:
            self.comm = MPI.COMM_WORLD

        self.best = float('inf')

        self.running = [None] * self.comm.Get_size()
        self.population = []
        self.retired = []

        self.checkpoint_file = checkpoint_file

        self.coordinator_message_processing_functions = [
            None,
            self._process_individual_request,
            self._process_loss_report,
        ]

        return

    def propulate(self, resume=False):
        self.load_checkpoint = resume
        if self.comm.Get_rank() == coordinator_rank:
            thread = threading.Thread(target=self._coordinate, name="coord_thread")
            thread.start()
        self._work()
        if self.comm.Get_rank() == coordinator_rank:
            thread.join()
        return

    # NOTE individual level checkpointing is left to the user
    def _work(self):
        g = 0
        rank = self.comm.Get_rank()

        while self.generations < 1 or g < self.generations:
            message = (IND_REQUEST_SUBTAG, g)
            self.comm.send(message, dest=coordinator_rank, tag=COORD_REQUEST_TAG)
            individual = self.comm.recv(source=coordinator_rank, tag=COORD_REPLY_TAG)

            loss = self.loss_fn(individual)
            # NOTE report loss to coordinator
            message = (LOSS_REPORT_SUBTAG, (loss, g))
            self.comm.send(message, dest=coordinator_rank, tag=COORD_REQUEST_TAG)


            g += 1

        return

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
            if os.path.isfile(checkpoint_file) and self.load_checkpoint == True:
                with open(checkpoint_file, 'rb') as f:
                    self.population, self.running = pickle.load(f)

        self.terminated_ranks = 0
        while self.terminated_ranks < self.comm.Get_size():
            status = MPI.Status()
            message = self.comm.recv(source=MPI.ANY_SOURCE, tag=COORD_REQUEST_TAG, status=status)
            tag = status.tag
            source = status.source
            # NOTE 'decode' message
            subtag, message = message
            # NOTE process message
            self.coordinator_message_processing_functions[subtag](source, message)

        return

    def _process_individual_request(self, source, message):
        ind = self._breed(message, source)

        if self.running[source] is not None:
            raise ValueError()
        self.running[source] = ind
        self.comm.send(ind, dest=source, tag=COORD_REPLY_TAG)
        # TODO save checkpoint
        return

    def _process_loss_report(self, source, message):
        loss, generation = message
        self.running[source].loss = loss
        self.population.append(self.running[source])
        self.running[source] = None
        if loss < self.best:
            self.best = loss
        if generation == self.generations-1:
            self.terminated_ranks += 1
        return

    def summarize(self):
        if self.comm.Get_rank() == coordinator_rank:
            import matplotlib.pyplot as plt
            xs = [i.generation for i in self.population]
            ys = [i.loss for i in self.population]
            zs = [i.rank for i in self.population]

            print(self.best)
            plt.scatter(xs,ys, c=zs)
            plt.show()
        return
