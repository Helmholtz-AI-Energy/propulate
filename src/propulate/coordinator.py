import os
import pickle

# NOTE MPI has to already initialized by propulator at this point for MPIs that have not been installed with thread_multiple
from mpi4py import MPI

from .population import Individual

from ._globals import INDIVIDUAL_TAG, LOSS_REPORT_TAG, INIT_TAG, POPULATION_TAG


class Coordinator():
    def __init__(self):
        comm = MPI.COMM_WORLD.Get_parent()
        self.comm = comm.Merge(False)
        comm.Disconnect()

        self.running = [None] * self.comm.Get_size()
        self.population = []
        self.best = float('inf')

        self.generations = self.comm.recv(source=1, tag=INIT_TAG)
        self.checkpoint_file = self.comm.recv(source=1, tag=INIT_TAG)
        self.propagator = self.comm.recv(source=1, tag=INIT_TAG)
        self.fallback_propagator = self.comm.recv(source=1, tag=INIT_TAG)
        self.load_checkpoint = False
        return

    def _breed(self, generation, rank):
        ind = None

        try:
            ind = self.propagator(self.population)
            if ind.loss is not None:
                raise ValueError("No propagator applied, individual already evaluated")
        # TODO fallback should be part of the propagator
        except ValueError:
            ind = self.fallback_propagator()

        ind.generation = generation
        ind.rank = rank

        return ind

    # TODO different algorithms
    # TODO fix checkpointing
    def _coordinate(self):
        if self.checkpoint_file is not None:
            if os.path.isfile(self.checkpoint_file) and self.load_checkpoint:
                with open(self.checkpoint_file, 'rb') as f:
                    self.population, self.running = pickle.load(f)

        size = self.comm.Get_size()
        for i in range(1, size):
            individual = self._breed(0, i)
            self.running[i] = individual

        if self.generations == 0:
            return

        for i in range(1, size):
            self.comm.isend(self.running[i], dest=i, tag=INDIVIDUAL_TAG)

        self.terminated_ranks = 0
        while self.terminated_ranks < self.comm.Get_size()-1:
            status = MPI.Status()
            message = self.comm.recv(source=MPI.ANY_SOURCE, tag=LOSS_REPORT_TAG, status=status)
            source = status.source

            loss, generation = message
            if loss < self.best:
                self.best = loss

            self.running[source].loss = loss
            self.population.append(self.running[source])

            self.comm.send((self.population, self.best), dest=source, tag=POPULATION_TAG)

            if generation == self.generations - 1:
                self.running[source] = None
                self.terminated_ranks += 1
            else:
                self.running[source] = self._breed(generation + 1, source)
                self.comm.isend(self.running[source], dest=source, tag=INDIVIDUAL_TAG)

    # NOTE this is here to work around the bug (?) in mpi4py that would sometimes cause an mpi_abort
    def __del__(self):
        MPI.Finalize()
