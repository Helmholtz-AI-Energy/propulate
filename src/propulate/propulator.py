import os
import random
import pickle

import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI
MPI.Init()

# from .population import Individual


from ._globals import INDIVIDUAL_TAG, LOSS_REPORT_TAG, INIT_TAG, POPULATION_TAG, COORDINATOR_RANK


class Propulator():
    def __init__(self, loss_fn, propagator, fallback_propagator, comm=None, generations=0, checkpoint_file=None):
        self.loss_fn = loss_fn
        self.propagator = propagator
        self.fallback_propagator = fallback_propagator
        if fallback_propagator.parents != 0 or fallback_propagator.offspring != 1:
            raise ValueError("Fallback propagator has create 1 offspring from 0 parents")
        self.generations = int(generations)
        if self.generations < -1:
            raise ValueError("Invalid number of generations, needs to be larger than -1, but was {}".format(self.generations))
        self.comm = comm if comm is not None else MPI.COMM_WORLD

        self._population = []
        self.retired = []

        self.checkpoint_file = str(checkpoint_file)

        coord_comm = self.comm.Spawn("python", ['-c', "import propulate; coordinator=propulate.Coordinator();coordinator._coordinate()"])
        self.coord_comm = coord_comm.Merge(True)
        coord_comm.Disconnect()

        if self.coord_comm.Get_rank() == 1:
            self.coord_comm.send(self.generations, dest=COORDINATOR_RANK, tag=INIT_TAG)
            self.coord_comm.send(self.checkpoint_file, dest=COORDINATOR_RANK, tag=INIT_TAG)
            self.coord_comm.send(self.propagator, dest=COORDINATOR_RANK, tag=INIT_TAG)
            self.coord_comm.send(self.fallback_propagator, dest=COORDINATOR_RANK, tag=INIT_TAG)


    def propulate(self, resume=False):
        self.load_checkpoint = resume
        self._work()

    # NOTE individual level checkpointing is left to the user
    def _work(self):
        generation = 0
        rank = self.comm.Get_rank()

        while self.generations == -1 or generation < self.generations:
            individual = self.coord_comm.recv(source=COORDINATOR_RANK, tag=INDIVIDUAL_TAG)

            loss = self.loss_fn(individual)
            # NOTE report loss to coordinator
            message = (loss, generation,)
            req = self.coord_comm.isend(message, dest=COORDINATOR_RANK, tag=LOSS_REPORT_TAG)

            generation += 1

        req.wait()

    # TODO
    # @property
    # def population(self):
    #
    #     return

    # TODO
    def summarize(self, out_file=None):
        return
        # if self.comm.Get_rank() == 0:
        #     import matplotlib.pyplot as plt
        #     xs = [i.generation for i in self.population]
        #     ys = [i.loss for i in self.population]
        #     zs = [i.rank for i in self.population]

        #     print("Best loss: ", self.best)
        #     fig, ax = plt.subplots()
        #     scatter = ax.scatter(xs,ys, c=zs)
        #     plt.xlabel("generation")
        #     plt.ylabel("loss")
        #     legend = ax.legend(*scatter.legend_elements(), title="rank")
        #     if out_file is None:
        #         plt.show()
        #     else:
        #         plt.savefig(outfile)
