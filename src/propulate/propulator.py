import os
import random
import pickle

import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI
MPI.Init()

from ._globals import INDIVIDUAL_TAG, LOSS_REPORT_TAG, INIT_TAG, POPULATION_TAG


# TODO top n results instead of top 1, for neural network ensembles
# TODO get rid of the fallback at this place. should be in the propagator if needed
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

        self.population = []
        self.retired = []

        self.checkpoint_file = str(checkpoint_file)

        coord_comm = self.comm.Spawn("python", ['-c', "import propulate; coordinator=propulate.Coordinator();coordinator._coordinate()"])
        self.coord_comm = coord_comm.Merge(False)
        coord_comm.Disconnect()

        self.coordinator_rank = self.comm.Get_size()

        if self.coord_comm.Get_rank() == 0:
            self.coord_comm.send(self.generations, dest=self.coordinator_rank, tag=INIT_TAG)
            self.coord_comm.send(self.checkpoint_file, dest=self.coordinator_rank, tag=INIT_TAG)
            self.coord_comm.send(self.propagator, dest=self.coordinator_rank, tag=INIT_TAG)
            self.coord_comm.send(self.fallback_propagator, dest=self.coordinator_rank, tag=INIT_TAG)


    def propulate(self, resume=False):
        self.load_checkpoint = resume
        self._work()

    # NOTE individual level checkpointing is left to the user
    def _work(self):
        generation = 0
        rank = self.comm.Get_rank()

        while self.generations == -1 or generation < self.generations:
            individual = self.coord_comm.recv(source=self.coordinator_rank, tag=INDIVIDUAL_TAG)

            loss = self.loss_fn(individual)
            # NOTE report loss to coordinator
            message = (loss, generation,)
            self.coord_comm.send(message, dest=self.coordinator_rank, tag=LOSS_REPORT_TAG)

            # NOTE receives population and stats from coordinator, so user has access to them
            self.population, self.best = self.coord_comm.recv(source=self.coordinator_rank, tag=POPULATION_TAG)

            generation += 1

    def summarize(self, out_file=None):
        # self.population = self.comm.gather(self.population, root=0)
        self.population = self.comm.allgather(self.population)
        self.population = max(self.population, key=len)
        if self.comm.Get_rank() == 0:
            import matplotlib.pyplot as plt
            xs = [x.generation for x in self.population]
            ys = [x.loss for x in self.population]
            zs = [x.rank for x in self.population]

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

    # NOTE this is here to work around the bug (?) in mpi4py that would sometimes cause an mpi_abort
    def __del__(self):
        MPI.Finalize()
