import os
import random
import pickle
import threading

from mpi4py import MPI

from .propagators import mutate, mate


COORD_REQUEST_TAG = 1
COORD_REPLY_TAG = 2

IND_REQUEST_SUBTAG = 1
LOSS_REPORT_SUBTAG = 2
RANK_TERMINATE_SUBTAG = 3

coordinator_rank = 0


# TODO discombobulate num_ranks and pop_size and generation_size
# TODO reproducability
class Propulator():
    def __init__(self, loss, limits, comm, num_generations):
        self.pop_size = comm.Get_size()
        self.generations = num_generations
        self.comm = comm
        self.loss = loss
        self.limits = limits

        self.best = None

        self.load_checkpoint = False
        return

    def propulate(self, resume=False):
        self.load_checkpoint = resume
        if self.comm.Get_rank() == coordinator_rank:
            thread = threading.Thread(target=self.coordinate, name="coord_thread")
            thread.start()
        self.run()
        return

    # NOTE individual level checkpointing is left to the user
    # TODO rename
    def run(self):
        self.g = 0
        rank = self.comm.Get_rank()

        while self.g < self.generations:
            message = (IND_REQUEST_SUBTAG, self.g)
            self.comm.send(message, dest=coordinator_rank, tag=COORD_REQUEST_TAG)
            individual = self.comm.recv(source=coordinator_rank, tag=COORD_REPLY_TAG)

            loss = self.loss(individual)
            # NOTE report loss to coordinator
            message = (LOSS_REPORT_SUBTAG, (loss, self.g))
            self.comm.send(message, dest=coordinator_rank, tag=COORD_REQUEST_TAG)


            self.g += 1

        message = (RANK_TERMINATE_SUBTAG, self.g)
        self.comm.send(message, dest=coordinator_rank, tag=COORD_REQUEST_TAG)


        return

    # TODO move to retiring old generations
    # TODO use user provided parameters instead
    # TODO use user provided propagations functions instead
    def generate_individual(self):

        GEN_RANDOM_PROP = 0.1
        GEN_MATE_PROP = 0.5
        GEN_MATE_MUTATE_PROP = 0.2
        GEN_MUTATE_PROP = 1. - GEN_RANDOM_PROP - GEN_MATE_PROP - GEN_MATE_MUTATE_PROP

        # TODO clean up
        individual = Individual()
        pop = self.pop
        running = self.running
        limits = self.limits
        pop_size = self.pop_size

        x = random.random()
        if len(pop) >= pop_size:
            pop = sorted(pop, reverse=True, key=lambda ind: ind.loss)

        # NOTE randomly generate individual
        if len(pop) < pop_size or x < GEN_RANDOM_PROP:
            for limit in limits.keys():
                if type(limits[limit][0]) == int:
                    individual[limit] = random.randrange(*limits[limit])
                elif type(limits[limit][0]) == float:
                    individual[limit] = random.uniform(*limits[limit])
                elif type(limits[limit][0]) == str:
                    individual[limit] = random.choice(limits[limit])
                else:
                    assert False
        # NOTE breed individual from two parents in the top pop_size (and potentially mutate it)
        elif x < GEN_MATE_PROP + GEN_MATE_MUTATE_PROP + GEN_RANDOM_PROP:
            parent1, parent2 = random.sample(pop[:pop_size], 2)
            individual = mate(parent1, parent2)
            if x > GEN_MATE_PROP + GEN_RANDOM_PROP:
                individual = mutate(individual, limits)
        # NOTE mutate individual
        else:
            individual = mutate(random.choice(pop[:pop_size]), limits)
        # print("generated individual {}".format(individual))
        return individual

    def coordinate(self):
        self.running = {}
        self.pop = []
        checkpoint_file = "population_history.pkl"

        # TODO save/load proper history when moving to retired populations
        if os.path.isfile(checkpoint_file) and self.load_checkpoint == True:
            with open(checkpoint_file, 'rb') as f:
                pop, running = pickle.load(f)

        # TODO reorganize individuals hierarchically: generations of populations of individuals
        terminated = 0
        while terminated < self.pop_size:
            status = MPI.Status()
            # NOTE listen for job requests and loss reports
            message = self.comm.recv(source=MPI.ANY_SOURCE, tag=COORD_REQUEST_TAG, status=status)
            # NOTE message is (source, loss, generation, tag)
            tag = status.tag
            source = status.source
            subtag, message = message
            if subtag == IND_REQUEST_SUBTAG:
                if source in self.running:
                    self.pop.append(self.running[source])
                # print("IND_REQUEST from {}, for generation {}".format(source, message))
                ind = self.generate_individual()
                ind.g = message
                ind.r = source
                self.running[source] = ind
                self.comm.send(ind, dest=source, tag=COORD_REPLY_TAG)
                # print("sent answer {} to {}".format(ind, source))
                # NOTE save current population to file
                # TODO back up latest dump in case time is up while dumping
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump((self.pop, self.running), f)
            elif subtag == LOSS_REPORT_SUBTAG:
                print("LOSS_REPORT from {}:{:.2}, best:{}".format(source, message[0], self.best))
                # NOTE update the loss record, check if it applies to the currently running model,
                # NOTE job request might have arrived before the update (it actually should not because of locking communication)
                loss = message[0]
                g = message[1]
                if g == self.running[source].g:
                    self.running[source].loss = loss
                else:
                    for ind in self.pop:
                        if ind.r == source and ind.g == g:
                            ind.loss = loss
                            break
                    # TODO warn that a loss for an individual was reported that is not present for some reason
                    # TODO should never happen so maybe crash and burn?
                    print("bad things happened")
                if self.best is None or loss < self.best:
                    self.best = loss
            elif subtag == RANK_TERMINATE_SUBTAG:
                terminated += 1

            else:
                print("coordinator received invalid message")
                assert False

        return


# TODO multi objective optimization?
class Individual(dict):
    def __init__(self):
        super(Individual, self).__init__(list())
        self.g = None
        self.r = None
        self.loss = float("inf")
        return
