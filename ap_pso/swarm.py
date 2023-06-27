"""
This file contains the Swarm class, the technical equivalent to the Islands class of Propulate.
"""
from mpi4py import MPI

from ap_pso.particle import Particle
from ap_pso.propagators import Propagator
from ap_pso.utils import ExtendedPosition, TELL_TAG


class Swarm:
    """
    The swarm contains the particles.

    It governs, what the particles do, evaluates them and updates them.

    The particles, to be fair don't deserve to be classes at the moment,
    as they will lose all their methods quite soon to the swarm.

    Also, this class handles the internal MPI stuff, especially regarding communication between single workers within
    one swarm so that everybody is always up to date.
    """

    def __init__(self, num_workers: int, rank: int, propagator: Propagator, communicator: MPI.Comm):
        self.particles: list[Particle] = []
        self.propagator: Propagator = propagator
        self.swarm_best: ExtendedPosition = None
        self.archive: list[list[Particle]] = [] * num_workers
        self.rank: int = rank
        self.communicator = communicator
        self.size = num_workers

        # Create "randomly" initialised particles!

        # Communicate!

    def update(self):
        """
        This function runs an update on the worker's particle.

        The update is performed by the following steps:
        - First, a new particle with the stats of the old one plus one movement step update is created.
        - Then, the old particle is moved into archive.
        - In place of the old particle then the newly created is put.
        - At last, the new particle is communicated.
        """
        old_p = self.particles[self.rank]
        new_p = self.propagator(old_p)
        self.archive[self.rank].append(old_p)
        self.particles[self.rank] = new_p

    def _communicate_update(self):
        for i in range(self.size):
            if i != self.rank:
                self.communicator.send(self.particles[self.rank], i, TELL_TAG)

        while True:
            status = MPI.Status()
            if not self.communicator.iprobe(tag=TELL_TAG, status=status):
                break
            source = status.Get_source()
            incoming_p = self.communicator.recv(source=source, tag=TELL_TAG)
            self.archive[source].append(self.particles[source])
            self.particles[source] = incoming_p




