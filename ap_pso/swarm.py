"""
This file contains the Swarm class, the technical equivalent to the Islands class of Propulate.
"""
from typing import Optional

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

    def __init__(self, propagator: Propagator, communicator: MPI.Comm):
        """
        Swarm constructor.

        :param propagator: The propagator to be used to update and evaluate the particles within this swarm
        :param communicator: The communicator for this swarm's internal communication.
        """
        self.propagator: Propagator = propagator
        self.communicator = communicator

        self.rank: int = communicator.rank
        self.size: int = communicator.size

        self.particles: list[Particle] = []
        self.archive: list[list[Particle]] = [] * self.size

        self.swarm_best: Optional[ExtendedPosition] = None

        # Create "randomly" initialised particles!

        # Communicate!

    def update(self) -> None:
        """
        This method runs an update on the particles of the swarm. However, only the particle matching to the worker,
        on which the swarm object is lying, is updated (usually, multiple similar swarm objects exist at the same time
        within one communicator, all resembling the same swarm. This is a side effect of playing with MPI).

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

    def _communicate_update(self) -> None:
        """
        Private method to handle the swarm's internal updating communication.

        At the same time, it performs updates on the swarm-global best value.
        """
        for i in range(self.size):
            if i != self.rank:
                self.communicator.send(self.particles[self.rank], i, TELL_TAG)

        while True:
            status = MPI.Status()
            if not self.communicator.iprobe(tag=TELL_TAG, status=status):
                break
            source = status.Get_source()
            incoming_p: Particle = self.communicator.recv(source=source, tag=TELL_TAG)
            self.archive[source].append(self.particles[source])
            self.particles[source] = incoming_p
            if incoming_p.p_best is not None and (self.swarm_best is None or incoming_p.p_best < self.swarm_best):
                self.swarm_best = incoming_p.p_best

    def evaluate_particle(self) -> None:
        """
        This method calls the swarm's propagator's loss function
        on the particles and thus evaluates their fitness.

        The method always only considers the particle matching the rank of
        the worker within the swarm's internal communicator.

        After evaluation, the method also starts a communication round in
        order to update the rest of the workers on the swarm.
        """
        p: Particle = self.particles[self.rank]
        p.loss = self.propagator.loss_fn(p.position)
        p.update_p_best()
        self.particles[self.rank] = p
        self._communicate_update()
