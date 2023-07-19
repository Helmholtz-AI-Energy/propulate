import copy
import os
import pickle
import time
from operator import attrgetter
from pathlib import Path

import deepdiff
import numpy as np
from mpi4py import MPI

from ._globals import DUMP_TAG, INDIVIDUAL_TAG, MIGRATION_TAG, SYNCHRONIZATION_TAG


class Propulator:
    """
    Parallel propagator of populations with real migration.

    Individuals can only exist on one evolutionary island at a time, i.e., they are removed
    (i.e. deactivated for breeding) from the sending island upon emigration.
    """

    def __init__(
        self,
        loss_fn,
        propagator,
        isle_idx,
        comm=MPI.COMM_WORLD,
        generations=0,
        checkpoint_path=Path('./'),
        migration_topology=None,
        comm_inter=MPI.COMM_WORLD,
        migration_prob=0.,
        emigration_propagator=None,
        unique_ind=None,
        unique_counts=None,
        rng=None,
    ):
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
        checkpoint_path : Union[Path, str]
                          Path where checkpoints are loaded from and stored.
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
        rng : random.Random()
              random number generator
        """
        # Set class attributes.
        self.loss_fn = loss_fn  # callable loss function
        self.propagator = propagator  # evolutionary propagator
        if generations == 0:  # If number of iterations requested == 0.
            if MPI.COMM_WORLD.rank == 0:
                print("Requested number of generations is zero...[RETURN]")
            return
        self.generations = int(
            generations
        )  # number of generations (evaluations per individual)
        self.generation = 0 # current generation not yet evaluated
        self.isle_idx = int(isle_idx)  # isle index
        self.comm = comm  # intra-isle communicator
        self.comm_inter = comm_inter  # inter-isle communicator

        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.migration_prob = float(migration_prob)  # per-rank migration probability
        self.migration_topology = migration_topology  # migration topology
        self.unique_ind = unique_ind  # MPI.COMM_WORLD rank of each isle's worker 0
        self.unique_counts = unique_counts  # number of workers on each isle
        self.emigration_propagator = emigration_propagator  # emigration propagator
        self.emigrated = []  # emigrated individuals to be deactivated on sending isle
        self.rng = rng

        # Load initial population of evaluated individuals from checkpoint if exists.
        load_ckpt_file = self.checkpoint_path / f'island_{self.isle_idx}_ckpt.pkl'
        if not os.path.isfile(load_ckpt_file):  # If not exists, check for backup file.
            load_ckpt_file = load_ckpt_file.with_suffix('.bkp')

        if os.path.isfile(load_ckpt_file):
            with open(load_ckpt_file, "rb") as f:
                try:
                    self.population = pickle.load(f)
                    self.generation = max([x.generation for x in self.population if x.rank == self.comm.rank]) + 1
                    if self.comm.rank == 0:
                        print(
                            "NOTE: Valid checkpoint file found. "
                            f"Resuming from generation {self.generation} of loaded population..."
                        )
                except Exception:
                    self.population = []
                    if self.comm.rank == 0:
                        print(
                            "NOTE: No valid checkpoint file found. "
                            "Initializing population randomly..."
                        )
        else:
            self.population = []
            if self.comm.rank == 0:
                print(
                    "NOTE: No valid checkpoint file given. "
                    "Initializing population randomly..."
                )

    def propulate(self, logging_interval=10, DEBUG=1):
        """
        Run actual evolutionary optimization.""

        Parameters
        ----------

        logging_interval : int
                           Print each worker's progress every
                           `logging_interval`th generation.
        """
        self._work(logging_interval, DEBUG)

    def _get_active_individuals(self):
        """
        Get active individuals in current population list.

        Returns
        -------
        active_pop : list of propulate.population.Individuals
                     currently active individuals in population
        num_actives : int
                      number of currently active individuals
        """
        active_pop = [ind for ind in self.population if ind.active]
        num_actives = len(active_pop)

        return active_pop, num_actives

    def _breed(self):
        """
        Apply propagator to current population of active
        individuals to breed new individual.

        Returns
        -------
        ind : propulate.population.Individual
              newly bred individual
        """
        active_pop, _ = self._get_active_individuals()
        ind = self.propagator(
            active_pop
        )  # Breed new individual from active population.
        ind.generation = self.generation  # Set generation.
        ind.rank = self.comm.rank  # Set worker rank.
        ind.active = True  # If True, individual is active for breeding.
        ind.isle = self.isle_idx  # Set birth island.
        ind.current = self.comm.rank  # Set worker responsible for migration.
        ind.migration_steps = 0  # Set number of migration steps performed.
        return ind  # Return new individual.

    def _evaluate_individual(self, DEBUG):
        """
        Breed and evaluate individual.
        """
        ind = self._breed()  # Breed new individual.
        start_time = time.time()
        ind.loss = self.loss_fn(ind)  # Evaluate its loss.
        ind.evaltime = time.time()
        ind.evalperiod = ind.evaltime - start_time
        self.population.append(
            ind
        )  # Add evaluated individual to own worker-local population.
        if DEBUG == 2:
            print(
                f"I{self.isle_idx} W{self.comm.rank} G{self.generation}: BREEDING\n"
                f"Bred and evaluated individual {ind}.\n"
            )

        # Tell other workers in own isle about results to synchronize their populations.
        for r in range(self.comm.size):  # Loop over ranks in intra-isle communicator.
            if r == self.comm.rank:
                continue  # No self-talk.
            self.comm.send(copy.deepcopy(ind), dest=r, tag=INDIVIDUAL_TAG)

    def _receive_intra_isle_individuals(self, DEBUG):
        """
        Check for and possibly receive incoming individuals
        evaluated by other workers within own isle.

        Parameters
        ----------
        DEBUG : bool
                flag for additional debug prints
        """
        log_string = f"I{self.isle_idx} W{self.comm.rank} G{self.generation}: INTRA-ISLE SYNCHRONIZATION\n"
        probe_ind = True
        while probe_ind:
            stat = (
                MPI.Status()
            )  # Retrieve status of reception operation, including source and tag.
            probe_ind = self.comm.iprobe(
                source=MPI.ANY_SOURCE, tag=INDIVIDUAL_TAG, status=stat
            )
            # If True, continue checking for incoming messages. Tells whether message corresponding
            # to filters passed is waiting for reception via a flag that it sets.
            # If no such message has arrived yet, it returns False.
            log_string += f"Incoming individual to receive?...{probe_ind}\n"
            if probe_ind:
                # Receive individual and add it to own population.
                ind_temp = self.comm.recv(source=stat.Get_source(), tag=INDIVIDUAL_TAG)
                self.population.append(
                    ind_temp
                )  # Add received individual to own worker-local population.
                log_string += f"Added individual {ind_temp} from W{stat.Get_source()} to own population.\n"
        _, n_active = self._get_active_individuals()
        log_string += (
            f"After probing within isle: {n_active}/{len(self.population)} active.\n"
        )
        if DEBUG == 2:
            print(log_string)

    def _send_emigrants(self, DEBUG):
        """
        Perform migration, i.e. isle sends individuals out to other islands.

        Parameters
        ----------
        DEBUG : bool
                flag for additional debug prints
        """
        log_string = f"I{self.isle_idx} W{self.comm.rank} G{self.generation}: EMIGRATION\n"
        # Determine relevant line of migration topology.
        to_migrate = self.migration_topology[self.isle_idx, :]
        num_emigrants = np.sum(
            to_migrate
        )  # Determine overall number of emigrants to be sent out.
        eligible_emigrants = [
            ind
            for ind in self.population
            if ind.active and ind.current == self.comm.rank
        ]

        # Only perform migration if overall number of emigrants to be sent
        # out is smaller than current number of eligible emigrants.
        if num_emigrants <= len(eligible_emigrants):
            # Select all migrants to be sent out in this migration step.
            emigrator = self.emigration_propagator(
                num_emigrants
            )  # Set up emigration propagator.
            all_emigrants = emigrator(
                eligible_emigrants
            )  # Choose `offspring` eligible emigrants.
            self.rng.shuffle(all_emigrants)
            # Loop through relevant part of migration topology.
            offsprings_sent = 0
            for target_isle, offspring in enumerate(to_migrate):
                if offspring == 0:
                    continue
                # Determine MPI.COMM_WORLD ranks of workers on target isle.
                displ = self.unique_ind[target_isle]
                count = self.unique_counts[target_isle]
                dest_isle = np.arange(displ, displ + count)

                # Worker sends *different* individuals to each target isle.
                emigrants = all_emigrants[
                    offsprings_sent: offsprings_sent + offspring
                ]  # Choose `offspring` eligible emigrants.
                offsprings_sent += offspring
                log_string += f"Chose {len(emigrants)} emigrant(s): {emigrants}\n"

                # Deactivate emigrants on sending isle (true migration).
                for r in range(
                    self.comm.size
                ):  # Send emigrants to other intra-isle workers for deactivation.
                    if r == self.comm.rank:
                        continue  # No self-talk.
                    self.comm.send(
                        copy.deepcopy(emigrants), dest=r, tag=SYNCHRONIZATION_TAG
                    )
                    log_string += f"Sent {len(emigrants)} individual(s) {emigrants} to intra-isle W{r} to deactivate.\n"

                # Send emigrants to target island.
                departing = copy.deepcopy(emigrants)
                # Determine new responsible worker on target isle.
                for ind in departing:
                    ind.current = self.rng.randrange(0, count)
                for r in dest_isle:  # Loop through MPI.COMM_WORLD destination ranks.
                    MPI.COMM_WORLD.send(
                        copy.deepcopy(departing), dest=r, tag=MIGRATION_TAG
                    )
                    log_string += (
                        f"Sent {len(departing)} individual(s) to W{r-self.unique_ind[target_isle]} "
                        + f"on target I{target_isle}.\n"
                    )

                # Deactivate emigrants for sending worker.
                for emigrant in emigrants:
                    # Look for emigrant to deactivate in original population list.
                    to_deactivate = [
                        idx
                        for idx, ind in enumerate(self.population)
                        if ind == emigrant
                        and ind.migration_steps == emigrant.migration_steps
                    ]
                    assert len(to_deactivate) == 1  # There should be exactly one!
                    _, n_active_before = self._get_active_individuals()
                    self.population[
                        to_deactivate[0]
                    ].active = False  # Deactivate emigrant in population.
                    _, n_active_after = self._get_active_individuals()
                    log_string += (
                        f"Deactivated own emigrant {self.population[to_deactivate[0]]}. "
                        + f"Active before/after: {n_active_before}/{n_active_after}\n"
                    )
            _, n_active = self._get_active_individuals()
            log_string += (
                f"After emigration: {n_active}/{len(self.population)} active.\n"
            )

            if DEBUG == 2:
                print(log_string)

        else:
            if DEBUG == 2:
                print(
                    f"I{self.isle_idx} W{self.comm.rank} G{self.generation}: \n"
                    f"Population size {len(eligible_emigrants)} too small "
                    f"to select {num_emigrants} migrants."
                )

    def _receive_immigrants(self, DEBUG):
        """
        Check for and possibly receive immigrants send by other islands.

        Parameters
        ----------
        DEBUG : bool
                flag for additional debug prints
        """
        log_string = f"I{self.isle_idx} W{self.comm.rank} G{self.generation}: IMMIGRATION\n"
        probe_migrants = True
        while probe_migrants:
            stat = MPI.Status()
            probe_migrants = MPI.COMM_WORLD.iprobe(
                source=MPI.ANY_SOURCE, tag=MIGRATION_TAG, status=stat
            )
            log_string += f"Immigrant(s) to receive?...{probe_migrants}\n"
            if probe_migrants:
                immigrants = MPI.COMM_WORLD.recv(
                    source=stat.Get_source(), tag=MIGRATION_TAG
                )
                log_string += (
                    f"Received {len(immigrants)} immigrant(s) from global "
                    f"W{stat.Get_source()}: {immigrants}\n"
                )
                for immigrant in immigrants:
                    immigrant.migration_steps += 1
                    assert immigrant.active is True
                    catastrophic_failure = (
                        len(
                            [
                                ind
                                for ind in self.population
                                if ind == immigrant
                                and immigrant.migration_steps == ind.migration_steps
                                and immigrant.current == ind.current
                            ]
                        )
                        > 0
                    )
                    if catastrophic_failure:
                        raise RuntimeError(
                            log_string
                            + f"Identical immigrant {immigrant} already active on target I{self.isle_idx}."
                        )
                    self.population.append(
                        copy.deepcopy(immigrant)
                    )  # Append immigrant to population.
                    log_string += f"Added immigrant {immigrant} to population.\n"

                    # NOTE Do not remove obsolete individuals from population upon immigration
                    # as they should be deactivated in the next step anyways.

        _, n_active = self._get_active_individuals()
        log_string += f"After immigration: {n_active}/{len(self.population)} active.\n"

        if DEBUG == 2:
            print(log_string)

    def _check_emigrants_to_deactivate(self):
        """
        Redundant safety check for existence of emigrants that could not be deactived in population.
        """
        check = False
        # Loop over emigrants still to be deactivated.
        for idx, emigrant in enumerate(self.emigrated):
            existing_ind = [
                ind
                for ind in self.population
                if ind == emigrant and ind.migration_steps == emigrant.migration_steps
            ]
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

            log_string = (
                f"I{self.isle_idx} W{self.comm.rank} G{self.generation}:\n"
                + f"Currently in emigrated: {emigrant}\n"
                + f"I{self.isle_idx} W{self.comm.rank} G{self.generation}: Currently in population: {existing_ind}\n"
                + "Equivalence check: "
                + str(existing_ind[0] == emigrant)
                + str(compare_traits)
                + str(existing_ind[0].loss == self.emigrated[idx].loss)
                + str(existing_ind[0].active == emigrant.active)
                + str(existing_ind[0].current == emigrant.current)
                + str(existing_ind[0].isle == emigrant.isle)
                + str(existing_ind[0].migration_steps == emigrant.migration_steps)
            )
            print(log_string)

        return check

    def _deactivate_emigrants(self, DEBUG):
        """
        Check for and possibly receive emigrants from other intra-isle workers to be deactivated.

        Parameters
        ----------
        DEBUG : bool
                flag for additional debug prints
        """
        log_string = f"I{self.isle_idx} W{self.comm.rank} G{self.generation}: DEACTIVATION\n"
        probe_sync = True
        while probe_sync:
            stat = MPI.Status()
            probe_sync = self.comm.iprobe(
                source=MPI.ANY_SOURCE, tag=SYNCHRONIZATION_TAG, status=stat
            )
            log_string += f"Emigrants from others to be deactivated to be received?...{probe_sync}\n"
            if probe_sync:
                # Receive new emigrants.
                new_emigrants = self.comm.recv(
                    source=stat.Get_source(), tag=SYNCHRONIZATION_TAG
                )
                # Add new emigrants to list of emigrants to be deactivated.
                self.emigrated = self.emigrated + copy.deepcopy(new_emigrants)
                log_string += (
                    f"Got {len(new_emigrants)} new emigrant(s) {new_emigrants} "
                    + f"from W{stat.Get_source()} to be deactivated.\n"
                    + f"Overall {len(self.emigrated)} individuals to deactivate: {self.emigrated}\n"
                )
            # TODO In while loop or not?
            emigrated_copy = copy.deepcopy(self.emigrated)
            for emigrant in emigrated_copy:
                assert emigrant.active is True
                to_deactivate = [
                    idx
                    for idx, ind in enumerate(self.population)
                    if ind == emigrant
                    and ind.migration_steps == emigrant.migration_steps
                ]
                if len(to_deactivate) == 0:
                    log_string += (
                        f"Individual {emigrant} to deactivate not yet received.\n"
                    )
                    continue
                assert len(to_deactivate) == 1
                self.population[to_deactivate[0]].active = False
                to_remove = [
                    idx
                    for idx, ind in enumerate(self.emigrated)
                    if ind == emigrant
                    and ind.migration_steps == emigrant.migration_steps
                ]
                assert len(to_remove) == 1
                self.emigrated.pop(to_remove[0])
                log_string += (
                    f"Deactivated {self.population[to_deactivate[0]]}.\n"
                    + f"{len(self.emigrated)} individuals in emigrated.\n"
                )
        _, n_active = self._get_active_individuals()
        log_string += (
            "After synchronization: "
            + f"{n_active}/{len(self.population)} active.\n"
            + f"{len(self.emigrated)} individuals in emigrated.\n"
        )
        if DEBUG == 2:
            print(log_string)

    def _get_unique_individuals(self):
        """
        Get unique individuals (in terms of traits + loss) in current population.

        Returns
        ----------
        unique_inds : list of propulate.population.Individuals
                      list of unique individuals
        """
        unique_inds = []
        for individual in self.population:
            considered = False
            for ind in unique_inds:
                # Check for equivalence of traits only when
                # determining unique individuals. To do so, use
                # self.equals(other) member function of Individual()
                # class instead of `==` operator.
                if individual.equals(ind):
                    considered = True
                    break
            if not considered:
                unique_inds.append(individual)
        return unique_inds

    def _check_for_duplicates(self, active, DEBUG):
        """
        Check for duplicates in current population.

        For pollination, duplicates are allowed as emigrants are sent as copies
        and not deactivated on sending isle.

        Parameters
        ----------
        DEBUG : bool
                flag for additional debug prints
        """
        if active:
            population, _ = self._get_active_individuals()
        else:
            population = self.population
        unique_inds = []
        occurrences = []
        for individual in population:
            considered = False
            for ind in unique_inds:
                if individual == ind:
                    considered = True
                    break
            if not considered:
                num_copies = population.count(individual)
                if DEBUG == 2:
                    print(
                        f"I{self.isle_idx} W{self.comm.rank} G{self.generation}: "
                        f"{individual} occurs {num_copies} time(s)."
                    )
                unique_inds.append(individual)
                occurrences.append([individual, num_copies])
        return occurrences, unique_inds

    def _check_intra_isle_synchronization(self, populations):
        """
        Check synchronization of populations of workers within one isle.

        Parameters
        ----------
        populations : list of population lists with
                      propulate.population.Individual() instances

        Returns
        -------
        synchronized : bool
                       If True, populations are synchronized.
        """
        synchronized = True
        for population in populations:
            diffi = deepdiff.DeepDiff(population, populations[0], ignore_order=True)
            if len(diffi) == 0:
                continue
            print(
                f"I{self.isle_idx} W{self.comm.rank}: Population not synchronized:\n"
                f"{diffi}\n"
            )
            synchronized = False
        return synchronized

    def _work(self, logging_interval, DEBUG):
        """
        Execute evolutionary algorithm in parallel.
        """

        if self.comm.rank == 0:
            print(f"I{self.isle_idx} has {self.comm.size} workers.")

        dump = True if self.comm.rank == 0 else False
        migration = True if self.migration_prob > 0 else False
        MPI.COMM_WORLD.barrier()

        # Loop over generations.
        while self.generations <= -1 or self.generation < self.generations:

            if DEBUG == 1 and self.generation % int(logging_interval) == 0:
                print(
                    f"I{self.isle_idx} W{self.comm.rank}: In generation {self.generation}..."
                )

            # Breed and evaluate individual.
            self._evaluate_individual(DEBUG)

            # Check for and possibly receive incoming individuals from other intra-isle workers.
            self._receive_intra_isle_individuals(DEBUG)

            # Migration.
            if migration:

                # Emigration: Isle sends individuals out.
                # Happens on per-worker basis with certain probability.
                if self.rng.random() < self.migration_prob:
                    self._send_emigrants(DEBUG)

                # Immigration: Check for incoming individuals from other isles.
                self._receive_immigrants(DEBUG)

                # Emigration: Check for emigrants from other intra-isle workers to be deactivated.
                self._deactivate_emigrants(DEBUG)
                if DEBUG == 2:
                    check = self._check_emigrants_to_deactivate()
                    assert check is False

            if dump:  # Dump checkpoint.
                if DEBUG == 2:
                    print(
                        f"I{self.isle_idx} W{self.comm.rank} G{self.generation}: Dumping checkpoint..."
                    )
                save_ckpt_file = self.checkpoint_path / f'island_{self.isle_idx}_ckpt.pkl'
                if os.path.isfile(save_ckpt_file):
                    try:
                        os.replace(save_ckpt_file, save_ckpt_file.with_suffix(".bkp"))
                    except Exception as e:
                        print(e)
                with open(save_ckpt_file, "wb") as f:
                    pickle.dump((self.population), f)

                dest = self.comm.rank + 1 if self.comm.rank + 1 < self.comm.size else 0
                self.comm.send(copy.deepcopy(dump), dest=dest, tag=DUMP_TAG)
                dump = False

            stat = MPI.Status()
            probe_dump = self.comm.iprobe(
                source=MPI.ANY_SOURCE, tag=DUMP_TAG, status=stat
            )
            if probe_dump:
                dump = self.comm.recv(source=stat.Get_source(), tag=DUMP_TAG)
                if DEBUG == 2:
                    print(
                        f"I{self.isle_idx} W{self.comm.rank} G{self.generation}: "
                        f"Going to dump next: {dump}. Before: W{stat.Get_source()}"
                    )

            # Go to next generation.
            self.generation += 1

        # Having completed all generations, the workers have to wait for each other.
        # Once all workers are done, they should check for incoming messages once again
        # so that each of them holds the complete final population and the found optimum
        # irrespective of the order they finished.

        MPI.COMM_WORLD.barrier()
        if MPI.COMM_WORLD.rank == 0:
            print("OPTIMIZATION DONE.\nNEXT: Final checks for incoming messages...")
        MPI.COMM_WORLD.barrier()

        # Final check for incoming individuals evaluated by other intra-isle workers.
        self._receive_intra_isle_individuals(DEBUG)
        MPI.COMM_WORLD.barrier()

        if migration:
            # Final check for incoming individuals from other islands.
            self._receive_immigrants(DEBUG)
            MPI.COMM_WORLD.barrier()

            # Emigration: Final check for emigrants from other intra-isle workers to be deactivated.
            self._deactivate_emigrants(DEBUG)

            if DEBUG == 1:
                check = self._check_emigrants_to_deactivate()
                assert check is False
                MPI.COMM_WORLD.barrier()
                if len(self.emigrated) > 0:
                    print(
                        f"I{self.isle_idx} W{self.comm.rank} G{self.generation}: "
                        f"Finally {len(self.emigrated)} individual(s) in emigrated: {self.emigrated}:\n"
                        f"{self.population}"
                    )
                    self._deactivate_emigrants(DEBUG)
                    check = self._check_emigrants_to_deactivate()
                    assert check is False

            MPI.COMM_WORLD.barrier()

        # Final checkpointing on rank 0.
        save_ckpt_file = self.checkpoint_path / f'island_{self.isle_idx}_ckpt.pkl'
        if self.comm.rank == 0:  # Dump checkpoint.
            if os.path.isfile(save_ckpt_file):
                try:
                    os.replace(save_ckpt_file, save_ckpt_file.with_suffix(".bkp"))
                except Exception as e:
                    print(e)
                with open(save_ckpt_file, "wb") as f:
                    pickle.dump((self.population), f)

        MPI.COMM_WORLD.barrier()
        stat = MPI.Status()
        probe_dump = self.comm.iprobe(source=MPI.ANY_SOURCE, tag=DUMP_TAG, status=stat)
        if probe_dump:
            dump = self.comm.recv(source=stat.Get_source(), tag=DUMP_TAG)
        MPI.COMM_WORLD.barrier()

    def summarize(self, top_n=1, out_file="summary.png", DEBUG=1):
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
        active_pop, _ = self._get_active_individuals()
        total = self.comm_inter.reduce(len(active_pop), root=0)
        MPI.COMM_WORLD.barrier()
        if MPI.COMM_WORLD.rank == 0:
            print("\n###########")
            print("# SUMMARY #")
            print("###########\n")
            print(
                f"Number of currently active individuals is {total}. "
                f"\nExpected overall number of evaluations is {self.generations*MPI.COMM_WORLD.size}."
            )
        populations = self.comm.gather(self.population, root=0)
        # Only double-check number of occurrences of each individual for DEBUG level 2.
        if DEBUG == 2:
            occurrences, _ = self._check_for_duplicates(True, DEBUG)
            if self.comm.rank == 0:
                if self._check_intra_isle_synchronization(populations):
                    res_str = (
                        f"I{self.isle_idx}: Populations among workers synchronized."
                    )
                else:
                    res_str = f"I{self.isle_idx}: Populations among workers not synchronized:\n{populations}"
                res_str += f"I{self.isle_idx}: {len(active_pop)}/{len(self.population)} individuals active ({len(occurrences)} unique)."
                print(res_str)
        MPI.COMM_WORLD.barrier()
        best = None
        if DEBUG == 0:
            if self.comm.rank == 0:
                best = min(self.population, key=attrgetter("loss"))
                res_str = f"Top result on isle {self.isle_idx}: {best}\n"
                print(res_str)
            best = self.comm.bcast(best, root=0)
        else:
            if self.comm.rank == 0:
                unique_pop = self._get_unique_individuals()
                unique_pop.sort(key=lambda x: x.loss)
                best = unique_pop[:top_n]
                res_str = f"Top {top_n} result(s) on isle {self.isle_idx}:\n"
                for i in range(top_n):
                    res_str += f"({i+1}): {unique_pop[i]}\n"
                print(res_str)
            best = self.comm.bcast(best, root=0)

        if self.comm.rank == 0:
            import matplotlib.pyplot as plt

            xs = [x.generation for x in self.population]
            ys = [x.loss for x in self.population]
            zs = [x.rank for x in self.population]

            fig, ax = plt.subplots()
            scatter = ax.scatter(xs, ys, c=zs)
            plt.xlabel("Generation")
            plt.ylabel("Loss")
            ax.legend(*scatter.legend_elements(), title="Rank")
            plt.savefig(f"isle_{self.isle_idx}_{out_file}")
            plt.close()
            Best = self.comm_inter.gather(best, root=0)
        MPI.COMM_WORLD.barrier()
        if MPI.COMM_WORLD.rank != 0:
            Best = None
        Best = MPI.COMM_WORLD.bcast(Best, root=0)
        return Best
