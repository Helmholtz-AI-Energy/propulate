import logging
from typing import Union

import h5py
import numpy as np

from propulate import Migrator, Pollinator, Propulator
from propulate.population import Individual

log = logging.getLogger(__name__)


def final_synch(propulator: Propulator) -> None:
    """Perform final synchronization on completion of optimization."""
    propulator.propulate_comm.barrier()

    # Final check for incoming individuals evaluated by other intra-island workers.
    propulator._receive_intra_island_individuals()
    propulator._intra_send_cleanup()
    propulator._inter_send_cleanup()
    assert len(propulator.intra_buffers) == 0
    assert len(propulator.inter_buffers) == 0

    if propulator.migration_prob > 0.0:
        if isinstance(propulator, Migrator):
            # TODO sometimes receives duplicates here
            propulator._receive_immigrants()
            propulator.propulate_comm.barrier()

            # Emigration: Final check for emigrants from other intra-island workers to be deactivated.
            propulator._deactivate_emigrants()
            assert propulator._check_emigrants_to_deactivate() is False
            propulator.propulate_comm.barrier()
            assert len(propulator.emigrated) == 0

        elif isinstance(propulator, Pollinator):
            with h5py.File(propulator.checkpoint_path, "a", driver="mpio", comm=propulator.propulate_comm) as f:
                propulator._receive_immigrants(f)
            propulator.propulate_comm.barrier()

            # Immigration: Final check for individuals replaced by other intra-island workers to be deactivated.
            propulator._deactivate_replaced_individuals()
            propulator.propulate_comm.barrier()
            if len(propulator.replaced) > 0:
                log.error(f"{propulator.replaced}")
            assert len(propulator.replaced) == 0


def population_consistency_check(propulator: Propulator) -> None:
    """Check population sizes match expectations for island models."""
    active_pop = propulator._get_active_individuals()
    num_active = len(active_pop)

    # NOTE check that all workers on one island have the same number of individuals
    all_num_active = np.array(propulator.island_comm.allgather(num_active), dtype=int)
    if not np.all(all_num_active == num_active):
        log.error(f"Inconsistent number of total individuals: {all_num_active}")
        assert False

    if propulator.island_counts is not None:
        num_active = int(propulator.propulate_comm.allreduce(num_active / propulator.island_counts[propulator.island_idx]))

    propulator.propulate_comm.barrier()
    # if propulator.propulate_comm.rank == 0:
    #     log.info(
    #         "###########\n# SUMMARY #\n###########\n"
    #         f"Number of currently active individuals is {num_active}.\n"
    #         f"Expected overall number of evaluations is {propulator.generations*propulator.propulate_comm.size}."
    #     )
    # Only double-check number of occurrences of each individual for DEBUG level 2.
    # populations = propulator.island_comm.gather(propulator.population, root=0)
    occurrences, _ = _check_for_duplicates(propulator)
    # if propulator.island_comm.rank == 0:
    #     if propulator._check_intra_island_synchronization(populations):
    #         log.info(f"Island {propulator.island_idx}: Populations among workers synchronized.")
    #     else:
    #         log.info(f"Island {propulator.island_idx}: Populations among workers not synchronized:\n{populations}")
    # log.info(
    #         f"Island {propulator.island_idx}: {len(active_pop)}/{len(propulator.population)} "
    #         f"individuals active ({len(occurrences)} unique)"
    #     )
    propulator.propulate_comm.barrier()


def _check_for_duplicates(propulator: Propulator) -> tuple[list[list[Union[Individual, int]]], list[Individual]]:
    """
    Check for duplicates in current population.

    Parameters
    ----------
    propulator : Propulator
        Propulator to check.

    Returns
    -------
    List[List[propulate.population.Individual | int]]
        The individuals and their occurrences.
    List[propulate.population.Individual]
        The unique individuals in the population.
    """
    active_population = propulator._get_active_individuals()
    unique_inds: list[Individual] = []
    occurrences: list[list[Union[Individual, int]]] = []
    for individual in active_population:
        considered = False
        for ind in unique_inds:
            if individual == ind:
                considered = True
                break
        if not considered:
            num_copies = active_population.count(individual)
            # log.debug(
            #     f"Island {propulator.island_idx} Worker {propulator.island_comm.rank} Generation {propulator.generation}: "
            #     f"{individual} occurs {num_copies} time(s)."
            # )
            unique_inds.append(individual)
            occurrences.append([individual, num_copies])
    return occurrences, unique_inds
