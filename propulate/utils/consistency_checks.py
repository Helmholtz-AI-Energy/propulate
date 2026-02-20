import logging

import h5py
import numpy as np

from propulate import Migrator, Pollinator, Propulator

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
            propulator._receive_intra_island_individuals()
            propulator._receive_immigrants()
            propulator.propulate_comm.barrier()

            # Emigration: Final check for emigrants from other intra-island workers to be deactivated.
            propulator._deactivate_emigrants()
            assert propulator._check_emigrants_to_deactivate() is False
            propulator.propulate_comm.barrier()
            assert len(propulator.emigrated) == 0

        elif isinstance(propulator, Pollinator):
            propulator._receive_intra_island_individuals()
            with h5py.File(propulator.checkpoint_path, "a", driver="mpio", comm=propulator.propulate_comm) as f:
                propulator._receive_immigrants(f)
            propulator.propulate_comm.barrier()

            # Immigration: Final check for individuals replaced by other intra-island workers to be deactivated.
            propulator._deactivate_replaced_individuals()
            propulator.propulate_comm.barrier()
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

    if propulator.island_sizes is not None:
        num_active = int(propulator.propulate_comm.allreduce(num_active / propulator.island_sizes[propulator.island_idx]))

    propulator.propulate_comm.barrier()
