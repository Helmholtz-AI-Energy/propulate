import logging
import math
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np

from propulate import Migrator, Pollinator, Propulator
from propulate.population import Individual

log = logging.getLogger(__name__)

Key = Tuple[int, int, int]


def final_synch(propulator: Propulator) -> None:
    """
    Perform final synchronization on completion of optimization.

    ``Propulator.propulate`` already receives all outstanding messages before returning, so this is a no-op that
    verifies nothing is left in flight.
    """
    log.debug("Starting final sync.")
    with h5py.File(propulator.checkpoint_path, "a", driver="mpio", comm=propulator.propulate_comm) as f:
        propulator._final_synchronization(f)
    assert len(propulator.send_requests) == 0
    assert len(propulator.send_buffers) == 0
    propulator.propulate_comm.barrier()


def _key(ind: Individual) -> Key:
    """Get the identity (birth island, birth island rank, generation) of an individual."""
    return (int(ind.island), int(ind.island_rank), int(ind.generation))


def _content(ind: Individual) -> Tuple:
    """Get a comparable summary of an individual's content (loss, position, velocity)."""
    velocity = None if ind.velocity is None else tuple(np.round(ind.velocity, 5).tolist())
    return (round(float(ind.loss), 5), tuple(np.round(ind.position, 5).tolist()), velocity)


def _active(propulator: Propulator) -> Dict[Key, Tuple]:
    """Get the content of all active individuals by identity."""
    return {_key(ind): _content(ind) for ind in propulator.population.values() if ind.active > 0}


def population_consistency_check(propulator: Propulator) -> None:
    """
    Check that the workers' populations match each other, the checkpoint, and the expected sizes.

    Assumes a completed run in which every worker evaluated ``propulator.generations`` generations.
    """
    active = _active(propulator)
    active_keys = sorted(active)

    # NOTE check that all workers on one island hold exactly the same active individuals (by identity and content, not
    # by order)
    all_active = propulator.island_comm.allgather(active)
    for rank, other in enumerate(all_active):
        if other != all_active[0]:
            keys, keys0 = set(other), set(all_active[0])
            log.error(
                f"Island {propulator.island_idx}: active population of worker {rank} differs from worker 0: "
                f"only on {rank}: {sorted(keys - keys0)}, only on 0: {sorted(keys0 - keys)}, "
                f"different content: {sorted(k for k in keys & keys0 if other[k] != all_active[0][k])}"
            )
            assert False

    # NOTE after all messages have been processed, every entry is either active once or inactive, and every active
    # individual is evaluated
    assert all(ind.active in (0, 1) for ind in propulator.population.values()), [
        ind for ind in propulator.population.values() if ind.active not in (0, 1)
    ]
    assert all(not math.isnan(ind.loss) for ind in propulator.population.values() if ind.active > 0)

    # NOTE check that the active population matches the island's column of ``active_on_island`` in the checkpoint
    propulator.propulate_comm.barrier()  # All writes to the checkpoint have completed and the file is closed.
    checkpoint_keys = []
    with h5py.File(propulator.checkpoint_path, "r", driver=None) as f:
        for island in range(len(propulator.island_sizes)):
            for island_rank in range(propulator.island_sizes[island]):
                active_on_island = f[f"{island}"][f"{island_rank}"]["active_on_island"][:, propulator.island_idx]
                losses = f[f"{island}"][f"{island_rank}"]["loss"][:]
                for generation in np.flatnonzero(active_on_island):
                    assert not np.isnan(losses[generation])
                    checkpoint_keys.append((int(island), int(island_rank), int(generation)))
    if sorted(checkpoint_keys) != active_keys:
        log.error(
            f"Island {propulator.island_idx} Worker {propulator.island_comm.rank}: checkpoint does not match population: "
            f"only in checkpoint: {sorted(set(checkpoint_keys) - set(active_keys))}, "
            f"only in population: {sorted(set(active_keys) - set(checkpoint_keys))}"
        )
        assert False

    # NOTE check the distribution of individuals over islands
    island_displs = np.cumsum(np.concatenate((np.array([0]), propulator.island_sizes)))
    all_keys = propulator.propulate_comm.allgather(active_keys)
    island_keys = [all_keys[displ] for displ in island_displs[:-1]]  # Workers of an island hold the same set.
    bred: List[Key] = [
        (island, island_rank, generation)
        for island in range(len(propulator.island_sizes))
        for island_rank in range(propulator.island_sizes[island])
        for generation in range(propulator.generations)
    ]
    if isinstance(propulator, Migrator):
        # NOTE with real migration, each individual is active on exactly one island
        everywhere = sorted(key for keys in island_keys for key in keys)
        assert everywhere == bred, (
            f"duplicated: {sorted({k for k in everywhere if everywhere.count(k) > 1})}, "
            f"missing: {sorted(set(bred) - set(everywhere))}, unexpected: {sorted(set(everywhere) - set(bred))}"
        )
    elif isinstance(propulator, Pollinator):
        # NOTE with pollination, each immigrant replaces one individual, so each island holds as many as it bred
        sizes = [len(keys) for keys in island_keys]
        expected = [int(x * propulator.generations) for x in propulator.island_sizes]
        assert sizes == expected, f"active population sizes per island {sizes} != expected {expected}"
        assert all(set(keys) <= set(bred) for keys in island_keys)
    elif len(propulator.island_sizes) > 1:
        raise ValueError("Unknown Propulator type.")
    else:
        # NOTE a single island may also be checked right after loading a checkpoint, before all generations exist
        assert set(island_keys[0]) <= set(bred)

    propulator.propulate_comm.barrier()


def assert_populations_match(expected: Iterable[Individual], actual: Iterable[Individual], subset: bool = False) -> None:
    """
    Check that two collections hold the same individuals irrespective of order.

    Individuals are compared by identity (birth island, birth island rank, generation) and content (position, loss,
    velocity).

    Parameters
    ----------
    expected : Iterable[propulate.population.Individual]
        The expected individuals.
    actual : Iterable[propulate.population.Individual]
        The actual individuals.
    subset : bool, optional
        If True, ``actual`` may hold additional individuals. Default is False.
    """

    def as_dict(individuals: Iterable[Individual]) -> Dict[Key, Individual]:
        individuals = list(individuals)
        result = {_key(ind): ind for ind in individuals}
        assert len(result) == len(individuals), "Duplicate identities in population"
        return result

    expected_dict, actual_dict = as_dict(expected), as_dict(actual)
    missing = sorted(set(expected_dict) - set(actual_dict))
    assert len(missing) == 0, f"Missing individuals: {missing}"
    if not subset:
        extra = sorted(set(actual_dict) - set(expected_dict))
        assert len(extra) == 0, f"Unexpected individuals: {extra}"
    for key, ind in expected_dict.items():
        other = actual_dict[key]
        assert np.allclose(ind.position, other.position, rtol=1e-5), f"Position of {key} differs: {ind} vs {other}"
        if ind.velocity is None or other.velocity is None:
            assert ind.velocity is None and other.velocity is None, f"Velocity of {key} differs: {ind} vs {other}"
        else:
            assert np.allclose(ind.velocity, other.velocity, rtol=1e-5), f"Velocity of {key} differs: {ind} vs {other}"
        if np.isnan(ind.loss):
            assert np.isnan(other.loss), f"Loss of {key} differs: {ind} vs {other}"
        else:
            assert math.isclose(float(ind.loss), float(other.loss), rel_tol=1e-5), f"Loss of {key} differs: {ind} vs {other}"
