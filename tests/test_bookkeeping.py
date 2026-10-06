"""
Deterministic regression tests for the population bookkeeping of the island models.

Workers of an island learn about the same events (an individual was bred, migrated in, migrated out, was replaced)
through independent messages that MPI delivers in no particular order relative to each other. These tests replay the
events in every possible order on a single worker, without any MPI traffic, and check that the resulting population is
the same.
"""

import copy
import itertools
import math
import pathlib
import random
from typing import Callable, Dict, List, Tuple

import h5py
import numpy as np
import pytest
from mpi4py import MPI

from propulate import Migrator, Pollinator, Propulator
from propulate.population import Individual
from propulate.propagators import SelectMax, SelectMin
from propulate.utils import get_default_propagator
from propulate.utils.benchmark_functions import get_function_search_space

ISLAND_SIZES = np.array([2, 2])  # Two islands with two workers each; the worker under test is worker 0 on island 0.


def _make_propulator(kind: type, path: pathlib.Path, generations: int = 10, migration_prob: float = 0.5) -> Propulator:
    """Set up a propulator of the given kind that plays the role of worker 0 on island 0 without MPI peers."""
    rng = random.Random(42)
    benchmark_function, limits = get_function_search_space("sphere")
    propagator = get_default_propagator(pop_size=4, limits=limits, rng=rng)
    return kind(
        loss_fn=benchmark_function,
        propagator=propagator,
        rng=rng,
        generations=generations,
        island_idx=0,
        island_comm=MPI.COMM_SELF,
        propulate_comm=MPI.COMM_SELF,
        checkpoint_path=path,
        migration_topology=np.array([[0, 1], [1, 0]]),
        migration_prob=migration_prob,
        emigration_propagator=SelectMin,
        immigration_propagator=SelectMax,
        island_displs=np.array([0, 2]),
        island_sizes=ISLAND_SIZES,
    )


def _individual(limits: Dict, island: int, island_rank: int, generation: int, loss: float) -> Individual:
    """Create an evaluated individual with the given identity."""
    ind = Individual({"a": 0.1 * generation, "b": 0.2 * island_rank}, limits)
    ind.island = island
    ind.island_rank = island_rank
    ind.prop_rank = int(np.cumsum(np.concatenate(([0], ISLAND_SIZES)))[island] + island_rank)
    ind.generation = generation
    ind.loss = loss
    return ind


def _placeholder(ind: Individual) -> Individual:
    """Create the unevaluated placeholder of an individual as loaded from a checkpoint of a run in progress."""
    placeholder = copy.deepcopy(ind)
    placeholder.loss = float("nan")
    placeholder.active = 0
    return placeholder


def _key(ind: Individual) -> Tuple[int, int, int]:
    return (ind.island, ind.island_rank, ind.generation)


@pytest.fixture
def rank_tmp_path(mpi_tmp_path: pathlib.Path) -> pathlib.Path:
    """Checkpoint directory private to this rank: the propulators under test do not share a checkpoint file."""
    path = mpi_tmp_path / f"rank_{MPI.COMM_WORLD.rank}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _check_all_orders(
    events: List[Callable[[Propulator], None]],
    make: Callable[[], Propulator],
    check: Callable[[Propulator], None],
) -> None:
    """Replay the events in every order on a fresh propulator and run the check on the result."""
    for order in itertools.permutations(range(len(events))):
        propulator = make()
        for idx in order:
            events[idx](propulator)
        try:
            check(propulator)
        except AssertionError as e:
            raise AssertionError(f"Failed for event order {order}: {e}") from e


@pytest.mark.parametrize("from_placeholder", [False, True])
def test_migrator_bookkeeping_is_order_independent(rank_tmp_path: pathlib.Path, from_placeholder: bool) -> None:
    """
    An individual bred by another worker that migrates away, comes back and leaves again.

    The breeding message may arrive after the deactivation notice or even after the individual's return from the other
    island, and copies carry the sender's bookkeeping state, which must be ignored.
    """
    propulator = _make_propulator(Migrator, rank_tmp_path)
    limits = propulator.propagator.limits
    ind = _individual(limits, island=0, island_rank=1, generation=3, loss=1.5)

    def make() -> Propulator:
        propulator.population = {}
        if from_placeholder:
            propulator.population[_key(ind)] = _placeholder(ind)
        return propulator

    def bred(p: Propulator) -> None:
        p._process_bred_individual(copy.deepcopy(ind))

    def emigrated(p: Propulator) -> None:
        p._process_emigrants([copy.deepcopy(ind)])

    def returned(p: Propulator) -> None:
        stale = copy.deepcopy(ind)
        stale.active = 2  # The sending worker's counter is irrelevant to the receiver.
        stale.migrator_island_rank = 7
        p._process_immigrants([stale])

    def check_active(p: Propulator) -> None:
        assert list(p.population) == [_key(ind)]
        entry = p.population[_key(ind)]
        assert entry.active == 1 and entry.loss == ind.loss
        assert entry.migrator_island_rank == 1  # The breeding worker is responsible for a native.
        assert p._get_active_individuals() == [entry]

    def check_inactive(p: Propulator) -> None:
        assert list(p.population) == [_key(ind)]
        entry = p.population[_key(ind)]
        assert entry.active == 0 and entry.loss == ind.loss
        assert p._get_active_individuals() == []

    _check_all_orders([bred, emigrated, returned], make, check_active)
    _check_all_orders([bred, emigrated, returned, emigrated], make, check_inactive)


def test_migrator_responsibility_is_independent_of_copies(rank_tmp_path: pathlib.Path) -> None:
    """Exactly one worker per island is responsible for an individual, regardless of which copy it holds."""
    propulator = _make_propulator(Migrator, rank_tmp_path)
    limits = propulator.propagator.limits
    native = _individual(limits, island=0, island_rank=1, generation=3, loss=1.5)
    immigrant = _individual(limits, island=1, island_rank=1, generation=4, loss=2.5)
    for ind in (native, immigrant):
        stale = copy.deepcopy(ind)
        stale.migrator_island_rank = 0  # A copy claiming that this worker is responsible.
        propulator._process_immigrants([stale])
    assert propulator._responsible_rank(propulator.population[_key(native)]) == 1
    assert propulator._responsible_rank(propulator.population[_key(immigrant)]) == 4 % ISLAND_SIZES[0] == 0
    other = _individual(limits, island=1, island_rank=0, generation=5, loss=2.5)
    assert propulator._responsible_rank(other) == 5 % ISLAND_SIZES[0] == 1
    assert propulator.population[_key(native)].migrator_island_rank == 1


@pytest.mark.parametrize("from_placeholder", [False, True])
def test_pollinator_bookkeeping_is_order_independent(rank_tmp_path: pathlib.Path, from_placeholder: bool) -> None:
    """
    A native bred by another worker is pollinated to the other island, comes back as a copy and is replaced at home.

    Copies of an already known individual and a late breeding message must not reactivate a replaced individual.
    """
    propulator = _make_propulator(Pollinator, rank_tmp_path)
    limits = propulator.propagator.limits
    ind = _individual(limits, island=0, island_rank=1, generation=3, loss=1.5)

    def make() -> Propulator:
        propulator.population = {}
        if from_placeholder:
            propulator.population[_key(ind)] = _placeholder(ind)
        return propulator

    def bred(p: Propulator) -> None:
        p._process_bred_individual(copy.deepcopy(ind))

    def replaced(p: Propulator) -> None:
        p._process_replaced_individuals([copy.deepcopy(ind)])

    def returned(p: Propulator) -> None:
        with h5py.File(p.checkpoint_path, "a", driver="mpio", comm=MPI.COMM_SELF) as f:
            p._process_immigrants([copy.deepcopy(ind)], f)

    def check_active(p: Propulator) -> None:
        assert list(p.population) == [_key(ind)]
        entry = p.population[_key(ind)]
        assert entry.active == 1 and entry.loss == ind.loss

    def check_inactive(p: Propulator) -> None:
        assert list(p.population) == [_key(ind)]
        entry = p.population[_key(ind)]
        assert entry.active == 0 and entry.loss == ind.loss

    _check_all_orders([bred, returned], make, check_active)
    _check_all_orders([bred, returned, replaced], make, check_inactive)


def test_pollinator_replaces_once_per_new_immigrant(rank_tmp_path: pathlib.Path) -> None:
    """The responsible worker replaces one individual per immigrant that is new to the island, not per copy received."""
    propulator = _make_propulator(Pollinator, rank_tmp_path)
    limits = propulator.propagator.limits
    natives = [_individual(limits, island=0, island_rank=0, generation=g, loss=float(g)) for g in range(4)]
    for native in natives:
        propulator.population[_key(native)] = native
    # Generation 4 % 2 == 0, so this worker is responsible for the immigrant on island 0.
    immigrant = _individual(limits, island=1, island_rank=1, generation=4, loss=0.5)

    with h5py.File(propulator.checkpoint_path, "a", driver="mpio", comm=MPI.COMM_SELF) as f:
        propulator._process_immigrants([copy.deepcopy(immigrant)], f)
        assert len(propulator._get_active_individuals()) == len(natives)  # One in, one out.
        assert propulator.population[_key(immigrant)].active == 1
        assert propulator.population[_key(natives[3])].active == 0  # The worst one was replaced.
        assert f["0"]["0"]["active_on_island"][3, 0] == 0
        assert f["1"]["1"]["active_on_island"][4, 0] == 1

        # Copies of the same individual from other senders do not trigger further replacements.
        propulator._process_immigrants([copy.deepcopy(immigrant)], f)
        propulator._process_immigrants([copy.deepcopy(immigrant)], f)
        assert len(propulator._get_active_individuals()) == len(natives)
        assert [ind.active for ind in natives] == [1, 1, 1, 0]

        # A replaced individual is not reactivated by another copy of it.
        propulator._process_immigrants([copy.deepcopy(natives[3])], f)
        assert natives[3].active == 0
        assert f["0"]["0"]["active_on_island"][3, 0] == 0


def test_placeholder_is_completed_by_any_evaluated_copy(rank_tmp_path: pathlib.Path) -> None:
    """A placeholder loaded from a checkpoint is completed with the data of whichever evaluated copy arrives first."""
    propulator = _make_propulator(Migrator, rank_tmp_path)
    limits = propulator.propagator.limits
    ind = _individual(limits, island=0, island_rank=1, generation=3, loss=1.5)
    propulator.population[_key(ind)] = _placeholder(ind)

    entry, created = propulator._upsert_individual(copy.deepcopy(ind))
    assert created and entry.loss == ind.loss and entry.active == 0
    assert propulator.population[_key(ind)] is entry
    same, created = propulator._upsert_individual(copy.deepcopy(ind))
    assert not created and same is entry

    # An unevaluated copy never replaces an evaluated entry.
    again, created = propulator._upsert_individual(_placeholder(ind))
    assert not created and again is entry and not math.isnan(entry.loss)


def test_restart_does_not_breed_a_finished_generation_again(rank_tmp_path: pathlib.Path) -> None:
    """
    A worker resumes after the last generation it finished even if that individual is no longer in its population.

    The last individual a worker evaluated may have migrated away or been replaced by the time the run stops. It is
    then not loaded from the checkpoint on the worker's island, but its generation must not be bred again: that would
    give a second individual the same identity and overwrite the first one's checkpoint row.
    """

    def make(generations: int) -> Propulator:
        rng = random.Random(42)
        benchmark_function, limits = get_function_search_space("sphere")
        propagator = get_default_propagator(pop_size=4, limits=limits, rng=rng)
        return Propulator(
            loss_fn=benchmark_function,
            propagator=propagator,
            rng=rng,
            generations=generations,
            island_comm=MPI.COMM_SELF,
            propulate_comm=MPI.COMM_SELF,
            checkpoint_path=rank_tmp_path,
        )

    first = make(10)
    first.propulate()
    last_key = (0, 0, 9)
    position = first.population[last_key].position.copy()
    del first
    with h5py.File(rank_tmp_path / "ckpt.hdf5", "r+") as f:
        assert f["generations"][0] == 9
        f["0"]["0"]["active_on_island"][9, 0] = 0  # Pretend the last individual is no longer active on the island.

    second = make(12)
    assert last_key not in second.population
    assert second.generation == 10
    second.propulate()
    assert sorted(second.population) == [(0, 0, g) for g in range(9)] + [(0, 0, 10), (0, 0, 11)]
    with h5py.File(rank_tmp_path / "ckpt.hdf5", "r") as f:
        assert f["generations"][0] == 11
        assert np.allclose(f["0"]["0"]["position"][9, 0], position, rtol=1e-5)
        assert f["0"]["0"]["active_on_island"][9, 0] == 0
