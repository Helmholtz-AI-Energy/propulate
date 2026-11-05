import pathlib
import random

import numpy as np
import pytest
from mpi4py import MPI

from propulate import Islands, Propulator
from propulate.population import Individual
from propulate.propagators import Conditional, Propagator
from propulate.propagators.pso import (
    BasicPSO,
    CanonicalPSO,
    ConstrictionPSO,
    InitUniformPSO,
    VelocityClampingPSO,
)
from propulate.utils.benchmark_functions import get_function_search_space, sphere

limits = get_function_search_space("sphere")[1]
rank = MPI.COMM_WORLD.rank
rng = random.Random(42 + rank)


@pytest.fixture(
    params=[
        BasicPSO(
            inertia=0.729,
            c_cognitive=1.49445,
            c_social=1.49445,
            rank=rank,
            limits=limits,
            rng=rng,
        ),
        VelocityClampingPSO(
            inertia=0.729,
            c_cognitive=1.49445,
            c_social=1.49445,
            rank=rank,
            limits=limits,
            rng=rng,
            v_limits=0.6,
        ),
        ConstrictionPSO(
            c_cognitive=2.05,
            c_social=2.05,
            rank=rank,
            limits=limits,
            rng=rng,
        ),
        CanonicalPSO(c_cognitive=2.05, c_social=2.05, rank=rank, limits=limits, rng=rng),
    ]
)
def pso_propagator(request: pytest.FixtureRequest) -> Propagator:
    """Iterate over PSO propagator variants."""
    return request.param


@pytest.mark.mpi
def test_pso(pso_propagator: Propagator, mpi_tmp_path: pathlib.Path) -> None:
    """
    Test single worker using Propulator to optimize a benchmark function using the default genetic propagator.

    Parameters
    ----------
    pso_propagator : BasicPSO
        The PSO propagator variant to test.
    mpi_tmp_path : pathlib.Path
        The temporary checkpoint directory.
    """
    # Set up evolutionary operator.
    init = InitUniformPSO(limits, rng=rng, rank=rank)
    propagator = Conditional(1, pso_propagator, init)

    # Set up propulator performing actual optimization.
    propulator = Propulator(
        loss_fn=sphere,
        propagator=propagator,
        rng=rng,
        generations=10,
        checkpoint_path=mpi_tmp_path,
    )

    # Run optimization and print summary of results.
    propulator.propulate()


def test_basic_pso_set_worker_context_updates_rank_and_world_size() -> None:
    """Worker context API should update PSO rank semantics."""
    pso = BasicPSO(
        inertia=0.729,
        c_cognitive=1.49445,
        c_social=1.49445,
        rank=99,
        limits=limits,
        rng=random.Random(1),
    )
    pso.set_worker_context(worker_rank=5, worker_size=24)
    assert pso.rank == 5
    assert pso.world_size == 24


def test_basic_pso_mismatched_rank_can_fail_with_empty_personal_history() -> None:
    """Document pre-fix failure mode: mismatched rank -> empty own_p -> min([]) crash."""
    pso = BasicPSO(
        inertia=0.729,
        c_cognitive=1.49445,
        c_social=1.49445,
        rank=99,  # intentionally mismatched
        limits={"a": (-5.12, 5.12), "b": (-5.12, 5.12)},
        rng=random.Random(3),
    )
    inds = []
    for r in (0, 1):
        ind = Individual(
            np.array([0.1 + r, 0.2 + r], dtype=float),
            {"a": (-5.12, 5.12), "b": (-5.12, 5.12)},
            velocity=np.array([0.01, 0.01], dtype=float),
            generation=1,
            rank=r,
        )
        ind.loss = float(r + 1)
        inds.append(ind)
    with pytest.raises(ValueError, match="min\\(\\) arg is an empty sequence"):
        pso._prepare_data(inds)


def _parallel_sphere(params, comm: MPI.Comm = MPI.COMM_SELF) -> float:
    """Simple 2D parallel loss for ranks_per_worker=2 integration test."""
    vals = np.array(list(params.values()), dtype=float)
    if comm != MPI.COMM_SELF:
        term = float(vals[comm.rank] ** 2)
        return float(comm.allreduce(term))
    return float(np.sum(vals**2))


@pytest.mark.mpi(min_size=4)
def test_pso_multi_rank_workers_no_rank_mismatch_crash(mpi_tmp_path: pathlib.Path) -> None:
    """Regression: with ranks_per_worker>1, PSO should run without empty-history crash."""
    local_limits = {"a": (-5.12, 5.12), "b": (-5.12, 5.12)}
    local_rank = MPI.COMM_WORLD.rank
    local_rng = random.Random(77 + local_rank)

    pso = BasicPSO(
        inertia=0.729,
        c_cognitive=1.49445,
        c_social=1.49445,
        rank=local_rank,  # global rank on purpose; worker context API must align it
        limits=local_limits,
        rng=local_rng,
    )
    init = InitUniformPSO(local_limits, rng=local_rng, rank=local_rank)
    propagator = Conditional(1, pso, init)

    islands = Islands(
        loss_fn=_parallel_sphere,
        propagator=propagator,
        rng=local_rng,
        generations=5,
        num_islands=1,
        migration_probability=0.0,
        pollination=False,
        checkpoint_path=mpi_tmp_path,
        ranks_per_worker=2,
    )
    islands.propulate(logging_interval=50, debug=0)
