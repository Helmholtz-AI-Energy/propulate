"""Verify rank-local pytest directories and shared pytest-mpi directories."""

from pathlib import Path

import pytest
from mpi4py import MPI


def test_mpi_temporary_directories(tmp_path: Path, mpi_tmp_path: Path, pytestconfig: pytest.Config) -> None:
    """Keep housekeeping isolated while all ranks can write to the shared path."""
    comm = MPI.COMM_WORLD
    paths = comm.allgather((str(tmp_path), str(mpi_tmp_path)))
    assert len({local for local, shared in paths}) == comm.size
    assert len({shared for local, shared in paths}) == 1
    assert str(mpi_tmp_path) == paths[0][0]

    if comm.size > 1:
        # Checking the root as well as the base detects a shared cleanup namespace.
        root = tmp_path.parent if pytestconfig.option.basetemp is not None else tmp_path.parents[2]
        roots = comm.allgather(str(root))
        assert len(set(roots)) == comm.size

    (mpi_tmp_path / f"worker-{comm.rank}.txt").write_text(str(comm.rank))
    comm.barrier()
    contents = [(mpi_tmp_path / f"worker-{rank}.txt").read_text() for rank in range(comm.size)]
    assert contents == [str(rank) for rank in range(comm.size)]
