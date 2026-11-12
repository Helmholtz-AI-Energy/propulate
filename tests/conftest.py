"""Keep pytest's temporary-directory housekeeping separate for each MPI rank."""

import os
import tempfile
from pathlib import Path

import pytest
from mpi4py import MPI


def pytest_configure(config: pytest.Config) -> None:
    """Isolate rank-local temporary roots without changing shared MPI fixtures."""
    if MPI.COMM_WORLD.size <= 1:
        return

    rank_directory = f"rank-{MPI.COMM_WORLD.rank}"
    if config.option.basetemp is not None:
        # An explicit base is cleared by pytest; never let ranks clear the same one.
        base = Path(config.option.basetemp)
        base.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(base / rank_directory)
        return

    root = Path(os.environ.get("PYTEST_DEBUG_TEMPROOT", tempfile.gettempdir()))
    root = root / "propulate-mpi" / rank_directory
    root.mkdir(parents=True, exist_ok=True)
    patch = pytest.MonkeyPatch()
    patch.setenv("PYTEST_DEBUG_TEMPROOT", str(root))
    config.add_cleanup(patch.undo)
