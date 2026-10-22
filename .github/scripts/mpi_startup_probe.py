"""Temporarily isolate MPI extension loading, initialization, and finalization."""

import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.finalize = False

print("Before MPI extension import", flush=True)
from mpi4py import MPI  # noqa: E402

print("After MPI extension import", flush=True)
print(MPI.Get_library_version(), flush=True)
print("Before MPI initialization", flush=True)
provided = MPI.Init_thread()
print(f"After MPI initialization; thread level={provided}", flush=True)
MPI.Finalize()
print("After MPI finalization", flush=True)
