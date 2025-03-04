import numpy as np
from mpi4py import MPI

# NOTE this is a test to for large message sizes with Isend instead of Send

world = MPI.COMM_WORLD

data = np.random.rand(10, 10000)
recv_buffer = np.empty((10, 10000), dtype=data.dtype)

reqs = []
for r in range(world.size):
    if r != world.rank:
        print(f"rank {world.rank} sending data to {r}")
        reqs.append(world.Isend([data, MPI.FLOAT], dest=r))

messages_received = 0
while messages_received < world.size - 1:
    probe = True
    while probe:
        stat = MPI.Status()
        probe = world.iprobe(source=MPI.ANY_SOURCE, status=stat)
        if probe:
            print(f"rank {world.rank} receiving data from {stat.Get_source()}")
            world.Recv([recv_buffer, MPI.FLOAT], source=stat.Get_source())
            messages_received += 1

print(f"{len(reqs)} {f'{world.rank}'*8}")
print(MPI.Request.Testsome(reqs))
print(f"{len(reqs)} {f'{world.rank}'*8}")
# for req in reqs:
#     req.Wait()
