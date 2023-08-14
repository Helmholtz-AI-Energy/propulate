if [ "$1" = "clear" ]
then
  rm -rf ./checkpoints
fi
# Options: birastrigin, bisphere, bukin, eggcrate, griewank, himmelblau, keane, leon, quartic, rastrigin, rosenbrock, schwefel, sphere, step

mpirun -n 4 python "$(dirname "$0")"/pso_example.py birastrigin 100
if [ "$1" = "clear" ]
then
  rm -rf ./checkpoints
fi
mpirun -n 4 python "$(dirname "$0")"/pso_example.py bisphere 100
if [ "$1" = "clear" ]
then
  rm -rf ./checkpoints
fi
mpirun -n 4 python "$(dirname "$0")"/pso_example.py bukin 100
if [ "$1" = "clear" ]
then
  rm -rf ./checkpoints
fi
mpirun -n 4 python "$(dirname "$0")"/pso_example.py eggcrate 100
if [ "$1" = "clear" ]
then
  rm -rf ./checkpoints
fi
mpirun -n 4 python "$(dirname "$0")"/pso_example.py griewank 100
if [ "$1" = "clear" ]
then
  rm -rf ./checkpoints
fi
mpirun -n 4 python "$(dirname "$0")"/pso_example.py schwefel 100
if [ "$1" = "clear" ]
then
  rm -rf ./checkpoints
fi
mpirun -n 4 python "$(dirname "$0")"/pso_example.py himmelblau 100
if [ "$1" = "clear" ]
then
  rm -rf ./checkpoints
fi
mpirun -n 4 python "$(dirname "$0")"/pso_example.py keane 100
if [ "$1" = "clear" ]
then
  rm -rf ./checkpoints
fi
mpirun -n 4 python "$(dirname "$0")"/pso_example.py leon 100
if [ "$1" = "clear" ]
then
  rm -rf ./checkpoints
fi
mpirun -n 4 python "$(dirname "$0")"/pso_example.py quartic 100
if [ "$1" = "clear" ]
then
  rm -rf ./checkpoints
fi
mpirun -n 4 python "$(dirname "$0")"/pso_example.py rastrigin 100
if [ "$1" = "clear" ]
then
  rm -rf ./checkpoints
fi
mpirun -n 4 python "$(dirname "$0")"/pso_example.py rosenbrock 100
if [ "$1" = "clear" ]
then
  rm -rf ./checkpoints
fi
mpirun -n 4 python "$(dirname "$0")"/pso_example.py sphere 100
if [ "$1" = "clear" ]
then
  rm -rf ./checkpoints
fi
mpirun -n 4 python "$(dirname "$0")"/pso_example.py step 100
