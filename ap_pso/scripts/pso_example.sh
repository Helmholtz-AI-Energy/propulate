if [ "$1" = "clear" ]
then
  rm -rf ./checkpoints
fi
mpirun -n 4 python "$(dirname "$0")"/pso_example.py sphere 100
