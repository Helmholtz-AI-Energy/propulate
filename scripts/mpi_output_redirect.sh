#!/bin/bash
# adapted from https://www.open-mpi.org/community/lists/users/2012/02/18362.php
# example usage: mpirun -n 8 scripts/parallel_test_filter.sh pytest --with-mpi
ARGS=$@
mkdir -p .logs
if [[ $OMPI_COMM_WORLD_RANK == 0 ]]
then
    $ARGS 2>&1 | tee .logs/$OMPI_COMM_WORLD_RANK.log
else
    $ARGS 1>.logs/$OMPI_COMM_WORLD_RANK.log 2>.logs/$OMPI_COMM_WORLD_RANK.log
fi
