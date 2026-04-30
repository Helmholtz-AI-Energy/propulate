#!/bin/bash
# adapted from https://www.open-mpi.org/community/lists/users/2012/02/18362.php
# example usage: mpirun -n 8 scripts/parallel_test_filter.sh pytest --with-mpi
ARGS=$@
if [[ $OMPI_COMM_WORLD_RANK == 0 ]]
then
    $ARGS
else
    $ARGS 1>/dev/null 2>/dev/null
fi
