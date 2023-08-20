#!/bin/bash
BASE_DIR="/pfs/work7/workspace/scratch/pa1164-propulate_bm_1/async-parallel-pso/"
for I in {0..3}
do
  for FUNCTION in "sphere" "rosenbrock" "step" "quartic" "rastrigin" "griewank" "schwefel" "bisphere" "birastrigin"
  do
    for RACE in {0..4}
    do
      EXECUTION_DIR="${BASE_DIR}/ap_pso/bm/bm_${I}_${FUNCTION}_${RACE}/bm_start.sh"
      NODES=1
      QUEUE="single"
      case "$RACE" in
        0)
          ;;
        1)
          NODES=$((NODES * 2))
          QUEUE="multiple"
          ;;
        2)
          NODES=$((NODES * 4))
          QUEUE="multiple"
          ;;
        3)
          NODES=$((NODES * 8))
          QUEUE="multiple"
          ;;
        4)
          NODES=$((NODES * 16))
          QUEUE="multiple"
          ;;
        5)
          ;;
        6)
          ;;
        *)
          echo "Error: Race $RACE was called."
          exit
          ;;
      esac
      sbatch -p "${QUEUE}" -N "${NODES}" "${EXECUTION_DIR}"
    done
  done
done
