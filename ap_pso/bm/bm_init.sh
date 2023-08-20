#!/bin/bash
BASE_DIR="/pfs/work7/workspace/scratch/pa1164-propulate_bm_1/async-parallel-pso/"
for I in {0..3}
do
  for FUNCTION in "sphere" "rosenbrock" "step" "quartic" "rastrigin" "griewank" "schwefel" "bisphere" "birastrigin"
  do
    for RACE in {0..4}
    do
      EXECUTION_DIR="${BASE_DIR}/ap_pso/bm/bm_${I}_${FUNCTION}_${RACE}"
      mkdir "$EXECUTION_DIR"
      mkdir "${EXECUTION_DIR}/images"
      NODES=40
      ITERATIONS=2000
      QUEUE="single"
      case "$RACE" in
        0)
          ;;
        1)
          NODES=$((NODES * 2))
          ITERATIONS=$((ITERATIONS / 2))
          QUEUE="multiple"
          ;;
        2)
          NODES=$((NODES * 4))
          ITERATIONS=$((ITERATIONS / 4))
          QUEUE="multiple"
          ;;
        3)
          NODES=$((NODES * 8))
          ITERATIONS=$((ITERATIONS / 8))
          QUEUE="multiple"
          ;;
        4)
          NODES=$((NODES * 16))
          ITERATIONS=$((ITERATIONS / 16))
          QUEUE="multiple"
          ;;
        5)
          ITERATIONS=$((ITERATIONS * 10))
          ;;
        6)
          ITERATIONS=-1
          ;;
        *)
          echo "Error: Race $RACE was called."
          ;;
      esac
      SCRIPT="#!/bin/bash
#SBATCH --nodes=${NODES}
#SBATCH --partition=${QUEUE}
#SBATCH --job-name=${FUNCTION}_${I}_${RACE}
#SBATCH --time=15:00
#SBATCH --mem=10000
#SBATCH --cpus-per-task=40
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pa1164@partner.kit.edu

cd \$(ws_find propulate_bm_1)
ml purge
ml restore propulate
source ${BASE_DIR}/../.venvs/async-parallel-pso/bin/activate

mpirun python -u ${BASE_DIR}/ap_pso/bm/pso_benchmark.py ${FUNCTION} ${ITERATIONS} ${I} ${EXECUTION_DIR}
deactivate
"
      echo "${SCRIPT}" > "${EXECUTION_DIR}/bm_start.sh"
    done
  done
done
