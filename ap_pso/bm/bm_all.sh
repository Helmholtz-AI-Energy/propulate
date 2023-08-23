#!/bin/bash
BASE_DIR="/pfs/work7/workspace/scratch/pa1164-propulate_bm_1/async-parallel-pso"
for RACE in {0..4}
do
  NODES=$(( 2 ** RACE ))
  TASKS=$(( 64 * NODES ))
  ITERATIONS=$(( 2000 / NODES ))
  QUEUE="multiple_il"
  if [[ $RACE -eq 0 ]]
  then
    NODES=2
  fi
  if [[ $RACE -eq 4 ]]
  then
    QUEUE="multiple_il"
  fi
  SCRIPT="#!/bin/bash
#SBATCH --nodes=${NODES}
#SBATCH --ntasks=${TASKS}
#SBATCH --partition=${QUEUE}
#SBATCH --job-name=\"all_${RACE}\"
#SBATCH --time=4:00:00
#SBATCH --mem=40000
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pa1164@partner.kit.edu

cd \$(ws_find propulate_bm_1)
ml purge
ml restore propulate
source ${BASE_DIR}/../.venvs/async-parallel-pso/bin/activate
"
  for FUNCTION in "sphere" "rosenbrock" "step" "quartic" "rastrigin" "griewank" "schwefel" "bisphere" "birastrigin"
  do
    DIRNAME="bm_P_${FUNCTION}_${RACE}"
    RESULTS_DIR="${BASE_DIR}/ap_pso/bm/results2/${DIRNAME}"
    mkdir "$RESULTS_DIR"

    SCRIPT+="mpirun --bind-to core --map-by core python -u ${BASE_DIR}/scripts/islands_example.py -f ${FUNCTION} -g ${ITERATIONS} -ckpt ${RESULTS_DIR} -i 1 -migp 0
"
  done
  for FUNCTION in "sphere" "rosenbrock" "step" "quartic" "rastrigin" "griewank" "schwefel" "bisphere" "birastrigin"
  do
    DIRNAME="bm_H_${FUNCTION}_${RACE}"
    RESULTS_DIR="${BASE_DIR}/ap_pso/bm/results2/${DIRNAME}"
    mkdir "$RESULTS_DIR"

    SCRIPT+="mpirun --bind-to core --map-by core python -u ${BASE_DIR}/ap_pso/bm/hyppopy_benchmark.py ${FUNCTION} ${ITERATIONS} ${RESULTS_DIR}
"
  done
  for FUNCTION in "sphere" "rosenbrock" "step" "quartic" "rastrigin" "griewank" "schwefel" "bisphere" "birastrigin"
  do
    for PSO in {0..3}
    do
      DIRNAME="bm_${PSO}_${FUNCTION}_${RACE}"
      RESULTS_DIR="${BASE_DIR}/ap_pso/bm/results2/${DIRNAME}"
      mkdir "$RESULTS_DIR"

      SCRIPT+="mpirun --bind-to core --map-by core python -u ${BASE_DIR}/ap_pso/bm/pso_benchmark.py ${FUNCTION} ${ITERATIONS} ${PSO} ${RESULTS_DIR}
"
    done
  done
  SCRIPT+="deactivate
"
  FILE="${BASE_DIR}/ap_pso/bm/start_bm_A_${RACE}.sh"
  echo "${SCRIPT}" > "${FILE}"
  sbatch -p "${QUEUE}" -N "${NODES}" -n "${TASKS}" --cpus-per-task 1 "${FILE}"
done
