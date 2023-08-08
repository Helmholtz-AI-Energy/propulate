#!/bin/bash

# TODO --cpu-bind=cores output-name srun

# Use GNU getopt to parse customized command-line arguments
CMA_COMMAND=$(getopt -o: --long generation:,pop_size:,num_isles:,migration_probability:,exploration:,select_worst_all_time:,dimension:,mainpath:,directory:,nodes:,cpus: -n 'getopt_errors_cma' -- "$@")

if [ $? != 0 ] ; then echo "Command-line error occurred. Terminating..." >&2 ; exit 1 ; fi

eval set -- "$CMA_COMMAND"

# Default values
GENERATION=100
POP_SIZE=""
NUM_ISLES=2
MIGRATION_PROBABILITY=0.9
EXPLORATION=false
SELECT_WORST_ALL_TIME=false
DIMENSION=2
MAIN_PATH="propulate/scripts/cma_es_benchmark.py"
DIRECTORY="benchmark"
NODES=1
CPUS_PER_NODE=8
#WORKERS=0 # TODO

# Iterate through parsed options
while true; do
  case "$1" in
    --generation ) GENERATION="$2"; shift 2 ;;
    --pop_size ) POP_SIZE="$2"; shift 2 ;;
    --num_isles ) NUM_ISLES="$2"; shift 2 ;;
    --migration_probability ) MIGRATION_PROBABILITY="$2"; shift 2 ;;
    --exploration ) EXPLORATION="$2"; shift 2 ;;
    --select_worst_all_time ) SELECT_WORST_ALL_TIME="$2"; shift 2 ;;
    --dimension ) DIMENSION="$2"; shift 2 ;;
    --mainpath ) MAIN_PATH="$2"; shift 2 ;;
    --directory ) DIRECTORY="$2"; shift 2 ;;
    --nodes ) NODES="$2"; shift 2 ;;
    --cpus ) CPUS_PER_NODE="$2"; shift 2 ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

NTASKS=$(($NODES * $CPUS_PER_NODE))

JOB_NAME=$((${generation}gens_${dimension}dims_${num_isles}isles_${NTASKS}cpus))

# Construct SBATCH base command
BASE_SCRIPT="#!/bin/bash

#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${JOB_NAME}.out
#SBATCH --time=48:00:00
#SBATCH --partition=dev_multiple
#SBATCH --nodes=$NODES
#SBATCH --ntasks=$NTASKS
#SBATCH --gres=cpu:$CPUS_PER_NODE
#SBATCH --reservation=BA-Arbeit

module purge
module load compiler/gnu/11.2
module load mpi/openmpi/4.1

source venv3.9/bin/activate

srun python -u $MAIN_PATH \
--nodes=$NODES \
--ntasks=$NTASKS
--generation $GENERATION \
--pop_size $POP_SIZE \
--migration_probability $MIGRATION_PROBABILITY \
--dimension $DIMENSION"

# Check if the directory already exists
if [ -d "$DIRECTORY" ]; then
  echo "$DIRECTORY directory already exists."
  exit 1
fi

PARAMETER_SUMMARY="generation: ${GENERATION}
pop_size: ${POP_SIZE}
nodes: ${NODES}
ntasks: ${NTASKS}
num_isles: ${NUM_ISLES}
migration_probability: ${MIGRATION_PROBABILITY}
exploration: ${EXPLORATION}
select_worst_all_time: ${SELECT_WORST_ALL_TIME}
dimension: ${DIMENSION}"

# make directories
mkdir -p "$DIRECTORY/default_prop/migr"
mkdir -p "$DIRECTORY/default_prop/polli"
mkdir -p "$DIRECTORY/cma/basic"
mkdir -p "$DIRECTORY/cma/active"
mkdir -p "$DIRECTORY/cma_island/active/migr"
mkdir -p "$DIRECTORY/cma_island/basic/migr"
mkdir -p "$DIRECTORY/cma_island/active/polli"
mkdir -p "$DIRECTORY/cma_island/basic/polli"

echo "$PARAMETER_SUMMARY" > "$DIRECTORY/parameters.txt"

# TODO do one function with multiple dimensions
# Script for each configuration
FUNCTIONS=("sphere" "rosenbrock" "step" "quartic" "rastrigin" "griewank" "schwefel")

PREFIX_SCRIPT_DEFAULT="$BASE_SCRIPT --num_isles $NUM_ISLES --default_propulate true"
PREFIX_SCRIPT_CMA="$BASE_SCRIPT --num_isles 1 --default_propulate false --exploration $EXPLORATION --select_worst_all_time $SELECT_WORST_ALL_TIME"
PREFIX_SCRIPT_CMA_ISLAND="$BASE_SCRIPT --num_isles $NUM_ISLES --default_propulate false --exploration $EXPLORATION --select_worst_all_time $SELECT_WORST_ALL_TIME"

for func in "${FUNCTIONS[@]}"; do
  # Default Propulate Benchmark
  echo "$PREFIX_SCRIPT_DEFAULT --function \"$func\" --pollination false" > "$DIRECTORY/default_prop/migr/default_prop_migr_${func}.sh"
  echo "$PREFIX_SCRIPT_DEFAULT --function \"$func\" --pollination true" > "$DIRECTORY/default_prop/polli/default_prop_polli_${func}.sh"

  # Basic CMA
  echo "$PREFIX_SCRIPT_CMA --function \"$func\" --active_cma false" > "$DIRECTORY/cma/basic/cma_basic_${func}.sh"
  # Active CMA
  echo "$PREFIX_SCRIPT_CMA --function \"$func\" --active_cma true" > "$DIRECTORY/cma/active/cma_active_${func}.sh"

  # CMA + Island Benchmark
  echo "$PREFIX_SCRIPT_CMA_ISLAND --function \"$func\" --active_cma false --pollination false" > "$DIRECTORY/cma_island/basic/migr/cma_basic_migr_${func}.sh"
  echo "$PREFIX_SCRIPT_CMA_ISLAND --function \"$func\" --active_cma false --pollination true" > "$DIRECTORY/cma_island/basic/polli/cma_basic_polli_${func}.sh"
  echo "$PREFIX_SCRIPT_CMA_ISLAND --function \"$func\" --active_cma true --pollination false" > "$DIRECTORY/cma_island/active/migr/cma_active_migr_${func}.sh"
  echo "$PREFIX_SCRIPT_CMA_ISLAND --function \"$func\" --active_cma true --pollination true" > "$DIRECTORY/cma_island/active/polli/cma_active_polli_${func}.sh"

  # Submit jobs
  cd "$DIRECTORY/default_prop/migr" || exit 1 && sbatch default_prop_migr_${func}.sh && cd ../../
  cd "default_prop/polli" || exit 1 && sbatch default_prop_polli_${func}.sh && cd ../../
  cd "cma/basic" || exit 1 && sbatch cma_basic_${func}.sh && cd ../../
  cd "cma/active" || exit 1 && sbatch cma_active_${func}.sh && cd ../../
  cd "cma_island/basic/migr" || exit 1 && sbatch cma_basic_migr_${func}.sh && cd ../../../
  cd "cma_island/basic/polli" || exit 1 && sbatch cma_basic_polli_${func}.sh && cd ../../../
  cd "cma_island/active/migr" || exit 1 && sbatch cma_active_migr_${func}.sh && cd ../../../
  cd "cma_island/active/polli" || exit 1 && sbatch cma_active_polli_${func}.sh && cd ../../../


