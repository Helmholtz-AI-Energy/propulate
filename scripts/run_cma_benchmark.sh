#!/bin/bash

# TODO --cpu-bind=cores output-name srun?
# TODO STORE TIMES ETC?
# TODO Bispehre

# Use GNU getopt to parse customized command-line arguments
CMA_COMMAND=$(getopt -o: --long generation:,pop_size:,num_isles:,migration_probability:,pool_size:,exploration:,select_worst_all_time:,dimension:,main_path:,directory:,nodes: -n 'getopt_errors_cma' -- "$@")

if [ $? != 0 ] ; then echo "Command-line error occurred. Terminating..." >&2 ; exit 1 ; fi

eval set -- "$CMA_COMMAND"

# Default values
GENERATION=100
POP_SIZE=""
NUM_ISLES=2
MIGRATION_PROBABILITY=0.9
POOL_SIZE=2
EXPLORATION=false
SELECT_WORST_ALL_TIME=false
DIMENSION=2
MAIN_PATH="cma_es_benchmark.py"
DIRECTORY="benchmark"
NODES=4
CPUS_PER_NODE=8

# Iterate through parsed options
while true; do
  case "$1" in
    --generation ) GENERATION="$2"; shift 2 ;;
    --pop_size ) POP_SIZE="$2"; shift 2 ;;
    --num_isles ) NUM_ISLES="$2"; shift 2 ;;
    --migration_probability ) MIGRATION_PROBABILITY="$2"; shift 2 ;;
    --pool_size ) POOL_SIZE="$2"; shift 2 ;;
    --exploration ) EXPLORATION="$2"; shift 2 ;;
    --select_worst_all_time ) SELECT_WORST_ALL_TIME="$2"; shift 2 ;;
    --dimension ) DIMENSION="$2"; shift 2 ;;
    --main_path ) MAIN_PATH="$2"; shift 2 ;;
    --directory ) DIRECTORY="$2"; shift 2 ;;
    --nodes ) NODES="$2"; shift 2 ;;
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
#SBATCH --nodes=$NODES
#SBATCH --ntasks=$NTASKS
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=uxyme@student.kit.edu
#SBATCH --partition=dev_multiple

module purge
module load mpi/openmpi/4.1

source propulate_venv/bin/activate

srun python -u $MAIN_PATH \
--generation $GENERATION \
--num_isles $NUM_ISLES
--pop_size $POP_SIZE \
--pool_size $POOL_SIZE \
--exploration $EXPLORATION \
--select_worst_all_time $SELECT_WORST_ALL_TIME \
--dimension $DIMENSION \
--migration_probability $MIGRATION_PROBABILITY"


# Check if the directory already exists
if [ -d "$DIRECTORY" ]; then
  echo "$DIRECTORY directory already exists."
  exit 1
fi

PARAMETER_SUMMARY="generation: ${GENERATION}
pop_size: ${POP_SIZE}
nodes: ${NODES}
ntasks: ${NTASKS}
cpus_per_node: ${CPUS_PER_NODE}
num_isles: ${NUM_ISLES}
migration_probability: ${MIGRATION_PROBABILITY}
pool_size: ${POOL_SIZE}
exploration: ${EXPLORATION}
select_worst_all_time: ${SELECT_WORST_ALL_TIME}
dimension: ${DIMENSION}"

