#!/bin/bash

# TODO --cpu-bind=cores output-name srun?
# TODO STORE TIMES ETC?
# TODO CPUS CPUSPERNODE RAUS

# Use GNU getopt to parse customized command-line arguments
CMA_COMMAND=$(getopt -o: --long generation:,pop_size:,num_isles:,migration_probability:,pool_size:,exploration:,select_worst_all_time:,dimension:,main_path:,directory:,nodes:,cpus: -n 'getopt_errors_cma' -- "$@")

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
NODES=1
CPUS_PER_NODE=8
#WORKERS=0

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
#SBATCH --nodes=$NODES
#SBATCH --ntasks=$NTASKS
#SBATCH --gres=cpu:$CPUS_PER_NODE
#SBATCH --partition=multiple

module purge
module load compiler/gnu/12.1
module load mpi/openmpi/4.1

source propulate_venv/bin/activate

srun python -u $MAIN_PATH \
--generation $GENERATION \
--pop_size $POP_SIZE \
--pool_size $pool_size \
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

# make directories
mkdir -p "$DIRECTORY/default_prop/migr"
mkdir -p "$DIRECTORY/default_prop/polli"
mkdir -p "$DIRECTORY/cma/basic"
mkdir -p "$DIRECTORY/cma/active"
mkdir -p "$DIRECTORY/cma_island/active/migr"
mkdir -p "$DIRECTORY/cma_island/basic/migr"
#mkdir -p "$DIRECTORY/cma_island/active/polli"
#mkdir -p "$DIRECTORY/cma_island/basic/polli"

mkdir -p "$DIRECTORY/multi_dim/default_prop/polli"
mkdir -p "$DIRECTORY/multi_dim/cma/active"
mkdir -p "$DIRECTORY/multi_dim/cma_island/active/migr"

echo "$PARAMETER_SUMMARY" > "$DIRECTORY/parameters.txt"

# Script for each configuration
FUNCTIONS=("sphere" "rosenbrock" "step" "quartic" "rastrigin" "griewank" "schwefel")

PREFIX_SCRIPT_DEFAULT="$BASE_SCRIPT --dimension $DIMENSION --num_isles $NUM_ISLES --default_propulate true"
PREFIX_SCRIPT_CMA="$BASE_SCRIPT --dimension $DIMENSION --num_isles 1 --default_propulate false --exploration $EXPLORATION --select_worst_all_time $SELECT_WORST_ALL_TIME"
PREFIX_SCRIPT_CMA_ISLAND="$BASE_SCRIPT --dimension $DIMENSION --num_isles $NUM_ISLES --default_propulate false --exploration $EXPLORATION --select_worst_all_time $SELECT_WORST_ALL_TIME"

PREFIX_SCRIPT_DEFAULT_MULTI="$BASE_SCRIPT --num_isles $NUM_ISLES --default_propulate true"
PREFIX_SCRIPT_CMA_MULTI="$BASE_SCRIPT --num_isles 1 --default_propulate false --exploration $EXPLORATION --select_worst_all_time $SELECT_WORST_ALL_TIME"
PREFIX_SCRIPT_CMA_ISLAND_MULTI="$BASE_SCRIPT --num_isles $NUM_ISLES --default_propulate false --exploration $EXPLORATION --select_worst_all_time $SELECT_WORST_ALL_TIME"

for func in "${FUNCTIONS[@]}"; do
  for ((run=1; run<=10; run++)); do
    # Default Propulate Benchmark
    echo "$PREFIX_SCRIPT_DEFAULT --function \"$func\" --pollination false" > "$DIRECTORY/default_prop/migr/default_prop_migr_${func}_run${run}.sh"
    echo "$PREFIX_SCRIPT_DEFAULT --function \"$func\" --pollination true" > "$DIRECTORY/default_prop/polli/default_prop_polli_${func}_run${run}.sh"

    # Basic CMA
    echo "$PREFIX_SCRIPT_CMA --function \"$func\" --active_cma false" > "$DIRECTORY/cma/basic/cma_basic_${func}_run${run}.sh"
    # Active CMA
    echo "$PREFIX_SCRIPT_CMA --function \"$func\" --active_cma true" > "$DIRECTORY/cma/active/cma_active_${func}_run${run}.sh"

    # CMA + Island
    echo "$PREFIX_SCRIPT_CMA_ISLAND --function \"$func\" --active_cma false --pollination false" > "$DIRECTORY/cma_island/basic/migr/cma_basic_migr_${func}_run${run}.sh"
    #echo "$PREFIX_SCRIPT_CMA_ISLAND --function \"$func\" --active_cma false --pollination true" > "$DIRECTORY/cma_island/basic/polli/cma_basic_polli_${func}_run${run}.sh"
    echo "$PREFIX_SCRIPT_CMA_ISLAND --function \"$func\" --active_cma true --pollination false" > "$DIRECTORY/cma_island/active/migr/cma_active_migr_${func}_run${run}.sh"
    #echo "$PREFIX_SCRIPT_CMA_ISLAND --function \"$func\" --active_cma true --pollination true" > "$DIRECTORY/cma_island/active/polli/cma_active_polli_${func}_run${run}.sh"

    # Default Propulate Multi Dimensions
    echo "$PREFIX_SCRIPT_DEFAULT_MULTI --dimension 10 --function \"$func\" --pollination true" > "$DIRECTORY/multi_dim/default_prop/polli/multi_10_default_prop_polli_${func}_run${run}.sh"
    echo "$PREFIX_SCRIPT_DEFAULT_MULTI --dimension 100 --function \"$func\" --pollination true" > "$DIRECTORY/multi_dim/default_prop/polli/multi_100_default_prop_polli_${func}_run${run}.sh"
    # CMA Active Multi Dimensions
    echo "$PREFIX_SCRIPT_CMA_MULTI --dimension 10 --function \"$func\" --active_cma true" > "$DIRECTORY/multi_dim/cma/active/multi10_cma_active_${func}_run${run}.sh"
    echo "$PREFIX_SCRIPT_CMA_MULTI --dimension 100 --function \"$func\" --active_cma true" > "$DIRECTORY/multi_dim/cma/active/multi100_cma_active_${func}_run${run}.sh"
    # CMA Active + Island Multi Dimensions
    echo "$PREFIX_SCRIPT_CMA_ISLAND_MULTI --dimension 10 --function \"$func\" --active_cma true --pollination false" > "$DIRECTORY/multi_dim/cma_island/active/migr/multi10_cma_active_migr_${func}_run${run}.sh"
    echo "$PREFIX_SCRIPT_CMA_ISLAND_MULTI --dimension 100 --function \"$func\" --active_cma true --pollination false" > "$DIRECTORY/multi_dim/cma_island/active/migr/multi100_cma_active_migr_${func}_run${run}.sh"

    # Submit jobs and redirect stdout to a separate output file for each job
    sbatch "$DIRECTORY/default_prop/migr/default_prop_migr_${func}_run${run}.sh" > "$DIRECTORY/default_prop/migr/default_prop_migr_${func}_run${run}.out"
    sbatch "$DIRECTORY/default_prop/polli/default_prop_polli_${func}_run${run}.sh" > "$DIRECTORY/default_prop/polli/default_prop_polli_${func}_run${run}.out"
    sbatch "$DIRECTORY/cma/basic/cma_basic_${func}_run${run}.sh" > "$DIRECTORY/cma/basic/cma_basic_${func}_run${run}.out"
    sbatch "$DIRECTORY/cma/active/cma_active_${func}_run${run}.sh" > "$DIRECTORY/cma/active/cma_active_${func}_run${run}.out"
    sbatch "$DIRECTORY/cma_island/basic/migr/cma_basic_migr_${func}_run${run}.sh" > "$DIRECTORY/cma_island/basic/migr/cma_basic_migr_${func}_run${run}.out"
    #sbatch "$DIRECTORY/cma_island/basic/polli/cma_basic_polli_${func}.sh" > "$DIRECTORY/cma_island/basic/polli/cma_basic_polli_${func}_run${run}.out"
    sbatch "$DIRECTORY/cma_island/active/migr/cma_active_migr_${func}_run${run}.sh" > "$DIRECTORY/cma_island/active/migr/cma_active_migr_${func}_run${run}.out"
    #sbatch "$DIRECTORY/cma_island/active/polli/cma_active_polli_${func}.sh" > "$DIRECTORY/cma_island/active/polli/cma_active_polli_${func}_run${run}.out"

    sbatch "$DIRECTORY/multi_dim/default_prop/polli/multi_10_default_prop_polli_${func}_run${run}.sh" > "$DIRECTORY/multi_dim/default_prop/polli/multi_10_default_prop_polli_${func}_run${run}.out"
    sbatch "$DIRECTORY/multi_dim/default_prop/polli/multi_100_default_prop_polli_${func}_run${run}.sh" > "$DIRECTORY/multi_dim/default_prop/polli/multi_100_default_prop_polli_${func}_run${run}.out"
    sbatch "$DIRECTORY/multi_dim/cma/active/multi10_cma_active_${func}_run${run}.sh" > "$DIRECTORY/multi_dim/cma/active/multi10_cma_active_${func}_run${run}.out"
    sbatch "$DIRECTORY/multi_dim/cma/active/multi100_cma_active_${func}_run${run}.sh" > "$DIRECTORY/multi_dim/cma/active/multi100_cma_active_${func}_run${run}.out"
    sbatch "$DIRECTORY/multi_dim/cma_island/active/migr/multi10_cma_active_migr_${func}_run${run}.sh" > "$DIRECTORY/multi_dim/cma_island/active/migr/multi10_cma_active_migr_${func}_run${run}.out"
    sbatch "$DIRECTORY/multi_dim/cma_island/active/migr/multi100_cma_active_migr_${func}_run${run}.sh" > "$DIRECTORY/multi_dim/cma_island/active/migr/multi100_cma_active_migr_${func}_run${run}.out"
  done
done
