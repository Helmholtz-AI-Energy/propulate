#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=128
#SBATCH --partition=multiple_il
#SBATCH --job-name=rc_1_1
#SBATCH --time=0:30:00
#SBATCH --mem=40000
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pa1164@partner.kit.edu

cd $(ws_find propulate_bm_1)
ml purge
ml restore propulate
source /pfs/work7/workspace/scratch/pa1164-propulate_bm_1/async-parallel-pso/.venvs/async-parallel-pso/bin/activate
mpirun --bind-to core --map-by core -mca btl ^ofi python -u /pfs/work7/workspace/scratch/pa1164-propulate_bm_1/async-parallel-pso/ap_pso/bm/pso_benchmark.py schwefel 1000 3 /pfs/work7/workspace/scratch/pa1164-propulate_bm_1/async-parallel-pso/ap_pso/bm/results3/bm_3_schwefel_1