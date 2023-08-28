#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks=512
#SBATCH --partition=multiple_il
#SBATCH --job-name=rc_3_1
#SBATCH --time=2:00:00
#SBATCH --mem=40000
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pa1164@partner.kit.edu

cd $(ws_find propulate_bm_1)
ml purge
ml restore propulate
source /pfs/work7/workspace/scratch/pa1164-propulate_bm_1/async-parallel-pso/.venvs/async-parallel-pso/bin/activate
mpirun --bind-to core --map-by core -mca btl ^ofi python -u /pfs/work7/workspace/scratch/pa1164-propulate_bm_1/async-parallel-pso/scripts/islands_example.py -f quartic -g 250 -ckpt /pfs/work7/workspace/scratch/pa1164-propulate_bm_1/async-parallel-pso/ap_pso/bm/results3/bm_P_quartic_3 -i 1 -migp 0 -v 0
mpirun --bind-to core --map-by core -mca btl ^ofi python -u /pfs/work7/workspace/scratch/pa1164-propulate_bm_1/async-parallel-pso/ap_pso/bm/pso_benchmark.py sphere 250 2 /pfs/work7/workspace/scratch/pa1164-propulate_bm_1/async-parallel-pso/ap_pso/bm/results3/bm_2_sphere_3