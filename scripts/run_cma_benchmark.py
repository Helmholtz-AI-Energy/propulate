import subprocess

num_islands = 4
dimension = 2

configurations = {
    "default_prop_migr": {
        "checkpoint_path": "default_prop/migr",
        "num_islands": num_islands,
        "pollination": False,
        "propagator": "default",
        "dimension": dimension,
    },
    "default_prop_polli": {
        "checkpoint_path": "default_prop/polli",
        "num_islands": num_islands,
        "pollination": True,
        "propagator": "default",
        "dimension": dimension,
    },
    "cma_basic": {
        "checkpoint_path": "cma/basic",
        "num_islands": 1,
        "pollination": False,
        "propagator": "cmaBasic",
        "dimension": dimension,
    },
    "cma_active": {
        "checkpoint_path": "cma/active",
        "num_islands": 1,
        "pollination": False,
        "propagator": "cmaActive",
        "dimension": dimension,
    },
    "cma_island_basic_migr": {
        "checkpoint_path": "cma_island/basic/migr",
        "num_islands": num_islands,
        "pollination": False,
        "propagator": "cmaBasic",
        "dimension": dimension,
    },
    "cma_island_active_migr": {
        "checkpoint_path": "cma_island/active/migr",
        "num_islands": num_islands,
        "pollination": False,
        "propagator": "cmaActive",
        "dimension": dimension,
    },
    "multi_dim_default_prop_polli_10": {
        "checkpoint_path": "multi_dim/default_prop/polli",
        "num_islands": num_islands,
        "pollination": True,
        "propagator": "default",
        "dimension": 10,
    },
    "multi_dim_cma_active_10": {
        "checkpoint_path": "multi_dim/cma/active",
        "num_islands": 1,
        "pollination": False,
        "propagator": "cmaActive",
        "dimension": 10,
    },
    "multi_dim_cma_island_active_migr_10": {
        "checkpoint_path": "multi_dim/cma_island/active/migr",
        "num_islands": num_islands,
        "pollination": False,
        "propagator": "cmaActive",
        "dimension": 10,
    },
    "multi_dim_default_prop_polli_30": {
        "checkpoint_path": "multi_dim/default_prop/polli",
        "num_islands": num_islands,
        "pollination": True,
        "propagator": "default",
        "dimension": 30,
    },
    "multi_dim_cma_active_30": {
        "checkpoint_path": "multi_dim/cma/active",
        "num_islands": 1,
        "pollination": False,
        "propagator": "cmaActive",
        "dimension": 30,
    },
    "multi_dim_cma_island_active_migr_30": {
        "checkpoint_path": "multi_dim/cma_island/active/migr",
        "num_islands": num_islands,
        "pollination": False,
        "propagator": "cmaActive",
        "dimension": 30,
    },
}


def main():
    for key, config in configurations.items():
        print("Current config:", config)
        job_name = f"{key}{dimension}dims_{num_islands}islands_polli{config['pollination']}"
        outfile = f"{job_name}.out"
        job_script_name = f"{job_name}.sh"
        scriptcontent = f"""#!/bin/bash
                #SBATCH --ntasks=40
                #SBATCH --nodes=1
                #SBATCH --job-name={job_name}
                #SBATCH --mail-type=BEGIN,END,FAIL
                #SBATCH --mail-user=uxyme@student.kit.edu
                #SBATCH --partition=dev_single
                #SBATCH --output={outfile}

                module purge
                module load devel/python/3.8.6_gnu_10.2
                module load compiler/gnu/10.2
                module load mpi/openmpi/4.1

                source /pfs/work7/workspace/scratch/fp5870-propulate/propulate_venv/bin/activate

                mpirun python /pfs/work7/workspace/scratch/fp5870-propulate/propulate/scripts/cma_es_benchmark.py --checkpoint_path {config["checkpoint_path"]} --num_islands {config["num_islands"]} --generation 100 --dimension {config["dimension"]} --pool_size 3 --propagator {config["propagator"]}
                    """
        if config["pollination"]:
            scriptcontent += " --pollination"

        with open(job_script_name, "wt") as f:
            f.write(
                scriptcontent
            )

        subprocess.run(f"sbatch -p dev_single {job_script_name}", shell=True)


if __name__ == "__main__":
    main()
