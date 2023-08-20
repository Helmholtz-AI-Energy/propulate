#!/usr/bin/env python3
import os.path
import pickle
import random
import sys
import warnings
from pathlib import Path

from hyppopy.HyppopyProject import HyppopyProject
from hyppopy.MPIBlackboxFunction import MPIBlackboxFunction
from hyppopy.SolverPool import SolverPool
from hyppopy.solvers.MPISolverWrapper import MPISolverWrapper
from hyppopy.solvers.OptunitySolver import OptunitySolver
from mpi4py import MPI

from scripts.function_benchmark import get_function_search_space

if __name__ == "__main__":
    assert len(sys.argv) >= 4

    ############
    # SETTINGS #
    ############

    function_name = sys.argv[1]  # Get function to optimize from command-line.
    max_iterations = int(sys.argv[2])
    CHECKPOINT_PLACE = sys.argv[3]
    POP_SIZE = 2 * MPI.COMM_WORLD.size  # Set size of breeding population.

    function, limits = get_function_search_space(function_name)
    rng = random.Random(MPI.COMM_WORLD.rank)

    project = HyppopyProject()
    for key in limits:
        project.add_hyperparameter(name=key, domain="uniform", data=list(limits[key]), type=float)
    project.add_setting(name="max_iterations", value=max_iterations)
    project.add_setting(name="solver", value="optunity")

    blackbox = MPIBlackboxFunction(blackbox_func=function, mpi_comm=MPI.COMM_WORLD)

    solver = OptunitySolver(project)
    solver = MPISolverWrapper(solver=solver, mpi_comm=MPI.COMM_WORLD)
    solver.blackbox = blackbox

    solver.run()
    df, best = solver.get_results()

    path = Path(f"{CHECKPOINT_PLACE}/result_{MPI.COMM_WORLD.rank}.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.isfile(path):
        try:
            os.replace(path, path.with_suffix(".bkp"))
            warnings.warn("Results file already existing! Possibly overwriting data!")
        except OSError as e:
            print(e)
    with open(path, "wb") as f:
        pickle.dump((df, best), f)
