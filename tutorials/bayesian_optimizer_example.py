"""Extended example script comparing Bayesian Optimization and PSO.

This script runs both Bayesian Optimization and PSO on the same benchmark function
and creates a comparison plot showing how the best fitness evolves over function
evaluations. Fitness history is extracted from the propulator's population history.
"""

import argparse
import pathlib
import random
from typing import Any, Dict, List, Optional, Tuple, Union, cast

# Ensure a non-interactive backend is used in headless/mpi runs BEFORE importing pyplot.
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI

from propulate import Propulator

# Import PSO bits
from propulate.propagators import Conditional

# Import Bayesian optimizer bits
from propulate.propagators.bayesopt import (
    BayesianOptimizer,
    MultiStartAcquisitionOptimizer,
)
from propulate.propagators.pso import BasicPSO, InitUniformPSO
from propulate.utils import set_logger_config
from propulate.utils.benchmark_functions import (
    get_function_search_space,
)

LimitsType = Dict[str, Union[Tuple[float, float], Tuple[int, int], Tuple[str, ...]]]


def extract_fitness_history(propulator: Any) -> Tuple[List[int], List[float]]:
    """Extract fitness history from propulator's population.

    - Maintains a running best value over evaluations
    - Repeats the previous best when encountering non-finite losses
    - Returns numpy arrays for easier downstream sanitization
    """
    if not propulator.population:
        return [], []

    # Sort individuals by generation to get chronological order
    sorted_individuals = sorted(propulator.population, key=lambda x: x.generation)

    evaluations = []
    best_fitness_history = []
    best_so_far = np.inf

    for i, individual in enumerate(sorted_individuals, 1):
        loss = individual.loss
        # Update best only for finite values
        if np.isfinite(loss) and loss < best_so_far:
            best_so_far = loss
        evaluations.append(i)
        # If best_so_far is still inf (no finite observation yet), store nan to avoid plotting issues
        best_fitness_history.append(best_so_far if np.isfinite(best_so_far) else np.nan)

    return evaluations, best_fitness_history


def run_optimizer(
    name: str,
    propagator: Any,
    benchmark_function: Any,
    config: argparse.Namespace,
    comm: Any,
    rng: random.Random,
    function_name: Optional[str] = None,
) -> Tuple[List[int], List[float]]:
    """Run an optimizer and return fitness history data extracted from population."""
    # Use function_name for checkpoint path if provided to ensure unique paths
    if function_name:
        checkpoint_path = f"{config.checkpoint}/function_{function_name}_{name}"
    else:
        checkpoint_path = f"{config.checkpoint}_{name}"

    propulator = Propulator(
        loss_fn=benchmark_function,
        propagator=propagator,
        rng=rng,
        island_comm=comm,
        generations=config.generations,
        checkpoint_path=checkpoint_path,
    )

    if comm.rank == 0:
        print(f"\n{'=' * 50}")
        print(f"Running {name} optimization...")
        print(f"{'=' * 50}")

    propulator.propulate(logging_interval=config.logging_interval, debug=config.verbosity)

    # Extract fitness history from propulator's population
    evaluations, fitness_history = extract_fitness_history(propulator)

    if comm.rank == 0:
        print(f"\n{name} completed!")
        if fitness_history:
            print(f"Best fitness: {fitness_history[-1]:.6f}")
            print(f"Total evaluations: {len(evaluations)}")
        else:
            print("No evaluations performed (resumed from completed checkpoint)")

    return evaluations, fitness_history


def parse_extended_arguments(comm: Any) -> Tuple[argparse.Namespace, Dict[str, Any]]:
    """Parse arguments with additional output path option."""
    # Create a new parser that includes both standard and custom arguments
    parser = argparse.ArgumentParser(
        prog="Bayesian Optimizer Comparison Example",
        description="Compare Bayesian Optimization and PSO with plotting capabilities.",
    )

    # Add all the standard arguments manually
    parser.add_argument(
        "--function",
        type=str,
        choices=[
            "bukin",
            "eggcrate",
            "himmelblau",
            "keane",
            "leon",
            "rastrigin",
            "schwefel",
            "sphere",
            "step",
            "rosenbrock",
            "quartic",
            "bisphere",
            "birastrigin",
            "griewank",
            "all",
        ],
        default="sphere",
        help="Function to optimize (use 'all' to run comparison for all functions)",
    )
    parser.add_argument("--generations", type=int, default=25, help="Number of generations")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random number generator")
    parser.add_argument("--verbosity", type=int, default=1, help="Verbosity level")
    parser.add_argument("--checkpoint", type=str, default="./", help="Path for checkpoints")
    parser.add_argument("--pop_size", type=int, default=2 * comm.size, help="Population size")
    parser.add_argument("--top_n", type=int, default=1, help="Top N results to show")
    parser.add_argument("--logging_interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--logging_level", type=int, default=20, help="Logging level")  # INFO = 20
    parser.add_argument("--ranks_per_worker", type=int, default=1, help="Ranks per worker")

    # Add our custom arguments
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Custom output path for the comparison plot. If not specified, will use checkpoint directory.",
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default="optimization_comparison.png",
        help="Name of the comparison plot file. Default: optimization_comparison.png",
    )
    # BO tuning flags
    parser.add_argument("--bo-acq", type=str, choices=["EI", "PI", "UCB"], default="EI", help="Acquisition type")
    parser.add_argument("--bo-xi", type=float, default=0.05, help="Exploration parameter for EI/PI")
    parser.add_argument("--bo-kappa", type=float, default=2.5, help="Exploration parameter for UCB")
    parser.add_argument("--bo-n-initial", type=int, default=None, help="Number of initial design points (default ~10*d)")
    parser.add_argument(
        "--bo-initial-design", type=str, choices=["sobol", "random", "lhs"], default="sobol", help="Initial design strategy"
    )
    parser.add_argument("--bo-n-candidates", type=int, default=None, help="Number of random candidates for acquisition optimizer")
    parser.add_argument("--bo-n-restarts", type=int, default=None, help="Number of polishing restarts for acquisition optimizer")
    parser.add_argument("--bo-sparse", action="store_true", help="Enable training set sparsification")
    parser.add_argument("--bo-max-points", type=int, default=500, help="Max points to keep when sparsification is enabled")
    # Exploration schedule / acquisition annealing controls
    parser.add_argument("--bo-p-explore-start", type=float, default=0.2, help="Initial epsilon-greedy exploration probability")
    parser.add_argument("--bo-p-explore-end", type=float, default=0.02, help="Final epsilon-greedy exploration probability")
    parser.add_argument(
        "--bo-p-explore-tau",
        type=float,
        default=150.0,
        help="Decay constant (higher = slower decay) for epsilon-greedy exploration",
    )
    parser.add_argument(
        "--bo-no-anneal", action="store_true", help="Disable automatic annealing of acquisition parameter (xi/kappa)"
    )
    # Dynamic acquisition switching
    parser.add_argument(
        "--bo-second-acq", type=str, choices=["EI", "PI", "UCB"], default=None, help="Optional second acquisition type to switch to"
    )
    parser.add_argument("--bo-acq-switch-gen", type=int, default=None, help="Generation after which to switch acquisition")
    parser.add_argument("--bo-second-xi", type=float, default=None, help="xi for second acquisition (if EI/PI)")
    parser.add_argument("--bo-second-kappa", type=float, default=None, help="kappa for second acquisition (if UCB)")
    # Presets for convenience
    parser.add_argument(
        "--bo-preset",
        type=str,
        choices=["balanced", "explore", "exploit", "aggressive"],
        default=None,
        help="Preset configuration overriding individual BO flags unless explicitly set",
    )

    # Parse all arguments
    config = parser.parse_args()

    # Apply presets (only if provided and corresponding flag left at default)
    if config.bo_preset:
        preset = config.bo_preset
        if preset == "balanced":
            config.bo_acq = config.bo_acq or "EI"
            config.bo_xi = 0.05 if parser.get_default("bo_xi") == config.bo_xi else config.bo_xi
            if config.bo_n_candidates is None:
                config.bo_n_candidates = 1024
            if config.bo_n_restarts is None:
                config.bo_n_restarts = 15
        elif preset == "explore":
            config.bo_acq = "UCB"
            config.bo_kappa = 3.5 if parser.get_default("bo_kappa") == config.bo_kappa else config.bo_kappa
            if config.bo_n_candidates is None:
                config.bo_n_candidates = 2048
            if config.bo_n_restarts is None:
                config.bo_n_restarts = 20
            config.bo_p_explore_start = 0.3
            config.bo_p_explore_end = 0.02
        elif preset == "exploit":
            config.bo_acq = "EI"
            config.bo_xi = 0.02
            if config.bo_n_candidates is None:
                config.bo_n_candidates = 512
            if config.bo_n_restarts is None:
                config.bo_n_restarts = 10
            config.bo_p_explore_start = 0.15
            config.bo_p_explore_end = 0.01
        elif preset == "aggressive":
            config.bo_acq = "EI"
            config.bo_xi = 0.1
            if config.bo_n_candidates is None:
                config.bo_n_candidates = 4096
            if config.bo_n_restarts is None:
                config.bo_n_restarts = 25
            config.bo_p_explore_start = 0.35
            config.bo_p_explore_end = 0.015
            config.bo_p_explore_tau = 250

    # Create the hp_set dictionary (empty since we don't use PSO hyperparameters from CLI here)
    hp_set: Dict[str, Any] = {}

    return config, hp_set


def run_single_function_comparison(
    function_name: str,
    config: argparse.Namespace,
    comm: Any,
) -> Tuple[List[int], List[float]]:
    """Run optimization comparison for a single function."""
    rng = random.Random(config.seed + comm.rank)  # Separate RNG for optimization.
    benchmark_function, limits = get_function_search_space(function_name)  # Get function + limits.
    limits_bo = cast(LimitsType, limits)
    limits_pso = cast(Dict[str, Tuple[float, float]], limits)

    # Problem dimension used by both BO and PSO configuration
    dim = len(limits)

    # =============================================================================
    # Setup Bayesian Optimizer
    # =============================================================================

    # Build optional acquisition optimizer based on flags
    acq_optimizer = None
    if config.bo_n_candidates is not None or config.bo_n_restarts is not None:
        acq_optimizer = MultiStartAcquisitionOptimizer(
            n_candidates=config.bo_n_candidates or max(256, 64 * dim),
            n_restarts=config.bo_n_restarts or max(5, min(20, 2 * dim)),
        )

    # Set up Bayesian optimizer propagator with configurable acquisition and initial design.
    second_params = {}
    if config.bo_second_acq:
        if config.bo_second_acq in ("EI", "PI") and config.bo_second_xi is not None:
            second_params["xi"] = config.bo_second_xi
        if config.bo_second_acq == "UCB" and config.bo_second_kappa is not None:
            second_params["kappa"] = config.bo_second_kappa

    bayes_propagator = BayesianOptimizer(
        limits=limits_bo,
        rank=comm.rank,
        world_size=comm.size,
        optimizer=acq_optimizer,
        acquisition_type=config.bo_acq,  # EI, PI, or UCB
        acquisition_params={"xi": config.bo_xi, "kappa": config.bo_kappa},
        rank_stretch=True,  # Diversify acquisition params across ranks
        factor_min=0.5,
        factor_max=2.0,
        sparse=config.bo_sparse,
        sparse_params={"max_points": config.bo_max_points},
        n_initial=config.bo_n_initial,
        initial_design=config.bo_initial_design,
        rng=rng,
        p_explore_start=config.bo_p_explore_start,
        p_explore_end=config.bo_p_explore_end,
        p_explore_tau=config.bo_p_explore_tau,
        anneal_acquisition=not config.bo_no_anneal,
        second_acquisition_type=config.bo_second_acq,
        acq_switch_generation=config.bo_acq_switch_gen,
        second_acquisition_params=second_params,
    )

    # =============================================================================
    # Setup PSO
    # =============================================================================

    # Set up PSO propagator with reasonable default parameters
    pso_propagator = BasicPSO(
        inertia=0.5,  # Inertia weight
        c_cognitive=2.0,  # Cognitive factor
        c_social=2.0,  # Social factor
        rank=comm.rank,
        limits=limits_pso,
        rng=rng,
    )

    # Initialize PSO with uniform random initialization
    pso_init = InitUniformPSO(limits_pso, rng=rng, rank=comm.rank)

    # Use a reasonable population size for PSO (default is often 20-40)
    pop_size = min(40, max(20, 2 * dim))  # Scale with problem dimension
    pso_propagator_with_init = Conditional(pop_size, pso_propagator, pso_init)

    # =============================================================================
    # Run both optimizers
    # =============================================================================

    # Run Bayesian Optimization
    bayes_evals, bayes_fitness = run_optimizer(
        "Bayesian_Optimization",
        bayes_propagator,
        benchmark_function,
        config,
        comm,
        random.Random(config.seed + comm.rank),
        function_name,
    )

    # Reset random seed for fair comparison
    comm.barrier()  # Synchronize before second run

    # Run PSO
    pso_evals, pso_fitness = run_optimizer(
        "PSO", pso_propagator_with_init, benchmark_function, config, comm, random.Random(config.seed + comm.rank), function_name
    )

    # =============================================================================
    # Create comparison plot (only on rank 0)
    # =============================================================================

    if comm.rank == 0:
        plt.figure(figsize=(12, 8))

        # Only plot if we have data
        if len(bayes_fitness) > 0:
            y_b = np.array(bayes_fitness, dtype=float)
            # Replace non-finite values with NaN to avoid backend conversion issues
            y_b[~np.isfinite(y_b)] = np.nan
            plt.semilogy(bayes_evals, y_b, "b-", linewidth=2, label="Bayesian Optimization", alpha=0.8)

        if len(pso_fitness) > 0:
            y_p = np.array(pso_fitness, dtype=float)
            y_p[~np.isfinite(y_p)] = np.nan
            plt.semilogy(pso_evals, y_p, "r-", linewidth=2, label="PSO", alpha=0.8)

        # Handle case where no data is available
        if len(bayes_fitness) == 0 and len(pso_fitness) == 0:
            plt.text(
                0.5,
                0.5,
                "No optimization data available\n(likely resumed from completed checkpoint)",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
                fontsize=14,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"),
            )
            plt.xlim(0, 1)
            plt.ylim(0, 1)

        plt.xlabel("Function Evaluations", fontsize=12)
        plt.ylabel("Best Fitness (log scale)", fontsize=12)
        plt.title(f"Optimization Comparison: {function_name}", fontsize=14, fontweight="bold")

        # Only show legend if we have data
        if len(bayes_fitness) > 0 or len(pso_fitness) > 0:
            plt.legend(fontsize=12)

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Determine output path for the plot
        plot_filename = f"optimization_comparison_{function_name}.png"
        if config.output_path is not None:
            # Use custom output path
            output_dir = pathlib.Path(config.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_path = output_dir / plot_filename
        else:
            # Use checkpoint directory as default
            plot_path = pathlib.Path(config.checkpoint) / plot_filename

        # Save the plot
        # Ensure path-like is converted to string and be robust to tight bbox issues
        try:
            plt.savefig(str(plot_path), dpi=300, bbox_inches="tight")
        except Exception:
            # Fallback without tight bbox if backend complains
            plt.savefig(str(plot_path), dpi=300)
        print(f"\nComparison plot saved to: {plot_path}")

        # Print final comparison summary
        print(f"\n{'=' * 60}")
        print("FINAL COMPARISON SUMMARY")
        print(f"{'=' * 60}")
        print(f"Function: {function_name}")
        print(f"Generations: {config.generations}")

        # Check if we have any evaluations (handle checkpoint resume case)
        if len(bayes_fitness) == 0 or len(pso_fitness) == 0:
            print("WARNING: No new evaluations performed (likely resumed from completed checkpoint)")
            print("Consider using a fresh checkpoint directory or increasing generations")
            if len(bayes_fitness) == 0:
                print("Bayesian Optimization - No evaluations performed")
            else:
                print(f"Bayesian Optimization - Final best: {bayes_fitness[-1]:.6f} in {len(bayes_evals)} evaluations")

            if len(pso_fitness) == 0:
                print("PSO                   - No evaluations performed")
            else:
                print(f"PSO                   - Final best: {pso_fitness[-1]:.6f} in {len(pso_evals)} evaluations")
        else:
            print(f"Bayesian Optimization - Final best: {bayes_fitness[-1]:.6f} in {len(bayes_evals)} evaluations")
            print(f"PSO                   - Final best: {pso_fitness[-1]:.6f} in {len(pso_evals)} evaluations")

            if bayes_fitness[-1] < pso_fitness[-1]:
                improvement = ((pso_fitness[-1] - bayes_fitness[-1]) / pso_fitness[-1]) * 100
                print(f"Bayesian Optimization achieved {improvement:.2f}% better final fitness")
            else:
                improvement = ((bayes_fitness[-1] - pso_fitness[-1]) / bayes_fitness[-1]) * 100
                print(f"PSO achieved {improvement:.2f}% better final fitness")

        print(f"Output plot: {plot_path}")
        print(f"{'=' * 60}")

    # Return BO results for summary collection (only BO since PSO not needed for aggregate BO summary)
    return bayes_evals, bayes_fitness


if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        print(
            "#################################################\n"
            "# PROPULATE: Parallel Propagator of Populations #\n"
            "#################################################\n"
            "# Comparing Bayesian Optimization vs PSO        #\n"
            "#################################################\n"
        )

    config, _ = parse_extended_arguments(comm)

    # Set up separate logger for Propulate optimization.
    set_logger_config(
        level=config.logging_level,  # Logging level
        log_file=f"{config.checkpoint}/{pathlib.Path(__file__).stem}.log",  # Logging path
        log_to_stdout=True,  # Print log on stdout.
        log_rank=False,  # Do not prepend MPI rank to logging messages.
        colors=True,  # Use colors.
    )

    # Define all available functions for the "all" option
    all_functions = [
        "bukin",
        "eggcrate",
        "himmelblau",
        "keane",
        "leon",
        "rastrigin",
        "schwefel",
        "sphere",
        "step",
        "rosenbrock",
        "quartic",
        "bisphere",
        "birastrigin",
        "griewank",
    ]

    # Determine which functions to run
    if config.function == "all":
        functions_to_run = all_functions
        if comm.rank == 0:
            print(f"Running comparison for all {len(functions_to_run)} functions...")
            # Clean up any existing checkpoint files to avoid conflicts
            import glob
            import os

            checkpoint_pattern = f"{config.checkpoint}/function_*"
            existing_checkpoints = glob.glob(checkpoint_pattern)
            if existing_checkpoints:
                print(f"Found {len(existing_checkpoints)} existing checkpoint directories. Removing them...")
                for checkpoint_dir in existing_checkpoints:
                    if os.path.isdir(checkpoint_dir):
                        import shutil

                        shutil.rmtree(checkpoint_dir)
                        print(f"Removed: {checkpoint_dir}")
    else:
        functions_to_run = [config.function]

    # Collect results for optional summary
    all_results = []
    for func_idx, function_name in enumerate(functions_to_run):
        if comm.rank == 0 and len(functions_to_run) > 1:
            print(f"\n{'=' * 80}")
            print(f"FUNCTION {func_idx + 1}/{len(functions_to_run)}: {function_name.upper()}")
            print(f"{'=' * 80}")
        evals_bo, fit_bo = run_single_function_comparison(function_name, config, comm)
        # run_single_function_comparison already prints and saves plots; we re-run nothing.
        if comm.rank == 0:
            final_bo = fit_bo[-1] if fit_bo else float("nan")
            all_results.append((function_name, final_bo))

    if comm.rank == 0 and len(all_results) > 1:
        import csv
        import os

        summary_path = pathlib.Path(config.output_path or config.checkpoint) / "bo_results_summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["function", "bo_final_fitness"])
            for fn, val in all_results:
                writer.writerow([fn, val])
        print(f"\nSummary CSV written to: {summary_path}")
