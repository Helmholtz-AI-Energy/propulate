"""Benchmark function module."""
import argparse
import logging
from typing import Callable, Dict, Tuple

import numpy as np
from mpi4py import MPI


def rosenbrock(params: Dict[str, float]) -> float:
    """
    Rosenbrock function. This function has a narrow minimum inside a parabola-shaped valley.

    Input domain: -2.048 <= (x, y) <= 2.048
    Global minimum 0 at (x, y) = (1, 1)

    Parameters
    ----------
    params : Dict[str, float]
        The function parameters.

    Returns
    -------
    float
        The function value.
    """
    x = np.array(list(params.values()))
    return 100 * (x[0] ** 2 - x[1]) ** 2 + (1 - x[0]) ** 2


def step(params: Dict[str, float]) -> float:
    """
    Step function.

    This function represents the problem of flat surfaces. Plateaus pose obstacles to optimizers as they lack
    information about which direction is favorable.

    Input domain: -5.12 <= x_i <= 5.12, i = 1,...,N
    Global minimum -5N at (x_i)_N <= (-5)_N

    The Propulate paper uses N = 5.

    Parameters
    ----------
    params : Dict[str, float]
        The function parameters.

    Returns
    -------
    float
        The function value.
    """
    x = np.array(list(params.values()))
    return np.sum(x.astype(int), dtype=float).item()


def quartic(params: Dict[str, float]) -> float:
    """
    Quartic function.

    A unimodal function padded with Gaussian noise. As it never returns the same value on the same point,
    algorithms that do not perform well on this function will do poorly on noisy data.

    Input domain: -1.28 <= x_i <= 1.28, i = 1,...,N
    Global minimum ∑ Gauss(0, 1)_i at (x_i)_N = (0)_N
    The Propulate paper uses N = 30.

    Parameters
    ----------
    params : Dict[str, float]
        The function parameters.

    Returns
    -------
    float
        The function value.
    """
    x = np.array(list(params.values()))
    idx = np.arange(1, len(x) + 1)
    gauss = np.random.normal(size=len(x))
    return abs(np.sum(idx * x**4 + gauss))


def rastrigin(params: Dict[str, float]) -> float:
    """
    Rastrigin function: continuous, non-convex, separable, differentiable, multimodal.

    A non-linear and highly multimodal function. Its surface is determined by two external variables, controlling
    the modulation’s amplitude and frequency. The local minima are located at a rectangular grid with size 1.
    Their functional values increase with the distance to the global minimum.

    Input domain: -5.12 <= x_i <= 5.12, i = 1,...,N
    Global minimum 0 at (x_i)_N = (0)_N
    The Propulate paper uses N = 20.

    Parameters
    ----------
    params : Dict[str, float]
        The function parameters.

    Returns
    -------
    float
        The function value.
    """
    a = 10.0
    x = np.array(list(params.values()))
    return a * len(x) + np.sum(x**2 - a * np.cos(2 * np.pi * x))


def griewank(params: Dict[str, float]) -> float:
    """
    Griewank function.

    Griewank's product creates subpopulations strongly codependent to parallel GAs, while the summation produces a
    parabola. Its local optima lie above parabola level but decrease with increasing dimensions, i.e., the larger the
    search range, the flatter the function.

    Input domain: -600 <= x_i <= 600, i = 1,...,N
    Global minimum 0 at (x_i)_N = (0)_N
    The Propulate paper uses N = 10.

    Parameters
    ----------
    params : Dict[str, float]
        The function parameters.

    Returns
    -------
    float
        The function value.
    """
    x = np.array(list(params.values()))
    idx = np.arange(1, len(x) + 1)
    return 1 + 1.0 / 4000 * np.sum(x**2) - np.prod(np.cos(x / np.sqrt(idx)))


def schwefel(params: Dict[str, float]) -> float:
    """
    Schwefel 2.20 function: continuous, convex, separable, non-differentiable, non-multimodal.

    This function has a second-best minimum far away from the global optimum.

    Input domain: -500 <= x_i <= 500, i = 1,...,N
    Global minimum 0 at (x_i)_N = (420.968746)_N
    The Propulate paper uses N = 10.

    Parameters
    ----------
    params : Dict[str, float]
        The function parameters.

    Returns
    -------
    float
        The function value.
    """
    v = 418.982887
    x = np.array(list(params.values()))
    return v * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def bisphere(params: Dict[str, float]) -> float:
    """
    Lunacek's double-sphere benchmark function.

    Lunacek, M., Whitley, D., & Sutton, A. (2008, September).
    The impact of global structure on search.
    In International Conference on Parallel Problem Solving from Nature
    (pp. 498-507). Springer, Berlin, Heidelberg.

    This function's landscape structure is the minimum of two quadratic functions, each creating a single funnel in the
    search space. The spheres are placed along the positive search-space diagonal, with the optimal and sub-optimal
    sphere in the middle of the positive and negative quadrant, respectively. Their distance and the barrier's height
    increase with dimensionality, creating a globally non-separable underlying surface.

    Input domain: -5.12 <= x_i <= 5.12, i = 1,...,N
    Global minimum 0 at (x_i)_N = (µ_1)_N with µ_1 = 2.5
    The Propulate paper uses N = 30.

    Parameters
    ----------
    params : Dict[str, float]
        The function parameters.

    Returns
    -------
    float
        The function value.
    """
    x = np.array(list(params.values()))
    n = len(x)
    d = 1
    s = 1 - np.sqrt(1 / (2 * np.sqrt(n + 20) - 8.2))
    mu1 = 2.5
    mu2 = -np.sqrt((mu1**2 - d) / s)
    return min(np.sum((x - mu1) ** 2), d * n + s * np.sum((x - mu2) ** 2)).item()


def birastrigin(params: Dict[str, float]) -> float:
    """
    Lunacek's double-Rastrigin benchmark function.

    Lunacek, M., Whitley, D., & Sutton, A. (2008, September).
    The impact of global structure on search.
    In International Conference on Parallel Problem Solving from Nature
    (pp. 498-507). Springer, Berlin, Heidelberg.

    A double-funnel version of Rastrigin. This function isolates global structure as the main difference impacting
    problem difficulty on a well understood test case.

    Input domain: -5.12 <= x_i <= 5.12, i = 1,...,N
    Global minimum 0 at (x_i)_N = (µ_1)_N with µ_1 = 2.5
    The Propulate paper uses N = 30.

    Parameters
    ----------
    params : Dict[str, float]
        The function parameters.

    Returns
    -------
    float
        The function value.
    """
    x = np.array(list(params.values()))
    n = len(x)
    d = 1
    s = 1 - np.sqrt(1 / (2 * np.sqrt(n + 20) - 8.2))
    mu1 = 2.5
    mu2 = -np.sqrt((mu1**2 - d) / s)
    return min(
        np.sum((x - mu1) ** 2), d * n + s * np.sum((x - mu2) ** 2)
    ) + 10 * np.sum(1 - np.cos(2 * np.pi * (x - mu1)))


def bukin_n6(params: Dict[str, float]) -> float:
    """
    Bukin N.6 function: continuous, convex, non-separable, non-differentiable, multimodal.

    Input domain: -15 <= x <= -5, -3 <= y <= 3
    Global minimum 0 at (x, y) = (-10, 1)

    Parameters
    ----------
    params : Dict[str, float]
        The function parameters.

    Returns
    -------
    float
        The function value.
    """
    x = np.array(list(params.values()))
    return 100 * np.sqrt(np.abs(x[1] - 0.01 * x[0] ** 2)) + 0.01 * np.abs(x[0] + 10)


def egg_crate(params: Dict[str, float]) -> float:
    """
    Egg-crate function: continuous, non-convex, separable, differentiable, multimodal.

    Input domain: -5 <= x, y <= 5
    Global minimum -1 at (x, y) = (0, 0)

    Parameters
    ----------
    params : Dict[str, float]
        The function parameters.

    Returns
    -------
    float
        The function value.
    """
    x = np.array(list(params.values()))
    return x[0] ** 2 + x[1] ** 2 + 25 * (np.sin(x[0]) ** 2 + np.sin(x[1]) ** 2)


def himmelblau(params: Dict[str, float]) -> float:
    """
    Himmelblau function: continuous, non-convex, non-separable, differentiable, multimodal.

    Input domain: -6 <= x, y <= 6
    Global minimum 0 at (x, y) = (3, 2)

    Parameters
    ----------
    params : Dict[str, float]
        The function parameters.

    Returns
    -------
    float
        The function value.
    """
    x = np.array(list(params.values()))
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def keane(params: Dict[str, float]) -> float:
    """
    Keane function: continuous, non-convex, non-separable, differentiable, multimodal.

    Input domain: -10 <= x, y <= 10
    Global minimum 0.6736675 at (x, y) = (1.3932491, 0) and (x, y) = (0, 1.3932491)

    Parameters
    ----------
    params : Dict[str, float]
        The function parameters.

    Returns
    -------
    float
        The function value.
    """
    x = np.array(list(params.values()))
    return (
        -(np.sin(x[0] - x[1]) ** 2)
        * np.sin(x[0] + x[1]) ** 2
        / np.sqrt(x[0] ** 2 + x[1] ** 2)
    )


def leon(params: Dict[str, float]) -> float:
    """
    Leon function: continuous, non-convex, non-separable, differentiable, non-multimodal, non-random, non-parametric.

    Input domain: 0 <= x, y <= 10
    Global minimum 0 at (x, y) =(1, 1)

    Parameters
    ----------
    params : Dict[str, float]
        The function parameters.

    Returns
    -------
    float
        The function value.
    """
    x = np.array(list(params.values()))
    return 100 * (x[1] - x[0] ** 3) ** 2 + (1 - x[0]) ** 2


def sphere(params: Dict[str, float]) -> float:
    """
    Sphere function: continuous, convex, separable, differentiable, unimodal.

    Input domain: -5.12 <= x, y <= 5.12
    Global minimum 0 at (x, y) = (0, 0)

    Parameters
    ----------
    params : Dict[str, float]
        The function parameters.

    Returns
    -------
    float
        The function value.
    """
    return np.sum(np.array(list(params.values())) ** 2).item()


def get_function_search_space(
    fname: str,
) -> Tuple[Callable, Dict[str, Tuple[float, float]]]:
    """
    Get search space limits and function from function name.

    Parameters
    ----------
    fname : str
        The function name.

    Returns
    -------
    Callable
        The callable function.
    Dict[str, tuple[float, float]]
        The search space.
    """
    if fname == "bukin":
        function = bukin_n6
        limits = {
            "a": (-15.0, -5.0),
            "b": (-3.0, 3.0),
        }
    elif fname == "eggcrate":
        function = egg_crate
        limits = {
            "a": (-5.0, 5.0),
            "b": (-5.0, 5.0),
        }
    elif fname == "himmelblau":
        function = himmelblau
        limits = {
            "a": (-6.0, 6.0),
            "b": (-6.0, 6.0),
        }
    elif fname == "keane":
        function = keane
        limits = {
            "a": (0.0, 10.0),
            "b": (0.0, 10.0),
        }
    elif fname == "leon":
        function = leon
        limits = {
            "a": (0.0, 10.0),
            "b": (0.0, 10.0),
        }
    elif fname == "sphere":
        function = sphere
        limits = {
            "a": (-5.12, 5.12),
            "b": (-5.12, 5.12),
        }
    elif fname == "rosenbrock":
        function = rosenbrock
        limits = {
            "a": (-2.048, 2.048),
            "b": (-2.048, 2.048),
        }
    elif fname == "step":
        function = step
        limits = {
            "a": (-5.12, 5.12),
            "b": (-5.12, 5.12),
            "c": (-5.12, 5.12),
            "d": (-5.12, 5.12),
            "e": (-5.12, 5.12),
        }
    elif fname == "quartic":
        function = quartic
        limits = {
            "a": (-1.28, 1.28),
            "b": (-1.28, 1.28),
            "c": (-1.28, 1.28),
            "d": (-1.28, 1.28),
            "e": (-1.28, 1.28),
            "f": (-1.28, 1.28),
            "g": (-1.28, 1.28),
            "h": (-1.28, 1.28),
            "i": (-1.28, 1.28),
            "j": (-1.28, 1.28),
            "k": (-1.28, 1.28),
            "l": (-1.28, 1.28),
            "m": (-1.28, 1.28),
            "n": (-1.28, 1.28),
            "o": (-1.28, 1.28),
            "p": (-1.28, 1.28),
            "q": (-1.28, 1.28),
            "r": (-1.28, 1.28),
            "s": (-1.28, 1.28),
            "t": (-1.28, 1.28),
            "u": (-1.28, 1.28),
            "v": (-1.28, 1.28),
            "w": (-1.28, 1.28),
            "x": (-1.28, 1.28),
            "y": (-1.28, 1.28),
            "z": (-1.28, 1.28),
            "A1": (-1.28, 1.28),
            "B1": (-1.28, 1.28),
            "C1": (-1.28, 1.28),
            "D1": (-1.28, 1.28),
        }
    elif fname == "bisphere":
        function = bisphere
        limits = {
            "a": (-5.12, 5.12),
            "b": (-5.12, 5.12),
            "c": (-5.12, 5.12),
            "d": (-5.12, 5.12),
            "e": (-5.12, 5.12),
            "f": (-5.12, 5.12),
            "g": (-5.12, 5.12),
            "h": (-5.12, 5.12),
            "i": (-5.12, 5.12),
            "j": (-5.12, 5.12),
            "k": (-5.12, 5.12),
            "l": (-5.12, 5.12),
            "m": (-5.12, 5.12),
            "n": (-5.12, 5.12),
            "o": (-5.12, 5.12),
            "p": (-5.12, 5.12),
            "q": (-5.12, 5.12),
            "r": (-5.12, 5.12),
            "s": (-5.12, 5.12),
            "t": (-5.12, 5.12),
            "u": (-5.12, 5.12),
            "v": (-5.12, 5.12),
            "w": (-5.12, 5.12),
            "x": (-5.12, 5.12),
            "y": (-5.12, 5.12),
            "z": (-5.12, 5.12),
            "A1": (-5.12, 5.12),
            "B1": (-5.12, 5.12),
            "C1": (-5.12, 5.12),
            "D1": (-5.12, 5.12),
        }
    elif fname == "birastrigin":
        function = birastrigin
        limits = {
            "a": (-5.12, 5.12),
            "b": (-5.12, 5.12),
            "c": (-5.12, 5.12),
            "d": (-5.12, 5.12),
            "e": (-5.12, 5.12),
            "f": (-5.12, 5.12),
            "g": (-5.12, 5.12),
            "h": (-5.12, 5.12),
            "i": (-5.12, 5.12),
            "j": (-5.12, 5.12),
            "k": (-5.12, 5.12),
            "l": (-5.12, 5.12),
            "m": (-5.12, 5.12),
            "n": (-5.12, 5.12),
            "o": (-5.12, 5.12),
            "p": (-5.12, 5.12),
            "q": (-5.12, 5.12),
            "r": (-5.12, 5.12),
            "s": (-5.12, 5.12),
            "t": (-5.12, 5.12),
            "u": (-5.12, 5.12),
            "v": (-5.12, 5.12),
            "w": (-5.12, 5.12),
            "x": (-5.12, 5.12),
            "y": (-5.12, 5.12),
            "z": (-5.12, 5.12),
            "A1": (-5.12, 5.12),
            "B1": (-5.12, 5.12),
            "C1": (-5.12, 5.12),
            "D1": (-5.12, 5.12),
        }
    elif fname == "rastrigin":
        function = rastrigin
        limits = {
            "a": (-5.12, 5.12),
            "b": (-5.12, 5.12),
            "c": (-5.12, 5.12),
            "d": (-5.12, 5.12),
            "e": (-5.12, 5.12),
            "f": (-5.12, 5.12),
            "g": (-5.12, 5.12),
            "h": (-5.12, 5.12),
            "i": (-5.12, 5.12),
            "j": (-5.12, 5.12),
            "k": (-5.12, 5.12),
            "l": (-5.12, 5.12),
            "m": (-5.12, 5.12),
            "n": (-5.12, 5.12),
            "o": (-5.12, 5.12),
            "p": (-5.12, 5.12),
            "q": (-5.12, 5.12),
            "r": (-5.12, 5.12),
            "s": (-5.12, 5.12),
            "t": (-5.12, 5.12),
        }

    elif fname == "griewank":
        function = griewank
        limits = {
            "a": (-600.0, 600.0),
            "b": (-600.0, 600.0),
            "c": (-600.0, 600.0),
            "d": (-600.0, 600.0),
            "e": (-600.0, 600.0),
            "f": (-600.0, 600.0),
            "g": (-600.0, 600.0),
            "h": (-600.0, 600.0),
            "i": (-600.0, 600.0),
            "j": (-600.0, 600.0),
        }
    elif fname == "schwefel":
        function = schwefel
        limits = {
            "a": (-500.0, 500.0),
            "b": (-500.0, 500.0),
            "c": (-500.0, 500.0),
            "d": (-500.0, 500.0),
            "e": (-500.0, 500.0),
            "f": (-500.0, 500.0),
            "g": (-500.0, 500.0),
            "h": (-500.0, 500.0),
            "i": (-500.0, 500.0),
            "j": (-500.0, 500.0),
        }
    else:
        ValueError(f"Function {fname} undefined...exiting")

    return function, limits


def parse_arguments(
    propulate_comm: MPI.Comm = MPI.COMM_WORLD,
) -> Tuple[argparse.Namespace, Dict[str, bool]]:
    """
    Set up argument parser for Propulate optimization of simple mathematical functions.

    Parameters
    ----------
    propulate_comm : MPI.Comm, optional
        The communicator used to run the Propulate optimization. Default is ``MPI.COMM_WORLD``.

    Returns
    -------
    Namespace
        The namespace of all parsed arguments.
    Dict[str, bool]
        A dictionary logging if one of the PSO hyperparameters was actually set. Only relevant for PSO.
    """
    parser = argparse.ArgumentParser(
        prog="Simple Propulator example",
        description="Set up and run a basic Propulator optimization of mathematical functions.",
    )
    parser.add_argument(  # Function to optimize
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
        ],
        default="sphere",
    )
    parser.add_argument(
        "--generations", type=int, default=1000
    )  # Number of generations
    parser.add_argument(
        "--seed", type=int, default=0
    )  # Seed for Propulate random number generator
    parser.add_argument("--verbosity", type=int, default=1)  # Verbosity level
    parser.add_argument(
        "--checkpoint", type=str, default="./"
    )  # Path for loading and writing checkpoints.
    parser.add_argument(
        "--pop_size", type=int, default=2 * propulate_comm.size
    )  # Breeding pool size
    parser.add_argument(
        "--crossover_probability", type=float, default=0.7
    )  # Crossover probability
    parser.add_argument(
        "--mutation_probability", type=float, default=0.4
    )  # Mutation probability
    parser.add_argument("--random_init_probability", type=float, default=0.1)
    parser.add_argument("--top_n", type=int, default=1)
    parser.add_argument("--logging_interval", type=int, default=10)
    parser.add_argument("--logging_level", type=int, default=logging.INFO)

    # -------- Island-model specific arguments (ignored if not needed) --------
    parser.add_argument(
        "--num_islands", type=int, default=2
    )  # Number of separate evolutionary islands
    parser.add_argument(
        "--migration_probability", type=float, default=0.9
    )  # Migration probability
    parser.add_argument("--num_migrants", type=int, default=1)
    parser.add_argument("--pollination", action="store_true")

    # -------- PSO-specific arguments (ignored if not needed) --------
    parser.add_argument(
        "--variant",
        type=str,
        choices=["Basic", "VelocityClamping", "Constriction", "Canonical"],
        default="Basic",
    )  # PSO variant to run
    hp_set: Dict[str, bool] = {
        "inertia": False,
        "cognitive": False,
        "social": False,
    }

    class ParamSettingCatcher(argparse.Action):
        """Extend ``argparse``'s ``Action`` class to enable logging if one of the PSO hyperparameters was set."""

        def __call__(self, parser, namespace, values, option_string=None):  # type: ignore
            hp_set[self.dest] = True
            super().__call__(parser, namespace, values, option_string)

    parser.add_argument(
        "--inertia", type=float, default=0.729, action=ParamSettingCatcher
    )  # Inertia weight
    parser.add_argument(
        "--cognitive", type=float, default=1.49445, action=ParamSettingCatcher
    )  # Cognitive factor
    parser.add_argument(
        "--social", type=float, default=1.49445, action=ParamSettingCatcher
    )  # Social factor
    parser.add_argument(
        "--clamping_factor", type=float, default=0.6
    )  # Velocity clamping factor

    # -------- CMA-ES specific arguments (ignored if not needed)
    parser.add_argument("--adapter", type=str, default="basic")

    # -------- Multi-rank worker specific arguments (ignored if not needed)
    parser.add_argument(
        "--ranks_per_worker", type=int, default=2
    )  # Number of sub ranks that each worker will use

    return parser.parse_args(), hp_set
