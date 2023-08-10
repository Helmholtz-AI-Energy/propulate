from typing import Callable, Dict, Tuple
import numpy as np


def bukin_n6(params: Dict[str, float]) -> float:
    """
    Bukin N.6 function: continuous, convex, non-separable, non-differentiable, multimodal

    Input domain: -15 <= x <= -5, -3 <= y <= 3
    Global minimum 0: (x, y) = (-10, 1)

    Parameters
    ----------
    params: dict
            function parameters

    Returns
    -------
    float: function value
    """
    x = params["x"]
    y = params["y"]
    return 100 * np.sqrt(np.abs(y - 0.01 * x ** 2)) + 0.01 * np.abs(x + 10)


def egg_crate(params: Dict[str, float]) -> float:
    """
    Egg-crate function: continuous, non-convex, separable, differentiable, multimodal

    Input domain: -5 <= x, y <= 5
    Global minimum -1 at (x, y) = (0, 0)

    Parameters
    ----------
    params: dict
            function parameters

    Returns
    -------
    float: function value
    """
    x = params["x"]
    y = params["y"]
    return x ** 2 + y ** 2 + 25 * (np.sin(x) ** 2 + np.sin(y) ** 2)


def himmelblau(params: Dict[str, float]) -> float:
    """
    Himmelblau function: continuous, non-convex, non-separable, differentiable, multimodal

    Input domain: -6 <= x, y <= 6
    Global minimum 0 at (x, y) = (3, 2)

    Parameters
    ----------
    params: dict
            function parameters

    Returns
    -------
    float: function value
    """
    x = params["x"]
    y = params["y"]
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def keane(params: Dict[str, float]) -> float:
    """
    Keane function: continuous, non-convex, non-separable, differentiable, multimodal

    Input domain: -10 <= x, y <= 10
    Global minimum 0.6736675 at (x, y) = (1.3932491, 0) and (x, y) = (0, 1.3932491)

    Parameters
    ------
    params: dict
            function parameters

    Returns
    -------
    float: function value
    """
    x = params["x"]
    y = params["y"]
    return -np.sin(x - y) ** 2 * np.sin(x + y) ** 2 / np.sqrt(x ** 2 + y ** 2)


def leon(params: Dict[str, float]) -> float:
    """
    Leon function: continuous, non-convex, non-separable, differentiable, non-multimodal, non-random, non-parametric

    Input domain: 0 <= x, y <= 10
    Global minimum 0 at (x, y) =(1, 1)

    Parameters
    ----------
    params: dict
            function parameters

    Returns
    -------
    float: function value
    """
    x = params["x"]
    y = params["y"]
    return 100 * (y - x ** 3) ** 2 + (1 - x) ** 2


def rastrigin(params: Dict[str, float]) -> float:
    """
    Rastrigin function: continuous, non-convex, separable, differentiable, multimodal

    Input domain: -5.12 <= x, y <= 5.12
    Global minimum -20 at (x, y) = (0, 0)

    Parameters
    ----------
    params: dict
            function parameters

    Returns
    -------
    float: function value
    """
    x = params["x"]
    y = params["y"]
    return x ** 2 - 10 * np.cos(2 * np.pi * x) + y ** 2 - 10 * np.cos(2 * np.pi * y)


def schwefel(params: Dict[str, float]) -> float:
    """
    Schwefel 2.20 function: continuous, convex, separable, non-differentiable, non-multimodal

    Input domain: -100 <= x, y <= 100
    Global minimum 0 at (x, y) = (0, 0)

    Params
    ------
    params: dict
            function parameters

    Returns
    -------
    float: function value
    """
    params = np.array(list(params.values()))
    return len(params) * 418.982887 - np.sum(params * np.sin(np.sqrt(np.abs(params))))


def sphere(params: Dict[str, float]) -> float:
    """
    Sphere function: continuous, convex, separable, differentiable, unimodal

    Input domain: -5.12 <= x, y <= 5.12
    Global minimum 0 at (x, y) = (0, 0)
    Params
    ------
    params : dict
             function parameters
    Returns
    -------
    float : function value
    """
    return np.sum(np.array(list(params.values())) ** 2)


def step(params: Dict[str, float]) -> float:
    """
    Step function

    Input domain: -5 <= (x_i) <= 5
    Global minimum 0 at (x_i) = -5, i = 1,...,N

    Params
    ------
    params: dict
            function parameters

    Returns
    -------
    float: function value
    """
    params = np.array(list(params.values()))
    return np.sum(np.floor(params))


def get_function_search_space(fname: str) -> (Callable, Dict[str, Tuple[float, float]]):
    """
    Get search space limits and function from function name.

    Params
    ------
    fname: str
           function name

    Returns
    -------
    Callable: function
    dict: search space
    """
    if fname == "bukin":
        function = bukin_n6
        limits = {
            "x": (-15.0, -5.0),
            "y": (-3.0, 3.0),
        }
    elif fname == "eggcrate":
        function = egg_crate
        limits = {
            "x": (-5.0, 5.0),
            "y": (-5.0, 5.0),
        }
    elif fname == "himmelblau":
        function = himmelblau
        limits = {
            "x": (-6.0, 6.0),
            "y": (-6.0, 6.0),
        }
    elif fname == "keane":
        function = keane
        limits = {
            "x": (-10.0, 10.0),
            "y": (-10.0, 10.0),
        }
    elif fname == "leon":
        function = leon
        limits = {
            "x": (0.0, 10.0),
            "y": (0.0, 10.0),
        }
    elif fname == "rastrigin":
        function = rastrigin
        limits = {
            "x": (-5.12, 5.12),
            "y": (-5.12, 5.12),
        }
    elif fname == "schwefel":
        function = schwefel
        limits = {
            "x": (-500.0, 500.0),
            "y": (-500.0, 500.0),
        }
    elif fname == "sphere":
        function = sphere
        limits = {
            "x": (-5.12, 5.12),
            "y": (-5.12, 5.12),
        }
    elif fname == "step":
        function = step
        limits = {
            "x": (-5., 5.),
            "y": (-5., 5.),
        }
    else:
        ValueError(f"Function {fname} undefined...exiting")

    return function, limits
