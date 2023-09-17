# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import colorlog
import random
import sys
from mpi4py import MPI
from typing import Dict, Union, Tuple

from .propagators import (
    Compose,
    Conditional,
    InitUniform,
    IntervalMutationNormal,
    MateUniform,
    PointMutation,
    Propagator,
    SelectMin,
    SelectUniform,
)


def get_default_propagator(
    pop_size: int,
    limits: Union[
        Dict[str, Tuple[float, float]],
        Dict[str, Tuple[int, int]],
        Dict[str, Tuple[str, ...]],
    ],
    mate_prob: float,
    mut_prob: float,
    random_prob: float,
    sigma_factor: float = 0.05,
    rng: random.Random = None,
) -> Propagator:
    """
    Get Propulate's default evolutionary optimization propagator.

    Parameters
    ----------
    pop_size: int
              number of individuals in breeding population
    limits: dict
            (hyper-)parameters to be optimized, i.e., search space
    mate_prob: float
               uniform-crossover probability
    mut_prob: float
              point-mutation probability
    random_prob: float
                 random-initialization probability
    sigma_factor: float
                  scaling factor for obtaining std from search-space boundaries for interval mutation
    rng: random.Random
         random number generator

    Returns
    -------
    propagators.Propagator
        A basic evolutionary optimization propagator.
    """
    if any(
        isinstance(limits[x][0], float) for x in limits
    ):  # Check for existence of at least one continuous trait.
        propagator = Compose(
            [  # Compose propagator out of basic evolutionary operators with Compose(...).
                SelectMin(pop_size),
                SelectUniform(offspring=2, rng=rng),
                MateUniform(mate_prob, rng=rng),
                PointMutation(limits, probability=mut_prob, rng=rng),
                IntervalMutationNormal(
                    limits, sigma_factor=sigma_factor, probability=1.0, rng=rng
                ),
                InitUniform(limits, parents=1, probability=random_prob, rng=rng),
            ]
        )
    else:
        propagator = Compose(
            [  # Compose propagator out of basic evolutionary operators with Compose(...).
                SelectMin(pop_size),
                SelectUniform(offspring=2, rng=rng),
                MateUniform(mate_prob, rng=rng),
                PointMutation(limits, probability=mut_prob, rng=rng),
                InitUniform(limits, parents=1, probability=random_prob, rng=rng),
            ]
        )

    init = InitUniform(limits, rng=rng)
    propagator = Conditional(
        pop_size, propagator, init
    )  # Initialize random if population size < specified `pop_size`.

    return propagator


def set_logger_config(
    level: int = logging.INFO,
    log_file: Union[str, Path] = None,
    log_to_stdout: bool = True,
    log_rank: bool = False,
    colors: bool = True,
) -> None:
    """
    Set up the logger. Should only need to be done once.
    Generally, logging should only be done on the master rank.

    Parameters
    ----------
    level: logging.INFO, ...
           default level for logging
           default: INFO
    log_file: str, Path
              file to save the log to
              default: None
    log_to_stdout: bool
                   flag indicating if the log should be printed on stdout
                   default: True
    log_rank: bool
              flag for prepending the MPI rank to the logging message
    colors: bool
            flag for using colored logs
            default: True
    """
    rank = f"{MPI.COMM_WORLD.Get_rank()}:" if log_rank else ""
    # Get base logger for Propulate.
    base_logger = logging.getLogger("propulate")
    simple_formatter = logging.Formatter(
        f"{rank}:[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    )
    if colors:
        formatter = colorlog.ColoredFormatter(
            fmt=f"{rank}[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s]"
            f"[%(log_color)s%(levelname)s%(reset)s] - %(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
            secondary_log_colors={},
        )
        std_handler = logging.StreamHandler(stream=sys.stdout)
        std_handler.setFormatter(formatter)
    else:
        std_handler = logging.StreamHandler(stream=sys.stdout)
        std_handler.setFormatter(simple_formatter)

    if log_to_stdout:
        base_logger.addHandler(std_handler)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(simple_formatter)
        base_logger.addHandler(file_handler)
    base_logger.setLevel(level)
    return
