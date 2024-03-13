# -*- coding: utf-8 -*-
import logging
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Union, Tuple

import colorlog
import numpy as np
from mpi4py import MPI

from .population import Individual, Particle
from .propagators import (
    Compose,
    Conditional,
    InitUniform,
    IntervalMutationNormal,
    CrossoverUniform,
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
    crossover_prob: float = 0.7,
    mutation_prob: float = 0.4,
    random_init_prob: float = 0.1,
    sigma_factor: float = 0.05,
    rng: Optional[random.Random] = None,
) -> Propagator:
    """
    Get Propulate's default evolutionary optimization propagator.

    Parameters
    ----------
    pop_size : int
        The number of individuals in the breeding population.
    limits : Dict[str, Tuple[float, float]] | Dict[str, Tuple[int, int]] | Dict[str, Tuple[str, ...]]
        The (hyper-)parameters to be optimized, i.e., the search space.
    crossover_prob : float, optional
        The uniform-crossover probability. Default is 0.7.
    mutation_prob : float, optional
        The point-mutation probability. Default is 0.4.
    random_init_prob : float, optional
        The random-initialization probability. Default is 0.1.
    sigma_factor : float
        The scaling factor for obtaining the standard deviation from the search-space boundaries for interval mutation.
        Default is 0.05.
    rng : random.Random, optional
        The separate random number generator for the Propulate optimization.

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
                CrossoverUniform(crossover_prob, rng=rng),
                PointMutation(limits, probability=mutation_prob, rng=rng),
                IntervalMutationNormal(
                    limits, sigma_factor=sigma_factor, probability=1.0, rng=rng
                ),
                InitUniform(limits, parents=1, probability=random_init_prob, rng=rng),
            ]
        )
    else:
        propagator = Compose(
            [  # Compose propagator out of basic evolutionary operators with Compose(...).
                SelectMin(pop_size),
                SelectUniform(offspring=2, rng=rng),
                CrossoverUniform(crossover_prob, rng=rng),
                PointMutation(limits, probability=mutation_prob, rng=rng),
                InitUniform(limits, parents=1, probability=random_init_prob, rng=rng),
            ]
        )

    init = InitUniform(limits, rng=rng)
    propagator = Conditional(
        pop_size, propagator, init
    )  # Initialize random if population size < specified `pop_size`.
    return propagator


def set_logger_config(
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    log_to_stdout: bool = True,
    log_rank: bool = False,
    colors: bool = True,
) -> None:
    """
    Set up the logger. Should only need to be done once. Generally, logging should only be done on the master rank.

    Parameters
    ----------
    level : int
        The default level for logging. Default is ``logging.INFO``.
    log_file : str | Path, optional
        The file to save the log to.
    log_to_stdout : bool
        A flag indicating if the log should be printed on stdout. Default is True.
    log_rank : bool
        A flag for prepending the MPI rank to the logging message. Default is False.
    colors : bool
        A flag for using colored logs. Default is True.
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


def make_particle(individual: Individual) -> Particle:
    """
    Convert individuals to particles.

    Parameters
    ----------
    individual : propulate.Individual
        An individual to be converted to a particle.

    Returns
    --------
    propulate.Particle
        The converted individual.
    """
    p = Particle(generation=individual.generation)
    p.position = np.zeros(len(individual))
    p.velocity = np.zeros(len(individual))
    for i, k in enumerate(individual):
        p.position[i] = individual[k]
        p[k] = individual[k]
    return p
