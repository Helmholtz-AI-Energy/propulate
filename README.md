![Propulate Logo](./LOGO.svg)

# propulate


Parallel propagator of populations.

## Description

Evolution-inspired hyperparameter-optimization in MPI-parallelized fashion.
To be more efficient generations are less well separated than they usually are in evolutionary algorithms.
Instead, a new individual is generated from a pool of currently active, already evaluated individuals that may be from any generation.
Individuals may be removed from the breeding population based on different criteria.

## Documentation

For usage example see scripts.

## Installation

Pull and run ``pip install -e .`` or ``python setup.py develop``.
Requires an MPI implementation (currently only tested with OpenMPI) and ``mpi4py``.

## To Dos

- weight/parameter succession from parents or hall of fame
- more algorithms and operators, covariance matrix adaptation evolution strategy
