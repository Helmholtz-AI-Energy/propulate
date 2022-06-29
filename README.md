![Propulate Logo](./LOGO.svg)

# Parallel Propagator of Populations
## Project Status
[![License: BSD-3](https://img.shields.io/badge/License-BSD--3-blue)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

``Propulate`` is a massively parallel evolutionary hyperparameter optimizer based on the island model with asynchronous propagation of populations and asynchronous migration. 
In contrast to classical GAs, ``Propulate`` maintains a continuous population of already evaluated individuals with a softened notion of the typically strictly separated, discrete generations. 
Our contributions include:
- A novel parallel genetic algorithm based on a fully asynchronized island model with independently processing workers. 
- Massive parallelism by asynchronous propagation of continuous populations and migration via efficient communication using the message passing interface.
- Optimized use efficiency of parallel hardware by minimizing idle times in distributed computing environments.

To be more efficient, the generations are less well separated than they usually are in evolutionary algorithms.
New individuals are generated from a pool of currently active, already evaluated individuals that may be from any generation. 
Individuals may be removed from the breeding population based on different criteria.

## Documentation

For usage example, see scripts.

## Installation

Pull and run ``pip install -e .`` or ``python setup.py develop``.
Requires an MPI implementation (currently only tested with OpenMPI) and ``mpi4py``.

## To Dos

- weight/parameter succession from parents or hall of fame
- more algorithms and operators, covariance matrix adaptation evolution strategy

## Acknowledgments
*This work is supported by the Helmholtz AI platform grant.*
<div align="center"; style="position:absolute;top:50%;left:50%;">
  <a href="http://www.kit.edu/english/index.php"><img src=./.figs/logo_KIT.svg height="50px" hspace="5%" vspace="0px"></a><a href="https://www.helmholtz.ai"><img src=./.figs/logo_HelmholtzAI.svg height="25px" hspace="5%" vspace="0px"></a>
</div>
