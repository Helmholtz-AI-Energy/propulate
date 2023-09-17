![Propulate Logo](./LOGO.svg)

# Parallel Propagator of Populations

[![DOI](https://zenodo.org/badge/495731357.svg)](https://zenodo.org/badge/latestdoi/495731357)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)
[![License: BSD-3](https://img.shields.io/badge/License-BSD--3-blue)](https://opensource.org/licenses/BSD-3-Clause)
![PyPI](https://img.shields.io/pypi/v/propulate)
![PyPI - Downloads](https://img.shields.io/pypi/dm/propulate)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/7785/badge)](https://www.bestpractices.dev/projects/7785)
[![](https://img.shields.io/badge/Contact-marie.weiel%40kit.edu-orange)](mailto:marie.weiel@kit.edu)
[![Documentation Status](https://readthedocs.org/projects/propulate/badge/?version=latest)](https://propulate.readthedocs.io/en/latest/?badge=latest)

# **Click [here](https://www.scc.kit.edu/en/aboutus/16956.php) to watch our 3 min introduction video!** 

## What `Propulate` can do for you

`Propulate` is an HPC-tailored software for solving optimization problems in parallel. It is openly accessible and easy to use. Compared to a widely used competitor, `Propulate` is consistently faster - at least an order of magnitude for a set of typical benchmarks - and in some cases even more accurate.

Inspired by biology, `Propulate` borrows mechanisms from biological evolution, such as selection, recombination, and mutation. Evolution begins with a population of solution candidates, each with randomly initialized genes. It is an iterative "survival of the fittest" process where the population at each iteration can be viewed as a generation. For each generation, the fitness of each candidate in the population is evaluated. The genes of the fittest candidates are incorporated in the next generation.

Like in nature, `Propulate` does not wait for all compute units to finish the evaluation of the current generation. Instead, the compute units communicate the currently available information and use that to breed the next candidate immediately. This avoids waiting idly for other units and thus a load imbalance.
Each unit is responsible for evaluating a single candidate. The result is a fitness level corresponding with that candidate’s genes, allowing us to compare and rank all candidates. This information is sent to other compute units as soon as it becomes available.
When a unit is finished evaluating a candidate and communicating the resulting fitness, it breeds the candidate for the next generation using the fitness values of all candidates it evaluated and received from other units so far. 

`Propulate` can be used for hyperparameter optimization and neural architecture search. 
It was already successfully applied in several accepted scientific publications. Applications include grid load forecasting, remote sensing, and structural molecular biology.

## In more technical terms

``Propulate`` is a massively parallel evolutionary hyperparameter optimizer based on the island model with asynchronous propagation of populations and asynchronous migration.
In contrast to classical GAs, ``Propulate`` maintains a continuous population of already evaluated individuals with a softened notion of the typically strictly separated, discrete generations.
Our contributions include:
- A novel parallel genetic algorithm based on a fully asynchronized island model with independently processing workers.
- Massive parallelism by asynchronous propagation of continuous populations and migration via efficient communication using the message passing interface.
- Optimized use efficiency of parallel hardware by minimizing idle times in distributed computing environments.

To be more efficient, the generations are less well separated than they usually are in evolutionary algorithms.
New individuals are generated from a pool of currently active, already evaluated individuals that may be from any generation.
Individuals may be removed from the breeding population based on different criteria.

You can find the corresponding publication [here](https://doi.org/10.1007/978-3-031-32041-5_6):  
>Taubert, O. *et al.* (2023). Massively Parallel Genetic Optimization Through Asynchronous Propagation of Populations. In: Bhatele, A., Hammond, J., Baboulin, M., Kruse, C. (eds) High Performance Computing. ISC High Performance 2023. Lecture Notes in Computer Science, vol 13948. Springer, Cham. [doi.org/10.1007/978-3-031-32041-5_6](https://doi.org/10.1007/978-3-031-32041-5_6)

## Documentation

Check out the full documentation at [https://propulate.readthedocs.io/](https://propulate.readthedocs.io/) :rocket:! Here you can 
find installation instructions, tutorials, theoretical background, and API references.

## Installation

You can install the latest stable release from PyPI: ``pip install propulate``  
If you need the latest updates, you can also install ``Propulate`` directly from the master branch at you own risk.
Pull and run ``pip install -e .`` or ``python setup.py develop``.  
``Propulate`` depends on ``mpi4py`` and requires an MPI implementation under the hood. Currently, it is only tested with 
OpenMPI.

## Acknowledgments
*This work is supported by the Helmholtz AI platform grant.*
![](./.figs/hai_kit_logos.svg)



