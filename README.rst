=========
propulate
=========


Parallel propagator of populations.


Description
===========

Evolution-inspired hyperparameter-optimization in MPI-parallelized fashion.
In order to be more efficient generations are less well separated than they often are in evolutionary algorithms.
Instead a new individual is generated from a pool of currently active already evaluated individuals that may be from any generation.
Individuals may be removed from the breeding population based on different criteria.

Documentation
=============

For usage example see scripts.

Installation
============

Pull and run ``pip install -e .`` or ``python setup.py develop``
Requires a MPI  implementation (currently only tested with  OpenMPI) and ``mpi4py`` 

TODO
====

- checkpointing + example
- soft reproducibility + tests
- add thread based coordinator back in for systems where that works with the MPI
- weight/parameter succession from parents or hall of fame
- 

Note
====

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.

