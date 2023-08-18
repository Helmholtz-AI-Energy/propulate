.. _tut_islands:

Multi-island optimization of a mathematical function
====================================================
You can find the corresponding ``Python`` script here:
https://github.com/Helmholtz-AI-Energy/propulate/blob/master/scripts/islands_example.py

Next, we want to go one step further and minimize the sphere function using ``Propulate``'s **asynchronous island model**.
For this purpose, ``Propulate`` provides a specific class called ``Islands``. The basic procedure, including defining
the search space, the loss function to optimize, and the evolutionary operator, is the same as for the asynchronous
evolutionary optimization without islands (or rather with only one island) before.
In addition, we need to set up a couple of more things to configure, that is the islands themselves as well as the migration
process between them. This includes

* the number of islands (``num_islands``) or, alternatively, the distribution of compute resources over the islands
  (``island_sizes``),
* the number of migrants (``num_migrants``)
* the migration topology (``migration_topology``) and probability (``migration_probability``),
* whether we want to perform actual migration or pollination (``pollination``), and
* how to choose the migrants from the population (``emigration_propagator`` and ``immigration_propagator``).

The migration topology is a quadratic matrix of size ``num_islands * num_islands`` where entry :math:`\left(i,j\right)`
specifies the number of individuals that island :math:`i` sends to island :math:`j` in case of migration. Below, you see
how to set up a fully connected topology, where each island sends ``num_migrants`` of its best individuals to each other
island. With ``num_migrants = 1``, this also is the default behaviour in ``Propulate``:

.. code-block:: python

    # Set up fully connected migration topology.
    migration_topology = config.num_migrants * np.ones(
        (config.num_islands, config.num_islands),
        dtype=int)
    np.fill_diagonal(migration_topology, 0)  # An island does not send migrants to itself.

Next, we set up the island model itself using the ``Islands`` class. In addition to the ``Propulator`` arguments defining
the islands' internal asynchronous evolutionary optimization process, ``Islands`` takes all migration-relevant arguments
for setting up the island model and migration:

.. code-block:: python

    islands = Islands(    # Set up island model.
        loss_fn=sphere,  # Loss function
        propagator=propagator,  # Evolutionary operator
        rng=rng,  # Separate random number generator for the optimization process
        generations=config.generations,  # Number of generations each worker performs
        num_islands=config.num_islands,  # Number of evolutionary islands
        migration_topology=migration_topology,  # Migration topology
        migration_probability=config.migration_probability,  # Migration probability
        emigration_propagator=SelectMin,  # How to choose emigrants
        immigration_propagator=SelectMax,  # How to choose individuals to be replaced by migrants in case of pollination
        pollination=config.pollination,  # Whether to perform actual migration or pollination
        checkpoint_path=config.checkpoint)  # Checkpoint path

This will set up an island model with ``num_island`` islands and distribute the available compute resources as equally
as possible over all islands. For example, consider a parallel computing environment with overall 40 processing elements.
If we set ``num_islands = 4``, we get four islands with ten workers each. If we set ``num_islands = 6``, we get six
islands, where four of them have seven workers and the remaining two have six workers. Alternatively, you can set the
worker distribution directly using ``island_sizes``, e.g., ``island_sizes = numpy.array([10, 10, 10, 10])`` for four
islands with ten workers each.

Now, we are ready to run the optimization:

.. code-block:: python

    islands.evolve(  # Run actual optimization.
        top_n=config.top_n, # Number of best individuals to print in the summary.
        logging_interval=config.logging_int, # Logging interval
        debug=config.verbosity)  # Verbosity level


.. note::
    ``Propulate`` creates a separate checkpoint for each island. Checkpoints are only compatible between runs that use
    the same island model and parallel computing environment.

You can run the example script ``islands_example.py``:

.. code-block::

    mpirun --use-hwthread-cpus python propulator_example.py