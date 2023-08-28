.. _tut_islands:

Multi-Island Optimization of a Mathematical Function
====================================================

.. note::

   You can find the corresponding ``Python`` script here:
   https://github.com/Helmholtz-AI-Energy/propulate/blob/master/tutorials/islands_example.py

Next, we want to go one step further and minimize the sphere function using ``Propulate``'s *asynchronous island model*.
``Propulate`` provides a specific class called ``Islands`` for this. The basic procedure, including defining
the search space, the loss function to optimize, and the evolutionary operator, is the same as for the asynchronous
evolutionary optimization without islands (or rather one island) before.
In addition, we need to configure a couple of more things, that is the islands themselves as well as the migration
process between them. This includes:

* the number of islands (``num_islands``) or, alternatively, the distribution of compute resources over the islands
  (``island_sizes``)
* the number of migrants (``num_migrants``)
* the migration topology (``migration_topology``) and probability (``migration_probability``)
* whether we want to perform actual migration or pollination (``pollination``)
* how to choose the migrants from the population (``emigration_propagator`` and ``immigration_propagator``).

The migration topology is a quadratic matrix of size ``num_islands * num_islands`` where entry :math:`\left(i,j\right)`
specifies the number of individuals that island :math:`i` sends to island :math:`j` in case of migration. Below, you see
how to set up a fully connected topology, where each island sends ``num_migrants`` of its best individuals to each other
island. With ``num_migrants = 1``, this is the default behaviour in ``Propulate``:

.. code-block:: python

    migration_topology = config.num_migrants * np.ones(  # Set up fully connected migration topology.
        (config.num_islands, config.num_islands),
        dtype=int)
    np.fill_diagonal(migration_topology, 0)  # An island does not send migrants to itself.

Next, we set up the island model itself using the ``Islands`` class. In addition to the ``Propulator`` arguments defining
the islands' internal asynchronous optimization process, ``Islands`` takes all migration-relevant arguments
to configure the islands and the migration between them:

.. code-block:: python

    islands = Islands(  # Set up island model.
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

This will instantiate an island model with ``num_island`` islands and distribute the available compute resources as equally
as possible over all islands. For example, consider a parallel computing environment with overall 40 processing elements.
If we set ``num_islands = 4``, we get four islands with ten workers each. If we set ``num_islands = 6``, we get six
islands, where four of them have seven workers and the remaining two have six workers. Alternatively, you can set the
worker distribution directly using ``island_sizes``, e.g., ``island_sizes = numpy.array([10, 10, 10, 10])`` for four
islands with ten workers each. This allows for heterogeneous setups if desired.

Now we are ready to run the optimization:

.. code-block:: python

    islands.evolve(  # Run actual optimization.
        top_n=config.top_n, # Number of best individuals to print in the summary.
        logging_interval=config.logging_int, # Logging interval
        debug=config.verbosity)  # Verbosity level


.. note::
    ``Propulate`` creates a separate checkpoint for each island. Checkpoints are only compatible between runs that use
    the same island model and parallel computing environment.

You can run the example script ``islands_example.py``:

.. code-block:: console

    $ mpirun --use-hwthread-cpus python propulator_example.py

With ten MPI ranks and two islands with five workers each, the output looks like this:

.. code-block:: text

    #################################################
    # PROPULATE: Parallel Propagator of Populations #
    #################################################

            ⠀⠀⠀⠈⠉⠛⢷⣦⡀⠀⣀⣠⣤⠤⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀        ⠀⠀⠀⠀⠀⣀⣻⣿⣿⣿⣋⣀⡀⠀⠀⢀⣠⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀        ⠀⠀⣠⠾⠛⠛⢻⣿⣿⣿⠟⠛⠛⠓⠢⠀⠀⠉⢿⣿⣆⣀⣠⣤⣀⣀⠀⠀⠀
    ⠀        ⠀⠘⠁⠀⠀⣰⡿⠛⠿⠿⣧⡀⠀⠀⢀⣤⣤⣤⣼⣿⣿⣿⡿⠟⠋⠉⠉⠀⠀
    ⠀        ⠀⠀⠀⠀⠠⠋⠀⠀⠀⠀⠘⣷⡀⠀⠀⠀⠀⠹⣿⣿⣿⠟⠻⢶⣄⠀⠀⠀⠀
    ⠀⠀        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣧⠀⠀⠀⠀⢠⡿⠁⠀⠀⠀⠀⠈⠀⠀⠀⠀
    ⠀⠀        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿⡄⠀⠀⢠⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⣾⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀        ⣤⣤⣤⣤⣤⣤⡤⠄⠀⠀⣀⡀⢸⡇⢠⣤⣁⣀⠀⠀⠠⢤⣤⣤⣤⣤⣤⣤⠀
    ⠀⠀⠀⠀⠀        ⠀⣀⣤⣶⣾⣿⣿⣷⣤⣤⣾⣿⣿⣿⣿⣷⣶⣤⣀⠀⠀⠀⠀⠀⠀
            ⠀⠀⠀⣠⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⣄⠀⠀⠀
    ⠀        ⠀⠼⠿⣿⣿⠿⠛⠉⠉⠉⠙⠛⠿⣿⣿⠿⠛⠛⠛⠛⠿⢿⣿⣿⠿⠿⠇⠀⠀
    ⠀        ⢶⣤⣀⣀⣠⣴⠶⠛⠋⠙⠻⣦⣄⣀⣀⣠⣤⣴⠶⠶⣦⣄⣀⣀⣠⣤⣤⡶⠀
            ⠀⠀⠈⠉⠉⠉⠀⠀⠀⠀⠀⠀⠀⠉⠉⠉⠉⠀⠀⠀⠀⠀⠉⠉⠉⠉⠀⠀⠀⠀

    Worker distribution [0 0 0 0 0 1 1 1 1 1] with island counts [5 5] and island displacements [0 5].
    Migration topology [[0 1]
     [1 0]] has shape (2, 2).
    NOTE: Island migration probability 0.9 results in per-rank migration probability 0.18.
    Starting parallel optimization process.
    NOTE: No valid checkpoint file given. Initializing population randomly...
    Island 1 has 5 workers.
    No pollination.
    NOTE: No valid checkpoint file given. Initializing population randomly...
    Island 0 has 5 workers.
    Island 1 Worker 1: In generation 0...
    Island 1 Worker 2: In generation 0...
    Island 0 Worker 2: In generation 0...
    Island 0 Worker 3: In generation 0...
    Island 0 Worker 0: In generation 0...
    Island 1 Worker 0: In generation 0...
    Island 1 Worker 3: In generation 0...
    Island 0 Worker 4: In generation 0...
    Island 1 Worker 4: In generation 0...
    Island 0 Worker 1: In generation 0...

    ...

    Island 0 Worker 0: In generation 980...
    Island 1 Worker 4: In generation 990...
    Island 1 Worker 2: In generation 940...
    Island 0 Worker 0: In generation 990...
    Island 1 Worker 2: In generation 950...
    Island 1 Worker 2: In generation 960...
    Island 1 Worker 2: In generation 970...
    Island 1 Worker 2: In generation 980...
    Island 1 Worker 2: In generation 990...
    OPTIMIZATION DONE.
    NEXT: Final checks for incoming messages...

    ###########
    # SUMMARY #
    ###########

    Number of currently active individuals is 10000.
    Expected overall number of evaluations is 10000.
    Top 1 result(s) on island 1:
    (1): [{'a': '-1.84E-4', 'b': '9.84E-4'}, loss 1.00E-6, island 1, worker 0, generation 248]

    Top 1 result(s) on island 0:
    (1): [{'a': '-1.84E-4', 'b': '9.84E-4'}, loss 1.00E-6, island 1, worker 0, generation 248]
