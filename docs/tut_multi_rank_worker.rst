.. _tut_multi_rank_worker:

Using Propulate with Multi-Rank Workers
=======================================
.. note::

   You can find the corresponding ``Python`` script here:
   https://github.com/Helmholtz-AI-Energy/propulate/blob/master/tutorials/multi_rank_workers_example.py

In addition to the already explained functionality, ``Propulate`` |:dna:| enables using multi-rank workers for an
internally parallelized evaluation of the loss function. This is useful for, e.g., data-parallel training of neural
networks during the hyperparameter optimization, where each individual network is trained on multiple GPUs.
To show you how this works, let us look at a simple toy example where we minimize a parallel version of the sphere
function. If you find this boring and already got the concept of multi-rank workers, you can jump directly to the
``PyTorch DistributedDataParallel`` tutorial :ref:`here<tut_ddp>`.

We consider eight processors distributed over two islands and two ranks per worker. This means one island has two
workers with two ranks each, i.e., ``ranks_per_worker=2``. The setup is illustrated below:

.. figure:: images/multi-rank-worker.png
   :width: 100 %
   :align: center

   **Distribution and assignment of processors with multi-rank workers.** In this example, the overall eight processors
   (MPI world, grey) are distributed over two islands (red). Each island has two workers (dark blue) with two ranks (white)
   each. Each worker and island has its own communicator. In addition, there is the so-called ``Propulate`` communicator
   (bright blue) consisting of each worker's internal rank 0.

Each rank of a worker calculates one of the squared terms :math:`x_i^2` in the (in this example) two-dimensional sphere
function:

.. math::

    f_\text{parallel sphere}\left(x_i; i=0,\dots,\texttt{ranks_per_worker}\right)=\sum_{i} x_i^2

In general, the parallel sphere function's dimension should equal the number of ranks per worker ``ranks_per_worker``.
The definition of the corresponding ``Python`` function is shown below. The only difference compared to the single-rank
worker case is that the loss function additionally takes in the worker's sub communicator as an input argument. The
splitting of all available processes into the required communicators on island and worker level is done by ``Propulate``
|:dna:| internally. The only thing you need to take care of is that the loss function returns the final evaluated value
on the worker's rank 0 as these ranks are also part of the ``Propulate`` communicator and responsible for the actual
optimization process. All communication required between a worker's ranks to achieve this must be implemented within the
loss function. In the parallel sphere function below, this is done by summing up the squared terms over all ranks in a
worker with ``allreduce``:

.. code-block:: python
  :emphasize-lines: 1, 14-15, 26


  def parallel_sphere(params: Dict[str, float], comm: MPI.Comm = MPI.COMM_SELF) -> float:
      """
      Parallel sphere function to showcase using multi-rank workers in Propulate.

      Sphere function: continuous, convex, separable, differentiable, unimodal

      Input domain: -5.12 <= x, y <= 5.12
      Global minimum 0 at (x, y) = (0, 0)

      Parameters
      ----------
      params : Dict[str, float]
          The function parameters.
      comm : MPI.Comm
          The communicator of the worker. Default is MPI.COMM_SELF, corresponding to the single-rank worker case.

      Returns
      -------
      float
          The function value.
      """
      if comm != MPI.COMM_SELF:  # Multi-rank worker case
          term = (
              list(params.values())[comm.rank] ** 2
          )  # Each rank squares one of the inputs.
          return comm.allreduce(term)  # Return the sum over all squared inputs.
      else:  # Backup for single-rank worker case
          return np.sum(np.array(list(params.values())) ** 2).item()

This function will be minimized in the main part of the script below. We use all available processors from the MPI world
communicator. The ``parse_arguments()`` function retrieves all user-provided command-line arguments and sets default
values for all other required parameters. In addition, we configure our logger and ``Propulate``'s random number
generator and define the search space for our parallel sphere function:

.. code-block:: python

  if __name__ == "__main__":
      full_world_comm = MPI.COMM_WORLD  # Get full world communicator.

      config, _ = parse_arguments()  # Parse user-provided command-line arguments.

      propulate.set_logger_config(
          log_file=f"{config.checkpoint}/{pathlib.Path(__file__).stem}.log"
      )  # Set up logger.

      rng = random.Random(
          config.seed + full_world_comm.rank
      )  # Separate random number generator for optimization.

      limits = {
          "a": (-5.12, 5.12),
          "b": (-5.12, 5.12),
      }  # Set search-space limits.

As before, we are now ready to set up the propagator used to breed new individuals from existing ones. We again use
``Propulate``'s evolutionary operator:

.. code-block:: python

      # Set up evolutionary operator.
      propagator = get_default_propagator(  # Get default evolutionary operator.
          pop_size=config.pop_size,  # Breeding pool size
          limits=limits,  # Search-space limits
          crossover_prob=config.crossover_probability,  # Crossover probability
          mutation_prob=config.mutation_probability,  # Mutation probability
          random_init_prob=config.random_init_probability,  # Random-initialization probability
          rng=rng,  # Separate random number generator for Propulate optimization
      )

Next, we set up the island model and run the actual optimization. The only difference to the single-rank worker case is
the ``ranks_per_worker`` argument which must be passed to the instantiated ``Islands`` object as shown below. Internally,
``Propulate`` takes care of splitting the available ranks into the required communicators. You only need to make sure
that the overall number of processors is evenly divisible by the number of ranks per worker:

.. code-block:: python

      # Set up island model.
      islands = Islands(
          loss_fn=parallel_sphere,  # Loss function to be minimized
          propagator=propagator,  # Propagator, i.e., evolutionary operator to be used
          rng=rng,  # Separate random number generator for Propulate optimization
          generations=config.generations,  # Overall number of generations
          num_islands=config.num_islands,  # Number of islands
          migration_probability=config.migration_probability,  # Migration probability
          emigration_propagator=SelectMin,  # Selection policy for emigrants
          immigration_propagator=SelectMax,  # Selection policy for immigrants
          pollination=config.pollination,  # Whether to use pollination or migration
          checkpoint_path=config.checkpoint,  # Checkpoint path
          # ----- SPECIFIC FOR MULTI-RANK UCS ----
          ranks_per_worker=config.ranks_per_worker,  # Number of ranks per (multi rank) worker
      )

      # Run actual optimization.
      islands.evolve(
          top_n=config.top_n,  # Print top-n best individuals on each island in summary.
          logging_interval=config.logging_interval,  # Logging interval
          debug=config.verbosity,  # Debug level
      )

You can run the script via:

.. code-block:: console

    $ mpirun -n 8 python multi_rank_workers_example.py

The output produced looks like this:

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

    [2024-06-04 12:35:23,858][propulate.islands][INFO] - Worker distribution [0 0 1 1] with island counts [2 2] and island displacements [0 2].
    [2024-06-04 12:35:23,859][propulate.islands][INFO] - Migration topology [[0 1]
     [1 0]] has shape (2, 2).
    [2024-06-04 12:35:23,859][propulate.islands][INFO] - NOTE: Island migration probability 0.9 results in per-rank migration probability 0.45.
    Starting parallel optimization process.
    [2024-06-04 12:35:23,859][propulate.islands][INFO] - Use island model with real migration.
    [2024-06-04 12:35:23,859][propulate.propulator][INFO] - No valid checkpoint file given. Initializing population randomly...
    [2024-06-04 12:35:23,859][propulate.migrator][INFO] - Island 0 has 2 workers.
    [2024-06-04 12:35:23,859][propulate.propulator][INFO] - No valid checkpoint file given. Initializing population randomly...
    [2024-06-04 12:35:23,860][propulate.migrator][INFO] - Island 1 has 2 workers.
    [2024-06-04 12:35:23,860][propulate.migrator][INFO] - Island 0 Worker 0: In generation 0...
    [2024-06-04 12:35:23,860][propulate.migrator][INFO] - Island 1 Worker 1: In generation 0...
    [2024-06-04 12:35:23,860][propulate.migrator][INFO] - Island 0 Worker 1: In generation 0...
    [2024-06-04 12:35:23,860][propulate.migrator][INFO] - Island 1 Worker 0: In generation 0...
    [2024-06-04 12:35:23,866][propulate.migrator][INFO] - Island 0 Worker 0: In generation 10...
    [2024-06-04 12:35:23,866][propulate.migrator][INFO] - Island 1 Worker 0: In generation 10...
    [2024-06-04 12:35:23,867][propulate.migrator][INFO] - Island 0 Worker 1: In generation 10...
    [2024-06-04 12:35:23,870][propulate.migrator][INFO] - Island 1 Worker 1: In generation 10...
    ...
    [2024-06-04 12:35:31,474][propulate.migrator][INFO] - Island 1 Worker 1: In generation 990...
    [2024-06-04 12:35:31,503][propulate.migrator][INFO] - Island 0 Worker 1: In generation 970...
    [2024-06-04 12:35:31,604][propulate.migrator][INFO] - Island 0 Worker 0: In generation 980...
    [2024-06-04 12:35:31,674][propulate.migrator][INFO] - Island 0 Worker 1: In generation 980...
    [2024-06-04 12:35:31,724][propulate.migrator][INFO] - Island 0 Worker 0: In generation 990...
    [2024-06-04 12:35:31,778][propulate.migrator][INFO] - Island 0 Worker 1: In generation 990...
    [2024-06-04 12:35:31,865][propulate.migrator][INFO] - OPTIMIZATION DONE.
    [2024-06-04 12:35:31,865][propulate.migrator][INFO] - NEXT: Final checks for incoming messages...
    [2024-06-04 12:35:31,950][propulate.propulator][INFO] - ###########
    # SUMMARY #
    ###########
    Number of currently active individuals is 4000.
    Expected overall number of evaluations is 4000.
    [2024-06-04 12:35:34,363][propulate.propulator][INFO] - Top 1 result(s) on island 1:
    (1): [{'a': '2.47E-3', 'b': '-4.79E-3'}, loss 2.90E-5, island 1, worker 1, generation 965]

    [2024-06-04 12:35:34,378][propulate.propulator][INFO] - Top 1 result(s) on island 0:
    (1): [{'a': '2.47E-3', 'b': '-4.79E-3'}, loss 2.90E-5, island 1, worker 1, generation 965]

Even though we have eight ranks overall, only four workers are created – two for island 0 and two for island 1–, where
each worker has two internal ranks for the parallelized evaluation of the loss function.

.. warning::

   Combining multi-rank workers with surrogate models in ``Propulate`` |:dna:| has not yet been tested and might cause
   issues. Please be cautious when using these features together. We are actively working on this and will provide
   support for their combination soon |:rocket:|.
