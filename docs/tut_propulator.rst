.. _tut_propulator:

Evolutionary Optimization of a Mathematical Function
====================================================
You can find the corresponding ``Python`` script here:
https://github.com/Helmholtz-AI-Energy/propulate/blob/master/scripts/propulator_example.py

The basic optimization mechanism in ``Propulate`` is that of Darwinian evolution, i.e.,
beneficial traits are selected, recombined, and mutated to breed more fit individuals.
To show you how ``Propulate`` works, we use its *basic asynchronous evolutionary optimizer* to minimize
two-dimensional mathematical functions. Let us consider the sphere function:

.. math::
    f_\mathrm{sphere}\left(x,y\right)=x^2+y^2

The sphere function is smooth, unimodal, strongly convex, symmetric, and thus easy to optimize. Its global minimum is
:math:`f_\mathrm{sphere}\left(x^*,y^*\right)=0` at :math:`x^*=y^*=0` at the orange star.

.. image:: images/sphere.png
   :width: 80 %
   :align: center

|

How to Use Propulate - A Recipe
-----------------------------------

As the very first step, we need to define the key ingredients that define the optimization problem we want to solve:

* The **search space** of the parameters to be optimized as a ``Python`` dictionary.
  ``Propulate`` can handle three different parameter types:

    - A tuple of ``float`` for a continuous parameter, e.g., ``{"learning_rate": (0.0001, 0.01)}``
    - A tuple of ``int`` for an ordinal parameter, e.g., ``{"conv_layers": (2, 10)}``
    - A tuple of ``str`` for a categorical parameter, e.g., ``{"activation": ("relu", "sigmoid", "tanh")}``

  .. note::
    The boundaries for continuous and ordinal parameters are inclusive.

  All-together, a search space dictionary might look like this:

  .. code-block:: python

    limits = {"learning_rate": (0.001, 0.01),
              "conv_layers": (2, 10),
              "activation": ("relu", "sigmoid", "tanh")}

  The sphere function has two continuous parameters, :math:`x` and :math:`y`, and we consider
  :math:`x,y \in\left[-5.12, 5.12\right]`. The search space in our example thus looks like this:

  .. code-block:: python

    limits = {"x": (-5.12, 5.12),
              "y": (-5.12, 5.12)}

* The fitness or *loss function* (also known as the objective function). This is the function we want to optimize in order
  to find the best parameters. The loss function can be any ``Python`` function with the following characteristics:

    - Its input is a set of parameters to be optimized as a ``Python`` dictionary.
    - Its output is a scalar fitness or loss that determines how good the tested parameter set is.
    - This objective function can be a black box.
    - ``Propulate`` is a minimizer. If you want to maximize a fitness function, you need to choose the sign appropriately
      and invert your scalar fitness by multiplying it with :math:`-1`.

  In this example, the loss function whose minimum we want to find is the sphere function
  :math:`f_\mathrm{sphere}\left(x,y\right)`:

  .. code-block:: python

    def sphere(params: Dict[str, float]) -> float:
        """
        Sphere function: continuous, convex, separable, differentiable, unimodal

        Input domain: -5.12 <= x, y <= 5.12
        Global minimum 0 at (x, y) = (0, 0)

        Parameters
        ----------
        params: dict[str, float]
                function parameters
        Returns
        -------
        float
            function value
        """
        return numpy.sum(numpy.array(list(params.values())) ** 2)


Next, we need to define the evolutionary operator or propagator that we want to use to breed new individuals during the
optimization process. ``Propulate`` provides a reasonable default propagator via a utility function that serves as a
good start for the most optimization problems. You can adapt its hyperparameters, such as crossover and mutation
probability, as you wish. In the example script, you can pass those hyperparameters as command-line options (this is the
``config`` in the code snippet below) or just use the default values. You also need to pass a separate random number
generator that is used exclusively from the actual evolutionary optimizer (and not in the objective function):

.. code-block:: python

    rng = random.Random(config.seed+MPI.COMM_WORLD.rank)  # Separate random number generator for optimization.
    # Set up evolutionary operator.
    propagator = propulate.utils.get_default_propagator(  # Get default evolutionary operator.
        pop_size=config.pop_size,  # Breeding pool size
        limits=limits,  # Search-space limits
        mate_prob=config.crossover_probability,  # Crossover probability
        mut_prob=config.mutation_probability,  # Mutation probability
        random_prob=config.random_init_probability,  # Random-initialization probability
        rng=rng)  # Random number generator for the optimization process

We also need to set up the actual evolutionary optimizer, i.e., a so-called ``Propulator`` instance. This will handle the
parallel asynchronous optimization process for us.

.. code-block:: python

    propulator = Propulator(  # Set up propulator performing actual optimization.
        loss_fn=sphere,  # Loss function to minimize
        propagator=propagator,  # Evolutionary operator to use
        comm=MPI.COMM_WORLD,  # Communicator to use
        generations=config.generations,  # Number of generations
        checkpoint_path=config.checkpoint,  # Checkpoint path
        rng=rng)  # Random number generator for optimization process

Now it's time to run the actual optimization. Overall, ``generations * MPI.COMM_WORLD.size`` evaluations will be performed:

.. code-block:: python

    # Run optimization and print summary of results.
    propulator.propulate(logging_interval=config.logging_int, debug=config.verbosity)
    propulator.summarize(top_n=config.top_n, debug=config.verbosity)

Let's Get Your Hands Dirty (At Least a Bit)
-------------------------------------------
Do the following to run the example script:

#. Make sure you have a working MPI installation on your machine.
#. If you have not already done this, create a fresh virtual environment with ``Python``: ``python3 -m venv best-venv-ever``
#. Activate it: ``source best-venv-ever/bin/activate``
#. Upgrade ``pip``: ``pip install --upgrade pip``
#. Install ``Propulate``: ``pip install propulate``
#. Run the example script ``propulator_example.py``: ``mpirun --use-hwthread-cpus python propulator_example.py``

Or just copy and paste:

.. code-block::

    python3 -m venv best-venv-ever
    source best-venv-ever/bin/activate
    pip install --upgrade pip
    pip install propulate
    mpirun --use-hwthread-cpus python propulator_example.py

Checkpointing
-------------
``Propulate`` automatically creates checkpoints of your population in regular intervals during the optimization. You can
pass the ``Propulator`` a path via its ``checkpoint_path`` argument where it should write those checkpoints to. This
also is the path where it will look for existing checkpoint files to start an optimization run from. As a default, it
will use your current working directory.

.. warning::
    If you start an optimization run requesting 100 generations from a checkpoint file with 100 generations,
    the optimizer will return immediately.
.. warning::
    If you start an optimization run from existing checkpoints, those checkpoints must be compatible with your current
    parallel computing environment. This means that if you use a checkpoint created in a setting with 20 processing
    elements in a different computing environment with, e.g., 10 processing elements, the behaviour is undefined.
