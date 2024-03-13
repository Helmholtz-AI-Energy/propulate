.. _quick-start:

Quick Start
===========

To start using ``Propulate``, the first step is to install it using ``pip``. We recommend to install ``Propulate`` in a
separate ``Python3`` virtual environment:

.. code-block:: console

    $ python3 -m venv ./propulate
    $ source ./propulate/bin/activate
    $ pip install --upgrade pip
    $ pip install propulate

This will install the ``Propulate`` optimizer functionality along with some basic example scripts to get you started.
You can find those scripts in the Github repository's `subfolder`_ ``tutorials/``.

To quickly test whether you installation was successful and everything works as expected, check out the minimum
working example provided below:

.. code-block:: python

    """Minimum working example showing how to use Propulate."""
    import propulate
    from mpi4py import MPI
    import random

    # Set the communicator and the optimization parameters
    comm = MPI.COMM_WORLD
    rng = random.Random(MPI.COMM_WORLD.rank)
    population_size = comm.size * 2
    generations = 100
    checkpoint = "./propulate_checkpoints"
    propulate.utils.set_logger_config()

    # Define the function to minimize and the search space, e.g., a 2D sphere function on (-5.12, 5.12)^2.
    def loss_fn(params):
        """Loss function to minimize."""
        return params["x"] ** 2 + params["y"] ** 2

    limits = {"x": (-5.12, 5.12), "y": (-5.12, 5.12)}

    # Initialize the propagator and propulator with default parameters.
    propagator = propulate.utils.get_default_propagator(
        pop_size=population_size,
        limits=limits,
        rng=rng
    )
    propulator = propulate.Propulator(
        loss_fn=loss_fn,
        propagator=propagator,
        rng=rng,
        island_comm=comm,
        generations=generations,
        checkpoint_path=checkpoint,
    )

    # Run optimization and get summary of results.
    propulator.propulate()
    propulator.summarize()

``Propulate`` has more functionality and configurable options available, to accomodate various use cases and workflows.
To learn more about how to use ``Propulate`` and adapt it to your needs, check out the :ref:`usage` section of the
documentation |:raised_hands:|.


.. Links
.. _subfolder: https://github.com/Helmholtz-AI-Energy/propulate/tree/master/tutorials