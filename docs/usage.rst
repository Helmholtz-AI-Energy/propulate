.. _usage:

Tutorials
=========

The tutorials below guide you through the exemplary usage scripts on our `Github`_ page step-by-step.
If you want to use ``Propulate`` for your own applications, you can use these scripts as templates.

Evolutionary optimization of a mathematical function
----------------------------------------------------
You can find the corresponding ``Python`` script here:
https://github.com/Helmholtz-AI-Energy/propulate/blob/master/scripts/propulator_example.py

The basic optimization mechanism in ``Propulate`` is that of Darwinian evolution, i.e.,
beneficial traits are selected, recombined, and mutated to breed more fit individuals.
To show you how ``Propulate`` works, we use its **basic asynchronous evolutionary optimizer** to minimize
two-dimensional mathematical functions. Let us consider the sphere function:

.. math::
    f_\mathrm{sphere}\left(x,y\right)=x^2+y^2

The sphere function is smooth, unimodal, strongly convex, symmetric, and thus easy to optimize. Its global minimum is
:math:`f_\mathrm{sphere}\left(x^*,y^*\right)=0` at :math:`x^*=y^*=0`.

.. image:: images/sphere.png
   :width: 80 %
   :align: center

|

How to use ``Propulate`` - A recipe
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

* The fitness or **loss function** (also known as the objective function). This is the function we want to optimize in order
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

Let's get your hands dirty (at least a bit)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
^^^^^^^^^^^^^
``Propulate`` automatically creates checkpoints of your population in regular intervals during the optimization. You can
pass the ``Propulator`` a path via its ``checkpoint_path`` argument where it should write those checkpoints to. This
also is the path where it will look for existing checkpoint files to start an optimization run from. As a default, it
will use your current working directory.

.. warning::
    If you start an optimization run requesting 100 generations from a checkpoint file with 100 generations,
    nothing will happen.
.. warning::
    If you start an optimization run from existing checkpoints, those checkpoints must be compatible with your current
    parallel computing environment. This means that if you use a checkpoint created in a setting with 20 processing
    elements in a different computing environment with, e.g., 10 processing elements, weird things will happen.


Multi-island optimization of a mathematical function
----------------------------------------------------
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

Next, we set up the island model itself using the ``Islands`` class. In addition to the `Propulator` arguments defining
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

Hyperparameter optimization of a neural network
-----------------------------------------------
You can find the corresponding ``Python`` script here:
https://github.com/Helmholtz-AI-Energy/propulate/blob/master/scripts/torch_example.py

Almost all machine-learning algorithms have non-learnable hyperparameters that influence the training and in
particular their predictive capacity.
``Propulate`` is intended to help AI practitioners optimizing those hyperparameters efficiently. Below, we show you how
to do this using a simple toy example. We want to train a simple convolutional neural network in ``Pytorch`` and
``Pytorch-Lightning`` for MNIST classification and we want to know the best hyperparameters in terms of prediction
accuracy of our network for this.

.. image:: images/mnist.png
    :width: 100 %

We consider:

* the number of convolutional layers ``conv_layers``
* the activation function to use ``activation``
* the learning rate ``learning_rate``

Thus, our search space dictionary looks as follows:

.. code-block:: python

    limits = {
        "conv_layers": (2, 10),  # number of convolutional layers, int for ordinal
        "activation": ("relu", "sigmoid", "tanh"),  # activation function to use, str for categorical
        "learning_rate": (0.01, 0.0001)}  # learning rate, float for continuous

When tuning the hyperparameters of an ML model, evaluating an individual during the optimization corresponds to training
a neural network instance using a specific combination of hyperparameters to be optimized. In addition, we need some
model performance metric to assign each evaluated individual, i.e., tested hyperparameter combination, a scalar loss.
Here, we choose the model's (negative) validation accuracy for this. Remember that the ``Propulate`` loss function takes
in a combination of those parameters that we want to optimize and returns a scalar value telling us how good this
parameter combination actually was. For hyperparameter optimization, the loss function thus takes in a hyperparameter
combination of our model, trains the model using this specific hyperparameter combination, and returns its validation
accuracy as a loss for the evolutionary optimization.
Below, we show you how to do this using the example of the most important code snippets. We start with defining the
neural network which looks like this:

.. code-block:: python

    class Net(LightningModule):
        """Neural network class."""
        def __init__(
                self,
                conv_layers: int,
                activation: torch.nn.modules.activation,
                lr: float,
                loss_fn: torch.nn.modules.loss
        ) -> None:
            """
            Set up neural network.

            Parameters
            ----------
            conv_layers: int
                         number of convolutional layers
            activation: torch.nn.modules.activation
                        activation function to use
            lr: float
                learning rate
            loss_fn: torch.nn.modules.loss
                     loss function
            """
            super(Net, self).__init__()

            self.lr = lr  # Set learning rate
            self.loss_fn = loss_fn  # Set loss function for neural network training.
            self.best_accuracy = 0.0  # Initialize the model's best validation accuracy.
            layers = []  # Set up the model architecture (depending on number of convolutional layers specified).
            layers += [nn.Sequential(nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding=1),
                                     activation()),]
            layers += [nn.Sequential(nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=1),
                                     activation())
                       for _ in range(conv_layers - 1)]

            self.fc = nn.Linear(in_features=7840,
                                out_features=10)  # MNIST has 10 classes.
            self.conv_layers = nn.Sequential(*layers)
            self.val_acc = Accuracy("multiclass", num_classes=10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.

            Parameters
            ----------
            x: torch.Tensor
               data sample

            Returns
            -------
            torch.Tensor
                The model's predictions for input data sample
            """
            ...
            return x

        def training_step(
                self,
                batch: Tuple[torch.Tensor, torch.Tensor],
                batch_idx: int
        ) -> torch.Tensor:
            """
            Calculate loss for training step in Lightning train loop.

            Parameters
            ----------
            batch: Tuple[torch.Tensor, torch.Tensor]
                   input batch
            batch_idx: int
                       batch index

            Returns
            -------
            torch.Tensor
                training loss for input batch
            """
            x, y = batch
            return self.loss_fn(self(x), y)

        def validation_step(
                self,
                batch: Tuple[torch.Tensor, torch.Tensor],
                batch_idx: int
        ) -> torch.Tensor:
            """
            Calculate loss for validation step in Lightning validation loop during training.

            Parameters
            ----------
            batch: Tuple[torch.Tensor, torch.Tensor]
                   current batch
            batch_idx: int
                       batch index

            Returns
            -------
            torch.Tensor
                validation loss for input batch
            """
            x, y = batch
            pred = self(x)
            loss = self.loss_fn(pred, y)
            val_acc = self.val_acc(torch.nn.functional.softmax(pred, dim=-1), y)
            if val_acc > self.best_accuracy:  # This is the metric Propulate optimizes on!
                self.best_accuracy = val_acc
            return loss

        def configure_optimizers(self) -> torch.optim.sgd.SGD:
            """
            Configure optimizer.

            Returns
            -------
            torch.optim.sgd.SGD
                stochastic gradient descent optimizer
            """
            # The optimizer uses the learning rate which is one of the hyperparameters that we want to optimize.
            return torch.optim.SGD(self.parameters(), lr=self.lr)

We also need some helper function to load the MNIST data:

.. code-block:: python

    def get_data_loaders(batch_size):
        """
        Get MNIST train and validation dataloaders.

        Parameters
        ----------
        batch_size: int
                    batch size

        Returns
        -------
        DataLoader
            training dataloader
        DataLoader
            validation dataloader
        """
        ...
        return train_loader, val_loader

Now we are ready to set up the ``Propulate`` loss function that is minimized during the evolutionary optimization in
order to find the best hyperparameters for our model:

.. code-block:: python

    def ind_loss(
            params: Dict[str, Union[int, float, str]]
    ) -> float:
        """
        Loss function for evolutionary optimization with Propulate.
        We minimize the model's negative validation accuracy.

        Parameters
        ----------
        params: dict[str, int | float | str]]

        Returns
        -------
        float
            The trained model's negative validation accuracy
        """
        # Extract hyperparameter combination to test from input dictionary.
        conv_layers = params["conv_layers"]  # Number of convolutional layers
        activation = params["activation"]  # Activation function
        lr = params["lr"]  # Learning rate

        epochs = 2  # Number of epochs to train

        # Define the activation function mapping.
        activations = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
        activation = activations[activation]  # Get activation function.
        loss_fn = torch.nn.CrossEntropyLoss()  # Use cross-entropy loss for multi-class classification.

        model = Net(conv_layers, activation, lr, loss_fn)  # Set up neural network with specified hyperparameters.
        model.best_accuracy = 0.0  # Initialize the model's best validation accuracy.

        train_loader, val_loader = get_data_loaders(batch_size=8)  # Get training and validation data loaders.

        # Under the hood, the Lightning Trainer handles the training loop details.
        trainer = Trainer(max_epochs=epochs,  # Stop training once this number of epochs is reached.
                          accelerator="gpu",  # Pass accelerator type.
                          devices=[  # Devices to train on
                              MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE
                                  ],
                          enable_progress_bar=False,  # Disable progress bar.
                          )
        trainer.fit(  # Run full model training optimization routine.
            model=model,  # Model to train
            train_dataloaders=train_loader,  # Dataloader for training samples
            val_dataloaders=val_loader  # Dataloader for validation samples
        )
        # Return negative best validation accuracy as an individual's loss.
        return -model.best_accuracy.item()

Just as before, this loss function is fed into the asynchronous evolutionary optimizer (``Propulator``) or the
asynchronous island model (``Islands``) which takes care of the actual genetic optimization.

.. note::
    Running this script without any modifications requires compute nodes with four GPUs.

.. Links
.. _Github: https://github.com/Helmholtz-AI-Energy/propulate