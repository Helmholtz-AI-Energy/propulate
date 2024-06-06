.. _tut_surrogate:

Pruning with Surrogate Models
=============================

.. note::

   You can find the corresponding ``Python`` scripts here:
   https://github.com/Helmholtz-AI-Energy/propulate/tree/master/tutorials/surrogate

We all know that hyperparameter optimization is a critical aspect in machine learning that tries to find the most
effective settings for a model's non-trainable parameters to optimize the trained model's predictive performance.
Automated approaches like random search, grid search, Bayesian optimization, or population-based algorithms as
implemented in ``Propulate`` |:dna:| train the neural network over and over again, testing new hyperparameters every
time. As each evaluation typically corresponds to a full training of a neural network model, we aim at finding effective
hyperparameter settings with as little evaluations as possible. Even though ``Propulate`` already makes smart choices
about which hyperparameters to test next compared to, e.g., plain grid search, this is still very compute-intensive,
especially with newer models getting bigger and bigger.

Predicting the performance of hyperparameter configurations during the training process allows for early termination of
less promising configurations. To this end, ``Propulate`` features so-called surrogate models, which have access to
interim loss values from each evaluated neural network's training during the hyperparameter optimization and decide
whether to stop it early.
Our evaluation of static and probabilistic surrogate models for hyperparameter optimization in ``Propulate`` with
different datasets and neural networks showed a significant decrease in total run time and energy consumption while
still finding a loss within small bounds of the best loss found without early stopping.
Below, we will guide you through a basic example of how to use surrogate models for pruning in ``Propulate``.

As in the :ref:`hyperparameter optimization tutorial<tut_hpo>`, let us again consider the problem of MNIST
classification with a simple convolutional neural network. The neural network class ``Net`` and the ``get_dataloaders()``
function are left unchanged and reused as before:

.. code-block:: python

    GPUS_PER_NODE: int = 1
    NUM_WORKERS: int = (
        2  # Set this to the recommended number of workers in the PyTorch dataloader.
    )

    log_path = "torch_ckpts"
    log = logging.getLogger(__name__)  # Get logger instance.


    class Net(nn.Module):
        """Convolutional neural network class."""

        ...  # Reused from hyperparameter optimization tutorial without changes!


    def get_data_loaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
        """
        Get MNIST train and validation dataloaders.

        Parameters
        ----------
        batch_size : int
            The batch size.

        Returns
        -------
        torch.utils.data.DataLoader
            The training dataloader.
        torch.utils.data.DataLoader
            The validation dataloader.
        """
        ...  # Reused from hyperparameter optimization tutorial without changes!

        return train_loader, val_loader


The only thing that is different when using surrogate models is the individual's loss function ``ind_loss()``. As
mentioned before, surrogate models predict the performance of hyperparameter configurations during the training process
to stop less promising individuals early. To decide whether to stop an individual early, we need access to interim loss
values from each evaluated neural network's training. This is achieved by yielding the average validation loss of each
evaluated candidate in regular intervals (e.g., after each epoch) during training. These interim loss values are fed
into the surrogate model which decides whether to continue or cancel the training and updates itself accordingly based
on the provided value.

.. code-block:: python
    :emphasize-lines: 3, 14-15, 91

    def ind_loss(
        params: Dict[str, Union[int, float, str]],
    ) -> Generator[float, None, None]:
        """
        Loss function for evolutionary optimization with Propulate. Minimize the model's negative validation accuracy.

        Parameters
        ----------
        params : Dict[str, int | float | str]
            The parameters to be optimized.

        Returns
        -------
        Generator[float, None, None]
            Yields the negative validation accuracy in regular intervals during training of the model.
        """
        # Extract hyperparameter combination to test from input dictionary.
        conv_layers = int(params["conv_layers"])  # Number of convolutional layers
        activation = str(params["activation"])  # Activation function
        lr = float(params["lr"])  # Learning rate

        epochs: int = 2  # Number of epochs to train

        rank: int = MPI.COMM_WORLD.rank  # Get rank of current worker.

        num_gpus = torch.cuda.device_count()  # Number of GPUs available
        if num_gpus == 0:
            device = torch.device("cpu")
        else:
            device_index = rank % num_gpus
            device = torch.device(
                f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"
            )

        log.info(f"Rank: {rank}, Using device: {device}")

        activations = {
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
        }  # Define activation function mapping.
        activation = activations[activation]  # Get activation function.
        loss_fn = (
            torch.nn.CrossEntropyLoss()
        )  # Use cross-entropy loss for multi-class classification.

        model = Net(conv_layers, activation, lr, loss_fn).to(
            device
        )  # Set up neural network with specified hyperparameters.
        model.best_accuracy = 0.0  # Initialize the model's best validation accuracy.

        train_loader, val_loader = get_data_loaders(
            batch_size=8
        )  # Get training and validation dataloaders.

        # Configure optimizer.
        optimizer = model.configure_optimizers()

        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
            # Training loop
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                # Zero out gradients.
                optimizer.zero_grad()
                # Forward + backward pass and optimizer step to update parameters.
                loss = model.training_step((data, target))
                loss.backward()
                optimizer.step()
                # Update loss.
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            log.info(f"Epoch {epoch+1}: Avg Training Loss: {avg_train_loss}")

            # Validation loop
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(val_loader):
                    data, target = data.to(device), target.to(device)
                    # Forward pass
                    loss = model.validation_step((data, target))
                    # Update loss.
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            log.info(f"Epoch {epoch+1}: Avg Validation Loss: {avg_val_loss}")

            yield avg_val_loss


Now we have all the ingredients to perform a hyperparameter optimization with early stopping in ``Propulate`` |:dna:|:

.. code-block:: python

    if __name__ == "__main__":
        comm = MPI.COMM_WORLD
        if comm.rank == 0:  # Download data at the top, then we don't need to later.
            MNIST(download=True, root=".", transform=None, train=True)
            MNIST(download=True, root=".", transform=None, train=False)
        comm.Barrier()

        num_generations = 3  # Number of generations
        pop_size = 2 * comm.size  # Breeding population size
        limits = {
            "conv_layers": (2, 10),
            "activation": ("relu", "sigmoid", "tanh"),
            "lr": (0.01, 0.0001),
        }  # Define search space.
        rng = random.Random(
            comm.rank
        )  # Set up separate random number generator for evolutionary optimizer.
        set_seeds(42 * comm.rank)  # Set seed for torch.
        propagator = get_default_propagator(  # Get default evolutionary operator.
            pop_size=pop_size,  # Breeding population size
            limits=limits,  # Search space
            crossover_prob=0.7,  # Crossover probability
            mutation_prob=0.4,  # Mutation probability
            random_init_prob=0.1,  # Random-initialization probability
            rng=rng,  # Random number generator for evolutionary optimizer
        )
        islands = Islands(  # Set up island model.
            loss_fn=ind_loss,  # Loss function to optimize
            propagator=propagator,  # Evolutionary operator
            rng=rng,  # Random number generator
            generations=num_generations,  # Number of generations per worker
            num_islands=1,  # Number of islands
            checkpoint_path=log_path,
            surrogate_factory=lambda: surrogate.StaticSurrogate(),
            # Alternatively, you can use a dynamic surrogate model here:
            # surrogate_factory=lambda: surrogate.DynamicSurrogate(limits),
        )
        islands.evolve(  # Run evolutionary optimization.
            top_n=1,  # Print top-n best individuals on each island in summary.
            logging_interval=1,  # Logging interval
            debug=2,  # Verbosity level
        )
