.. _tut_ddp:

Using Propulate with ``PyTorch DistributedDataParallel``
========================================================
.. note::

   You can find the corresponding ``Python`` script here:
   https://github.com/Helmholtz-AI-Energy/propulate/blob/master/tutorials/torch_ddp_example.py

In this tutorial, you will learn how to use ``Propulate``'s |:dna:| multi-rank workers to train each individual network
on multiple GPUs in a data-parallel fashion with ``PyTorch``'s ``DistributedDataParallel`` module.

We will start with a short overview of data-parallel neural networks and how to train them in ``PyTorch``. Feel free to
skip the introduction if you are already familiar with this |:rocket:|.

Data-parallel neural networks (DPNNs) in ``PyTorch``
----------------------------------------------------

Data-parallel training of neural networks involves distributing the training data across multiple processors or
machines, where each processor independently computes gradients on a subset of the data, followed by aggregating the
gradients to update the model parameters. Thus, we can accelerate computation and increase training throughput.
``PyTorch`` provides the ``DistributedDataParallel`` (``DDP``) module for this, which abstracts away some of the
complexities of implementing data-parallel training in a distributed setting. From the official `documentation`_:

    *"Distributed data-parallel training is a widely adopted single-program multiple-data training paradigm. The model
    is replicated on every process, and every model replica will be fed with a different set of input data samples. The
    DistributedDataParallel module takes care of gradient communication to keep model replicas synchronized and overlaps
    it with the gradient computations to speed up training. It implements data parallelism at the module level which can
    run across multiple machines. Applications using DDP should spawn multiple processes and create a single DDP
    instance per process. DDP uses collective communications in the torch.distributed package to synchronize gradients
    and buffers. More specifically, DDP registers an autograd hook for each parameter given by model.parameters() and
    the hook will fire when the corresponding gradient is computed in the backward pass. Then DDP uses that signal to
    trigger gradient synchronization across processes. The recommended way to use DDP is to spawn one process for each
    model replica. DDP processes can be placed on the same machine or across machines, but GPU devices cannot be shared
    across processes."*

The ``torch.distributed package`` supports three built-in backends for communication between processors. This `table`_
shows which functions are available for use with CPU/CUDA tensors. Since many supercomputers connect GPUs with NVLink
within a node and an Infiniband Interconnect between nodes, we use the officially recommended `NCCL`_ backend here. The
NVIDIA Collective Communication Library (NCCL) implements multi-GPU and multi-node communication functions optimized for
NVIDIA GPUs and networks. It provides routines, such as all-gather, all-reduce, broadcast, reduce, reduce-scatter, and
point-to-point transmit and receive.

How to train a DPNN with ``PyTorch``'s ``DistributedDataParallel`` Module
-------------------------------------------------------------------------

Below is a recipe for training a DPNN with ``DDP`` in ``PyTorch``:

**1. Initialize the Distributed Environment**
   Before using ``DDP``, you need to define and initialize the distributed environment. This involves setting up the
   communication backend (NCCL for us), specifying the so-called process group, and assigning a unique rank and the
   world size to each process in the process group. To use ``DDP`` for evaluating each individual neural network in
   ``Propulate`` |:dna:| in a data-parallel fashion, you need to set up a separate process group for each worker.

**2. Load the Data**
  Data parallelism means splitting the input data across the processes in the process group and
  computing the forward and backward passes independently on each rank. This enables parallel processing and reduces the
  training time. For each individual, you load the training and validation datasets and distribute them equally over the
  processes in its process group so that each process holds a different, exclusive subset of each dataset. ``PyTorch``
  provides a dedicated sampler for this, the so-called ``DistributedSampler``.

**3. Model Instantiation and Replication**
  Afterwards, you need to replicate the model across the processes in each
  group. Each replica will process a subset of the input data provided by the ``DistributedSampler``. To do so, you
  instantiate the model just as in the serial case and wrap it with ``DDP``. This ensures that the gradients computed
  during the backward pass are synchronized across all replicas.

**4. Training Loop**
  For each individual, repeat for a specified number of iterations or until convergence is reached:

  - **Forward pass:** Each replica of the model independently processes its portion of the input data.
  - **Backward pass and gradient synchronization:** The gradients are computed independently on each replica. They are then
    synchronized across all replicas using a function called "all-reduce". This step ensures that the model parameters
    are updated consistently across all processes.
  - **Optimization step:** Once the gradients are synchronized, the optimizer performs an optimization step to update the
    model parameters. This step is performed independently and redundantly on each replica.
  - **Validation:** After updating the model parameters, you can compute the current model's accuracy on the training and
    validation dataset. As each process only holds a portion of each dataset, you need to implement some more
    communication to obtain the accuracy on each whole dataset.
  - **Evaluation:** After training, you can evaluate the final model's performance using a held-out test dataset. The
    evaluation is typically performed on a single process without the need for data parallelism.

Distributed Dataloading
-----------------------
To train a DPNN, each process needs to load an exclusive subset of the dataset. ``PyTorch`` provides a dedicated sampler
to distribute and load data in a distributed training setting, the so-called ``DistributedSampler``. It enables
efficient data loading across multiple processes by partitioning the dataset into smaller subsets that are processed
independently by each process. The ``DistributedSampler`` works in conjunction with ``DDP``. It ensures that each
process operates on a unique subset of the dataset, avoiding redundant computation and enabling parallelism. Below, you
can find an overview of how this works:

**1. Data Partitioning**
  The ``DistributedSampler`` partitions the dataset into smaller subsets based on the number of processes involved in
  the distributed training. Each process is responsible for processing a specific subset of the data.

**2. Shuffling and Sampling**
  Optionally, the ``DistributedSampler`` can shuffle the dataset before partitioning it to introduce randomness into the
  training. This helps prevent biases and improves the model's generalization. The shuffling is typically performed on a
  single process, and the shuffled indices are then broadcast to other processes.

**3. Data Loading**
  During training, each process loads its assigned subset of the dataset using the ``DistributedSampler``. The sampler
  provides indices corresponding to the samples in the process's partition of the dataset.

**4. Parallel Processing**
  Once the data is loaded, each process operates independently on its portion of the dataset. Forward and backward
  passes, as well as the optimization step, are performed separately on each process.

**5. Synchronization**
  After each training iteration, the processes synchronize to ensure that the model parameters and gradients are
  consistent across all processes. This synchronization is handled by ``DDP``.

**6. Iteration and Epoch Completion**
  The ``DistributedSampler`` manages the completion of iterations and epochs. It ensures that each process finishes
  processing its assigned subset of the data before moving on to the next iteration or epoch. The ``DistributedSampler``
  may also reshuffle the dataset at the end of each epoch to introduce further randomness.

Using ``Propulate`` with ``PyTorch``'s ``DistributedDataParallel``
------------------------------------------------------------------

In this tutorial, we again consider the simple convolutional neural network for MNIST classification from before:

.. code-block:: python

    class Net(nn.Module):
        """
        Toy neural network class.

        Attributes
        ----------
        conv_layers : torch.nn.modules.container.Sequential
            The model's convolutional layers.
        fc : nn.Linear
            The fully connected output layer.

        Methods
        -------
        forward()
            The forward pass.
        """

        ... # Reused from hyperparameter optimization tutorial without changes!


As you already know, each individual corresponds to an instance of this neural network trained with a specific set of
hyperparameters that we want to optimize. However, in contrast to before, each network is trained on multiple GPUs in a
data-parallel fashion instead of a single GPU.

We use ``PyTorch``'s ``DistributedSampler`` to split the input data across the processes in each process group and thus
enable data-parallel training of each individual. This is what happens in the ``get_data_loaders()`` function below:

.. code-block:: python

    def get_data_loaders(
        batch_size: int, subgroup_comm: MPI.Comm
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Get MNIST train and validation dataloaders.

        Parameters
        ----------
        batch_size : int
            The batch size.
        subgroup_comm: MPI.Comm
            The MPI communicator object for the local class

        Returns
        -------
        torch.utils.data.DataLoader
            The training dataloader.
        torch.utils.data.DataLoader
            The validation dataloader.
        """
        data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        train_dataset = MNIST(
            download=False, root=".", transform=data_transform, train=True
        )
        val_dataset = MNIST(download=False, root=".", transform=data_transform, train=False)
        if (
            subgroup_comm.size > 1
        ):  # Make the samplers use the torch world to distribute data
            train_sampler = datadist.DistributedSampler(train_dataset)
            val_sampler = datadist.DistributedSampler(val_dataset)
        else:
            train_sampler = None
            val_sampler = None
        num_workers = NUM_WORKERS
        log.info(f"Use {num_workers} workers in dataloader.")

        train_loader = DataLoader(
            dataset=train_dataset,  # Use MNIST training dataset.
            batch_size=batch_size,  # Batch size
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            shuffle=(train_sampler is None),  # Shuffle data only if no sampler is provided.
            sampler=train_sampler,
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            batch_size=1,  # Batch size
            shuffle=False,  # Do not shuffle data.
            sampler=val_sampler,
        )
        return train_loader, val_loader


As already mentioned before, we need one ``PyTorch`` process group per individual for ``DDP`` in ``Propulate`` |:dna:|.
The ``torch_process_group_init`` function below sets up one of these group for each worker based on its communicator:

.. code-block:: python

    def torch_process_group_init(subgroup_comm: MPI.Comm, method) -> None:
        """
        Create the torch process group of each multi-rank worker from a subgroup of the MPI world.

        Parameters
        ----------
        subgroup_comm : MPI.Comm
            The split communicator for the multi-rank worker's subgroup. This is provided to the individual's loss function
            by the ``Islands`` class if there are multiple ranks per worker.
        method : str
            The method to use to initialize the process group.
            Options: [``nccl-slurm``, ``nccl-openmpi``, ``gloo``]
            If CUDA is not available, ``gloo`` is automatically chosen for the method.
        """
        global _DATA_PARALLEL_GROUP
        global _DATA_PARALLEL_ROOT

        comm_rank, comm_size = subgroup_comm.rank, subgroup_comm.size

        # Get master address and port.
        # Don't want different groups to use the same port.
        subgroup_id = MPI.COMM_WORLD.rank // comm_size
        port = 29500 + subgroup_id

        if comm_size == 1:
            return
        master_address = socket.gethostname()
        # Each multi-rank worker rank needs to get the hostname of rank 0 of its subgroup.
        master_address = subgroup_comm.bcast(str(master_address), root=0)

        # Save environment variables.
        os.environ["MASTER_ADDR"] = master_address
        # Use the default PyTorch port.
        os.environ["MASTER_PORT"] = str(port)

        if not torch.cuda.is_available():
            method = "gloo"
            log.info("No CUDA devices found: Falling back to gloo.")
        else:
            log.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
            num_cuda_devices = torch.cuda.device_count()
            device_number = MPI.COMM_WORLD.rank % num_cuda_devices
            log.info(f"device count: {num_cuda_devices}, device number: {device_number}")
            torch.cuda.set_device(device_number)

        time.sleep(0.001 * comm_rank)  # Avoid DDOS'ing rank 0.
        if method == "nccl-openmpi":  # Use NCCL with OpenMPI.
            dist.init_process_group(
                backend="nccl",
                rank=comm_rank,
                world_size=comm_size,
            )

        elif method == "nccl-slurm":  # Use NCCL with a TCP store.
            wireup_store = dist.TCPStore(
                host_name=master_address,
                port=port,
                world_size=comm_size,
                is_master=(comm_rank == 0),
                timeout=dt.timedelta(seconds=60),
            )
            dist.init_process_group(
                backend="nccl",
                store=wireup_store,
                world_size=comm_size,
                rank=comm_rank,
            )
        elif method == "gloo":  # Use gloo.
            wireup_store = dist.TCPStore(
                host_name=master_address,
                port=port,
                world_size=comm_size,
                is_master=(comm_rank == 0),
                timeout=dt.timedelta(seconds=60),
            )
            dist.init_process_group(
                backend="gloo",
                store=wireup_store,
                world_size=comm_size,
                rank=comm_rank,
            )
        else:
            raise NotImplementedError(
                f"Given 'method' ({method}) not in [nccl-openmpi, nccl-slurm, gloo]!"
            )

        # Call a barrier here in order for sharp to use the default comm.
        if dist.is_initialized():
            dist.barrier()
            disttest = torch.ones(1)
            if method != "gloo":
                disttest = disttest.cuda()

            dist.all_reduce(disttest)
            assert disttest[0] == comm_size, "Failed test of dist!"
        else:
            disttest = None
        log.info(
            f"Finish subgroup torch.dist init: world size: {dist.get_world_size()}, rank: {dist.get_rank()}"
        )

The ``torch_process_group_init`` function is called in the very beginning of the loss function ``ind_loss()`` in
``Propulate`` |:dna:|. As before, ``ind_loss()`` takes in the hyperparameters to be optimized, trains the neural network
using this hyperparameters (now in a data-parallel fashion using ``DDP``), and returns the trained model's validation
loss as a measure of its predictive performance. The main difference to the single-GPU case is that we need to wrap our
initial model with ``DDP`` and use the ``DistributedSampler`` when getting the train and validation dataloaders:

.. code-block:: python

    def ind_loss(
        params: Dict[str, Union[int, float, str]], subgroup_comm: MPI.Comm
    ) -> float:
        """
        Loss function for evolutionary optimization with Propulate. Minimize the model's negative validation accuracy.

        Parameters
        ----------
        params : Dict[str, int | float | str]
            The hyperparameters to be optimized evolutionarily.
        subgroup_comm : MPI.Comm
            Each multi-rank worker's subgroup communicator.

        Returns
        -------
        float
            The trained model's validation loss.
        """
        torch_process_group_init(subgroup_comm, method=SUBGROUP_COMM_METHOD)
        # Extract hyperparameter combination to test from input dictionary.
        conv_layers = params["conv_layers"]  # Number of convolutional layers
        activation = params["activation"]  # Activation function
        lr = params["lr"]  # Learning rate
        gamma = params["gamma"]  # Learning rate reduction factor

        epochs = 20

        activations = {
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
        }  # Define activation function mapping.
        activation = activations[activation]  # Get activation function.
        loss_fn = torch.nn.NLLLoss()

        # Set up neural network with specified hyperparameters.
        model = Net(conv_layers, activation)

        train_loader, val_loader = get_data_loaders(
            batch_size=8, subgroup_comm=subgroup_comm
        )  # Get training and validation data loaders.

        if torch.cuda.is_available():
            device = MPI.COMM_WORLD.rank % GPUS_PER_NODE
            model = model.to(device)
        else:
            device = "cpu"

        if dist.is_initialized() and dist.get_world_size() > 1:
            model = DDP(model)  # Wrap model with DDP.

        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
        log_interval = 10000
        best_val_loss = 1000000
        early_stopping_count, early_stopping_limit = 0, 5
        set_new_best = False
        model.train()
        for epoch in range(epochs):  # Loop over epochs.
            # ------------ Train loop ------------
            for batch_idx, (data, target) in enumerate(
                train_loader
            ):  # Loop over training batches.
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % log_interval == 0 or batch_idx == len(train_loader) - 1:
                    log.info(
                        f"Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} "
                        f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                    )
            # ------------ Validation loop ------------
            model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += loss_fn(output, target).item()  # Sum up batch loss.
                    pred = output.argmax(
                        dim=1, keepdim=True
                    )  # Get the index of the max log-probability.
                    correct += pred.eq(target.view_as(pred)).sum().item()

            val_loss /= len(val_loader.dataset)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                set_new_best = True

            log.info(
                f"\nTest set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} "
                f"({100. * correct / len(val_loader.dataset):.0f}%)\n"
            )

            if not set_new_best:
                early_stopping_count += 1
            if early_stopping_count >= early_stopping_limit:
                log.info("hit early stopping count, breaking")
                break

            # ------------ Scheduler step ------------
            scheduler.step()
            set_new_best = False

        # Return best validation loss as an individual's loss (trained so lower is better).
        dist.destroy_process_group()
        return best_val_loss

Now we have all the ingredients to start the actual optimization in ``Propulate`` |:dna:|. We want to optimize

- the number of convolutional layers, ``conv_layers``,
- the activation function, ``activation``,
- the learning rate, ``lr``, and
- the multiplicative factor of learning rate decay in the scheduler, ``gamma``.

Make sure to adapt the number of GPUs per node, ``GPUS_PER_NODE``, as well as the method used to initialize the process
groups, ``SUBGROUP_COMM_METHOD``, for your own needs and hardware:

.. code-block:: python

    GPUS_PER_NODE: int = 4  # This example script was tested on a single node with 4 GPUs.
    NUM_WORKERS: int = (
        2  # Set this to the recommended number of workers in the PyTorch dataloader.
    )
    SUBGROUP_COMM_METHOD = "nccl-slurm"
    log_path = "torch_ckpts"
    log = logging.getLogger("propulate")  # Get logger instance.

    if __name__ == "__main__":
        config, _ = parse_arguments()

        comm = MPI.COMM_WORLD
        if comm.rank == 0:  # Download data at the top, then we don't need to later.
            MNIST(download=True, root=".", transform=None, train=True)
            MNIST(download=True, root=".", transform=None, train=False)
        comm.Barrier()
        pop_size = 2 * comm.size  # Breeding population size
        limits = {
            "conv_layers": (2, 10),
            "activation": ("relu", "sigmoid", "tanh"),
            "lr": (0.01, 0.0001),
            "gamma": (0.5, 0.999),
        }  # Define search space.
        rng = random.Random(
            comm.rank
        )  # Set up separate random number generator for evolutionary optimizer.

        # Set up separate logger for Propulate optimization.
        set_logger_config(
            level=logging.INFO,  # Logging level
            log_file=f"{log_path}/{pathlib.Path(__file__).stem}.log",  # Logging path
            log_to_stdout=True,  # Print log on stdout.
            log_rank=False,  # Do not prepend MPI rank to logging messages.
            colors=True,  # Use colors.
        )
        if comm.rank == 0:
            log.info("Starting Torch DDP tutorial!")

        propagator = get_default_propagator(  # Get default evolutionary operator.
            pop_size=pop_size,  # Breeding population size
            limits=limits,  # Search space
            crossover_prob=0.7,  # Crossover probability
            mutation_prob=0.4,  # Mutation probability
            random_init_prob=0.1,  # Random-initialization probability
            rng=rng,  # Separate random number generator for Propulate optimization
        )

        # Set up island model.
        islands = Islands(
            loss_fn=ind_loss,  # Loss function to be minimized
            propagator=propagator,  # Propagator, i.e., evolutionary operator to be used
            rng=rng,  # Separate random number generator for Propulate optimization
            generations=config.generations,  # Overall number of generations
            num_islands=config.num_islands,  # Number of islands
            migration_probability=config.migration_probability,  # Migration probability
            pollination=config.pollination,  # Whether to use pollination or migration
            checkpoint_path=config.checkpoint,  # Checkpoint path
            # ----- SPECIFIC FOR MULTI-RANK UCS -----
            ranks_per_worker=2,  # Number of ranks per (multi rank) worker
        )

        # Run actual optimization.
        islands.evolve(
            top_n=config.top_n,  # Print top-n best individuals on each island in summary.
            logging_interval=config.logging_interval,  # Logging interval
            debug=config.verbosity,  # Debug level
        )


.. warning::

   Combining multi-rank workers with surrogate models in ``Propulate`` |:dna:| has not yet been tested and might cause
   issues. Please be cautious when using these features together. We are actively working on this and will provide
   support for their combination soon |:rocket:|.



.. _documentation: https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html
.. _table: https://pytorch.org/docs/stable/distributed.html#backends
.. _NCCL: https://developer.nvidia.com/nccl
