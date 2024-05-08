"""
Toy example for HP optimization / NAS in Propulate, using a simple CNN trained on MNIST in a data-parallel fashion.

This script was tested on two compute nodes with 4 GPUs each. Note that you need to adapt ``GPUS_PER_NODE`` in l. 29.
"""

import datetime as dt
import logging
import os
import pathlib
import random
import socket
import time
from typing import Dict, Tuple, Union

import torch
import torch.distributed as dist
import torch.utils.data.distributed as datadist
from mpi4py import MPI
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

from propulate import Islands
from propulate.utils import get_default_propagator, set_logger_config
from propulate.utils.benchmark_functions import parse_arguments

GPUS_PER_NODE: int = 4  # This example script was tested on a single node with 4 GPUs.
NUM_WORKERS: int = (
    2  # Set this to the recommended number of workers in the PyTorch dataloader.
)
SUBGROUP_COMM_METHOD = "nccl-slurm"
log_path = "torch_ckpts"
log = logging.getLogger("propulate")  # Get logger instance.


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

    def __init__(
        self,
        conv_layers: int,
        activation: torch.nn.modules.activation,
    ) -> None:
        """
        Initialize the neural network.

        Parameters
        ----------
        conv_layers : int
            The number of convolutional layers to use.
        activation : torch.nn.modules.activation
            The activation function to use.
        """
        super().__init__()
        layers = []  # Set up the model architecture (depending on number of convolutional layers specified).
        layers += [
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding=1),
                activation(),
            ),
        ]
        layers += [
            nn.Sequential(
                nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=1),
                activation(),
            )
            for _ in range(conv_layers - 1)
        ]

        self.fc = nn.Linear(in_features=7840, out_features=10)  # MNIST has 10 classes.
        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
           The data sample.

        Returns
        -------
        torch.Tensor
            The model's predictions for input data sample.
        """
        b, c, w, h = x.size()
        x = self.conv_layers(x)
        x = x.view(b, 10 * 28 * 28)
        x = self.fc(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output


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

    # Get master address and port
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


if __name__ == "__main__":
    config, _ = parse_arguments()

    comm = MPI.COMM_WORLD
    if comm.rank == 0:  # Download data at the top, then we don't need to later.
        dataset = MNIST(download=True, root=".", transform=None, train=True)
        dataset = MNIST(download=True, root=".", transform=None, train=False)
        del dataset
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
