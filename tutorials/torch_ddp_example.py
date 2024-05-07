"""
Toy example for HP optimization / NAS in Propulate, using a simple CNN trained on the MNIST dataset.

This script was tested on a single compute node with 4 GPUs. Note that you need to adapt ``GPUS_PER_NODE`` (see ll. 25).
"""
import logging
import pathlib
import random
from typing import Dict, Tuple, Union

import torch
import os
import socket
from mpi4py import MPI
from torch import nn
import time
from torch.utils.data import DataLoader
import torch.utils.data.distributed as datadist
from torch.optim.lr_scheduler import StepLR
from torch import optim
import torch.distributed as dist
import datetime as dt
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor
import torch.nn.functional as F

from propulate import Islands
from propulate.utils import get_default_propagator, set_logger_config
from propulate.utils.benchmark_functions import parse_arguments


GPUS_PER_NODE: int = 4  # This example script was tested on a single node with 4 GPUs.
NUM_WORKERS: int = (
    2  # Set this to the recommended number of workers in the PyTorch dataloader.
)
SUBGROUP_COMM_METHOD = "nccl-slurm"
log_path = "torch_ckpts"
log = logging.getLogger(__name__)  # Get logger instance.


class Net(nn.Module):
    def __init__(
        self,
        conv_layers: int,
        activation: torch.nn.modules.activation,
    ):
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
        output = F.log_softmax(x, dim=1)
        return output


def get_data_loaders(batch_size: int, subgroup_comm: MPI.Comm) -> Tuple[DataLoader, DataLoader]:
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
    train_dataset = MNIST(download=False, root=".", transform=data_transform, train=True)
    val_dataset = MNIST(download=False, root=".", transform=data_transform, train=False)
    if subgroup_comm.size > 1:  # need to make the samplers use the torch world to distributed data
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
        shuffle=(train_sampler is None),  # Shuffle data.
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
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_ROOT
    # done want different groups to use the same port
    subgroup_id = MPI.COMM_WORLD.rank // subgroup_comm.size
    port = 29500 + subgroup_id
    # get master address and port
    # os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"

    comm_size = subgroup_comm.Get_size()
    if comm_size == 1:
        return
    master_address = socket.gethostname()
    # each subgroup needs to get the hostname of rank 0 of that group
    master_address = subgroup_comm.bcast(str(master_address), root=0)

    # save env vars
    os.environ["MASTER_ADDR"] = master_address
        # use the default pytorch port
    os.environ["MASTER_PORT"] = str(port)

    comm_rank = subgroup_comm.Get_rank()

    nccl_world_size = comm_size
    nccl_world_rank = comm_rank
    # print(subgroup_comm.rank, subgroup_comm.size, master_address, port)
    if not torch.cuda.is_available():
        method = "gloo"
        log.info("No CUDA devices found: falling back to gloo")
    else:
        log.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
        num_cuda_devices = torch.cuda.device_count()
        log.info(f"device count: {num_cuda_devices}, device number: {comm_rank % num_cuda_devices}")
        torch.cuda.set_device(comm_rank % num_cuda_devices)


    time.sleep(0.001 * comm_rank)  # avoid DDOS'ing rank 0
    if method == "nccl-openmpi":
        dist.init_process_group(
            backend="nccl",
            rank=subgroup_comm.rank,
            world_size=subgroup_comm.size,
        )

    elif method == "nccl-slurm":
        wireup_store = dist.TCPStore(
            host_name=master_address,
            port=port,
            world_size=nccl_world_size,
            is_master=(nccl_world_rank == 0),
            timeout=dt.timedelta(seconds=60),
        )
        dist.init_process_group(
            backend="nccl",
            store=wireup_store,
            world_size=nccl_world_size,
            rank=nccl_world_rank,
        )
    elif method == "gloo":
        wireup_store = dist.TCPStore(
            host_name=master_address,
            port=port,
            world_size=nccl_world_size,
            is_master=(nccl_world_rank == 0),
            timeout=dt.timedelta(seconds=60),
        )
        dist.init_process_group(
            backend="gloo",
            store=wireup_store,
            world_size=nccl_world_size,
            rank=nccl_world_rank,
        )
    else:
        raise NotImplementedError(f"Given 'method' ({method}) not in [nccl-openmpi, nccl-slurm, gloo]")

    # make sure to call a barrier here in order for sharp to use the default comm:
    if dist.is_initialized():
        dist.barrier()
        disttest = torch.ones(1)
        if method != "gloo":
            disttest = disttest.cuda()

        dist.all_reduce(disttest)
        assert disttest[0] == nccl_world_size, "failed test of dist!"
    else:
        disttest = None
    log.info(f"Finish subgroup torch.dist init: world size: {dist.get_world_size()}, rank: {dist.get_rank()}")


def ind_loss(params: Dict[str, Union[int, float, str]], subgroup_comm: MPI.Comm) -> float:
    """
    Loss function for evolutionary optimization with Propulate. Minimize the model's negative validation accuracy.

    Parameters
    ----------
    params : Dict[str, int | float | str]
        The hyperparameters to be optimized evolutionarily.

    Returns
    -------
    float
        The trained model's negative validation accuracy.
    """
    torch_process_group_init(subgroup_comm, method=SUBGROUP_COMM_METHOD)
    # Extract hyperparameter combination to test from input dictionary.
    conv_layers = params["conv_layers"]  # Number of convolutional layers
    activation = params["activation"]  # Activation function
    lr = params["lr"]  # Learning rate
    gamma = params["gamma"]

    epochs = 100

    activations = {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
    }  # Define activation function mapping.
    activation = activations[activation]  # Get activation function.
    loss_fn = torch.nn.NLLLoss()

    model = Net(conv_layers, activation)  
    # Set up neural network with specified hyperparameters.
    # model.best_accuracy = 0.0  # Initialize the model's best validation accuracy.

    train_loader, val_loader = get_data_loaders(
        batch_size=8, subgroup_comm=subgroup_comm
    )  # Get training and validation data loaders.

    if torch.cuda.is_available():
        device = MPI.COMM_WORLD.rank % GPUS_PER_NODE
        model = model.to(device)
    else:
        device = "cpu"
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    log_interval = 100
    best_val_loss = 1000000
    early_stopping_count, early_stopping_limit = 0, 5
    set_new_best = False
    model.train()
    for epoch in range(epochs):
        # train loop ====================================================
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0 or batch_idx == len(train_loader) - 1:
                log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        # val loop ======================================================
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += loss_fn(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(val_loader.dataset)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            set_new_best = True

        log.info(f'\nTest set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({100. * correct / len(val_loader.dataset):.0f}%)\n')

        if not set_new_best:
            early_stopping_count += 1
        if early_stopping_count >= early_stopping_limit:
            log.info(f"hit early stopping count, breaking")
            break
        
        # scheduler step ================================================
        scheduler.step()
        set_new_best = False
        
    # Return best validation loss as an individual's loss (trained so lower is better)
    return best_val_loss


if __name__ == "__main__":
    config, _ = parse_arguments()

    comm = MPI.COMM_WORLD
    if comm.rank == 0:  # Download data at the top, then we dont need to later
        train_loader = DataLoader(
            dataset=MNIST(
                download=True, root=".", transform=None, train=True
            ),  # Use MNIST training dataset.
            batch_size=2,  # Batch size
            num_workers=1,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True,  # Shuffle data.
        )
        del train_loader
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
    propagator = get_default_propagator(  # Get default evolutionary operator.
        pop_size=pop_size,  # Breeding population size
        limits=limits,  # Search space
        crossover_prob=0.7,  # Crossover probability
        mutation_prob=0.4,  # Mutation probability
        random_init_prob=0.1,  # Random-initialization probability
        rng=rng,  # Separate random number generator for Propulate optimization
    )

    # Set up separate logger for Propulate optimization.
    set_logger_config(
        level=logging.INFO,  # Logging level
        log_file=f"{log_path}/{pathlib.Path(__file__).stem}.log",  # Logging path
        log_to_stdout=True,  # Print log on stdout.
        log_rank=False,  # Do not prepend MPI rank to logging messages.
        colors=True,  # Use colors.
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
        # ----- SPECIFIC FOR MULTI-RANK UCS ----
        ranks_per_worker=2,  # Number of ranks per (multi rank) worker
    )

    # Run actual optimization.
    islands.evolve(
        top_n=config.top_n,  # Print top-n best individuals on each island in summary.
        logging_interval=config.logging_interval,  # Logging interval
        debug=config.verbosity,  # Debug level
    )
