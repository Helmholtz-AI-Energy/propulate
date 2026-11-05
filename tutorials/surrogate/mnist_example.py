import logging
import os
import random
import sys
from typing import Dict, Generator, Tuple, Union

import torch
from mpi4py import MPI
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

from propulate import Islands, surrogate
from propulate.utils import get_default_propagator

GPUS_PER_NODE: int = 1
NUM_WORKERS: int = 2  # Set this to the recommended number of workers in the PyTorch dataloader.

log_path = "torch_ckpts"
log = logging.getLogger(__name__)  # Get logger instance.

sys.path.append(os.path.abspath("../../"))


class Net(nn.Module):
    """Convolutional neural network class."""

    def __init__(self, conv_layers: int, activation: nn.Module, lr: float, loss_fn: nn.Module) -> None:
        """
        Set up neural network.

        Parameters
        ----------
        conv_layers : int
            The number of convolutional layers.
        activation : torch.nn.modules.activation
            The activation function to use.
        lr : float
            The learning rate.
        loss_fn : torch.nn.modules.loss
            The loss function.
        """
        super(Net, self).__init__()

        self.lr = lr  # Set learning rate.
        self.loss_fn = loss_fn  # Set the loss function used for training the model.
        self.best_accuracy = 0.0  # Initialize the model's best validation accuracy.
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
        self.val_acc = Accuracy("multiclass", num_classes=10)
        self.train_acc = Accuracy("multiclass", num_classes=10)

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
        return x

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Calculate loss for training step in Lightning train loop.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            The input batch.

        Returns
        -------
        torch.Tensor
            The training loss for the input batch.
        """
        x, y = batch
        pred = self(x)
        loss_val = self.loss_fn(pred, y)
        return loss_val

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Calculate loss for validation step in Lightning validation loop during training.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            The current batch.

        Returns
        -------
        torch.Tensor
            The validation loss for the input batch.
        """
        x, y = batch
        pred = self(x)
        loss_val = self.loss_fn(pred, y)
        return loss_val

    def configure_optimizers(self) -> torch.optim.SGD:
        """
        Configure the sgd optimizer.

        Returns
        -------
        torch.optim.sgd.SGD
            A stochastic gradient descent optimizer.
        """
        return torch.optim.SGD(self.parameters(), lr=self.lr)


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
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    num_workers = NUM_WORKERS
    log.info(f"Use {num_workers} workers in dataloader.")

    # Note that the MNIST dataset has already been downloaded before globally by rank 0 in the main part.
    train_loader = DataLoader(
        dataset=MNIST(download=False, root=".", transform=data_transform, train=True),  # Use MNIST training dataset.
        batch_size=batch_size,  # Batch size
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        shuffle=True,  # Shuffle data.
    )
    val_loader = DataLoader(
        dataset=MNIST(download=False, root=".", transform=data_transform, train=False),  # Use MNIST testing dataset.
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        batch_size=1,  # Batch size
        shuffle=False,  # Do not shuffle data.
    )
    return train_loader, val_loader


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
        device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")

    log.info(f"Rank: {rank}, Using device: {device}")

    activations = {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
    }  # Define activation function mapping.
    activation = activations[activation]  # Get activation function.
    loss_fn = torch.nn.CrossEntropyLoss()  # Use cross-entropy loss for multi-class classification.

    model = Net(conv_layers, activation, lr, loss_fn).to(device)  # Set up neural network with specified hyperparameters.
    model.best_accuracy = 0.0  # Initialize the model's best validation accuracy.

    train_loader, val_loader = get_data_loaders(batch_size=8)  # Get training and validation dataloaders.

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
        log.info(f"Epoch {epoch + 1}: Avg Training Loss: {avg_train_loss}")

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
        log.info(f"Epoch {epoch + 1}: Avg Validation Loss: {avg_val_loss}")

        yield avg_val_loss


def set_seeds(seed_value: int = 42) -> None:
    """
    Set seed for reproducibility.

    Parameters
    ----------
    seed_value : int, optional
        The seed to use. Default is 42.
    """
    random.seed(seed_value)  # Python random module
    torch.manual_seed(seed_value)  # PyTorch random number generator for CPU
    torch.cuda.manual_seed(seed_value)  # PyTorch random number generator for all GPUs
    torch.cuda.manual_seed_all(seed_value)  # PyTorch random number generator for multi-GPU
    torch.backends.cudnn.deterministic = True  # Use deterministic algorithms.
    torch.backends.cudnn.benchmark = False  # Disable to be deterministic.
    os.environ["PYTHONHASHSEED"] = str(seed_value)  # Python hash seed


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    if comm.rank == 0:  # Download data at the top, then we don't need to later.
        MNIST(download=True, root=".", transform=None, train=True)
        MNIST(download=True, root=".", transform=None, train=False)
    comm.Barrier()

    num_generations = 3  # Number of generations
    pop_size = 2 * comm.size  # Breeding population size
    limits: Dict[str, Union[Tuple[int, int], Tuple[float, float], Tuple[str, ...]]] = {
        "conv_layers": (2, 10),
        "activation": ("relu", "sigmoid", "tanh"),
        "lr": (0.01, 0.0001),
    }  # Define search space.
    rng = random.Random(comm.rank)  # Set up separate random number generator for evolutionary optimizer.
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
        # surrogate_factory=lambda: surrogate.DynamicSurrogate(limits),
    )
    islands.propulate(  # Run evolutionary optimization.
        logging_interval=1,  # Logging interval
        debug=2,  # Verbosity level
    )
    islands.summarize(
        top_n=1,  # Print top-n best individuals on each island in summary.
        debug=2,  # Verbosity level
    )
