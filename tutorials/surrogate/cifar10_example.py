import enum
import logging
import os
import random
import sys
from typing import Dict, Generator, Tuple, Union

import torch
from mpi4py import MPI
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

from propulate import Islands, surrogate
from propulate.utils import get_default_propagator

GPUS_PER_NODE: int = 1

log_path = "torch_ckpts"
log = logging.getLogger(__name__)  # Get logger instance.

sys.path.append(os.path.abspath("../../"))


class Permutation(enum.Enum):
    """
    Enum class for permutations of layers in convolution block.

    Each letter represents a layer in the block. The default permutation is "ABC".
    """

    ABC = (0, 1, 2)
    ACB = (0, 2, 1)
    BAC = (1, 0, 2)
    BCA = (1, 2, 0)
    CAB = (2, 0, 1)
    CBA = (2, 1, 0)


def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Calculate accuracy of model's predictions.

    Parameters
    ----------
    outputs : torch.Tensor
        The model's predictions.
    labels : torch.Tensor
        The true labels.

    Returns
    -------
    torch.Tensor
        The accuracy of the model's predictions.
    """
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def shuffle_array(array: list, permutation: Permutation) -> list:
    """
    Shuffle the order of elements in an array according to a given permutation.

    Parameters
    ----------
    array : list
        The array to shuffle.
    permutation : Permutation
        The permutation to apply to the array.

    Returns
    -------
    list
        The shuffled array.
    """
    return [array[i] for i in permutation.value]


def conv_block(
    in_channels: int,
    out_channels: int,
    perm: Permutation = Permutation.ABC,
    pool: bool = False,
) -> nn.Sequential:
    """
    Create a ResNet block, consisting of the convolution layer, batch normalization, and ReLU activation function.

    The order of these layers can be shuffled according to a given permutation.

    Parameters
    ----------
    in_channels : int
        The number of input channels.
    out_channels : int
        The number of output channels.
    perm : Permutation
        Shuffle layers according to given permutation.
    pool : bool
        Whether to apply max pooling.

    Returns
    -------
    nn.Sequential
        The ResNet convolution block.
    """
    # Adjust when batch normalization is called before convolution.
    batch_norm_in = out_channels if perm in [Permutation.ABC, Permutation.ACB, Permutation.CAB] else in_channels

    # Shuffle layers to sometimes get bad performance.
    layers = shuffle_array(
        [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(batch_norm_in),
            nn.ReLU(inplace=True),
        ],
        perm,
    )
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class Net(nn.Module):
    """Residual neural network class."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        perm: Permutation,
        lr: float,
        loss_fn: nn.Module,
        weight_decay: float = 0.0,
    ) -> None:
        """
        Set up ResNet neural network.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        num_classes : int
            The number of classes in the dataset.
        perm : Permutation
              Shuffle layers according to given permutation.
        lr : float
            The learning rate.
        loss_fn : torch.nn.modules.loss
            The loss function.
        weight_decay : float
            The weight decay.
        """
        super().__init__()

        self.lr = lr  # Set learning rate.
        self.loss_fn = loss_fn  # Set the loss function used for training the model.
        self.best_accuracy = 0.0  # Initialize the model's best validation accuracy.
        self.weight_decay = weight_decay

        self.val_acc = Accuracy("multiclass", num_classes=10)
        self.train_acc = Accuracy("multiclass", num_classes=10)

        self.conv1 = conv_block(in_channels, 64, perm)
        self.conv2 = conv_block(64, 128, perm, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128, perm), conv_block(128, 128, perm))

        self.conv3 = conv_block(128, 256, perm, pool=True)
        self.conv4 = conv_block(256, 512, perm, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512, perm), conv_block(512, 512, perm))

        self.classifier = nn.Sequential(nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, num_classes))

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
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Calculate loss for training step in Lightning train loop.

        Parameters
        ----------
        batch: Tuple[torch.Tensor, torch.Tensor]
            The input batch.
        batch_idx: int
            The batch index.

        Returns
        -------
        torch.Tensor
            The training loss for the input batch.
        """
        x, y = batch
        pred = self(x)
        loss_val = self.loss_fn(pred, y)
        return loss_val

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Calculate loss for validation step in Lightning validation loop during training.

        Parameters
        ----------
        batch: Tuple[torch.Tensor, torch.Tensor]
            The current batch.
        batch_idx: int
            The batch index.

        Returns
        -------
        torch.Tensor
            The validation loss for the input batch.
        """
        x, y = batch
        pred = self(x)
        loss_val = self.loss_fn(pred, y)
        return loss_val

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the adam optimizer.

        Returns
        -------
        torch.optim.Optimizer
            An instance of the Adam optimizer with the specified learning rate and weight decay.

        """
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def get_data_loaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR10 train and validation dataloaders.

    Parameters
    ----------
    batch_size: int
        The batch size.

    Returns
    -------
    torch.utils.data.DataLoader
        The training dataloader.
    torch.utils.data.DataLoader
        The validation dataloader.
    """
    data_transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Set empty DataLoader.
    train_loader = DataLoader(
        dataset=TensorDataset(torch.empty(0), torch.empty(0)),
        batch_size=batch_size,
        shuffle=False,
    )

    if MPI.COMM_WORLD.Get_rank() == 0:  # Only root downloads data.
        train_loader = DataLoader(
            dataset=CIFAR10(download=True, root=".", transform=data_transform, train=True),  # Use CIFAR10 training dataset.
            batch_size=batch_size,  # Batch size
            shuffle=True,  # Shuffle data.
        )

    # NOTE barrier only called, when dataset has not been downloaded yet
    if not hasattr(get_data_loaders, "barrier_called"):
        MPI.COMM_WORLD.Barrier()

        setattr(get_data_loaders, "barrier_called", True)

    if MPI.COMM_WORLD.Get_rank() != 0:
        train_loader = DataLoader(
            dataset=CIFAR10(download=False, root=".", transform=data_transform, train=True),  # Use CIFAR10 training dataset.
            batch_size=batch_size,  # Batch size
            shuffle=True,  # Shuffle data.
        )
    val_loader = DataLoader(
        dataset=CIFAR10(download=False, root=".", transform=data_transform, train=False),  # Use CIFAR testing dataset.
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
    params : Dict[str, int | float | str]]
        The parameters to be optimized.

    Returns
    -------
    Generator[float, None, None]
        Yields the negative validation accuracy in regular intervals during training of the model.
    """
    # Extract hyperparameter combination to test from input dictionary.
    lr = float(params["lr"])  # Learning rate
    perm = str(params["perm"])  # Permutation

    epochs: int = 2  # Number of epochs to train

    rank: int = MPI.COMM_WORLD.Get_rank()  # Get rank of current worker.

    num_gpus = torch.cuda.device_count()  # Number of GPUs available
    if num_gpus == 0:
        device = torch.device("cpu")
    else:
        device_index = rank % num_gpus
        device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")

    log.info(f"Rank: {rank}, Using device: {device}")

    permutations = {
        "ABC": Permutation.ABC,
        "ACB": Permutation.ACB,
        "BAC": Permutation.BAC,
        "BCA": Permutation.BCA,
        "CAB": Permutation.CAB,
        "CBA": Permutation.CBA,
    }

    permutation = permutations[perm]  # Get permutation.

    num_classes: int = 10  # Number of classes in CIFAR10 dataset.
    in_channels: int = 3  # Number of channels in CIFAR10 images.

    loss_fn = torch.nn.CrossEntropyLoss()  # Use cross-entropy loss for multi-class classification.

    # Use non-null weight decay.
    weight_decay = 1e-4
    grad_clip = 0.1

    model = Net(in_channels, num_classes, permutation, lr, loss_fn, weight_decay).to(
        device
    )  # Set up neural network with specified hyperparameters.
    model.best_accuracy = 0.0  # Initialize the model's best validation accuracy.

    train_loader, val_loader = get_data_loaders(batch_size=8)  # Get training and validation data loaders.

    # Configure optimizer.
    optimizer = model.configure_optimizers()

    # Initialize average validation loss parameter.
    avg_val_loss: float = 0.0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        # Training loop
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # Zero out gradients.
            optimizer.zero_grad()
            # Forward + backward + optimize.
            loss = model.training_step((data, target), batch_idx)
            loss.backward()

            # Use gradient clipping.
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)

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
                loss = model.validation_step((data, target), batch_idx)
                # Update loss.
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        log.info(f"Epoch {epoch + 1}: Avg Validation Loss: {avg_val_loss}")

        yield avg_val_loss


def set_seeds(seed_value: int = 42) -> None:
    """Set seed for reproducibility."""
    random.seed(seed_value)  # Python random module
    torch.manual_seed(seed_value)  # pytorch random number generator for CPU
    torch.cuda.manual_seed(seed_value)  # pytorch random number generator for all GPUs
    torch.cuda.manual_seed_all(seed_value)  # for multi-GPU.
    torch.backends.cudnn.deterministic = True  # use deterministic algorithms.
    torch.backends.cudnn.benchmark = False  # disable to be deterministic.
    os.environ["PYTHONHASHSEED"] = str(seed_value)  # python hash seed.


if __name__ == "__main__":
    num_generations = 3  # Number of generations
    pop_size = 2 * MPI.COMM_WORLD.size  # Breeding population size
    limits: Dict[str, Union[Tuple[int, int], Tuple[float, float], Tuple[str, ...]]] = {
        "lr": (0.01, 0.0001),
        "perm": ("ABC", "ACB", "BAC", "BCA", "CAB", "CBA"),
    }  # Define search space.
    rng = random.Random(MPI.COMM_WORLD.rank)  # Set up separate random number generator for evolutionary optimizer.
    set_seeds(42 * MPI.COMM_WORLD.Get_rank())  # set seed for torch
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
    )
