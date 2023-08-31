#!/usr/bin/env python3

import random
import sys
import time
from typing import Tuple, Dict, Union

import numpy as np
import torch
from lightning.pytorch import loggers
from torch import nn
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule, Trainer
from torchmetrics import Accuracy

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

from mpi4py import MPI

from ap_pso.propagators import *
from propulate import Islands
from propulate.propagators import Conditional

num_generations = 10
pop_size = 2 * MPI.COMM_WORLD.size
GPUS_PER_NODE = 1  # 4
log_path = "torch_ckpts"

limits = {
    "convlayers": (2.0, 10.0),
    "lr": (0.01, 0.0001),
}


class Net(LightningModule):
    def __init__(self, convlayers: int, activation, lr: float, loss_fn):
        super(Net, self).__init__()

        self.lr = lr
        self.loss_fn = loss_fn
        self.best_accuracy = 0.0
        layers = []
        layers += [
            nn.Sequential(nn.Conv2d(1,
                                    10,
                                    kernel_size=3,
                                    padding=1),
                          activation()),
        ]
        layers += [
            nn.Sequential(nn.Conv2d(10,
                                    10,
                                    kernel_size=3,
                                    padding=1),
                          activation())
            for _ in range(convlayers - 1)
        ]

        self.fc = nn.Linear(7840, 10)
        self.conv_layers = nn.Sequential(*layers)

        self.val_acc = Accuracy('multiclass', num_classes=10)
        self.train_acc = Accuracy("multiclass", num_classes=10)

    def forward(self, x):
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
        b, c, w, h = x.size()
        x = self.conv_layers(x)
        x = x.view(b, 10 * 28 * 28)
        x = self.fc(x)
        return x

    def training_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
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
        pred = self(x)
        loss_val = self.loss_fn(pred, y)
        self.log("train loss", loss_val)
        train_acc_val = self.train_acc(torch.nn.functional.softmax(pred, dim=-1), y)
        self.log("train_ acc", train_acc_val)
        return loss_val

    def validation_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
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
        loss_val = self.loss_fn(pred, y)
        val_acc_val = self.val_acc(torch.nn.functional.softmax(pred, dim=-1), y)
        self.log("val_loss", loss_val)
        self.log("val_acc", val_acc_val)
        return loss_val

    def configure_optimizers(self) -> torch.optim.SGD:
        """
        Configure optimizer.

        Returns
        -------
        torch.optim.sgd.SGD
            stochastic gradient descent optimizer
        """
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def on_validation_epoch_end(self):
        """
        Calculate and store the model's validation accuracy after each epoch.
        """
        val_acc_val = self.val_acc.compute()
        self.log("val_acc_val", val_acc_val)
        self.val_acc.reset()
        if val_acc_val > self.best_accuracy:
            self.best_accuracy = val_acc_val

def get_data_loaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
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
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    if MPI.COMM_WORLD.Get_rank() == 0:  # Only root downloads data.
        train_loader = DataLoader(
            dataset=MNIST(
                download=True, root=".", transform=data_transform,
            ),  # Use MNIST training dataset.
            batch_size=batch_size,  # Batch size
            shuffle=True,  # Shuffle data.
        )
        MPI.COMM_WORLD.Barrier()
    else:
        MPI.COMM_WORLD.Barrier()
        train_loader = DataLoader(
            dataset=MNIST(
                root=".", transform=data_transform
            ),  # Use MNIST training dataset.
            batch_size=batch_size,  # Batch size
            shuffle=True,  # Shuffle data.
        )
    val_loader = DataLoader(
        dataset=MNIST(
            root=".", transform=data_transform, train=False
        ),  # Use MNIST testing dataset.
        shuffle=False,  # Do not shuffle data.
    )
    return train_loader, val_loader

def ind_loss(params: Dict[str, Union[int, float, str]]) -> float:
    """
    Loss function for evolutionary optimization with Propulate. Minimize the model's negative validation accuracy.

    Parameters
    ----------
    params: dict[str, int | float | str]]

    Returns
    -------
    float
        The trained model's negative validation accuracy
    """
    # Extract hyperparameter combination to test from input dictionary.
    conv_layers = int(np.round(params["conv_layers"]))  # Number of convolutional layers
    if conv_layers < 1:
        return float(10 - 10 * conv_layers)
    activation = params["activation"]  # Activation function
    lr = params["lr"]  # Learning rate

    epochs = 2  # Number of epochs to train

    activations = {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
    }  # Define activation function mapping.
    activation = activations[activation]  # Get activation function.
    loss_fn = (
        torch.nn.CrossEntropyLoss()
    )  # Use cross-entropy loss for multi-class classification.

    model = Net(
        conv_layers, activation, lr, loss_fn
    )  # Set up neural network with specified hyperparameters.
    model.best_accuracy = 0.0  # Initialize the model's best validation accuracy.

    train_loader, val_loader = get_data_loaders(
        batch_size=8
    )  # Get training and validation data loaders.

    tb_logger = loggers.TensorBoardLogger(
        save_dir=log_path + "/lightning_logs"
    )  # Get tensor board logger.

    # Under the hood, the Lightning Trainer handles the training loop details.
    trainer = Trainer(
        max_epochs=epochs,  # Stop training once this number of epochs is reached.
        accelerator="gpu",  # Pass accelerator type.
        devices=[MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE],  # Devices to train on
        enable_progress_bar=True,  # Disable progress bar.
        logger=tb_logger,  # Logger
    )
    trainer.fit(  # Run full model training optimization routine.
        model=model,  # Model to train
        train_dataloaders=train_loader,  # Dataloader for training samples
        val_dataloaders=val_loader,  # Dataloader for validation samples
    )
    # Return negative best validation accuracy as an individual's loss.
    return -model.best_accuracy


if __name__ == "__main__":
    rng = random.Random(MPI.COMM_WORLD.rank)
    pso = [
        VelocityClampingPropagator(0.7298, 1.49618, 1.49618, MPI.COMM_WORLD.rank, limits, rng, 0.6),
        ConstrictionPropagator(2.49618, 2.49618, MPI.COMM_WORLD.rank, limits, rng),
        BasicPSOPropagator(0.7298, 0.5, 0.5, MPI.COMM_WORLD.rank, limits, rng),
        CanonicalPropagator(2.49618, 2.49618, MPI.COMM_WORLD.rank, limits, rng)
    ][int(sys.argv[1])]

    # propagator = get_default_propagator(pop_size, limits, 0.7, 0.4, 0.1, rng=rng)

    if MPI.COMM_WORLD.rank == 0:
        print("#-----------------------------------#")
        print(f"| Current time: {time.time_ns()} |")
        print("#-----------------------------------#")

    propagator = Conditional(pop_size, pso, PSOInitUniform(limits, rng=rng, rank=MPI.COMM_WORLD.rank))
    islands = Islands(ind_loss, propagator, rng, generations=num_generations, pollination=False,
                      migration_probability=0, checkpoint_path=log_path)
    islands.evolve(top_n=1, debug=2)

    if MPI.COMM_WORLD.rank == 0:
        print("#-----------------------------------#")
        print(f"| Current time: {time.time_ns()} |")
        print("#-----------------------------------#")
