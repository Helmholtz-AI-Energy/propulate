#!/usr/bin/env python3

import random

import torch
from torch import nn
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule, Trainer
from torchmetrics import Accuracy

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

from mpi4py import MPI

from propulate import Islands
from propulate.utils import get_default_propagator
# from propulate.propagators import SelectMin, SelectMax


num_generations = 3
pop_size = 2 * MPI.COMM_WORLD.size
GPUS_PER_NODE = 4

limits = {
    "convlayers": (2, 10),
    "activation": ("relu", "sigmoid", "tanh"),
    "lr": (0.01, 0.0001),
}


class Net(LightningModule):
    def __init__(self, convlayers, activation, lr, loss_fn):
        super(Net, self).__init__()

        self.lr = lr
        self.loss_fn = loss_fn
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

    def forward(self, x):
        b, c, w, h = x.size()
        x = self.conv_layers(x)
        x = x.view(b, 10 * 28 * 28)
        x = self.fc(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        val_acc = self.val_acc(torch.nn.functional.softmax(pred, dim=-1), y)
        if val_acc > self.best_accuracy:
            self.best_accuracy = val_acc
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer


def get_data_loaders(batch_size):
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(
        MNIST(download=False, root=".", transform=data_transform, train=True),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        MNIST(download=False, root=".", transform=data_transform, train=False),
        batch_size=1,
        shuffle=False,
    )
    return train_loader, val_loader


def ind_loss(params):
    convlayers = params["convlayers"]
    activation = params["activation"]
    lr = params["lr"]
    epochs = 2

    activations = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
    activation = activations[activation]
    loss_fn = torch.nn.CrossEntropyLoss()

    model = Net(convlayers, activation, lr, loss_fn)
    model.best_accuracy = 0.0

    train_loader, val_loader = get_data_loaders(8)
    trainer = Trainer(max_epochs=epochs,
                      accelerator='gpu',
                      devices=[
                          MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE
                              ],
                      enable_progress_bar=False,
                      )
    trainer.fit(model, train_loader, val_loader)

    return -model.best_accuracy.item()


if __name__ == "__main__":
    rng = random.Random(MPI.COMM_WORLD.rank)
    propagator = get_default_propagator(pop_size, limits, 0.7, 0.4, 0.1, rng=rng)
    islands = Islands(
            ind_loss,
            propagator,
            rng,
            generations=num_generations,
            num_isles=2,
            migration_probability=0.9,
        )
    islands.evolve(top_n=1, logging_interval=1, DEBUG=2)
