#!/usr/bin/env python3
import random
import sys
from propulate import Islands, Propulator
from mpi4py import MPI
from propulate.utils import get_default_propagator
from propulate.propagators import SelectBest, SelectWorst, SelectUniform
import numpy as np
from propulate import Propulator
from propulate.utils import get_default_propagator

import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

NUM_GENERATIONS = 1000
GPUS_PER_NODE = 4
POP_SIZE = 8

limits = {
        'convlayers': (2, 10),
        'activation': ('relu', 'sigmoid', 'tanh'),
        'lr': (0.01, 0.0001),
        }


class Net(nn.Module):
    def __init__(self, convlayers, activation):
        super(Net, self).__init__()

        layers = []
        layers += [nn.Sequential(nn.Conv2d(1, 10, kernel_size=3, padding=1), activation()),]
        layers += [nn.Sequential(nn.Conv2d(10, 10, kernel_size=3, padding=1), activation()) for _ in range(convlayers-1)]

        self.fc = nn.Linear(7840, 10)
        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x):
        b, c, w, h = x.size()
        x = self.conv_layers(x)
        x = x.view(b, 10*28*28)
        x = self.fc(x)
        return x


def get_data_loaders(batch_size):
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(MNIST(download=False, root="/hkfs/work/workspace/scratch/ku4408-propulate/propulate/scripts", transform=data_transform, train=True), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MNIST(download=False, root="/hkfs/work/workspace/scratch/ku4408-propulate/propulate/scripts", transform=data_transform, train=False), batch_size=1, shuffle=False)
    return train_loader, val_loader


def ind_loss(params):
    from mpi4py import MPI

    convlayers = params['convlayers']
    activation = params['activation']
    lr = params['lr']
    epochs = 2

    activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
    activation = activations[activation]

    rank = MPI.COMM_WORLD.Get_rank()

    device = "cuda:{}".format(rank % GPUS_PER_NODE)

    model = Net(convlayers, activation)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loader, val_loader = get_data_loaders(8)

    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'acc': Accuracy(), 'ce': Loss(loss_fn)}, device=device)
    trainer.best_accuracy = 0.

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics

        if metrics['acc'] > trainer.best_accuracy:
            trainer.best_accuracy = metrics['acc']

    trainer.run(train_loader, max_epochs=epochs)
    return -trainer.best_accuracy


if __name__ == "__main__":
    while True:
        #migration_topology = num_migrants*np.ones((4, 4), dtype=int)
        #np.fill_diagonal(migration_topology, 0)

        rng = random.Random(MPI.COMM_WORLD.rank)

        propagator = get_default_propagator(POP_SIZE, limits, .7, .4, .1, rng=rng)
        islands = Islands(ind_loss, propagator, generations=NUM_GENERATIONS,
                          num_isles=2, isle_sizes=[4, 4, 4, 4], #migration_topology=migration_topology, 
                          load_checkpoint = "pop_cpt.p", 
                          save_checkpoint="pop_cpt.p",
                          migration_probability=0.9, emigration_propagator=SelectBest, 
                          immigration_propagator=SelectWorst,
                          pollination=False,
                          rng=rng)
        islands.evolve(top_n=1, logging_interval=1, DEBUG=2)
        break
