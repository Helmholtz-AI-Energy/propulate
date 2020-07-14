#!/usr/bin/env python3

import os

from propulate import Propulator
from propulate.utils import get_default_propagator

checkpoint_file = "./test.pkl"

limits = {
        'x' : (-10., 10.),
        'y' : (-10., 10.),
        'z' : (-10., 10.),
        'u' : (-10., 10.),
        'v' : (-10., 10.),
        'w' : (-10., 10.),
        }

def loss(params):
    return sum([params[x]**2 for x in params])

if not os.path.isfile(checkpoint_file):
    print("No checkpoint found. Running 10 generations from scratch")
    num_generations = 10

    propagator, fallback = get_default_propagator(8, limits, .7, .4, .1)

    propulator = Propulator(loss, propagator, fallback, generations=num_generations, checkpoint_file=checkpoint_file)

    propulator.propulate()

    propulator.summarize()

else:
    print("Resuming from checkpoint. Running another 10 generations.")
    num_generations = 20

    propulator = Propulator(loss, propagator, fallback, generations=num_generations, checkpoint_file=checkpoint_file)

    propulator.propulate(resume=True)

    propulator.summarize()
