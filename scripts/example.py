#!/usr/bin/env python3

import random

from propulate import Propulator
from propulate.utils import get_default_propagator


random.seed(42)
num_generations = 100

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

propagator, fallback = get_default_propagator(8, limits, .7, .8, .1)

propulator = Propulator(loss, propagator, fallback, num_generations=num_generations)

propulator.propulate()

propulator.summarize()
