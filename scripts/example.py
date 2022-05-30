#!/usr/bin/env python3

from propulate import Propulator
from propulate.utils import get_default_propagator

NUM_GENERATIONS = 10

limits = {
    "x": (-10.0, 10.0),
    "y": (-10.0, 10.0),
    "z": (-10.0, 10.0),
    "u": (-10.0, 10.0),
    "v": (-10.0, 10.0),
    "w": (-10.0, 10.0),
}


def loss(params):
    return sum([params[x] ** 2 for x in params])


propagator = get_default_propagator(8, limits, 0.7, 0.4, 0.1)
propulator = Propulator(loss, propagator, generations=NUM_GENERATIONS)
propulator.propulate()
