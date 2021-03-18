#!/usr/bin/env python3

from propulate import Propulator
from propulate.utils import get_default_propagator

NUM_GENERATIONS = 10

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


propagator = get_default_propagator(8, limits, .7, .4, .1)
propulator = Propulator(loss, propagator, generations=NUM_GENERATIONS)
propulator.propulate()
propulator.summarize()
