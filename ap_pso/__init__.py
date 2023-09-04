"""
This package contains - except for the example and the init propagator everything I added to propulate to be able to
run PSO on it.
"""
__all__ = ["Particle", "propagators", "make_particle"]

from propulate.particle import Particle
from ap_pso.utils import make_particle
