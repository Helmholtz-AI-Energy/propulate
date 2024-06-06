"""This package bundles all classes that are used as propagators in Propulate's optimization routine."""

from .base import (
    Compose,
    Conditional,
    InitUniform,
    Propagator,
    SelectMax,
    SelectMin,
    SelectUniform,
    Stochastic,
)
from .cmaes import (
    ActiveCMA,
    BasicCMA,
    CMAAdapter,
    CMAParameter,
    CMAPropagator,
)
from .ga import (
    CrossoverMultiple,
    CrossoverSigmoid,
    CrossoverUniform,
    IntervalMutationNormal,
    PointMutation,
    RandomPointMutation,
)
from .nm import ParallelNelderMead
from .pso import (
    BasicPSO,
    CanonicalPSO,
    ConstrictionPSO,
    StatelessPSO,
    VelocityClampingPSO,
)

__all__ = [
    "Propagator",
    "Stochastic",
    "Conditional",
    "Compose",
    "SelectMin",
    "SelectMax",
    "SelectUniform",
    "InitUniform",
    "PointMutation",
    "RandomPointMutation",
    "IntervalMutationNormal",
    "CrossoverUniform",
    "CrossoverMultiple",
    "CrossoverSigmoid",
    "BasicPSO",
    "VelocityClampingPSO",
    "ConstrictionPSO",
    "CanonicalPSO",
    "StatelessPSO",
    "CMAAdapter",
    "CMAParameter",
    "CMAPropagator",
    "BasicCMA",
    "ActiveCMA",
    "ParallelNelderMead",
]
