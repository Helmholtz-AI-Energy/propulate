"""
This package bundles all classes that are used as propagators in Propulate's optimization routine.
"""

from .base import (
    Propagator,
    Stochastic,
    Conditional,
    Compose,
    SelectMin,
    SelectMax,
    SelectUniform,
    InitUniform,
)

from .ga import (
    PointMutation,
    RandomPointMutation,
    IntervalMutationNormal,
    MateUniform,
    MateMultiple,
    MateSigmoid,
)

from .pso import (
    BasicPSO,
    VelocityClampingPSO,
    ConstrictionPSO,
    CanonicalPSO,
    StatelessPSO,
)

from .cmaes import (
    CMAAdapter,
    CMAParameter,
    CMAPropagator,
    BasicCMA,
    ActiveCMA,
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
    "MateUniform",
    "MateMultiple",
    "MateSigmoid",
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
]
