from .propagators import Cascade, PointMutation, MateUniform, SelectBest, SelectUniform, InitUniform

def get_default_propagator(pop_size, limits, mate_prob, mut_prob):
    propagator = Cascade(
            [
                SelectBest(pop_size),
                SelectUniform(2),
                MateUniform(mate_prob),
                PointMutation(limits, mut_prob),
            ]
        )

    fallback = InitUniform(limits)
    return propagator, fallback
