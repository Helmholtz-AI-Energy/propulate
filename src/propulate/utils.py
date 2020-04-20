from .propagators import Cascade, PointMutation, MateUniform, SelectBest, SelectUniform, InitUniform, IntervalMutationNormal

def get_default_propagator(pop_size, limits, mate_prob, mut_prob, random_prob):
    propagator = Cascade(
            [
                SelectBest(pop_size),
                SelectUniform(2),
                MateUniform(mate_prob),
                PointMutation(limits, mut_prob),
                IntervalMutationNormal(limits, probability=1),
                InitUniform(limits, parents=1, probability=random_prob),
            ]
        )

    fallback = InitUniform(limits)
    return propagator, fallback
