from .propagators import Cascade, PointMutation, MateUniform, SelectBest, SelectUniform, InitUniform, IntervalMutationNormal

def get_default_propagator(pop_size, limits, mate_prob, mut_prob, random_prob, sigma_factor=0.05):
    propagator = Cascade(
            [
                SelectBest(pop_size),
                SelectUniform(2),
                MateUniform(mate_prob),
                PointMutation(limits, probability=mut_prob),
                IntervalMutationNormal(limits, sigma_factor=sigma_factor, probability=1),
                InitUniform(limits, parents=1, probability=random_prob),
            ]
        )

    fallback = InitUniform(limits)
    return propagator, fallback
