.. _pop::

Population-Based Algorithms
===========================
Population-based algorithms are optimization and search algorithms inspired by biological evolution and natural
selection. These algorithms maintain a set (or *population*) of candidate solutions where each solution represents a
unique point in the search space of the problem. They are designed to solve complex optimization problems by iteratively
evolving this population of candidate solution over multiple generations. The goal is to find the best solution within a
given search space without exhaustively evaluating every possible solution. The core idea is to mimic the process of
evolution, where candidates with better traits have a higher chance of surviving and reproducing, passing on their
advantageous traits to the next generation. Similarly, in optimization problems, the solutions that exhibit better
performance with respect to the optimization objective are favored and used to generate new candidate solutions.
Population-based algorithms have been applied to a wide range of optimization problems, including engineering design,
scheduling, machine learning parameter tuning, and more. They can explore diverse parts of the search space
simultaneously, which can help find global or near-global optima in complex optimization landscapes.

WHAT?
    "Survival of the fittest" metaheuristics inspired by biological evolution

WHY?
    Find good-enough solutions to global optimization problems efficiently.

HOW?
    Individuals
        Representation of candidate solutions in the search space. This is the vector of parameters to be optimized.
    Fitness function
        Scalar metric to evaluate how good an individual is. This is the metric to optimize on.
    Propagators
        Mechanisms for breeding new (hopefully better) individuals from current ones. This is what we do to the current
        population of individuals to obtain the next generation to be evaluated.

.. warning::

   As with all metaheuristics, these algorithms have hyperparameters themselves and their effectiveness strongly depends
   on choosing proper hyperparameters and problem-specific considerations.