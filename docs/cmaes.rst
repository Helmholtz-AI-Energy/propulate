.. _cmaes:

Covariance Matrix Adaption Evolution Strategy
=============================================

Covariance Matrix Adaptation Evolution Strategy (CMA-ES) is a robust and efficient evolutionary algorithm for solving
difficult non-linear, non-convex optimization problems. It was developed by Nikolaus Hansen and Andreas Ostermeier in
the late 1990s [1]. It leverages covariance matrix adaptation to efficiently explore and exploit the search space.

[1] *N. Hansen and A. Ostermeier, "Completely Derandomized Self-Adaptation in Evolution Strategies", in Evolutionary
Computation, vol. 9, no. 2, pp. 159-195, June 2001*, https://doi.org/10.1162/106365601750190398.

Basic Concepts
--------------

- **Population-Based Search:** CMA-ES operates on a population of candidate solutions, evolving them over generations
  to improve the overall fitness.
- **Covariance Matrix:** A key feature of CMA-ES is the adaptation of the covariance matrix, which models the
  distribution of the population and its correlations, allowing for effective search in the solution space.

Key Components
--------------
**1. Initialization**

- Initialize a population of candidate solutions with a mean vector :math:`m`, step size :math:`\sigma`, and
  covariance matrix :math:`C`.
- The initial mean vector is set based on prior knowledge or randomly.

**2. Sampling**

- Generate new candidate solutions by sampling from a multivariate normal distribution:

  .. math::

     x_k = m + \sigma \cdot N\left(0, C\right)

  where:

    - :math:`x_k` is the *k*-th candidate solution.
    - :math:`m` is the mean vector.
    - :math:`\sigma` is the step size.
    - :math:`N\left(0, C\right)` is a normal distribution with mean 0 and covariance matrix :math:`C`.

**3. Evaluation**

- Evaluate the fitness (or loss) of each candidate solution using a predefined fitness (loss) function.

**4. Selection**

- Select the best-performing candidates based on their fitness values.

**5. Recombination and Adaptation**

- Update the mean vector :math:`m` to the weighted average of the selected candidates:

  .. math::

     m_\text{new} = \sum_{i=1}^{\mu} w_i \cdot x_i

  where:

    - :math:`w_i` are the recombination weights.
    - :math:`x_i` are the selected candidate solutions.

- Adapt the covariance matrix :math:`C` using the evolution path and selected candidates:

  .. math::

     C_{new} = (1 - c_1 - c_\mu) \cdot C + c_1 \cdot p_c \cdot p_c^T + c_\mu \cdot \sum_{i=1}^{\mu} w_i \cdot (x_i - m)(x_i - m)^T

  where:

    - :math:`c_1` and :math:`c_\mu` are learning rates.
    - :math:`p_c` is the evolution path.

**6. Step Size Adaptation**

- Adapt the step size :math:`\sigma` based on the length of the evolution path:

  .. math::

     \sigma_text{new} = \sigma \cdot \exp \left( \frac{c_\sigma}{d_\sigma} \left( \frac{||p_\sigma||}{E(||N(0,I)||)} - 1 \right) \right)

  where:

    - :math:`c_\sigma` and :math:`d_\sigma` are learning rates.
    - :math:`p_\sigma` is the evolution path for the step size.
    - :math:`E(||N(0,I)||)` is the expected length of a random vector from the standard normal distribution.

Iteration
---------
The process of sampling, evaluating, selecting, and adapting continues iteratively until a stopping criterion is met,
such as a maximum number of generations or a satisfactory fitness level.

Advantages
----------
- CMA-ES is highly effective for solving complex, multi-modal optimization problems.
- It requires minimal parameter tuning and adapts dynamically to the search landscape.
- The algorithm's robustness and adaptability make it suitable for a wide range of applications.
