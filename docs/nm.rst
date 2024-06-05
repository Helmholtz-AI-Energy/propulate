.. _nm:

Nelder-Mead Optimization
========================

The Nelder-Mead optimization method, also known as the downhill simplex method, is a versatile, derivative-free
technique for finding the minimum of a function in a multidimensional space. It was developed by John Nelder and Roger
Mead in 1965. By iteratively modifying a simplex through geometric transformations, it effectively navigates the search
space to find optimal solutions.

[1] *J. A. Nelder and R. Mead, "A Simplex Method for Function Minimization", The Computer Journal, vol. 7, issue 4,
January 1965, pp. 308–313*, https://doi.org/10.1093/comjnl/7.4.308

Basic Concepts
--------------

- **Simplex:** The method operates on a simplex, which is a geometric figure consisting of :math:`n+1` vertices in an
  :math:`n+1`-dimensional space.
- **Non-Derivative Based:** The Nelder-Mead method does not require the calculation of derivatives, making it suitable
  for optimizing functions that are not differentiable or are noisy.

Key Components
--------------

**1. Initialization**

- Initialize a simplex with :math:`n+1` vertices, where :math:`n` is the number of dimensions of the function to be
  minimized.
- The initial vertices can be chosen based on prior knowledge or randomly.

**2. Simplex Operations**

- The method iteratively modifies the simplex through a series of geometric transformations: reflection, expansion,
  contraction, and shrinkage.
- These operations are designed to move the simplex toward regions of lower function values.

**Reflection**

.. math::
   x_r = x_0 + \alpha (x_0 - x_h)

where:

   - :math:`x_0` is the centroid of the simplex excluding the worst vertex.
   - :math:`x_h` is the worst vertex (highest function value).
   - :math:`\alpha` is the reflection coefficient, typically set to 1.

**Expansion**

.. math::

   x_e = x_0 + \gamma (x_r - x_0)

where:

   - :math:`x_r` is the reflected point.
   - :math:`\gamma` is the expansion coefficient, typically set to 2.

**Contraction**

.. math::

   x_c = x_0 + \rho (x_h - x_0)

where:

   - :math:`x_0` is the centroid of the simplex excluding the worst vertex.
   - :math:`x_h` is the worst vertex (highest function value).
   - :math:`\rho` is the contraction coefficient, typically set to 0.5.

**Shrinkage**

.. math::

   x_i = x_l + \sigma (x_i - x_l)

for all :math:`i`, where:

   - :math:`x_l` is the best vertex (lowest function value).
   - :math:`x_i` are the other vertices.
   - :math:`\sigma` is the shrinkage coefficient, typically set to 0.5.

Iteration
---------

The process of reflection, expansion, contraction, and shrinkage continues iteratively until a stopping criterion is
met, such as a maximum number of iterations or a convergence threshold.

Advantages
----------

- **Derivative-Free:** The Nelder-Mead method does not require the calculation of gradients, making it suitable for
  optimizing non-smooth or noisy functions.
- **Simplicity:** The algorithm is straightforward to implement and understand.
- **Flexibility:** It can be applied to a wide range of optimization problems.

Disadvantages
-------------

- **Scalability:** The method may struggle with high-dimensional problems.
- **Local Optima:** It can get stuck in local optima, especially in highly non-convex landscapes.
