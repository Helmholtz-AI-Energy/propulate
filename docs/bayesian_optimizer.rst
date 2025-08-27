.. _bayesopt:

Bayesian Optimization
=====================

Bayesian Optimization (BO) is a sample-efficient strategy for optimizing black-box, expensive, and potentially noisy
objective functions. It builds a *surrogate model* of the objective (commonly a Gaussian Process) and uses an
*acquisition function* to decide where to evaluate next, explicitly balancing exploration and exploitation [1, 2, 3]. 
To prevent repeated evaluation of identical candidates by different ranks, the acquisition parameters representing 
the tradeoff between exploration and exploitation are varied between ranks [4].

[1] *J. Mockus, V. Tiesis, and A. Zilinskas, "The application of Bayesian methods for seeking the extremum", 1978.*

[2] *D. R. Jones, M. Schonlau, and W. J. Welch, "Efficient Global Optimization of Expensive Black-Box Functions",
Journal of Global Optimization, 1998.*

[3] *J. Snoek, H. Larochelle, and R. P. Adams, "Practical Bayesian Optimization of Machine Learning Algorithms",
Advances in Neural Information Processing Systems (NeurIPS), 2012.*

[4] *Häse F, Roch LM, Kreisbeck C, Aspuru-Guzik A. "Phoenics: A Bayesian Optimizer for Chemistry", ACS Cent Sci. 2018.*


Basic Concepts
--------------

- **Surrogate Model (Gaussian Process):** BO models the objective :math:`f(\mathbf{x})` with a probabilistic
  surrogate giving a posterior mean :math:`\mu(\mathbf{x})` and standard deviation :math:`\sigma(\mathbf{x})`.
- **Acquisition Function:** A cheap-to-evaluate utility that uses :math:`\mu(\mathbf{x})` and :math:`\sigma(\mathbf{x})`
  to rank candidate points. Maximizing (or minimizing a negated form of) the acquisition chooses the next evaluation.
- **Exploration–Exploitation Trade-off:** The acquisition trades off sampling uncertain regions (exploration) and
  promising regions (exploitation).
- **Per-Rank Surrogates in Propulate:** In parallel runs, each MPI rank fits its own GP on its local data
  ,optionally using different acquisition hyperparameters via rank stretching (highly reccomended).

Key Components
--------------

**1. Gaussian Process Posterior**

Given training data :math:`X \in \mathbb{R}^{n \times d}`, :math:`\mathbf{y} \in \mathbb{R}^n`, kernel
:math:`k(\cdot,\cdot)`, and noise variance :math:`\sigma_n^2`, the GP posterior at :math:`\mathbf{x}` is

.. math::

   \mu(\mathbf{x}) = \mathbf{k}(\mathbf{x}, X) \left[K + \sigma_n^2 I\right]^{-1} \mathbf{y}, \qquad
   \sigma^2(\mathbf{x}) = k(\mathbf{x}, \mathbf{x}) - \mathbf{k}(\mathbf{x}, X)\left[K + \sigma_n^2 I\right]^{-1}\mathbf{k}(X, \mathbf{x}),

where :math:`K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)` and :math:`\mathbf{k}(\mathbf{x}, X) = [k(\mathbf{x}, \mathbf{x}_1),\ldots,k(\mathbf{x}, \mathbf{x}_n)]`.

**2. Acquisition Functions (minimization)**

- *Expected Improvement (EI)* with jitter :math:`\xi \ge 0` and current best :math:`f^\*`:

  .. math::

     Z(\mathbf{x}) = \frac{f^\* - \mu(\mathbf{x}) - \xi}{\sigma(\mathbf{x})}, \quad
     \operatorname{EI}(\mathbf{x}) = (f^\* - \mu(\mathbf{x}) - \xi)\Phi(Z) + \sigma(\mathbf{x})\phi(Z),

  with :math:`\Phi` and :math:`\phi` the standard normal CDF and PDF. When :math:`\sigma(\mathbf{x})=0`,
  :math:`\operatorname{EI}(\mathbf{x})=\max(f^\*-\mu(\mathbf{x})-\xi,\,0)`.

- *Probability of Improvement (PI):*

  .. math::

     \operatorname{PI}(\mathbf{x}) = \Phi\!\left( \frac{f^\* - \mu(\mathbf{x}) - \xi}{\sigma(\mathbf{x})} \right).

- *Upper Confidence Bound (UCB)* with exploration parameter :math:`\kappa > 0`:

  .. math::

     \operatorname{UCB}(\mathbf{x}) = \mu(\mathbf{x}) - \kappa\,\sigma(\mathbf{x}), \quad
     \text{(Propulate minimizes } -\operatorname{UCB}\text{).}

**3. Acquisition Optimization**

Propulate separates the *acquisition* from its *optimizer*. Any routine that finds a low value of the acquisition
(e.g., random search, multi-start L-BFGS, CMA on the acquisition) can be plugged in.

**4. Parallel & Resource-Aware Fitting**

- *SingleCPUFitter:* scikit-learn GP (normalized :math:`y`, optimizer L-BFGS-B).
- *MultiCPUFitter:* each MPI rank runs a subset of random restarts, the best model (by log marginal likelihood) is
  gathered on rank 0 and broadcast to all ranks. The model is serialized/deserialized to cross MPI boundaries.
- *SingleGPUFitter:* optional GPyTorch-based training on a CUDA device (if available).

**5. Rank Stretching (Diversity Across Ranks)**

To encourage heterogeneous exploration, per-rank scaling is applied linearly:

.. math::

   s_r = \text{factor\_min} + \frac{r}{S-1}\left(\text{factor\_max} - \text{factor\_min}\right),

where rank :math:`r \in \{0,\ldots,S-1\}`. For EI/PI, :math:`\xi \leftarrow s_r \cdot \xi`; for UCB,
:math:`\kappa \leftarrow s_r \cdot \kappa`.

**6. Sparse Subsampling**

When many points are available, Propulate can subsample up to :math:`N_{\max}` points before fitting the GP to keep
training and inference fast.

**7. BO Loop in Propulate**

1. **Cold start:** sample uniformly within user-provided limits.
2. **Fit surrogate:** train GP on :math:`(\mathbf{X}, \mathbf{y})` (possibly subsampled).
3. **Compute current best:** :math:`f^\* = \min(\mathbf{y})`.
4. **Optimize acquisition:** find :math:`\mathbf{x}_{\text{new}} = \arg\min a(\mathbf{x}; \mu, \sigma, f^\*)`.
5. **Evaluate objective** at :math:`\mathbf{x}_{\text{new}}` and add to the population.
6. **Repeat** until the stopping criterion (generations, time, or target value) is met.

Advantages
----------

- **Sample-Efficient:** Finds good solutions with few expensive evaluations.
- **Uncertainty-Aware:** Acquisition explicitly exploits predictive uncertainty.
- **Parallel-Friendly:** Independent per-rank surrogates plus rank stretching provide diverse exploration.
- **Backend-Agnostic Fitting:** Works on single CPU, multi-CPU (MPI), or GPU (GPyTorch) backends.
- **Flexible Acquisitions & Optimizers:** Swap EI/PI/UCB and choose any inner optimizer for the acquisition.
