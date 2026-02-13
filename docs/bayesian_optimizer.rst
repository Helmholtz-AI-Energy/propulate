.. _bayesopt:

Bayesian Optimization
=====================

Bayesian Optimization (BO) is a sample-efficient strategy for optimizing black-box, expensive, and potentially noisy
objective functions. It builds a *surrogate model* of the objective (commonly a Gaussian Process) and uses an
*acquisition function* to decide where to evaluate next, explicitly balancing exploration and exploitation [1, 2, 3].
To prevent repeated evaluation of identical candidates by different ranks, Propulate can scale acquisition parameters
across ranks (``rank_stretch``) [4].

[1] *J. Mockus, V. Tiesis, and A. Zilinskas, "The application of Bayesian methods for seeking the extremum", 1978.*

[2] *D. R. Jones, M. Schonlau, and W. J. Welch, "Efficient Global Optimization of Expensive Black-Box Functions",
Journal of Global Optimization, 1998.*

[3] *J. Snoek, H. Larochelle, and R. P. Adams, "Practical Bayesian Optimization of Machine Learning Algorithms",
Advances in Neural Information Processing Systems (NeurIPS), 2012.*

[4] *Häse F, Roch LM, Kreisbeck C, Aspuru-Guzik A. "Phoenics: A Bayesian Optimizer for Chemistry", ACS Cent Sci. 2018.*


Current Support in Propulate
----------------------------

Implemented in the current code base:

- ``BayesianOptimizer`` propagator with a Gaussian Process surrogate via scikit-learn on CPU (``SingleCPUFitter``).
- Acquisition functions ``EI``, ``PI``, and ``UCB`` (API name ``UCB``, implemented in minimization form as
  :math:`\mu - \kappa \sigma`).
- Mixed-type search spaces (float, int, categorical) via continuous relaxation and projection.
- Initial design strategies ``sobol`` (default), ``lhs``, and ``random``.
- Rank stretching for per-rank acquisition-parameter diversity.
- Sparse subsampling for large training sets.

Currently not supported:

- ``MultiCPUFitter``, ``SingleGPUFitter``, and ``MultiGPUFitter`` (these currently raise ``NotImplementedError``).


Basic Concepts
--------------

- **Surrogate Model (Gaussian Process):** BO models the objective :math:`f(\mathbf{x})` with a probabilistic
  surrogate giving a posterior mean :math:`\mu(\mathbf{x})` and standard deviation :math:`\sigma(\mathbf{x})`.
- **Acquisition Function:** A cheap-to-evaluate utility that uses :math:`\mu(\mathbf{x})` and :math:`\sigma(\mathbf{x})`
  to rank candidate points. Maximizing (or minimizing a negated form of) the acquisition chooses the next evaluation.
- **Exploration–Exploitation Trade-off:** The acquisition trades off sampling uncertain regions (exploration) and
  promising regions (exploitation).
- **Per-Rank Surrogates in Propulate:** In parallel runs, each MPI rank fits its own GP on its local data
  and can use different acquisition hyperparameters via rank stretching.

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

- *Upper Confidence Bound (UCB)* with exploration parameter :math:`\kappa > 0`
  (implemented in minimization form):

  .. math::

     \operatorname{LCB}(\mathbf{x}) = \mu(\mathbf{x}) - \kappa\,\sigma(\mathbf{x}).

  ``Propulate`` keeps the legacy name ``UCB`` in the API, but minimizes the
  above lower-confidence expression directly.

**3. Acquisition Optimization**

Propulate separates the *acquisition* from its *optimizer*. Any routine that finds a low value of the acquisition
(e.g., random search, multi-start L-BFGS, CMA on the acquisition) can be plugged in.

**4. Surrogate Fitting Backends (Current Status)**

- *SingleCPUFitter:* scikit-learn GP (normalized :math:`y`, optimizer L-BFGS-B).
- *MultiCPUFitter:* API placeholder; currently not implemented.
- *SingleGPUFitter:* API placeholder; currently not implemented.
- *MultiGPUFitter:* API placeholder; currently not implemented.

**5. Rank Stretching (Diversity Across Ranks)**

To encourage heterogeneous exploration, per-rank scaling is applied linearly:

.. math::

   s_r = \text{factor\_min} + \frac{r}{S-1}\left(\text{factor\_max} - \text{factor\_min}\right),

where rank :math:`r \in \{0,\ldots,S-1\}`. For EI/PI, :math:`\xi \leftarrow s_r \cdot \xi`; for UCB,
:math:`\kappa \leftarrow s_r \cdot \kappa` (applied to the implemented LCB form).

**6. Sparse Subsampling**

When many points are available, Propulate can perform sparse subsampling up to :math:`N_{\max}` points before fitting
the GP to keep training and inference fast.

**7. BO Loop in Propulate**

1. **Initial design:** sample according to ``initial_design`` (default: ``sobol``; alternatives: ``lhs``, ``random``).
2. **Fit surrogate:** train GP on :math:`(\mathbf{X}, \mathbf{y})` (optionally sparse-subsampled).
3. **Compute current best:** :math:`f^\* = \min(\mathbf{y})`.
4. **Optimize acquisition:** find :math:`\mathbf{x}_{\text{new}} = \arg\min a(\mathbf{x}; \mu, \sigma, f^\*)`.
5. **Evaluate objective** at :math:`\mathbf{x}_{\text{new}}` and add to the population.
6. **Repeat** until the stopping criterion (generations, time, or target value) is met.

Implemented Features
--------------------

**1. Mixed-Type Search Spaces**

``BayesianOptimizer`` supports continuous, integer, ordinal-integer, and categorical parameters:

- Floats are optimized continuously inside their bounds.
- Integer ranges are optimized continuously and projected by rounding and clipping.
- Ordinal integer sets are optimized continuously and projected to the nearest allowed value.
- Categoricals are one-hot encoded internally and projected back to a valid one-hot choice.

This allows using BO in heterogeneous HPO/NAS spaces while keeping a single GP/acquisition pipeline.

**2. Sparse Subsampling Behavior**

When ``sparse=True`` and the number of finite observations exceeds ``max_points``, the fitter uses a deterministic
subset:

- Always keep the top ``top_m`` best observations by loss.
- Fill the remaining budget with farthest-point selection for geometric diversity.

For mixed spaces, the diversity score combines normalized Euclidean distance on continuous dimensions and Hamming
distance on categorical blocks.

**3. Exploration Schedule (Epsilon-Greedy)**

Before optimizing the acquisition each generation, BO may perform random exploration with decaying probability

.. math::

   p_t = p_{\text{end}} + \left(p_{\text{start}} - p_{\text{end}}\right)\exp\!\left(-t / \tau\right),

configured via ``p_explore_start``, ``p_explore_end``, and ``p_explore_tau``.

**4. Acquisition Annealing and Switching**

- If ``anneal_acquisition=True``, ``xi`` (EI/PI) or ``kappa`` (UCB) is decayed over time.
- Optional dynamic switching is supported via
  ``second_acquisition_type``, ``acq_switch_generation``, and ``second_acquisition_params``.

**5. Hyperparameter-Optimization Schedule**

To reduce GP fitting overhead, BO does not re-optimize GP hyperparameters every generation:

- Wait until enough samples are available.
- Optimize for the first ``hp_opt_warmup_fits`` eligible fits.
- Afterwards optimize every ``hp_opt_period`` generations.

Minimal Usage
-------------

The example below shows the smallest practical BO setup with ``Propulator``.

.. code-block:: python

   import random
   from mpi4py import MPI

   from propulate import Propulator
   from propulate.propagators import BayesianOptimizer

   def sphere(params):
       return params["x"] ** 2 + params["y"] ** 2

   comm = MPI.COMM_WORLD
   rng = random.Random(42 + comm.rank)  # dedicated optimizer RNG per rank

   limits = {
       "x": (-5.12, 5.12),
       "y": (-5.12, 5.12),
   }

   propagator = BayesianOptimizer(
       limits=limits,
       rank=comm.rank,
       world_size=comm.size,
       acquisition_type="EI",
       acquisition_params={"xi": 0.05},
       n_initial=10,
       initial_design="sobol",
       rng=rng,
   )

   propulator = Propulator(
       loss_fn=sphere,
       propagator=propagator,
       rng=rng,
       island_comm=comm,
       generations=25,
       checkpoint_path="./bo_checkpoint",
   )

   propulator.propulate()
   propulator.summarize(top_n=3)

You can run this script with MPI:

.. code-block:: console

   mpirun --use-hwthread-cpus -n 4 python your_bo_script.py

Tutorial Example Script
-----------------------

Propulate ships an extended BO example that compares BO against PSO and writes comparison plots:
``tutorials/bayesian_optimizer_example.py``.

Quick start from the repository root:

.. code-block:: console

   mpirun --use-hwthread-cpus -n 4 python tutorials/bayesian_optimizer_example.py \
     --function sphere \
     --generations 25 \
     --checkpoint ./bo_runs

Useful BO-specific flags in that script include:
``--bo-acq``, ``--bo-xi``, ``--bo-kappa``, ``--bo-n-initial``, ``--bo-initial-design``, ``--bo-sparse``,
``--bo-max-points``, ``--bo-p-explore-start``, ``--bo-p-explore-end``, ``--bo-p-explore-tau``,
``--bo-second-acq``, and ``--bo-acq-switch-gen``.

Advantages
----------

- **Sample-Efficient:** Finds good solutions with few expensive evaluations.
- **Uncertainty-Aware:** Acquisition explicitly exploits predictive uncertainty.
- **Parallel-Friendly:** Independent per-rank surrogates plus rank stretching provide diverse exploration.
- **CPU-Ready Today:** Fully functional with a single-CPU GP fitter; additional fitter backends are scaffolded.
- **Flexible Acquisitions & Optimizers:** Swap EI/PI/UCB (UCB in implemented LCB form) and choose any inner optimizer
  for the acquisition.
