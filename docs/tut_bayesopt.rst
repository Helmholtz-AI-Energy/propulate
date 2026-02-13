.. _tut_bayesopt:

Bayesian Optimizer Tutorial
===========================

This page links the practical BO usage material in ``Propulate``:

- Theory and implementation details: :ref:`Bayesian Optimization page <bayesopt>`
- Runnable comparison script: ``tutorials/bayesian_optimizer_example.py``

Quick run from repository root:

.. code-block:: console

   mpirun --use-hwthread-cpus -n 4 python tutorials/bayesian_optimizer_example.py \
     --function sphere \
     --generations 25 \
     --checkpoint ./bo_runs

For a minimal custom setup with ``BayesianOptimizer`` + ``Propulator``, see
the ``Minimal Usage`` section on :ref:`the BO documentation page <bayesopt>`.
