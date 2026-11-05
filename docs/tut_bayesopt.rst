.. _tut_bayesopt:

Bayesian Optimizer Tutorial
===========================

This page links the practical BO usage material in ``Propulate``:

- Theory and implementation details: :ref:`Bayesian Optimization page <bayesopt>`
- Runnable comparison script: ``tutorials/bayesian_optimizer_example.py``

Quick run with default EI acquisition from repository root:

.. code-block:: console

   mpirun --use-hwthread-cpus -n 4 python tutorials/bayesian_optimizer_example.py \
     --function sphere \
     --generations 25 \
     --checkpoint ./bo_runs

Run with Thompson Sampling instead (no hyperparameters to tune; diversity is automatic):

.. code-block:: console

   mpirun --use-hwthread-cpus -n 4 python tutorials/bayesian_optimizer_example.py \
     --function sphere \
     --generations 25 \
     --bo-acq TS \
     --checkpoint ./bo_runs_ts

For minimal custom setups with ``BayesianOptimizer`` + ``Propulator``, including
Thompson Sampling and mixed-type examples, see :ref:`the BO documentation page <bayesopt>`.
