.. _tut_multi_rank_worker:

Using Propulate with Multi-Rank Workers
=======================================
.. note::

   You can find the corresponding ``Python`` script here:
   https://github.com/Helmholtz-AI-Energy/propulate/blob/master/tutorials/multi_rank_workers_example.py

In addition to the already explained functionality, ``Propulate`` enables using multi-rank workers for an internally
parallelized evaluation of the loss function. This is useful for, e.g., data-parallel training of neural networks during
the hyperparameter optimization, where each individual network is trained on multiple GPUs.

A more detailed explanation of the tutorial will be available soon |:rocket:|.
