.. _usage:

Tutorials
=========

The tutorials below guide you through the exemplary usage scripts on our `Github`_ page step-by-step.
If you want to use ``Propulate``, you can use these scripts as templates for your own applications.

Simple evolutionary optimization of mathematical functions
----------------------------------------------------------
Find the corresponding ``Python`` script here:
https://github.com/Helmholtz-AI-Energy/propulate/blob/master/scripts/propulator_example.py

To show you how ``Propulate`` works, we use it to minimize two-dimensional mathematical functions.
Let us consider the sphere function:

.. math::
    f_\mathrm{sphere}\left(x,y\right)=x^2+y^2

As an evolutionary algorithm, ``Propulate``'s basic optimization mechanism is that of Darwinian evolution, i.e.,
beneficial traits are selected, recombined, and mutated to breed more fit individuals.


Multi-island evolutionary optimization of mathematical functions
----------------------------------------------------------------
Find the corresponding ``Python`` script here: https://github.com/Helmholtz-AI-Energy/propulate/blob/master/scripts/islands_example.py

Next, we want to minimize the sphere function using ``Propulate``'s asynchronous island model.

On a higher level, Propulate employs an IM, which combines independent evolution of self-contained subpopulations with intermittent exchange of selected individuals.
To coordinate the search globally, each island occasionally delegates migrants to be included in the target islands' populations.
With worse performing islands typically receiving candidates from better performing ones, islands communicate genetic information competitively, thus increasing diversity among the subpopulations.
not only one population of individuals but multiple independent populations.

Hyperparameter optimization of a neural network
-----------------------------------------------
Find the corresponding ``Python`` script here: https://github.com/Helmholtz-AI-Energy/propulate/blob/master/scripts/torch_example.py


.. Links
.. _Github: https://github.com/Helmholtz-AI-Energy/propulate
