.. _island_model::

The Island Model
================

On top of its basic asynchronous population-based optimizer, ``Propulate`` implements an *asynchronous island model* |:island:|.
The island model is a common parallelization scheme for evolutionary algorithms. It combines independent evolution of
self-contained subpopulations (or islands) with intermittent exchange of selected individuals (migration).
To coordinate the search globally, each  island occasionally delegates migrants to be included in the target islands'
populations. With worse performing islands typically receiving candidates from better performing ones, islands
communicate genetic information competitively, thus increasing diversity among the subpopulations.

What this basically means is, that we do not only consider one population of individuals but multiple independent
populations. We call each of these populations an island. Those islands co-exist peacefully most of the time. But from
time to time, individuals migrate from one island to another. In this way, we can explore the
search space more comprehensively and prevent local trapping.

|im|  |im|

.. |im| image:: images/async_prop_im.png
   :width: 49 %

|

Propulate - Asynchronous Migration Between Islands
--------------------------------------------------
.. image:: images/async_migration_hgf.png
   :width: 100 %
   :align: center

|