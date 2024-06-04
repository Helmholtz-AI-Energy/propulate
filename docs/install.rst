.. _installation:

Installation
============

.. note::

   The following commands are tested for UNIX-based operating systems, i.e., MacOS and Linux. Some commands, in particular
   those for creating the virtual environment, differ for Windows machines.

We heavily recommend to install ``Propulate`` |:dna:| in its own virtual environment:

.. code-block:: console

   $ python3 -m venv ./propulate-venv
   $ source ./propulate-venv/bin/activate
   $ pip install --upgrade pip

The **latest stable release** can easily be installed from `PyPI`_ using ``pip``:

.. code-block:: console

    $ pip install propulate

If you need the **latest updates**, you can also install ``Propulate`` |:dna:| directly from the `Github master branch`_ at
your own risk:

.. code-block:: console

    $ pip install https://github.com/Helmholtz-AI-Energy/propulate

If you want to get the **source code and modify** it, you can clone the source code using ``git`` and install ``Propulate``
|:dna:| with ``pip`` in editable mode:

.. code-block:: console

    $ git clone https://github.com/Helmholtz-AI-Energy/propulate
    $ pip install -e .

If you plan to **contribute** to ``Propulate`` |:dna:|, you need to install it along with its developer dependencies:

.. code-block:: console

   $ pip install -e."[dev]"

If you wish to play around with the **tutorials**, you need to install ``Propulate`` |:dna:| along with some additional
dependencies required for running the tutorials:

.. code-block:: console

   $ pip install -e."[tutorials]"

.. note::

   ``Propulate`` |:dna:| uses the message passing interface (MPI) and requires an MPI implementation under the hood.
   Currently, it is only tested with `OpenMPI`_.

You can check whether your installation was successful by importing ``Propulate`` |:dna:| in ``Python``:

.. code-block:: python

   import propulate


.. Links
.. _PyPI: https://pypi.org/project/propulate/
.. _Github master branch: https://github.com/Helmholtz-AI-Energy/propulate
.. _OpenMPI: https://www.open-mpi.org/
