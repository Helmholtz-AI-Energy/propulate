.. _installation:

Installation
============

We heavily recommend to install ``Propulate`` in its own virtual environment:

.. code-block:: console

   $ python3 -m venv ./propulate
   $ source ./propulate/bin/activate
   $ pip install --upgrade pip

The latest stable release can easily be installed from `PyPI`_ using ``pip``:

.. code-block:: console

    $ pip install propulate

If you need the latest updates, you can also install ``Propulate`` directly from the `Github master branch`_ at you own risk:

.. code-block:: console

    $ pip install https://github.com/Helmholtz-AI-Energy/propulate

If you want to get the source code and modify it, you can clone the source code using ``git`` and install ``Propulate``
with ``pip`` or via ``setup.py``:

.. code-block:: console

    $ git clone https://github.com/Helmholtz-AI-Energy/propulate
    $ pip install -e .
   
Alternatively:

.. code-block:: console

   $ python setup.py develop

.. note::

   ``Propulate`` uses the message passing interface (MPI) and requires an MPI implementation under the hood.
   Currently, it is only tested with `OpenMPI`_.

You can check whether your installation was successful by importing ``Propulate`` in ``Python``:

.. code-block:: python

   import propulate


.. Links
.. _PyPI: https://pypi.org/project/propulate/
.. _Github master branch: https://github.com/Helmholtz-AI-Energy/propulate
.. _OpenMPI: https://www.open-mpi.org/
