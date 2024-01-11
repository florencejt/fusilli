.. _install_instructions:


How to Install
==============

Installation via PyPI
---------------------

``fusilli`` is available on PyPI and can be installed with pip:

.. code-block:: bash

    pip install fusilli

Setting Up a Virtual Environment
--------------------------------

It's recommended to use a virtual environment to install ``fusilli``, allowing you to isolate it from your system environment.

1. Install the package for creating virtual environments:

.. code-block:: bash

    pip install virtualenv

2. Create a directory for the virtual environment and navigate into it:

.. code-block:: bash

    mkdir fusilli_experiment_dir
    cd fusilli_experiment_dir

3. Create a virtual environment and install ``fusilli`` with its dependencies:

.. code-block:: bash

    python3.9 -m venv fusilli-env
    source fusilli-env/bin/activate
    pip install fusilli

