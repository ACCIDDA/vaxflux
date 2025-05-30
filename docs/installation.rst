Installation
============

Vaxflux is a Python package and can be installed using pip, conda, or any other standard package managers. It is compatible with Python 3.10, 3.11, and 3.12. However, vaxflux is currently in development and not available on PyPI or conda-forge, so it must be installed directly from GitHub.

Using `pip`
-----------

To install vaxflux using pip, run the following command in your terminal:

.. code-block:: shell

    pip install git+git://github.com/ACCIDDA/vaxflux.git

Alternatively, if you would prefer to install the package of https you can use:

.. code-block:: shell

    pip install git+https://github.com/ACCIDDA/vaxflux.git

Using `conda`
-------------

To install vaxflux using conda, you can create a new environment and install it from GitHub. Run the following commands in your terminal:

.. code-block:: shell

    conda install pip
    pip install git+https://github.com/ACCIDDA/vaxflux.git

Using `uv`
----------

To add vaxflux to your uv environment, you can use the following command:

.. code-block:: shell

    uv add vaxflux git+https://github.com/ACCIDDA/vaxflux.git
