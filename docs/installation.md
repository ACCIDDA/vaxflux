# Installation

Vaxflux is a Python package and can be installed using pip, conda, or any other
standard package managers. It is compatible with Python 3.10, 3.11, and 3.12.
However, vaxflux is currently in development and not available on PyPI or
conda-forge, so it must be installed directly from GitHub.

## Using `pip`

To install vaxflux using pip, run the following command in your terminal:

```shell
pip install git+https://github.com/ACCIDDA/vaxflux.git
```

## Using `conda`

To install vaxflux using conda, you can create a new environment and install it
from GitHub. Run the following commands in your terminal:

```shell
conda install pip
pip install git+https://github.com/ACCIDDA/vaxflux.git
```

## Using `uv`

To add vaxflux to your uv environment, you can use the following command:

```shell
uv add git+https://github.com/ACCIDDA/vaxflux.git
```

## Optional Dependencies

The vaxflux package also provides two optional dependency groups:

- `plot`: This adds additional dependencies that enable plotting functionality,
  without it you can still use the package to construct and fit models but will
  not be able to plot the results with built in functionality. It's recommended
  for local use that you include this optional dependency group, but it is not
  recommended for production usage (e.g. in an HPC environment).
- `demo`: This adds additional dependencies that are needed for running the
  jupyter notebooks in the `demos/` directory. This optional dependency group is
  generally not needed unless you are learning the package and want to run those
  demos locally yourself.

To install these optional dependency groups you can change the installation urls
above to `git+https://github.com/ACCIDDA/vaxflux.git[plot]` or
`git+https://github.com/ACCIDDA/vaxflux.git[demo]`.
