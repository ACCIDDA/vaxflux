# Installation

Vaxflux is a Python package and can be installed using pip, conda, or any other
standard package managers. It is compatible with Python 3.11 and 3.12.

## Install From PyPI

For most users, the recommended installation method is from PyPI:

## Using `pip`

To install vaxflux using pip, run the following command in your terminal:

```shell
pip install vaxflux
```

## Using `uv`

To add vaxflux to your uv environment, you can use the following command:

```shell
uv add vaxflux
```

## Install The Development Version From GitHub

If you install directly from GitHub, you will get the development version of
`vaxflux` rather than the latest PyPI release.

```shell
pip install git+https://github.com/ACCIDDA/vaxflux.git
```

For the development version, refer to the
[development documentation](https://accidda.github.io/vaxflux/dev/).

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

To install these optional dependency groups from PyPI:

```shell
pip install "vaxflux[plot]"
pip install "vaxflux[demo]"
```

To install them from the GitHub development version:

```shell
pip install "vaxflux[plot] @ git+https://github.com/ACCIDDA/vaxflux.git"
pip install "vaxflux[demo] @ git+https://github.com/ACCIDDA/vaxflux.git"
```
