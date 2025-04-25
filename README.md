# `vaxflux` - Seasonal Vaccine Uptake Modeling

`vaxflux` is a python package for constructing flexible seasonal vaccine uptake models.

## Getting Started

`vaxflux` is still a WIP so it is recommended to either install from github:

```bash
uv pip install "git+https://github.com/ACCIDDA/vaxflux"
```

or from source:

```bash
git clone git@github.com:ACCIDDA/vaxflux.git
cd vaxflux
uv sync
uv pip install --editable .
```

# Contributing

To contribute install the package from source (as described above). `vaxflux` uses several tools to maintain code quality which can be run as follows:

* `ruff`: To auto-format and lint python and jupyter notebook files,
* `mypy`: to type check the package, and
* `pytest` to run the unit tests.

The source for this package also includes a `Makefile` with these commands built in to make it easier for contributors to quality check their code before opening a PR. Importantly:

* `make` or `make all`: Will create the virtual environment if need be and run the tools above, and
* `make ci`: Will replicate the steps that the GitHub CI workflow will do to make it easier for contributors to debug CI failures locally.

The `Makefile` has only been tested with MacOS/Linux so it may not work as expected on Windows.
