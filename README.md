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

# Documentation

`vaxflux`'s documentation is not hosted yet, but can be viewed locally by installing from source (see above). Then in your clone of `vaxflux` you can run the following commands:

```bash
make docs
make serve
```

These commands will:

1. Build the documentation, namely the API reference which is not included in the git repository, and
2. Launch python's built-in webserver to view the documentation in your browser.

The documentation will be available for viewing at `http://localhost:8000/`. The port might vary, please consult `make serve` output to confirm.

# Contributing

To contribute install the package from source (as described above). `vaxflux` uses several tools to maintain code quality which can be run as follows:

* `ruff`: To auto-format and lint python and jupyter notebook files,
* `mypy`: to type check the package, and
* `pytest` to run the unit tests.

The source for this package also includes a `Makefile` with these commands built in to make it easier for contributors to quality check their code before opening a PR. Importantly:

* `make` or `make all`: Will create the virtual environment if need be and run the tools above, and
* `make ci`: Will replicate the steps that the GitHub CI workflow will do to make it easier for contributors to debug CI failures locally.

The `Makefile` has only been tested with MacOS/Linux so it may not work as expected on Windows.

# Funding Acknowledgement

This project was made possible by cooperative agreement CDC-RFA-FT-23-0069 from the CDC's Center for Forecasting and Outbreak Analytics. Its contents are solely the responsibility of the authors and do not necessarily represent the official views of the Centers for Disease Control and Prevention.
