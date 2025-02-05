# `vaxflux` - Seasonal Vaccine Uptake Modeling

`vaxflux` is a python package for constructing flexible vaccine uptake models.


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

* `uv run ruff format` to auto-format python and jupyter notebook files,
* `uv run ruff check` to lint python and jupyter notebook files,
* `uv run mypy .` to type check the package, and
* `uv run pytest` to run the unit tests.

Or (on MacOS/Linux) you can use the `./bin/lint` utility script which will run all of these commands in one go. There is also a GitHub action that will run these same set of checks for pull requests against `main`.
