UV_PROJECT_ENVIRONMENT ?= .venv
RM                     := rm -f
RMDIR                  := rm -rf

.PHONY: all check ci clean clean-docs docs format lint mypy pytest rstcheck serve

all: clean .venv lint mypy pytest docs

clean-docs:
	$(RMDIR) docs/__pycache__
	$(RMDIR) docs/api
	$(RMDIR) docs/_build
	$(RM) docs/changes.rst

clean:
	$(RMDIR) .mypy_cache
	$(RMDIR) .pytest_cache
	$(RMDIR) .ruff_cache
	$(RMDIR) .venv
	@$(MAKE) clean-docs
	$(RMDIR) src/vaxflux/__pycache__
	$(RMDIR) src/vaxflux.egg-info
	$(RM) uv.lock

.venv:
	uv sync --all-extras
	uv pip install --editable .

rstcheck: .venv
	$(UV_PROJECT_ENVIRONMENT)/bin/rstcheck --warn-unknown-settings --recursive docs/

docs/_build: .venv
	$(UV_PROJECT_ENVIRONMENT)/bin/sphinx-apidoc \
		--force \
		--remove-old \
		--separate \
		--doc-project 'API Reference' \
		--output-dir docs/api \
		src/vaxflux
	$(UV_PROJECT_ENVIRONMENT)/bin/python misc/edit_api_docs.py
	$(UV_PROJECT_ENVIRONMENT)/bin/sphinx-build -b html docs docs/_build

docs/changes.rst: .venv
	pandoc --from=markdown --to=rst --output=docs/changes.rst CHANGES.md

docs:
	@$(MAKE) clean-docs
	@$(MAKE) rstcheck
	@$(MAKE) docs/changes.rst
	@$(MAKE) docs/_build

serve: .venv docs
	$(UV_PROJECT_ENVIRONMENT)/bin/python -m http.server --directory docs/_build

format: .venv
	$(UV_PROJECT_ENVIRONMENT)/bin/ruff format

check: .venv
	$(UV_PROJECT_ENVIRONMENT)/bin/ruff check --fix --unsafe-fixes

lint: format check

mypy: .venv
	$(UV_PROJECT_ENVIRONMENT)/bin/mypy --strict .

pytest: .venv
	$(UV_PROJECT_ENVIRONMENT)/bin/pytest --doctest-modules

ci: .venv
	$(UV_PROJECT_ENVIRONMENT)/bin/ruff format --check
	$(UV_PROJECT_ENVIRONMENT)/bin/ruff check --no-fix
	$(UV_PROJECT_ENVIRONMENT)/bin/mypy --strict .
	$(UV_PROJECT_ENVIRONMENT)/bin/pytest --doctest-modules --exitfirst
