"""Sphinx documentation configuration file."""

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = "vaxflux"
copyright = "2025, ACCIDDA"
author = "ACCIDDA"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    "sphinx.ext.apidoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "use_edit_page_button": True,
}
html_context = {
    "github_user": "ACCIDDA",
    "github_repo": "vaxflux",
    "github_version": "main",
    "doc_path": "docs",
}
html_sidebars: dict[str, list[str]] = {
    "installation": [],
    "introduction": [],
    "getting-started": [],
    "model-details": [],
}

# -- Options for autodoc -----------------------------------------------------
autoclass_content = "both"
autodoc_member_order = "bysource"
