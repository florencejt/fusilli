# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath("../fusionlibrary"))
# sys.path.insert(0, os.path.abspath("../../"))
# sys.path.insert(0, os.path.abspath(".."))
# sys.path.insert(0, os.path.abspath("."))

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*MPS available but not used.*")
warnings.filterwarnings(
    "ignore", message="Checkpoint directory.*exists and is not empty."
)


project = "fusionlibrary"
copyright = "2023, Florence J Townend"
author = "Florence J Townend"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    # "sphinx.ext.intersphinx",  # for linking to other packages
    "sphinx.ext.mathjax",  # for mathjax
    "sphinx.ext.napoleon",  # for google style docstrings
    "sphinx.ext.viewcode",  # for linking to source code
    "sphinx_gallery.gen_gallery",  # for gallery
    "sphinx_rtd_theme",  # for readthedocs theme
]


# intersphinx_mapping = {
#     "numpy": ("https://docs.scipy.org/doc/numpy", None),
#     "python": ("https://docs.python.org/3", None),
#     "sklearn": ("http://scikit-learn.org/dev", None),
#     "torch": ("https://pytorch.org/docs/master", None),
#     "pytorch_lightning": (
#         "https://pytorch-lightning.readthedocs.io/en/stable/index.html#",
#         None,
#     ),
#     "jax": ("https://jax.readthedocs.io/en/latest/", None),
#     "numpyro": ("https://numpyro.readthedocs.io/en/latest/", None),
#     "jaxlib": ("https://jax.readthedocs.io/en/latest/", None),
# }


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autosummary_generate = True
autosummary_imported_members = False
add_module_names = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "pink_pasta_logo.png"
html_favicon = "pink_pasta_logo.png"

sphinx_gallery_conf = {
    "doc_module": "fusionlibrary",
    "examples_dirs": "examples",  # path to your example scripts
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    "ignore_pattern": "__init__.py",
    "run_stale_examples": False,
}


def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip


def setup(app):
    app.add_css_file("florencestheme.css")
    app.connect("autodoc-skip-member", skip)
