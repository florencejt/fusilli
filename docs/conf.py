# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import warnings
from sphinx_gallery.sorting import FileNameSortKey

sys.path.insert(0, os.path.abspath("../fusilli"))
sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*MPS available but not used.*")
warnings.filterwarnings("ignore", ".*GPU available but not used.*")

warnings.filterwarnings(
    "ignore", message="Checkpoint directory.*exists and is not empty."
)
warnings.filterwarnings("ignore", ".*samples in targets.*")

project = "fusilli"
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
    # "sphinx_rtd_theme",  # for readthedocs theme
    "renku_sphinx_theme",
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

# def include_public_members(app, what, name, obj, skip, options):
#     # Filter public members based on their names
#     return not name.startswith('_')

# def include_private_members(app, what, name, obj, skip, options):
#     # Filter private members based on their names
#     return name.startswith('_')

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "show-inheritance": True,
}
autosummary_generate = True
# autosummary_generate = [
#     "autosummary/public.rst",
#     "autosummary/private.rst",
# ]
# autosummary_toc_tree = {
#     'publicmembers': include_public_members,
#     'privatemembers': include_private_members,
# }
autosummary_imported_members = False
add_module_names = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "sphinx_rtd_theme"
# html_theme = "sphinx_wagtail_theme"
html_theme = "renku"
html_static_path = ["_static"]
html_logo = "_static/pink_pasta_logo.png"
html_favicon = "_static/pink_pasta_logo.png"

from sphinx_gallery.sorting import ExplicitOrder

sphinx_gallery_conf = {
    "doc_module": "fusilli",
    "examples_dirs": ["examples", "how_to_contribute"],  # path to your example scripts
    "gallery_dirs": [
        "auto_examples",
        "contributing_examples",
    ],  # path to where to save gallery generated output
    "ignore_pattern": r"__init__\.py",
    "run_stale_examples": False,
    "subsection_order": ExplicitOrder(
        [
            "examples/customising_behaviour",
            "examples/training_and_testing",
            "examples/model_comparison",
        ],
    ),
    "within_subsection_order": FileNameSortKey,
    'default_thumb_file': '_static/pink_pasta_logo.png',
}


def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip


def setup(app):
    app.add_css_file("florencestheme.css")
    app.connect("autodoc-skip-member", skip)
