"""Sphinx configuration for SilverSpoon-KD documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import silverspoon_kd  # E402 is allowed here: the import depends on the sys.path insert above

project = "SilverSpoon-KD"
copyright = "2026, Xaver R. Davey"
author = "Xaver R. Davey"
# Single source of truth for the version is ``silverspoon_kd.__version__``.
release = silverspoon_kd.__version__

# -- Extensions ---------------------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinx_copybutton",
]

# -- General ------------------------------------------------------------------

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
exclude_patterns = [
    "_build",
    ".DS_Store",
]

# -- MyST (Markdown) ---------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "fieldlist",
    "deflist",
]
myst_heading_anchors = 3

# -- Autodoc ------------------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "special-members": "__init__",
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "signature"
autodoc_typehints_format = "short"
autodoc_class_content = "class"

# -- Napoleon (Google-style docstrings) ---------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True

# -- Intersphinx (cross-project links) ---------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "transformers": ("https://huggingface.co/docs/transformers/main/en/", None),
}

# -- MathJax ------------------------------------------------------------------

mathjax3_config = {
    "tex": {
        "inlineMath": [["\\(", "\\)"], ["$", "$"]],
        "displayMath": [["\\[", "\\]"], ["$$", "$$"]],
        "processEscapes": True,
        "processEnvironments": True,
    },
}

# -- HTML output (Furo) -------------------------------------------------------

# Canonical URL of the hosted documentation (GitHub Pages behind a custom domain).
html_baseurl = "https://kd.silverspoon.dev/"

html_theme = "furo"
html_title = "SilverSpoon-KD"
html_favicon = "_static/favicon.png"
html_logo = "_static/favicon.png"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#3f51b5",
        "color-brand-content": "#3f51b5",
    },
    "dark_css_variables": {
        "color-brand-primary": "#7986cb",
        "color-brand-content": "#7986cb",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]
