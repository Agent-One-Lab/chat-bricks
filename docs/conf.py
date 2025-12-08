import os
import sys

# Ensure the project package is importable when building docs
sys.path.insert(0, os.path.abspath(".."))

project = "chat-bricks"
author = "chat-bricks team"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "fieldlist",
    "linkify",
]

html_theme = "alabaster"

