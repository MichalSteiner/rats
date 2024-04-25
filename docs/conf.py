# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'RATS'
copyright = '2024, Michal Steiner'
author = 'Michal Steiner'
release = '0.3.1 (alpha)'

import sys, os
sys.path.append('/media/chamaeleontis/Observatory_main/Code/observations_transits/rats')
sys.path.append('/media/chamaeleontis/Observatory_main/Code/observations_transits/OT/OT')

def patch_automodapi(app):
    """Monkey-patch the automodapi extension to exclude imported members"""
    from sphinx_automodapi import automodsumm
    from sphinx_automodapi.utils import find_mod_objs
    automodsumm.find_mod_objs = lambda *args: find_mod_objs(args[0], onlylocals=True)

def setup(app):
    app.connect("builder-inited", patch_automodapi)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
# 'sphinx_automodapi.automodapi'
    'sphinx_automodapi.automodapi',
    'nbsphinx'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store','**.ipynb_checkpoints']

# Intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'IPython': ('https://ipython.readthedocs.io/en/stable/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

numpydoc_show_class_members = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

nbsphinx_allow_errors = True

# html_theme = 'groundwork'
html_theme = 'renku'
html_static_path = ['_static']
