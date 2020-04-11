# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
import shutil
import importlib.util
import errno

from databricks import koalas
from databricks.koalas import utils
sys.path.insert(0, os.path.abspath('.'))

# Remove previously generated rst files. Ignore errors just in case it stops generating whole docs.
shutil.rmtree("%s/reference/api" % os.path.dirname(os.path.abspath(__file__)), ignore_errors=True)
try:
    os.mkdir("%s/reference/api" % os.path.dirname(os.path.abspath(__file__)))
except OSError as e:
    if e.errno != errno.EEXIST:
        raise


if "READTHEDOCS" in os.environ:
    # For Read the Docs, it has JDK 11 by default. Spark 2.x only supports JDK 8 at most.
    # Once Spark 3.0 is released, we can remove this as Spark 3.0 supports JDK 11.
    # For now, we should manually install JDK 8.
    import jdk
    if not os.path.isdir(jdk._JRE_DIR):
        jdk.install("8", jre=True)
    java_home = "%s/%s" % (jdk._JRE_DIR, os.listdir(jdk._JRE_DIR)[0])
    os.environ["JAVA_HOME"] = java_home
    os.environ["PATH"] = "%s/bin:%s" % (java_home, os.environ["PATH"])


# Lower the number of partitions to speed up documentation build
utils.default_session({"spark.sql.shuffle.partitions": "4"})


def gendoc():
    """Get releases from Github and generate reStructuredText files for release notes."""
    dev_dir = "%s/../../dev" % os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("gendoc", "%s/gendoc.py" % dev_dir)
    gendoc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gendoc)
    gendoc.download_pandoc_if_needed(dev_dir)
    source_dir = os.path.dirname(os.path.abspath(__file__))
    gendoc.gen_release_notes(source_dir)


gendoc()


# -- Project information -----------------------------------------------------

project = 'Koalas'
copyright = '2020, Databricks'
author = 'The Koalas Team'

# The full version, including alpha/beta/rc tags
release = os.environ.get('RELEASE_VERSION', koalas.__version__)


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '1.2'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'numpydoc',  # handle NumPy documentation formatted docstrings. Needs to install
    'matplotlib.sphinxext.plot_directive',  # For visualize plot result
    'nbsphinx',  # Converts Jupyter Notebook to reStructuredText files for Sphinx.
    # For ipython directive in reStructuredText files.
    'IPython.sphinxext.ipython_console_highlighting',
]

# matplotlib plot directive
plot_include_source = True
plot_formats = [("png", 90)]
plot_html_show_formats = False
plot_html_show_source_link = False
plot_pre_code = """import numpy as np
import pandas as pd
import databricks.koalas as ks"""

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# The master toctree document.
master_doc = 'index'

numpydoc_show_class_members = False

# -- Options for auto output -------------------------------------------------

autoclass_content = 'both'
autosummary_generate = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "../../icons/koalas-logo-docs.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'css/koalas.css',
]

# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = ['themes']

# If false, no index is generated.
html_use_index = False

# If false, no module index is generated.
html_domain_indices = False

# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = '../../icons/koalas-favicon.ico'


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'databricks.koalas', u'databricks.koalas Documentation',
     [u'Author'], 1)
]
