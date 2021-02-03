#!/usr/bin/env python

#
# Copyright (C) 2019 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import print_function

from io import open
import sys
from setuptools import setup
from os import path

DESCRIPTION = "Koalas: pandas API on Apache Spark"

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

try:
    exec(open('databricks/koalas/version.py').read())
except IOError:
    print("Failed to load Koalas version file for packaging. You must be in Koalas root dir.",
          file=sys.stderr)
    sys.exit(-1)
VERSION = __version__  # noqa

setup(
    name='koalas',
    version=VERSION,
    packages=[
        'databricks',
        'databricks.koalas',
        'databricks.koalas.indexes',
        'databricks.koalas.missing',
        'databricks.koalas.plot',
        'databricks.koalas.spark',
        'databricks.koalas.typedef',
        'databricks.koalas.usage_logging'],
    extras_require={
        'spark': ['pyspark>=2.4.0'],
        'mlflow': ['mlflow>=1.0'],
        'plotly': ['plotly>=4.8'],
        'matplotlib': ['matplotlib>=3.0.0,<3.3.0'],
    },
    python_requires='>=3.5,<3.9',
    install_requires=[
        'pandas>=0.23.2,<1.2.0',
        'pyarrow>=0.10',
        'numpy>=1.14,<1.20.0',
    ],
    author="Databricks",
    author_email="koalas@databricks.com",
    license='http://www.apache.org/licenses/LICENSE-2.0',
    url="https://github.com/databricks/koalas",
    project_urls={
        'Bug Tracker': 'https://github.com/databricks/koalas/issues',
        'Documentation': 'https://koalas.readthedocs.io/',
        'Source Code': 'https://github.com/databricks/koalas'
    },
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
