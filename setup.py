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

import sys
from setuptools import setup

DESCRIPTION = "Pandas DataFrame API on Apache Spark"

LONG_DESCRIPTION = """
Koalas makes data scientists more productive when interacting with big data,
by augmenting Apache Spark's Python DataFrame API to be compatible with
Pandas'.

Pandas is the de facto standard (single-node) dataframe implementation in
Python, while Spark is the de facto standard for big data processing.
With this package, data scientists can:

- Be immediately productive with Spark, with no learning curve, if one
  is already familiar with Pandas.
- Have a single codebase that works both with Pandas (tests, smaller datasets)
  and with Spark (distributed datasets).
"""

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
    packages=['databricks', 'databricks.koalas', 'databricks.koalas.missing'],
    extras_require={
        'spark': ['pyspark>=2.4.0'],
    },
    python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*',
    install_requires=[
        'pandas>=0.23',
        'pyarrow>=0.10',
        'numpy>=1.14',
    ],
    maintainer="Databricks",
    maintainer_email="koalas@databricks.com",
    license='http://www.apache.org/licenses/LICENSE-2.0',
    url="https://github.com/databricks/koalas",
    project_urls={
        'Bug Tracker': 'https://github.com/databricks/koalas/issues',
        # 'Documentation': '',
        'Source Code': 'https://github.com/databricks/koalas'
    },
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
)
