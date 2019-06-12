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
import pytest
import numpy
import tempfile
import atexit
import shutil
import uuid
from distutils.version import LooseVersion

from pyspark import __version__

from databricks import koalas
from databricks.koalas import utils


# Initialize Spark session that should be used in doctests or unittests.
# Delta requires Spark 2.4.2+. See
# https://github.com/delta-io/delta#compatibility-with-apache-spark-versions.
if LooseVersion(__version__) >= LooseVersion("3.0.0"):
    session = utils.default_session({"spark.jars.packages": "io.delta:delta-core_2.12:0.1.0"})
elif LooseVersion(__version__) >= LooseVersion("2.4.2"):
    session = utils.default_session({"spark.jars.packages": "io.delta:delta-core_2.11:0.1.0"})
else:
    session = utils.default_session()


@pytest.fixture(autouse=True)
def add_ks(doctest_namespace):
    doctest_namespace['ks'] = koalas


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace['np'] = numpy


@pytest.fixture(autouse=True)
def add_path(doctest_namespace):
    path = tempfile.mkdtemp()
    atexit.register(lambda: shutil.rmtree(path, ignore_errors=True))
    doctest_namespace['path'] = path


@pytest.fixture(autouse=True)
def add_db(doctest_namespace):
    db_name = str(uuid.uuid4()).replace("-", "")
    session.sql("CREATE DATABASE %s" % db_name)
    atexit.register(lambda: session.sql("DROP DATABASE IF EXISTS %s CASCADE" % db_name))
    doctest_namespace['db'] = db_name
