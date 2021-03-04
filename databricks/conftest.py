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
import os
import shutil
import uuid
import logging
from distutils.version import LooseVersion

import pandas as pd
import pyarrow as pa
from pyspark import __version__

from databricks import koalas as ks
from databricks.koalas import utils


shared_conf = {"spark.sql.shuffle.partitions": "4"}
# Initialize Spark session that should be used in doctests or unittests.
# Delta requires Spark 2.4.2+. See
# https://github.com/delta-io/delta#compatibility-with-apache-spark-versions.
if LooseVersion(__version__) >= LooseVersion("3.0.0"):
    shared_conf["spark.jars.packages"] = "io.delta:delta-core_2.12:0.7.0"
    session = utils.default_session(shared_conf)
elif LooseVersion(__version__) >= LooseVersion("2.4.2"):
    shared_conf["spark.jars.packages"] = "io.delta:delta-core_2.11:0.6.1"
    session = utils.default_session(shared_conf)
else:
    session = utils.default_session(shared_conf)

if os.getenv("DEFAULT_INDEX_TYPE", "") != "":
    ks.options.compute.default_index_type = os.getenv("DEFAULT_INDEX_TYPE")


@pytest.fixture(scope="session", autouse=True)
def session_termination():
    yield
    # Share one session across all the tests. Repeating starting and stopping sessions and contexts
    # seems causing a memory leak for an unknown reason in PySpark.
    session.stop()


@pytest.fixture(autouse=True)
def add_ks(doctest_namespace):
    doctest_namespace["ks"] = ks


@pytest.fixture(autouse=True)
def add_pd(doctest_namespace):
    if os.getenv("PANDAS_VERSION", None) is not None:
        assert pd.__version__ == os.getenv("PANDAS_VERSION")
    doctest_namespace["pd"] = pd


@pytest.fixture(autouse=True)
def add_pa(doctest_namespace):
    if os.getenv("PYARROW_VERSION", None) is not None:
        assert pa.__version__ == os.getenv("PYARROW_VERSION")
    doctest_namespace["pa"] = pa


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace["np"] = numpy


@pytest.fixture(autouse=True)
def add_path(doctest_namespace):
    path = tempfile.mkdtemp()
    atexit.register(lambda: shutil.rmtree(path, ignore_errors=True))
    doctest_namespace["path"] = path


@pytest.fixture(autouse=True)
def add_db(doctest_namespace):
    db_name = "db%s" % str(uuid.uuid4()).replace("-", "")
    session.sql("CREATE DATABASE %s" % db_name)
    atexit.register(lambda: session.sql("DROP DATABASE IF EXISTS %s CASCADE" % db_name))
    doctest_namespace["db"] = db_name


@pytest.fixture(autouse=os.getenv("KOALAS_USAGE_LOGGER", "") != "")
def add_caplog(caplog):
    with caplog.at_level(logging.INFO, logger="databricks.koalas.usage_logger"):
        yield


@pytest.fixture(autouse=True)
def check_options():
    orig_default_index_type = ks.options.compute.default_index_type
    yield
    assert ks.options.compute.default_index_type == orig_default_index_type
