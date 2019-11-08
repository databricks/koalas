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

if sys.version < '3':
    raise ImportError('Koalas does not support Python 2.')


from databricks.koalas.version import __version__


def assert_pyspark_version():
    import logging
    pyspark_ver = None
    try:
        import pyspark
    except ImportError:
        raise ImportError('Unable to import pyspark - consider doing a pip install with [spark] '
                          'extra to install pyspark with pip')
    else:
        pyspark_ver = getattr(pyspark, '__version__')
        if pyspark_ver is None or pyspark_ver < '2.4':
            logging.warning(
                'Found pyspark version "{}" installed. pyspark>=2.4.0 is recommended.'
                .format(pyspark_ver if pyspark_ver is not None else '<unknown version>'))


assert_pyspark_version()

from databricks.koalas.namespace import *
from databricks.koalas.frame import DataFrame
from databricks.koalas.series import Series
from databricks.koalas.typedef import Col, pandas_wraps

__all__ = ['read_csv', 'read_parquet', 'to_datetime', 'from_pandas',
           'get_dummies', 'DataFrame', 'Series', 'Col', 'pandas_wraps']


def _auto_patch():
    import os
    import logging
    # Autopatching is on by default.
    x = os.getenv("SPARK_KOALAS_AUTOPATCH", "true")
    if x.lower() in ("true", "1", "enabled"):
        logger = logging.getLogger('spark')
        logger.info("Patching spark automatically. You can disable it by setting "
                    "SPARK_KOALAS_AUTOPATCH=false in your environment")

    from pyspark.sql import dataframe as df
    df.DataFrame.to_koalas = DataFrame.to_koalas


_auto_patch()
