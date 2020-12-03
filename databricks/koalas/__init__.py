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
import os
import sys
from distutils.version import LooseVersion

from databricks.koalas.version import __version__  # noqa: F401


def assert_pyspark_version():
    import logging

    pyspark_ver = None
    try:
        import pyspark
    except ImportError:
        raise ImportError(
            "Unable to import pyspark - consider doing a pip install with [spark] "
            "extra to install pyspark with pip"
        )
    else:
        pyspark_ver = getattr(pyspark, "__version__")
        if pyspark_ver is None or pyspark_ver < "2.4":
            logging.warning(
                'Found pyspark version "{}" installed. pyspark>=2.4.0 is recommended.'.format(
                    pyspark_ver if pyspark_ver is not None else "<unknown version>"
                )
            )


assert_pyspark_version()

import pyspark
import pyarrow

if LooseVersion(pyspark.__version__) < LooseVersion("3.0"):
    if (
        LooseVersion(pyarrow.__version__) >= LooseVersion("0.15")
        and "ARROW_PRE_0_15_IPC_FORMAT" not in os.environ
    ):
        import logging

        logging.warning(
            "'ARROW_PRE_0_15_IPC_FORMAT' environment variable was not set. It is required to "
            "set this environment variable to '1' in both driver and executor sides if you use "
            "pyarrow>=0.15 and pyspark<3.0. "
            "Koalas will set it for you but it does not work if there is a Spark context already "
            "launched."
        )
        # This is required to support PyArrow 0.15 in PySpark versions lower than 3.0.
        # See SPARK-29367.
        os.environ["ARROW_PRE_0_15_IPC_FORMAT"] = "1"
elif "ARROW_PRE_0_15_IPC_FORMAT" in os.environ:
    raise RuntimeError(
        "Please explicitly unset 'ARROW_PRE_0_15_IPC_FORMAT' environment variable in both "
        "driver and executor sides. It is required to set this environment variable only "
        "when you use pyarrow>=0.15 and pyspark<3.0."
    )

if (
    LooseVersion(pyarrow.__version__) >= LooseVersion("2.0.0")
    and "PYARROW_IGNORE_TIMEZONE" not in os.environ
):
    import logging

    logging.warning(
        "'PYARROW_IGNORE_TIMEZONE' environment variable was not set. It is required to "
        "set this environment variable to '1' in both driver and executor sides if you use "
        "pyarrow>=2.0.0. "
        "Koalas will set it for you but it does not work if there is a Spark context already "
        "launched."
    )
    os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

from databricks.koalas.frame import DataFrame
from databricks.koalas.indexes import Index, MultiIndex
from databricks.koalas.series import Series
from databricks.koalas.groupby import NamedAgg

__all__ = [  # noqa: F405
    "read_csv",
    "read_parquet",
    "to_datetime",
    "from_pandas",
    "get_dummies",
    "DataFrame",
    "Series",
    "Index",
    "MultiIndex",
    "sql",
    "range",
    "concat",
    "melt",
    "get_option",
    "set_option",
    "reset_option",
    "read_sql_table",
    "read_sql_query",
    "read_sql",
    "options",
    "option_context",
    "NamedAgg",
]


def _auto_patch_spark():
    import os
    import logging

    # Attach a usage logger.
    logger_module = os.getenv("KOALAS_USAGE_LOGGER", "")
    if logger_module != "":
        try:
            from databricks.koalas import usage_logging

            usage_logging.attach(logger_module)
        except Exception as e:
            logger = logging.getLogger("databricks.koalas.usage_logger")
            logger.warning(
                "Tried to attach usage logger `{}`, but an exception was raised: {}".format(
                    logger_module, str(e)
                )
            )

    # Autopatching is on by default.
    x = os.getenv("SPARK_KOALAS_AUTOPATCH", "true")
    if x.lower() in ("true", "1", "enabled"):
        logger = logging.getLogger("spark")
        logger.info(
            "Patching spark automatically. You can disable it by setting "
            "SPARK_KOALAS_AUTOPATCH=false in your environment"
        )

        from pyspark.sql import dataframe as df

        df.DataFrame.to_koalas = DataFrame.to_koalas


def _auto_patch_pandas():
    import pandas as pd

    # In order to use it in test cases.
    global _frame_has_class_getitem
    global _series_has_class_getitem

    _frame_has_class_getitem = hasattr(pd.DataFrame, "__class_getitem__")
    _series_has_class_getitem = hasattr(pd.Series, "__class_getitem__")

    if sys.version_info >= (3, 7):
        # Just in case pandas implements '__class_getitem__' later.
        if not _frame_has_class_getitem:
            pd.DataFrame.__class_getitem__ = lambda params: DataFrame.__class_getitem__(params)

        if not _series_has_class_getitem:
            pd.Series.__class_getitem__ = lambda params: Series.__class_getitem__(params)


_auto_patch_spark()
_auto_patch_pandas()

# Import after the usage logger is attached.
from databricks.koalas.config import get_option, options, option_context, reset_option, set_option
from databricks.koalas.namespace import *  # F405
from databricks.koalas.sql import sql
