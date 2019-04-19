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

"""
Utilities to monkey patch PySpark used in databricks-koalas.
"""
from pyspark.sql import session, dataframe as df, column as col, functions as F
import pyspark
from decorator import decorator
import types
import logging

from databricks.koalas.frame import PandasLikeDataFrame
from databricks.koalas.series import PandasLikeSeries
from databricks.koalas.session import SparkSessionPatches
from databricks.koalas import namespace

logger = logging.getLogger('spark')

_TOUCHED_TEST = "_pandas_updated"


def patch_spark():
    """
    This function monkey patches Spark to make PySpark's behavior similar to Pandas.

    See the readme documentation for an exhaustive list of the changes performed by this function.

    Once this function is called, the behavior cannot be reverted.
    """
    # Directly patching the base does not work because DataFrame inherits from object
    # (known python limitation)
    # NormalDF = pyspark.sql.dataframe.DataFrame
    # PatchedDF = type("DataFrame0", (PandasLikeDataFrame, object), dict(NormalDF.__dict__))
    # pyspark.sql.dataframe.DataFrame = PatchedDF
    # pyspark.sql.DataFrame = PatchedDF

    # Just going to update the dictionary
    _inject(df.DataFrame, PandasLikeDataFrame)
    _inject(df.Column, PandasLikeSeries)
    # Override in all cases these methods to prevent any dispatching.
    df.Column.__repr__ = PandasLikeSeries.__repr__
    df.Column.__str__ = PandasLikeSeries.__str__
    # Replace the creation of the operators in columns
    _wrap_operators()
    # Wrap all the functions in the standard libraries
    _wrap_functions()
    # Inject a few useful functions.
    for func in ['from_pandas', 'read_csv', 'read_parquet']:
        setattr(session.SparkSession, func, getattr(SparkSessionPatches, func))
        setattr(pyspark, func, getattr(namespace, func))
    pyspark.to_datetime = namespace.to_datetime
    pyspark.get_dummies = namespace.get_dummies


@decorator
def wrap_column_function(f, *args, **kwargs):
    # Call the function first
    res = f(*args, **kwargs)
    if isinstance(res, col.Column):
        # Need to track where this column is coming from
        all_inputs = list(args) + list(kwargs.values())

        def ref_df(x):
            if isinstance(x, df.DataFrame):
                return x
            if isinstance(x, df.Column):
                if hasattr(x, "_spark_ref_dataframe"):
                    return x._spark_ref_dataframe
                else:
                    logger.debug("Found a column without reference: {}".format(str(x)))
            return None
        all_col_inputs = [ref_df(c) for c in all_inputs]
        all_df_inputs = list(dict([(id(f), f) for f in all_col_inputs if f is not None]).items())
        if len(all_df_inputs) > 1:
            logger.warning("Too many anchors to conclude")
        elif not all_df_inputs:
            logger.debug("Could not find anchors")
        else:
            (_, df_ref) = all_df_inputs[0]
            res._spark_ref_dataframe = df_ref
        return res


def _wrap_operators():
    attrs = ["__neg__", "__add__", "__sub__", "__mul__", "__div__", "__truediv__", "__mod__",
             "__eq__", "__ne__", "__lt__", "__le__", "__ge__", "__gt__", "__and__", "__or__"]
    if hasattr(col.Column, _TOUCHED_TEST):
        return
    for attr in attrs:
        oldfun = getattr(col.Column, attr)
        fun = wrap_column_function(oldfun)
        setattr(col.Column, attr, fun)
    setattr(col.Column, _TOUCHED_TEST, "")


def _wrap_functions():
    all_funs = F.__all__
    if hasattr(F, _TOUCHED_TEST):
        return
    for fname in all_funs:
        if fname in ('pandas_udf',):
            continue
        oldfun = getattr(F, fname)
        if isinstance(oldfun, types.FunctionType):
            fun = wrap_column_function(oldfun)
            setattr(F, fname, fun)
            setattr(F, '_spark_' + fname, oldfun)
    setattr(F, _TOUCHED_TEST, "")


def _inject(target_type, inject_type):
    # Make sure to resolve the base classes too.
    mro = list(inject_type.__mro__)
    mro.reverse()
    # Keep a duplicate of all the existing methods:
    setattr(target_type, "_spark_getattr", target_type.__getattr__)
    setattr(target_type, "_spark_getitem", target_type.__getitem__)
    for (key, fun) in list(target_type.__dict__.items()):
        # Skip the system attributes
        if key.startswith("__") or key.startswith("_spark_"):
            continue
        setattr(target_type, "_spark_" + key, fun)

    # Inject all the methods from the hierarchy:
    setattr(target_type, "__getattr__", inject_type.__getattr__)
    setattr(target_type, "__getitem__", inject_type.__getitem__)
    for attr in ["__iter__", "__len__", "__invert__", "__setitem__", "__dir__"]:
        if hasattr(inject_type, attr):
            setattr(target_type, attr, inject_type.__dict__[attr])
    for t in mro:
        if t == object:
            continue
        for (key, fun) in list(t.__dict__.items()):
            # Skip the system attributes
            if key.startswith("__") or key.startswith("_spark_"):
                continue
            setattr(target_type, key, fun)
