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
Commonly used utils in Koalas.
"""

import functools
from collections import OrderedDict
import os
from typing import Callable, Dict, List, Tuple, Union

from pyspark import sql as spark
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
import pandas as pd

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.


def combine_frames(this, *args, how="full"):
    """
    This method combines `this` DataFrame with a different `that` DataFrame or
    Series from a different DataFrame.

    It returns a DataFrame that has prefix `this_` and `that_` to distinct
    the columns names from both DataFrames

    It internally performs a join operation which can be expensive in general.
    So, if `OPS_ON_DIFF_FRAMES` environment variable is not set,
    this method throws an exception.
    """
    from databricks.koalas import Series
    from databricks.koalas import DataFrame
    from databricks.koalas.config import get_option

    if all(isinstance(arg, Series) for arg in args):
        assert all(arg._kdf is args[0]._kdf for arg in args), \
            "Currently only one different DataFrame (from given Series) is supported"
        if this is args[0]._kdf:
            return  # We don't need to combine. All series is in this.
        that = args[0]._kdf[[ser.name for ser in args]]
    elif len(args) == 1 and isinstance(args[0], DataFrame):
        assert isinstance(args[0], DataFrame)
        if this is args[0]:
            return  # We don't need to combine. `this` and `that` are same.
        that = args[0]
    else:
        raise AssertionError("args should be single DataFrame or "
                             "single/multiple Series")

    if get_option("compute.ops_on_diff_frames"):
        this_index_map = this._internal.index_map
        this_data_columns = this._internal.data_columns
        that_index_map = that._internal.index_map
        that_data_columns = that._internal.data_columns
        assert len(this_index_map) == len(that_index_map)

        join_scols = []
        merged_index_scols = []

        # If the same named index is found, that's used.
        for this_column, this_name in this_index_map:
            for that_col, that_name in that_index_map:
                if this_name == that_name:
                    # We should merge the Spark columns into one
                    # to mimic pandas' behavior.
                    this_scol = this._internal.scol_for(this_column)
                    that_scol = that._internal.scol_for(that_col)
                    join_scol = this_scol == that_scol
                    join_scols.append(join_scol)
                    merged_index_scols.append(
                        F.when(
                            this_scol.isNotNull(), this_scol
                        ).otherwise(that_scol).alias(this_column))
                    break
            else:
                raise ValueError("Index names must be exactly matched currently.")

        assert len(join_scols) > 0, "cannot join with no overlapping index names"

        index_columns = this._internal.index_columns
        joined_df = this._sdf.alias("this").join(
            that._sdf.alias("that"), on=join_scols, how=how)

        joined_df = joined_df.select(
            merged_index_scols +
            [this[c]._scol.alias("__this_%s" % this[c].name) for c in this_data_columns] +
            [that[c]._scol.alias("__that_%s" % that[c].name) for c in that_data_columns])

        new_data_columns = [c for c in joined_df.columns if c not in index_columns]
        return DataFrame(
            this._internal.copy(sdf=joined_df, data_columns=new_data_columns))
    else:
        raise ValueError("Cannot combine column argument because "
                         "it comes from a different dataframe")


def align_diff_frames(resolve_func, this, that, fillna=True, how="full"):
    """
    This method aligns two different DataFrames with a given `func`. Columns are resolved and
    handled within the given `func`.
    To use this, `OPS_ON_DIFF_FRAMES` environment variable should be enabled, for now.

    :param resolve_func: Takes aligned (joined) DataFrame, the column of the current DataFrame, and
        the column of another DataFrame. It returns an iterable that produces Series.

        >>> import os
        >>>
        >>> prev = os.environ.get("OPS_ON_DIFF_FRAMES", "false")
        >>> os.environ["OPS_ON_DIFF_FRAMES"] = "true"
        >>>
        >>> kdf1 = ks.DataFrame({'a': [9, 8, 7, 6, 5, 4, 3, 2, 1]})
        >>> kdf2 = ks.DataFrame({'a': [9, 8, 7, 6, 5, 4, 3, 2, 1]})
        >>>
        >>> def func(kdf, this_columns, that_columns):
        ...    kdf  # conceptually this is A + B.
        ...
        ...    # Within this function, Series from A or B can be performed against `kdf`.
        ...    this_column = this_columns[0]  # this is 'a' from kdf1.
        ...    that_column = that_columns[0]  # this is 'a' from kdf2.
        ...    new_series = kdf[this_column] - kdf[that_column]
        ...
        ...    # This new series will be placed in new DataFrame.
        ...    yield new_series.rename(this_column)  # or list(new_series)
        >>>
        >>>
        >>> align_diff_frames(func, kdf1, kdf2).sort_index()
           a
        0  0
        1  0
        2  0
        3  0
        4  0
        5  0
        6  0
        7  0
        8  0
        >>> os.environ["OPS_ON_DIFF_FRAMES"] = prev

    :param this: a DataFrame to align
    :param that: another DataFrame to align
    :param fillna: If True, it fills missing values in non-common columns in both `this` and `that`.
        Otherwise, it returns as are.
    :param how: join way. In addition, it affects how `resolve_func` resolves the column conflict.
        - full: `resolve_func` should resolve only common columns from 'this' and 'that' DataFrames.
            For instance, if 'this' has columns A, B, C and that has B, C, D, `this_columns` and
            'that_columns' in this function are B, C and B, C.
        - left: `resolve_func` should resolve columns including that columns.
            For instance, if 'this' has columns A, B, C and that has B, C, D, `this_columns` is
            B, C but `that_columns` are B, C, D.
    :return: Alined DataFrame
    """
    from databricks.koalas import DataFrame

    assert how == "full" or how == "left"

    this_data_columns = this._internal.data_columns
    that_data_columns = that._internal.data_columns
    common_columns = set(this_data_columns).intersection(that_data_columns)

    # 1. Full outer join given two dataframes.
    combined = combine_frames(this, that, how=how)

    # 2. Apply given function to transform the columns in a batch and keep the new columns.
    combined_data_columns = combined._internal.data_columns

    that_columns_to_apply = []
    this_columns_to_apply = []
    additional_that_columns = []
    columns_to_keep = []

    for combined_column in combined_data_columns:
        for common_column in common_columns:
            if combined_column == "__this_%s" % common_column:
                this_columns_to_apply.append(combined_column)
                break
            elif combined_column == "__that_%s" % common_column:
                that_columns_to_apply.append(combined_column)
                break
        else:
            if how == "left" and \
                    combined_column in ["__that_%s" % c for c in that_data_columns]:
                # In this case, we will drop `that_columns` in `columns_to_keep` but passes
                # it later to `func`. `func` should resolve it.
                # Note that adding this into a separate list (`additional_that_columns`)
                # is intentional so that `this_columns` and `that_columns` can be paired.
                additional_that_columns.append(combined_column)
            elif fillna:
                columns_to_keep.append(F.lit(None).cast(FloatType()).alias(combined_column))
            else:
                columns_to_keep.append(F.col(combined_column))

    that_columns_to_apply += additional_that_columns

    # Should extract columns to apply and do it in a batch in case
    # it adds new columns for example.
    kser_set = list(resolve_func(combined, this_columns_to_apply, that_columns_to_apply))
    columns_applied = [c._scol for c in kser_set]

    sdf = combined._sdf.select(
        combined._internal.index_scols + columns_applied + columns_to_keep)

    # 3. Restore the names back and deduplicate columns.
    this_columns = OrderedDict()
    # Add columns in an order of its original frame.
    new_data_columns = [c for c in sdf.columns if c not in combined._internal.index_columns]
    for this_data_column in this_data_columns:
        for new_column in new_data_columns:
            striped = new_column
            if new_column.startswith("__this_") or new_column.startswith("__that_"):
                striped = new_column[7:]  # cut out the prefix (either __this_ or __that_).

            # Respect the applied columns first if there are duplicated columns found.
            if striped not in this_columns and this_data_column == striped:
                this_columns[striped] = F.col(new_column).alias(striped)
                break

    # After that, we will add the rest columns.
    other_columns = OrderedDict()
    for new_column in new_data_columns:
        striped = new_column
        if new_column.startswith("__this_") or new_column.startswith("__that_"):
            striped = new_column[7:]  # cut out the prefix (either __this_ or __that_).

        # Respect the applied columns first if there are duplicated columns found.
        if striped not in this_columns:
            other_columns[striped] = F.col(new_column).alias(striped)

    sdf = sdf.select(
        combined._internal.index_scols +
        list(this_columns.values()) +
        list(other_columns.values()))

    new_data_columns = [c for c in sdf.columns if c not in combined._internal.index_columns]
    internal = combined._internal.copy(sdf=sdf, data_columns=new_data_columns)
    return DataFrame(internal)


def align_diff_series(func, this, *args, how="full"):
    from databricks.koalas.base import IndexOpsMixin

    cols = [arg for arg in args if isinstance(arg, IndexOpsMixin)]
    this_series = this
    combined = combine_frames(this._kdf, *cols, how=how)

    that_columns = [F.col("__that_%s" % arg.name)
                    if isinstance(arg, IndexOpsMixin) else arg for arg in args]

    scol = func(F.col("__this_%s" % this_series.name), *that_columns).alias(this_series.name)

    this_series._kdf = combined
    series = this_series._with_new_scol(scol)

    # To make the anchor frame to only have the new Series. This loses the original anchor.
    return series.to_frame()[this_series.name]


def default_session(conf=None):
    if conf is None:
        conf = dict()
    builder = spark.SparkSession.builder.appName("Koalas")
    for key, value in conf.items():
        builder = builder.config(key, value)
    return builder.getOrCreate()


def validate_arguments_and_invoke_function(pobj: Union[pd.DataFrame, pd.Series],
                                           koalas_func: Callable, pandas_func: Callable,
                                           input_args: Dict):
    """
    Invokes a pandas function.

    This is created because different versions of pandas support different parameters, and as a
    result when we code against the latest version, our users might get a confusing
    "got an unexpected keyword argument" error if they are using an older version of pandas.

    This function validates all the arguments, removes the ones that are not supported if they
    are simply the default value (i.e. most likely the user didn't explicitly specify it). It
    throws a TypeError if the user explicitly specify an argument that is not supported by the
    pandas version available.

    For example usage, look at DataFrame.to_html().

    :param pobj: the pandas DataFrame or Series to operate on
    :param koalas_func: koalas function, used to get default parameter values
    :param pandas_func: pandas function, used to check whether pandas supports all the arguments
    :param input_args: arguments to pass to the pandas function, often created by using locals().
                       Make sure locals() call is at the top of the function so it captures only
                       input parameters, rather than local variables.
    :return: whatever pandas_func returns
    """
    import inspect

    # Makes a copy since whatever passed in is likely created by locals(), and we can't delete
    # 'self' key from that.
    args = input_args.copy()
    del args['self']

    if 'kwargs' in args:
        # explode kwargs
        kwargs = args['kwargs']
        del args['kwargs']
        args = {**args, **kwargs}

    koalas_params = inspect.signature(koalas_func).parameters
    pandas_params = inspect.signature(pandas_func).parameters

    for param in koalas_params.values():
        if param.name not in pandas_params:
            if args[param.name] == param.default:
                del args[param.name]
            else:
                raise TypeError(
                    ("The pandas version [%s] available does not support parameter '%s' " +
                     "for function '%s'.") % (pd.__version__, param.name, pandas_func.__name__))

    args['self'] = pobj
    return pandas_func(**args)


def lazy_property(fn):
    """
    Decorator that makes a property lazy-evaluated.

    Copied from https://stevenloria.com/lazy-properties/
    """
    attr_name = '_lazy_' + fn.__name__

    @property
    @functools.wraps(fn)
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


def scol_for(sdf: spark.DataFrame, column_name: str) -> spark.Column:
    """ Return Spark Column for the given column name. """
    return sdf['`{}`'.format(column_name)]


def column_index_level(column_index: List[Tuple[str]]) -> int:
    """ Return the level of the column index. """
    if len(column_index) == 0:
        return 0
    else:
        levels = set(len(idx) for idx in column_index)
        assert len(levels) == 1, levels
        return list(levels)[0]
