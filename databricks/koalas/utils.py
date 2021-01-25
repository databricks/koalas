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
from distutils.version import LooseVersion
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import pyarrow
import pyspark
from pyspark import sql as spark
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
import pandas as pd
from pandas.api.types import is_list_like

# For running doctests and reference resolution in PyCharm.
from databricks import koalas as ks  # noqa: F401
from databricks.koalas.typedef.typehints import as_spark_type

if TYPE_CHECKING:
    # This is required in old Python 3.5 to prevent circular reference.
    from databricks.koalas.base import IndexOpsMixin
    from databricks.koalas.frame import DataFrame
    from databricks.koalas.internal import InternalFrame
    from databricks.koalas.series import Series


ERROR_MESSAGE_CANNOT_COMBINE = (
    "Cannot combine the series or dataframe because it comes from a different dataframe. "
    "In order to allow this operation, enable 'compute.ops_on_diff_frames' option."
)


def same_anchor(
    this: Union["DataFrame", "IndexOpsMixin", "InternalFrame"],
    that: Union["DataFrame", "IndexOpsMixin", "InternalFrame"],
) -> bool:
    """
    Check if the anchors of the given DataFrame or Series are the same or not.
    """
    from databricks.koalas.base import IndexOpsMixin
    from databricks.koalas.frame import DataFrame
    from databricks.koalas.internal import InternalFrame

    if isinstance(this, InternalFrame):
        this_internal = this
    else:
        assert isinstance(this, (DataFrame, IndexOpsMixin)), type(this)
        this_internal = this._internal

    if isinstance(that, InternalFrame):
        that_internal = that
    else:
        assert isinstance(that, (DataFrame, IndexOpsMixin)), type(that)
        that_internal = that._internal

    return (
        this_internal.spark_frame is that_internal.spark_frame
        and this_internal.index_level == that_internal.index_level
        and all(
            this_scol._jc.equals(that_scol._jc)
            for this_scol, that_scol in zip(
                this_internal.index_spark_columns, that_internal.index_spark_columns
            )
        )
    )


def combine_frames(this, *args, how="full", preserve_order_column=False):
    """
    This method combines `this` DataFrame with a different `that` DataFrame or
    Series from a different DataFrame.

    It returns a DataFrame that has prefix `this_` and `that_` to distinct
    the columns names from both DataFrames

    It internally performs a join operation which can be expensive in general.
    So, if `compute.ops_on_diff_frames` option is False,
    this method throws an exception.
    """
    from databricks.koalas.config import get_option
    from databricks.koalas.frame import DataFrame
    from databricks.koalas.internal import (
        InternalFrame,
        HIDDEN_COLUMNS,
        NATURAL_ORDER_COLUMN_NAME,
        SPARK_INDEX_NAME_FORMAT,
    )
    from databricks.koalas.series import Series

    if all(isinstance(arg, Series) for arg in args):
        assert all(
            same_anchor(arg, args[0]) for arg in args
        ), "Currently only one different DataFrame (from given Series) is supported"
        assert not same_anchor(this, args[0]), "We don't need to combine. All series is in this."
        that = args[0]._kdf[list(args)]
    elif len(args) == 1 and isinstance(args[0], DataFrame):
        assert isinstance(args[0], DataFrame)
        assert not same_anchor(
            this, args[0]
        ), "We don't need to combine. `this` and `that` are same."
        that = args[0]
    else:
        raise AssertionError("args should be single DataFrame or " "single/multiple Series")

    if get_option("compute.ops_on_diff_frames"):

        def resolve(internal, side):
            rename = lambda col: "__{}_{}".format(side, col)
            internal = internal.resolved_copy
            sdf = internal.spark_frame
            sdf = internal.spark_frame.select(
                [
                    scol_for(sdf, col).alias(rename(col))
                    for col in sdf.columns
                    if col not in HIDDEN_COLUMNS
                ]
                + list(HIDDEN_COLUMNS)
            )
            return internal.copy(
                spark_frame=sdf,
                index_spark_columns=[
                    scol_for(sdf, rename(col)) for col in internal.index_spark_column_names
                ],
                data_spark_columns=[
                    scol_for(sdf, rename(col)) for col in internal.data_spark_column_names
                ],
            )

        this_internal = resolve(this._internal, "this")
        that_internal = resolve(that._internal, "that")

        this_index_map = list(
            zip(this_internal.index_spark_column_names, this_internal.index_names)
        )
        that_index_map = list(
            zip(that_internal.index_spark_column_names, that_internal.index_names)
        )
        assert len(this_index_map) == len(that_index_map)

        join_scols = []
        merged_index_scols = []

        # Note that the order of each element in index_map is guaranteed according to the index
        # level.
        this_and_that_index_map = list(zip(this_index_map, that_index_map))

        this_sdf = this_internal.spark_frame.alias("this")
        that_sdf = that_internal.spark_frame.alias("that")

        # If the same named index is found, that's used.
        index_column_names = []
        for i, ((this_column, this_name), (that_column, that_name)) in enumerate(
            this_and_that_index_map
        ):
            if this_name == that_name:
                # We should merge the Spark columns into one
                # to mimic pandas' behavior.
                this_scol = scol_for(this_sdf, this_column)
                that_scol = scol_for(that_sdf, that_column)
                join_scol = this_scol == that_scol
                join_scols.append(join_scol)

                column_name = SPARK_INDEX_NAME_FORMAT(i)
                index_column_names.append(column_name)
                merged_index_scols.append(
                    F.when(this_scol.isNotNull(), this_scol).otherwise(that_scol).alias(column_name)
                )
            else:
                raise ValueError("Index names must be exactly matched currently.")

        assert len(join_scols) > 0, "cannot join with no overlapping index names"

        joined_df = this_sdf.join(that_sdf, on=join_scols, how=how)

        if preserve_order_column:
            order_column = [scol_for(this_sdf, NATURAL_ORDER_COLUMN_NAME)]
        else:
            order_column = []

        joined_df = joined_df.select(
            merged_index_scols
            + [
                scol_for(this_sdf, this_internal.spark_column_name_for(label))
                for label in this_internal.column_labels
            ]
            + [
                scol_for(that_sdf, that_internal.spark_column_name_for(label))
                for label in that_internal.column_labels
            ]
            + order_column
        )

        index_columns = set(index_column_names)
        new_data_columns = [
            col
            for col in joined_df.columns
            if col not in index_columns and col != NATURAL_ORDER_COLUMN_NAME
        ]
        level = max(this_internal.column_labels_level, that_internal.column_labels_level)

        def fill_label(label):
            if label is None:
                return ([""] * (level - 1)) + [None]
            else:
                return ([""] * (level - len(label))) + list(label)

        column_labels = [
            tuple(["this"] + fill_label(label)) for label in this_internal.column_labels
        ] + [tuple(["that"] + fill_label(label)) for label in that_internal.column_labels]
        column_label_names = (
            [None] * (1 + level - this_internal.column_labels_level)
        ) + this_internal.column_label_names
        return DataFrame(
            InternalFrame(
                spark_frame=joined_df,
                index_spark_columns=[scol_for(joined_df, col) for col in index_column_names],
                index_names=this_internal.index_names,
                column_labels=column_labels,
                data_spark_columns=[scol_for(joined_df, col) for col in new_data_columns],
                column_label_names=column_label_names,
            )
        )
    else:
        raise ValueError(ERROR_MESSAGE_CANNOT_COMBINE)


def align_diff_frames(
    resolve_func,
    this: "DataFrame",
    that: "DataFrame",
    fillna: bool = True,
    how: str = "full",
    preserve_order_column: bool = False,
) -> "DataFrame":
    """
    This method aligns two different DataFrames with a given `func`. Columns are resolved and
    handled within the given `func`.
    To use this, `compute.ops_on_diff_frames` should be True, for now.

    :param resolve_func: Takes aligned (joined) DataFrame, the column of the current DataFrame, and
        the column of another DataFrame. It returns an iterable that produces Series.

        >>> from databricks.koalas.config import set_option, reset_option
        >>>
        >>> set_option("compute.ops_on_diff_frames", True)
        >>>
        >>> kdf1 = ks.DataFrame({'a': [9, 8, 7, 6, 5, 4, 3, 2, 1]})
        >>> kdf2 = ks.DataFrame({'a': [9, 8, 7, 6, 5, 4, 3, 2, 1]})
        >>>
        >>> def func(kdf, this_column_labels, that_column_labels):
        ...    kdf  # conceptually this is A + B.
        ...
        ...    # Within this function, Series from A or B can be performed against `kdf`.
        ...    this_label = this_column_labels[0]  # this is ('a',) from kdf1.
        ...    that_label = that_column_labels[0]  # this is ('a',) from kdf2.
        ...    new_series = (kdf[this_label] - kdf[that_label]).rename(str(this_label))
        ...
        ...    # This new series will be placed in new DataFrame.
        ...    yield (new_series, this_label)
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
        >>> reset_option("compute.ops_on_diff_frames")

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
        - inner: Same as 'full' mode; however, internally performs inner join instead.
    :return: Aligned DataFrame
    """
    assert how == "full" or how == "left" or how == "inner"

    this_column_labels = this._internal.column_labels
    that_column_labels = that._internal.column_labels
    common_column_labels = set(this_column_labels).intersection(that_column_labels)

    # 1. Perform the join given two dataframes.
    combined = combine_frames(this, that, how=how, preserve_order_column=preserve_order_column)

    # 2. Apply the given function to transform the columns in a batch and keep the new columns.
    combined_column_labels = combined._internal.column_labels

    that_columns_to_apply = []
    this_columns_to_apply = []
    additional_that_columns = []
    columns_to_keep = []
    column_labels_to_keep = []

    for combined_label in combined_column_labels:
        for common_label in common_column_labels:
            if combined_label == tuple(["this", *common_label]):
                this_columns_to_apply.append(combined_label)
                break
            elif combined_label == tuple(["that", *common_label]):
                that_columns_to_apply.append(combined_label)
                break
        else:
            if how == "left" and combined_label in [
                tuple(["that", *label]) for label in that_column_labels
            ]:
                # In this case, we will drop `that_columns` in `columns_to_keep` but passes
                # it later to `func`. `func` should resolve it.
                # Note that adding this into a separate list (`additional_that_columns`)
                # is intentional so that `this_columns` and `that_columns` can be paired.
                additional_that_columns.append(combined_label)
            elif fillna:
                columns_to_keep.append(F.lit(None).cast(FloatType()).alias(str(combined_label)))
                column_labels_to_keep.append(combined_label)
            else:
                columns_to_keep.append(combined._internal.spark_column_for(combined_label))
                column_labels_to_keep.append(combined_label)

    that_columns_to_apply += additional_that_columns

    # Should extract columns to apply and do it in a batch in case
    # it adds new columns for example.
    if len(this_columns_to_apply) > 0 or len(that_columns_to_apply) > 0:
        kser_set, column_labels_applied = zip(
            *resolve_func(combined, this_columns_to_apply, that_columns_to_apply)
        )
        columns_applied = [c.spark.column for c in kser_set]
        column_labels_applied = list(column_labels_applied)
    else:
        columns_applied = []
        column_labels_applied = []

    applied = combined[columns_applied + columns_to_keep]
    applied.columns = pd.MultiIndex.from_tuples(
        column_labels_applied + column_labels_to_keep, names=combined.columns.names
    )

    # 3. Restore the names back and deduplicate columns.
    this_labels = OrderedDict()
    # Add columns in an order of its original frame.
    for this_label in this_column_labels:
        for new_label in applied._internal.column_labels:
            if new_label[1:] not in this_labels and this_label == new_label[1:]:
                this_labels[new_label[1:]] = new_label

    # After that, we will add the rest columns.
    other_labels = OrderedDict()
    for new_label in applied._internal.column_labels:
        if new_label[1:] not in this_labels:
            other_labels[new_label[1:]] = new_label

    kdf = applied[list(this_labels.values()) + list(other_labels.values())]
    kdf.columns = kdf.columns.droplevel()
    return kdf


def align_diff_series(func, this_series: "Series", *args, how: str = "full") -> "Series":
    from databricks.koalas.frame import DataFrame
    from databricks.koalas.series import Series, first_series

    cols = [arg for arg in args if isinstance(arg, Series)]
    combined = combine_frames(this_series.to_frame(), *cols, how=how)

    scol = func(
        combined["this"]._internal.data_spark_columns[0],
        *combined["that"]._internal.data_spark_columns
    )

    internal = combined._internal.copy(
        column_labels=this_series._internal.column_labels,
        data_spark_columns=[scol.alias(name_like_string(this_series.name))],
        column_label_names=this_series._internal.column_label_names,
    )
    return first_series(DataFrame(internal))


def is_testing():
    """ Indicates whether Koalas is currently running tests. """
    return "KOALAS_TESTING" in os.environ


def default_session(conf=None):
    if conf is None:
        conf = dict()
    should_use_legacy_ipc = False
    if LooseVersion(pyarrow.__version__) >= LooseVersion("0.15") and LooseVersion(
        pyspark.__version__
    ) < LooseVersion("3.0"):
        conf["spark.executorEnv.ARROW_PRE_0_15_IPC_FORMAT"] = "1"
        conf["spark.yarn.appMasterEnv.ARROW_PRE_0_15_IPC_FORMAT"] = "1"
        conf["spark.mesos.driverEnv.ARROW_PRE_0_15_IPC_FORMAT"] = "1"
        conf["spark.kubernetes.driverEnv.ARROW_PRE_0_15_IPC_FORMAT"] = "1"
        should_use_legacy_ipc = True

    builder = spark.SparkSession.builder.appName("Koalas")
    for key, value in conf.items():
        builder = builder.config(key, value)
    # Currently, Koalas is dependent on such join due to 'compute.ops_on_diff_frames'
    # configuration. This is needed with Spark 3.0+.
    builder.config("spark.sql.analyzer.failAmbiguousSelfJoin", False)

    if LooseVersion(pyspark.__version__) >= LooseVersion("3.0.1") and is_testing():
        builder.config("spark.executor.allowSparkContext", False)

    session = builder.getOrCreate()

    if not should_use_legacy_ipc:
        is_legacy_ipc_set = any(
            v == "1"
            for v in [
                session.conf.get("spark.executorEnv.ARROW_PRE_0_15_IPC_FORMAT", None),
                session.conf.get("spark.yarn.appMasterEnv.ARROW_PRE_0_15_IPC_FORMAT", None),
                session.conf.get("spark.mesos.driverEnv.ARROW_PRE_0_15_IPC_FORMAT", None),
                session.conf.get("spark.kubernetes.driverEnv.ARROW_PRE_0_15_IPC_FORMAT", None),
            ]
        )
        if is_legacy_ipc_set:
            raise RuntimeError(
                "Please explicitly unset 'ARROW_PRE_0_15_IPC_FORMAT' environment variable in "
                "both driver and executor sides. Check your spark.executorEnv.*, "
                "spark.yarn.appMasterEnv.*, spark.mesos.driverEnv.* and "
                "spark.kubernetes.driverEnv.* configurations. It is required to set this "
                "environment variable only when you use pyarrow>=0.15 and pyspark<3.0."
            )
    return session


def validate_arguments_and_invoke_function(
    pobj: Union[pd.DataFrame, pd.Series],
    koalas_func: Callable,
    pandas_func: Callable,
    input_args: Dict,
):
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
    :param koalas_func: Koalas function, used to get default parameter values
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
    del args["self"]

    if "kwargs" in args:
        # explode kwargs
        kwargs = args["kwargs"]
        del args["kwargs"]
        args = {**args, **kwargs}

    koalas_params = inspect.signature(koalas_func).parameters
    pandas_params = inspect.signature(pandas_func).parameters

    for param in koalas_params.values():
        if param.name not in pandas_params:
            if args[param.name] == param.default:
                del args[param.name]
            else:
                raise TypeError(
                    (
                        "The pandas version [%s] available does not support parameter '%s' "
                        + "for function '%s'."
                    )
                    % (pd.__version__, param.name, pandas_func.__name__)
                )

    args["self"] = pobj
    return pandas_func(**args)


def lazy_property(fn):
    """
    Decorator that makes a property lazy-evaluated.

    Copied from https://stevenloria.com/lazy-properties/
    """
    attr_name = "_lazy_" + fn.__name__

    @property
    @functools.wraps(fn)
    def wrapped_lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    def deleter(self):
        if hasattr(self, attr_name):
            delattr(self, attr_name)

    return wrapped_lazy_property.deleter(deleter)


def scol_for(sdf: spark.DataFrame, column_name: str) -> spark.Column:
    """ Return Spark Column for the given column name. """
    return sdf["`{}`".format(column_name)]


def column_labels_level(column_labels: List[Tuple]) -> int:
    """ Return the level of the column index. """
    if len(column_labels) == 0:
        return 1
    else:
        levels = set(1 if label is None else len(label) for label in column_labels)
        assert len(levels) == 1, levels
        return list(levels)[0]


def name_like_string(name: Optional[Union[str, Tuple]]) -> str:
    """
    Return the name-like strings from str or tuple of str

    Examples
    --------
    >>> name = 'abc'
    >>> name_like_string(name)
    'abc'

    >>> name = ('abc',)
    >>> name_like_string(name)
    'abc'

    >>> name = ('a', 'b', 'c')
    >>> name_like_string(name)
    '(a, b, c)'
    """
    if name is None:
        name = ("__none__",)
    elif is_list_like(name):
        name = tuple([str(n) for n in name])
    else:
        name = (str(name),)
    return ("(%s)" % ", ".join(name)) if len(name) > 1 else name[0]


def is_name_like_tuple(value: Any, allow_none: bool = True, check_type: bool = False) -> bool:
    """
    Check the given tuple is be able to be used as a name.

    Examples
    --------
    >>> is_name_like_tuple(('abc',))
    True
    >>> is_name_like_tuple((1,))
    True
    >>> is_name_like_tuple(('abc', 1, None))
    True
    >>> is_name_like_tuple(('abc', 1, None), check_type=True)
    True
    >>> is_name_like_tuple((1.0j,))
    True
    >>> is_name_like_tuple(tuple())
    False
    >>> is_name_like_tuple((list('abc'),))
    False
    >>> is_name_like_tuple(('abc', 1, None), allow_none=False)
    False
    >>> is_name_like_tuple((1.0j,), check_type=True)
    False
    """
    if value is None:
        return allow_none
    elif not isinstance(value, tuple):
        return False
    elif len(value) == 0:
        return False
    elif not allow_none and any(v is None for v in value):
        return False
    elif any(is_list_like(v) or isinstance(v, slice) for v in value):
        return False
    elif check_type:
        try:
            return all(v is None or as_spark_type(type(v)) is not None for v in value)
        except TypeError:
            return False
    else:
        return True


def is_name_like_value(
    value: Any, allow_none: bool = True, allow_tuple: bool = True, check_type: bool = False
) -> bool:
    """
    Check the given value is like a name.

    Examples
    --------
    >>> is_name_like_value('abc')
    True
    >>> is_name_like_value(1)
    True
    >>> is_name_like_value(None)
    True
    >>> is_name_like_value(('abc',))
    True
    >>> is_name_like_value(1.0j)
    True
    >>> is_name_like_value(list('abc'))
    False
    >>> is_name_like_value(None, allow_none=False)
    False
    >>> is_name_like_value(('abc',), allow_tuple=False)
    False
    >>> is_name_like_value(1.0j, check_type=True)
    False
    """
    if value is None:
        return allow_none
    elif isinstance(value, tuple):
        return allow_tuple and is_name_like_tuple(
            value, allow_none=allow_none, check_type=check_type
        )
    elif is_list_like(value) or isinstance(value, slice):
        return False
    elif check_type:
        try:
            return as_spark_type(type(value)) is not None
        except TypeError:
            return False
    else:
        return True


def validate_axis(axis=0, none_axis=0):
    """ Check the given axis is valid. """
    # convert to numeric axis
    axis = {None: none_axis, "index": 0, "columns": 1}.get(axis, axis)
    if axis not in (none_axis, 0, 1):
        raise ValueError("No axis named {0}".format(axis))
    return axis


def validate_bool_kwarg(value, arg_name):
    """ Ensures that argument passed in arg_name is of type bool. """
    if not (isinstance(value, bool) or value is None):
        raise ValueError(
            'For argument "{}" expected type bool, received '
            "type {}.".format(arg_name, type(value).__name__)
        )
    return value


def verify_temp_column_name(
    df: Union["DataFrame", spark.DataFrame], column_name_or_label: Union[Any, Tuple]
) -> Union[Any, Tuple]:
    """
    Verify that the given column name does not exist in the given Koalas or Spark DataFrame.

    The temporary column names should start and end with `__`. In addition, `column_name_or_label`
    expects a single string, or column labels when `df` is a Koalas DataFrame.

    >>> kdf = ks.DataFrame({("x", "a"): ['a', 'b', 'c']})
    >>> kdf["__dummy__"] = 0
    >>> kdf[("", "__dummy__")] = 1
    >>> kdf  # doctest: +NORMALIZE_WHITESPACE
       x __dummy__
       a           __dummy__
    0  a         0         1
    1  b         0         1
    2  c         0         1

    >>> verify_temp_column_name(kdf, '__tmp__')
    ('__tmp__', '')
    >>> verify_temp_column_name(kdf, ('', '__tmp__'))
    ('', '__tmp__')
    >>> verify_temp_column_name(kdf, '__dummy__')
    Traceback (most recent call last):
    ...
    AssertionError: ... `(__dummy__, )` ...
    >>> verify_temp_column_name(kdf, ('', '__dummy__'))
    Traceback (most recent call last):
    ...
    AssertionError: ... `(, __dummy__)` ...
    >>> verify_temp_column_name(kdf, 'dummy')
    Traceback (most recent call last):
    ...
    AssertionError: ... should be empty or start and end with `__`: ('dummy', '')
    >>> verify_temp_column_name(kdf, ('', 'dummy'))
    Traceback (most recent call last):
    ...
    AssertionError: ... should be empty or start and end with `__`: ('', 'dummy')

    >>> internal = kdf._internal.resolved_copy
    >>> sdf = internal.spark_frame
    >>> sdf.select(internal.data_spark_columns).show()  # doctest: +NORMALIZE_WHITESPACE
    +------+---------+-------------+
    |(x, a)|__dummy__|(, __dummy__)|
    +------+---------+-------------+
    |     a|        0|            1|
    |     b|        0|            1|
    |     c|        0|            1|
    +------+---------+-------------+

    >>> verify_temp_column_name(sdf, '__tmp__')
    '__tmp__'
    >>> verify_temp_column_name(sdf, '__dummy__')
    Traceback (most recent call last):
    ...
    AssertionError: ... `__dummy__` ... '(x, a)', '__dummy__', '(, __dummy__)', ...
    >>> verify_temp_column_name(sdf, ('', '__dummy__'))
    Traceback (most recent call last):
    ...
    AssertionError: <class 'tuple'>
    >>> verify_temp_column_name(sdf, 'dummy')
    Traceback (most recent call last):
    ...
    AssertionError: ... should start and end with `__`: dummy
    """
    from databricks.koalas.frame import DataFrame

    if isinstance(df, DataFrame):
        if isinstance(column_name_or_label, str):
            column_name = column_name_or_label

            level = df._internal.column_labels_level
            column_name_or_label = tuple([column_name_or_label] + ([""] * (level - 1)))
        else:
            column_name = name_like_string(column_name_or_label)

        assert any(len(label) > 0 for label in column_name_or_label) and all(
            label == "" or (label.startswith("__") and label.endswith("__"))
            for label in column_name_or_label
        ), "The temporary column name should be empty or start and end with `__`: {}".format(
            column_name_or_label
        )
        assert all(
            column_name_or_label != label for label in df._internal.column_labels
        ), "The given column name `{}` already exists in the Koalas DataFrame: {}".format(
            name_like_string(column_name_or_label), df.columns
        )
        df = df._internal.resolved_copy.spark_frame
    else:
        assert isinstance(column_name_or_label, str), type(column_name_or_label)
        assert column_name_or_label.startswith("__") and column_name_or_label.endswith(
            "__"
        ), "The temporary column name should start and end with `__`: {}".format(
            column_name_or_label
        )
        column_name = column_name_or_label

    assert isinstance(df, spark.DataFrame), type(df)
    assert (
        column_name not in df.columns
    ), "The given column name `{}` already exists in the Spark DataFrame: {}".format(
        column_name, df.columns
    )

    return column_name_or_label


def compare_null_first(left, right, comp):
    return (left.isNotNull() & right.isNotNull() & comp(left, right)) | (
        left.isNull() & right.isNotNull()
    )


def compare_null_last(left, right, comp):
    return (left.isNotNull() & right.isNotNull() & comp(left, right)) | (
        left.isNotNull() & right.isNull()
    )


def compare_disallow_null(left, right, comp):
    return left.isNotNull() & right.isNotNull() & comp(left, right)


def compare_allow_null(left, right, comp):
    return left.isNull() | right.isNull() | comp(left, right)
