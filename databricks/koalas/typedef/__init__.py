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

from databricks.koalas.typedef.typehints import *

__all__ = ["pandas_wraps", "as_spark_type", "infer_pd_series_spark_type"]


def _make_fun(f: typing.Callable, return_type: types.DataType, *args, **kwargs) -> "ks.Series":
    """
    This function calls the function f while taking into account some of the
    limitations of the pandas UDF support:
    - support for keyword arguments
    - support for scalar values (as long as they are picklable)
    - support for type hints and input checks.
    :param f: the function to call. It is expected to have field annotations (see below).
    :param return_sig: the return type
    :param args: the arguments of the function
    :param kwargs: the kwargs to pass to the function
    :return: the value of executing the function: f(*args, **kwargs)

    The way this function executes depends on the what is provided as arguments:
     - if one of the arguments is a koalas series or dataframe:
        - the function is wrapped as a Spark UDF
        - the series arguments are checked to be coming from the same original anchor
        - the non-series arguments are serialized into the spark UDF.

    The function is expected to have the following arguments:
    """
    from databricks.koalas.series import Series
    from pyspark.sql import Column
    from pyspark.sql.functions import pandas_udf

    # All the arguments.
    # None for columns or the value for non-columns
    frozen_args = []  # type: typing.List[typing.Any]
    # ks.Series for columns or None for the non-columns
    col_args = []  # type: typing.List[typing.Optional[Series]]
    for arg in args:
        if isinstance(arg, Series):
            frozen_args.append(None)
            col_args.append(arg)
        elif isinstance(arg, Column):
            raise ValueError(
                "A pyspark column was passed as an argument." " Pass a koalas series instead"
            )
        else:
            frozen_args.append(arg)
            col_args.append(None)

    # Value is none for kwargs that are columns, and the value otherwise
    frozen_kwargs = []  # type: typing.List[typing.Tuple[str, typing.Any]]
    # Value is a spark col for kwarg that is column, and None otherwise
    col_kwargs = []  # type: typing.List[typing.Tuple[str, Series]]
    for (key, arg) in kwargs.items():
        if isinstance(arg, Series):
            col_kwargs.append((key, arg))
        elif isinstance(arg, Column):
            raise ValueError(
                "A pyspark column was passed as an argument." " Pass a koalas series instead"
            )
        else:
            frozen_kwargs.append((key, arg))

    col_args_idxs = [idx for (idx, c) in enumerate(col_args) if c is not None]
    all_indexes = col_args_idxs + [key for (key, _) in col_kwargs]  # type: ignore
    if not all_indexes:
        # No argument is related to spark
        # The function is just called through without other considerations.
        return f(*args, **kwargs)

    # We detected some columns. They need to be wrapped in a UDF to spark.
    kser = _get_kser(args, kwargs)

    def clean_fun(*args2):
        assert len(args2) == len(all_indexes), "Missing some inputs:{}!={}".format(
            all_indexes, [str(c) for c in args2]
        )
        full_args = list(frozen_args)
        full_kwargs = dict(frozen_kwargs)
        for (arg, idx) in zip(args2, all_indexes):
            if isinstance(idx, int):
                full_args[idx] = arg
            else:
                assert isinstance(idx, str), str(idx)
                full_kwargs[idx] = arg
        return f(*full_args, **full_kwargs)

    wrapped_udf = pandas_udf(clean_fun, returnType=return_type)
    name_tokens = []
    spark_col_args = []
    for col in col_args:
        if col is not None:
            spark_col_args.append(col.spark_column)
            name_tokens.append(str(col.name))
    kw_name_tokens = []
    for (key, col) in col_kwargs:
        spark_col_args.append(col.spark_column)
        kw_name_tokens.append("{}={}".format(key, col.name))
    col = wrapped_udf(*spark_col_args)
    series = kser._with_new_scol(scol=col)  # type: 'ks.Series'
    all_name_tokens = name_tokens + sorted(kw_name_tokens)
    name = "{}({})".format(f.__name__, ", ".join(all_name_tokens))
    series = series.rename(name)
    return series


def _get_kser(args, kwargs):
    from databricks.koalas.series import Series

    all_cols = [arg for arg in args if isinstance(arg, Series)] + [
        arg for arg in kwargs.values() if isinstance(arg, Series)
    ]
    assert all_cols
    # TODO: check all the anchors
    return all_cols[0]


def pandas_wraps(function=None, return_col=None, return_scalar=None):
    """ This annotation makes a function available for Koalas.

    Spark requires more information about the return types than pandas, and sometimes more
    information is required. This annotations allows you to seamlessly write functions that
    work for both pandas and koalas.

    Examples
    --------

    Wrapping a function with python 3's type annotations:

    >>> from databricks.koalas import pandas_wraps
    >>> pdf = pd.DataFrame({"col1": [1, 2], "col2": [10, 20]}, dtype=np.int64)
    >>> df = ks.DataFrame(pdf)

    Consider a simple function that operates on pandas series of integers

    >>> def fun(col1):
    ...     return col1.apply(lambda x: x * 2)  # Arbitrary pandas code.
    >>> fun(pdf.col1)
    0    2
    1    4
    Name: col1, dtype: int64

    Koalas needs to know the return type in order to make this function accessible to Spark.
    The following function uses python built-in typing hint system to hint that this function
    returns a Series of integers:

    >>> @pandas_wraps
    ... def fun(col1) -> ks.Series[np.int64]:
    ...     return col1.apply(lambda x: x * 2)  # Arbitrary pandas code.

    This function works as before on pandas Series:

    >>> fun(pdf.col1)
    0    2
    1    4
    Name: col1, dtype: int64

    Now it also works on Koalas series:

    >>> fun(df.col1)
    0    2
    1    4
    Name: fun(col1), dtype: int64

    Alternatively, the type hint can be provided as an argument to the `pandas_wraps` decorator:

    >>> @pandas_wraps(return_col=np.int64)
    ... def fun(col1):
    ...     return col1.apply(lambda x: x * 2)  # Arbitrary pandas code.

    >>> fun(df.col1)
    0    2
    1    4
    Name: fun(col1), dtype: int64

    Unlike PySpark user-defined functions, the decorator supports arguments all of python's
    styles of arguments (named arguments, optional arguments, list and keyworded arguments).
    It will automatically distribute argument values that are not Koalas series. Here is an
    example of function with optional series arguments and non-series arguments:

    >>> @pandas_wraps(return_col=float)
    ... def fun(col1, col2 = None, arg1="x", **kwargs):
    ...    return 2.0 * col1 if arg1 == "x" else 3.0 * col1 * col2 * kwargs['col3']

    >>> fun(df.col1)
    0    2.0
    1    4.0
    Name: fun(col1), dtype: float32

    >>> fun(df.col1, col2=df.col2, arg1="y", col3=df.col2)
    0     300.0
    1    2400.0
    Name: fun(col1, col2=col2, col3=col2), dtype: float32

    Notes
    -----
    The arguments provided to the function must be picklable, or an error will be raised by Spark.
    The example below fails.

    >>> import sys
    >>> fun(df.col1, arg1=sys.stdout)  # doctest: +SKIP
    """
    import warnings
    from functools import wraps

    warnings.warn(
        "pandas_wraps is deprecated. Please use transform_batch instead.", DeprecationWarning,
    )

    def function_wrapper(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Extract the signature arguments from this function.
            sig_return = infer_return_type(f, return_col, return_scalar)
            if not isinstance(sig_return, SeriesType):
                raise ValueError(
                    "Expected the return type of this function to be of type column,"
                    " but found type {}".format(sig_return)
                )
            spark_return_type = sig_return.tpe
            return _make_fun(f, spark_return_type, *args, **kwargs)

        return wrapper

    if callable(function):
        return function_wrapper(function)
    else:
        return function_wrapper
