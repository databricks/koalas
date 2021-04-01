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
Koalas specific features.
"""
import inspect
from distutils.version import LooseVersion
from typing import Any, Optional, Tuple, Union, TYPE_CHECKING, cast
import types

import numpy as np  # noqa: F401
import pandas as pd
import pyspark
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructField, StructType

from databricks.koalas.internal import (
    InternalFrame,
    SPARK_INDEX_NAME_FORMAT,
    SPARK_DEFAULT_SERIES_NAME,
)
from databricks.koalas.typedef import infer_return_type, DataFrameType, ScalarType, SeriesType
from databricks.koalas.spark.utils import as_nullable_spark_type, force_decimal_precision_scale
from databricks.koalas.utils import (
    is_name_like_value,
    is_name_like_tuple,
    name_like_string,
    scol_for,
    verify_temp_column_name,
)

if TYPE_CHECKING:
    from databricks.koalas.frame import DataFrame
    from databricks.koalas.series import Series


class KoalasFrameMethods(object):
    """ Koalas specific features for DataFrame. """

    def __init__(self, frame: "DataFrame"):
        self._kdf = frame

    def attach_id_column(self, id_type: str, column: Union[Any, Tuple]) -> "DataFrame":
        """
        Attach a column to be used as identifier of rows similar to the default index.

        See also `Default Index type
        <https://koalas.readthedocs.io/en/latest/user_guide/options.html#default-index-type>`_.

        Parameters
        ----------
        id_type : string
            The id type.

            - 'sequence' : a sequence that increases one by one.

              .. note:: this uses Spark's Window without specifying partition specification.
                  This leads to move all data into single partition in single machine and
                  could cause serious performance degradation.
                  Avoid this method against very large dataset.

            - 'distributed-sequence' : a sequence that increases one by one,
              by group-by and group-map approach in a distributed manner.
            - 'distributed' : a monotonically increasing sequence simply by using PySpark’s
              monotonically_increasing_id function in a fully distributed manner.

        column : string or tuple of string
            The column name.

        Returns
        -------
        DataFrame
            The DataFrame attached the column.

        Examples
        --------
        >>> df = ks.DataFrame({"x": ['a', 'b', 'c']})
        >>> df.koalas.attach_id_column(id_type="sequence", column="id")
           x  id
        0  a   0
        1  b   1
        2  c   2

        >>> df.koalas.attach_id_column(id_type="distributed-sequence", column=0)
           x  0
        0  a  0
        1  b  1
        2  c  2

        >>> df.koalas.attach_id_column(id_type="distributed", column=0.0)
        ... # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
           x  0.0
        0  a  ...
        1  b  ...
        2  c  ...

        For multi-index columns:

        >>> df = ks.DataFrame({("x", "y"): ['a', 'b', 'c']})
        >>> df.koalas.attach_id_column(id_type="sequence", column=("id-x", "id-y"))
           x id-x
           y id-y
        0  a    0
        1  b    1
        2  c    2

        >>> df.koalas.attach_id_column(id_type="distributed-sequence", column=(0, 1.0))
           x   0
           y 1.0
        0  a   0
        1  b   1
        2  c   2
        """
        from databricks.koalas.frame import DataFrame

        if id_type == "sequence":
            attach_func = InternalFrame.attach_sequence_column
        elif id_type == "distributed-sequence":
            attach_func = InternalFrame.attach_distributed_sequence_column
        elif id_type == "distributed":
            attach_func = InternalFrame.attach_distributed_column
        else:
            raise ValueError(
                "id_type should be one of 'sequence', 'distributed-sequence' and 'distributed'"
            )

        assert is_name_like_value(column, allow_none=False), column
        if not is_name_like_tuple(column):
            column = (column,)

        internal = self._kdf._internal

        if len(column) != internal.column_labels_level:
            raise ValueError(
                "The given column `{}` must be the same length as the existing columns.".format(
                    column
                )
            )
        elif column in internal.column_labels:
            raise ValueError(
                "The given column `{}` already exists.".format(name_like_string(column))
            )

        # Make sure the underlying Spark column names are the form of
        # `name_like_string(column_label)`.
        sdf = internal.spark_frame.select(
            [
                scol.alias(SPARK_INDEX_NAME_FORMAT(i))
                for i, scol in enumerate(internal.index_spark_columns)
            ]
            + [
                scol.alias(name_like_string(label))
                for scol, label in zip(internal.data_spark_columns, internal.column_labels)
            ]
        )
        sdf = attach_func(sdf, name_like_string(column))

        return DataFrame(
            InternalFrame(
                spark_frame=sdf,
                index_spark_columns=[
                    scol_for(sdf, SPARK_INDEX_NAME_FORMAT(i)) for i in range(internal.index_level)
                ],
                index_names=internal.index_names,
                index_dtypes=internal.index_dtypes,
                column_labels=internal.column_labels + [column],
                data_spark_columns=(
                    [scol_for(sdf, name_like_string(label)) for label in internal.column_labels]
                    + [scol_for(sdf, name_like_string(column))]
                ),
                data_dtypes=(internal.data_dtypes + [None]),
                column_label_names=internal.column_label_names,
            ).resolved_copy
        )

    def apply_batch(self, func, args=(), **kwds) -> "DataFrame":
        """
        Apply a function that takes pandas DataFrame and outputs pandas DataFrame. The pandas
        DataFrame given to the function is of a batch used internally.

        See also `Transform and apply a function
        <https://koalas.readthedocs.io/en/latest/user_guide/transform_apply.html>`_.

        .. note:: the `func` is unable to access to the whole input frame. Koalas internally
            splits the input series into multiple batches and calls `func` with each batch multiple
            times. Therefore, operations such as global aggregations are impossible. See the example
            below.

            >>> # This case does not return the length of whole frame but of the batch internally
            ... # used.
            ... def length(pdf) -> ks.DataFrame[int]:
            ...     return pd.DataFrame([len(pdf)])
            ...
            >>> df = ks.DataFrame({'A': range(1000)})
            >>> df.koalas.apply_batch(length)  # doctest: +SKIP
                c0
            0   83
            1   83
            2   83
            ...
            10  83
            11  83

        .. note:: this API executes the function once to infer the type which is
            potentially expensive, for instance, when the dataset is created after
            aggregations or sorting.

            To avoid this, specify return type in ``func``, for instance, as below:

            >>> def plus_one(x) -> ks.DataFrame[float, float]:
            ...     return x + 1

            If the return type is specified, the output column names become
            `c0, c1, c2 ... cn`. These names are positionally mapped to the returned
            DataFrame in ``func``.

            To specify the column names, you can assign them in a pandas friendly style as below:

            >>> def plus_one(x) -> ks.DataFrame["a": float, "b": float]:
            ...     return x + 1

            >>> pdf = pd.DataFrame({'a': [1, 2, 3], 'b': [3, 4, 5]})
            >>> def plus_one(x) -> ks.DataFrame[zip(pdf.dtypes, pdf.columns)]:
            ...     return x + 1

            When the given function has the return type annotated, the original index of the
            DataFrame will be lost and a default index will be attached to the result DataFrame.
            Please be careful about configuring the default index. See also `Default Index Type
            <https://koalas.readthedocs.io/en/latest/user_guide/options.html#default-index-type>`_.


        Parameters
        ----------
        func : function
            Function to apply to each pandas frame.
        args : tuple
            Positional arguments to pass to `func` in addition to the
            array/series.
        **kwds
            Additional keyword arguments to pass as keywords arguments to
            `func`.

        Returns
        -------
        DataFrame

        See Also
        --------
        DataFrame.apply: For row/columnwise operations.
        DataFrame.applymap: For elementwise operations.
        DataFrame.aggregate: Only perform aggregating type operations.
        DataFrame.transform: Only perform transforming type operations.
        Series.koalas.transform_batch: transform the search as each pandas chunks.

        Examples
        --------
        >>> df = ks.DataFrame([(1, 2), (3, 4), (5, 6)], columns=['A', 'B'])
        >>> df
           A  B
        0  1  2
        1  3  4
        2  5  6

        >>> def query_func(pdf) -> ks.DataFrame[int, int]:
        ...     return pdf.query('A == 1')
        >>> df.koalas.apply_batch(query_func)
           c0  c1
        0   1   2

        >>> def query_func(pdf) -> ks.DataFrame["A": int, "B": int]:
        ...     return pdf.query('A == 1')
        >>> df.koalas.apply_batch(query_func)
           A  B
        0  1  2

        You can also omit the type hints so Koalas infers the return schema as below:

        >>> df.koalas.apply_batch(lambda pdf: pdf.query('A == 1'))
           A  B
        0  1  2

        You can also specify extra arguments.

        >>> def calculation(pdf, y, z) -> ks.DataFrame[int, int]:
        ...     return pdf ** y + z
        >>> df.koalas.apply_batch(calculation, args=(10,), z=20)
                c0        c1
        0       21      1044
        1    59069   1048596
        2  9765645  60466196

        You can also use ``np.ufunc`` and built-in functions as input.

        >>> df.koalas.apply_batch(np.add, args=(10,))
            A   B
        0  11  12
        1  13  14
        2  15  16

        >>> (df * -1).koalas.apply_batch(abs)
           A  B
        0  1  2
        1  3  4
        2  5  6

        """
        # TODO: codes here partially duplicate `DataFrame.apply`. Can we deduplicate?

        from databricks.koalas.groupby import GroupBy
        from databricks.koalas.frame import DataFrame
        from databricks import koalas as ks

        if not isinstance(func, types.FunctionType):
            assert callable(func), "the first argument should be a callable function."
            f = func
            func = lambda *args, **kwargs: f(*args, **kwargs)

        spec = inspect.getfullargspec(func)
        return_sig = spec.annotations.get("return", None)
        should_infer_schema = return_sig is None
        should_use_map_in_pandas = LooseVersion(pyspark.__version__) >= "3.0"

        original_func = func
        func = lambda o: original_func(o, *args, **kwds)

        self_applied = DataFrame(self._kdf._internal.resolved_copy)  # type: DataFrame

        if should_infer_schema:
            # Here we execute with the first 1000 to get the return type.
            # If the records were less than 1000, it uses pandas API directly for a shortcut.
            limit = ks.get_option("compute.shortcut_limit")
            pdf = self_applied.head(limit + 1)._to_internal_pandas()
            applied = func(pdf)
            if not isinstance(applied, pd.DataFrame):
                raise ValueError(
                    "The given function should return a frame; however, "
                    "the return type was %s." % type(applied)
                )
            kdf = ks.DataFrame(applied)  # type: DataFrame
            if len(pdf) <= limit:
                return kdf

            return_schema = force_decimal_precision_scale(
                as_nullable_spark_type(kdf._internal.to_internal_spark_frame.schema)
            )
            if should_use_map_in_pandas:
                output_func = GroupBy._make_pandas_df_builder_func(
                    self_applied, func, return_schema, retain_index=True
                )
                sdf = self_applied._internal.to_internal_spark_frame.mapInPandas(
                    lambda iterator: map(output_func, iterator), schema=return_schema
                )
            else:
                sdf = GroupBy._spark_group_map_apply(
                    self_applied, func, (F.spark_partition_id(),), return_schema, retain_index=True
                )

            # If schema is inferred, we can restore indexes too.
            internal = kdf._internal.with_new_sdf(sdf)
        else:
            return_type = infer_return_type(original_func)
            is_return_dataframe = isinstance(return_type, DataFrameType)
            if not is_return_dataframe:
                raise TypeError(
                    "The given function should specify a frame as its type "
                    "hints; however, the return type was %s." % return_sig
                )
            return_schema = cast(DataFrameType, return_type).spark_type

            if should_use_map_in_pandas:
                output_func = GroupBy._make_pandas_df_builder_func(
                    self_applied, func, return_schema, retain_index=False
                )
                sdf = self_applied._internal.to_internal_spark_frame.mapInPandas(
                    lambda iterator: map(output_func, iterator), schema=return_schema
                )
            else:
                sdf = GroupBy._spark_group_map_apply(
                    self_applied, func, (F.spark_partition_id(),), return_schema, retain_index=False
                )

            # Otherwise, it loses index.
            internal = InternalFrame(
                spark_frame=sdf,
                index_spark_columns=None,
                data_dtypes=cast(DataFrameType, return_type).dtypes,
            )

        return DataFrame(internal)

    def transform_batch(self, func, *args, **kwargs) -> Union["DataFrame", "Series"]:
        """
        Transform chunks with a function that takes pandas DataFrame and outputs pandas DataFrame.
        The pandas DataFrame given to the function is of a batch used internally. The length of
        each input and output should be the same.

        See also `Transform and apply a function
        <https://koalas.readthedocs.io/en/latest/user_guide/transform_apply.html>`_.

        .. note:: the `func` is unable to access to the whole input frame. Koalas internally
            splits the input series into multiple batches and calls `func` with each batch multiple
            times. Therefore, operations such as global aggregations are impossible. See the example
            below.

            >>> # This case does not return the length of whole frame but of the batch internally
            ... # used.
            ... def length(pdf) -> ks.DataFrame[int]:
            ...     return pd.DataFrame([len(pdf)] * len(pdf))
            ...
            >>> df = ks.DataFrame({'A': range(1000)})
            >>> df.koalas.transform_batch(length)  # doctest: +SKIP
                c0
            0   83
            1   83
            2   83
            ...

        .. note:: this API executes the function once to infer the type which is
            potentially expensive, for instance, when the dataset is created after
            aggregations or sorting.

            To avoid this, specify return type in ``func``, for instance, as below:

            >>> def plus_one(x) -> ks.DataFrame[float, float]:
            ...     return x + 1

            If the return type is specified, the output column names become
            `c0, c1, c2 ... cn`. These names are positionally mapped to the returned
            DataFrame in ``func``.

            To specify the column names, you can assign them in a pandas friendly style as below:

            >>> def plus_one(x) -> ks.DataFrame['a': float, 'b': float]:
            ...     return x + 1

            >>> pdf = pd.DataFrame({'a': [1, 2, 3], 'b': [3, 4, 5]})
            >>> def plus_one(x) -> ks.DataFrame[zip(pdf.dtypes, pdf.columns)]:
            ...     return x + 1

            When the given function returns DataFrame and has the return type annotated, the
            original index of the DataFrame will be lost and then a default index will be attached
            to the result. Please be careful about configuring the default index. See also
            `Default Index Type
            <https://koalas.readthedocs.io/en/latest/user_guide/options.html#default-index-type>`_.

        Parameters
        ----------
        func : function
            Function to transform each pandas frame.
        *args
            Positional arguments to pass to func.
        **kwargs
            Keyword arguments to pass to func.

        Returns
        -------
        DataFrame or Series

        See Also
        --------
        DataFrame.koalas.apply_batch: For row/columnwise operations.
        Series.koalas.transform_batch: transform the search as each pandas chunks.

        Examples
        --------
        >>> df = ks.DataFrame([(1, 2), (3, 4), (5, 6)], columns=['A', 'B'])
        >>> df
           A  B
        0  1  2
        1  3  4
        2  5  6

        >>> def plus_one_func(pdf) -> ks.DataFrame[int, int]:
        ...     return pdf + 1
        >>> df.koalas.transform_batch(plus_one_func)
           c0  c1
        0   2   3
        1   4   5
        2   6   7

        >>> def plus_one_func(pdf) -> ks.DataFrame['A': int, 'B': int]:
        ...     return pdf + 1
        >>> df.koalas.transform_batch(plus_one_func)
           A  B
        0  2  3
        1  4  5
        2  6  7

        >>> def plus_one_func(pdf) -> ks.Series[int]:
        ...     return pdf.B + 1
        >>> df.koalas.transform_batch(plus_one_func)
        0    3
        1    5
        2    7
        dtype: int64

        You can also omit the type hints so Koalas infers the return schema as below:

        >>> df.koalas.transform_batch(lambda pdf: pdf + 1)
           A  B
        0  2  3
        1  4  5
        2  6  7

        >>> (df * -1).koalas.transform_batch(abs)
           A  B
        0  1  2
        1  3  4
        2  5  6

        Note that you should not transform the index. The index information will not change.

        >>> df.koalas.transform_batch(lambda pdf: pdf.B + 1)
        0    3
        1    5
        2    7
        Name: B, dtype: int64

        You can also specify extra arguments as below.

        >>> df.koalas.transform_batch(lambda pdf, a, b, c: pdf.B + a + b + c, 1, 2, c=3)
        0     8
        1    10
        2    12
        Name: B, dtype: int64
        """
        from databricks.koalas.groupby import GroupBy
        from databricks.koalas.frame import DataFrame
        from databricks.koalas.series import first_series
        from databricks import koalas as ks

        assert callable(func), "the first argument should be a callable function."
        spec = inspect.getfullargspec(func)
        return_sig = spec.annotations.get("return", None)
        should_infer_schema = return_sig is None
        original_func = func
        func = lambda o: original_func(o, *args, **kwargs)

        names = self._kdf._internal.to_internal_spark_frame.schema.names
        should_by_pass = LooseVersion(pyspark.__version__) >= "3.0"

        def pandas_concat(series):
            # The input can only be a DataFrame for struct from Spark 3.0.
            # This works around to make the input as a frame. See SPARK-27240
            pdf = pd.concat(series, axis=1)
            pdf.columns = names
            return pdf

        def apply_func(pdf):
            return func(pdf).to_frame()

        def pandas_extract(pdf, name):
            # This is for output to work around a DataFrame for struct
            # from Spark 3.0.  See SPARK-23836
            return pdf[name]

        def pandas_series_func(f, by_pass):
            ff = f
            if by_pass:
                return lambda *series: first_series(ff(*series))
            else:
                return lambda *series: first_series(ff(pandas_concat(series)))

        def pandas_frame_func(f, field_name):
            ff = f
            return lambda *series: pandas_extract(ff(pandas_concat(series)), field_name)

        if should_infer_schema:
            # Here we execute with the first 1000 to get the return type.
            # If the records were less than 1000, it uses pandas API directly for a shortcut.
            limit = ks.get_option("compute.shortcut_limit")
            pdf = self._kdf.head(limit + 1)._to_internal_pandas()
            transformed = func(pdf)
            if not isinstance(transformed, (pd.DataFrame, pd.Series)):
                raise ValueError(
                    "The given function should return a frame; however, "
                    "the return type was %s." % type(transformed)
                )
            if len(transformed) != len(pdf):
                raise ValueError("transform_batch cannot produce aggregated results")
            kdf_or_kser = ks.from_pandas(transformed)

            if isinstance(kdf_or_kser, ks.Series):
                kser = cast(ks.Series, kdf_or_kser)

                spark_return_type = force_decimal_precision_scale(
                    as_nullable_spark_type(kser.spark.data_type)
                )
                return_schema = StructType(
                    [StructField(SPARK_DEFAULT_SERIES_NAME, spark_return_type)]
                )
                output_func = GroupBy._make_pandas_df_builder_func(
                    self._kdf, apply_func, return_schema, retain_index=False
                )

                pudf = pandas_udf(
                    pandas_series_func(output_func, should_by_pass),
                    returnType=spark_return_type,
                    functionType=PandasUDFType.SCALAR,
                )
                columns = self._kdf._internal.spark_columns
                # TODO: Index will be lost in this case.
                internal = self._kdf._internal.copy(
                    column_labels=kser._internal.column_labels,
                    data_spark_columns=[
                        (pudf(F.struct(*columns)) if should_by_pass else pudf(*columns)).alias(
                            kser._internal.data_spark_column_names[0]
                        )
                    ],
                    data_dtypes=kser._internal.data_dtypes,
                    column_label_names=kser._internal.column_label_names,
                )
                return first_series(DataFrame(internal))
            else:
                kdf = cast(DataFrame, kdf_or_kser)
                if len(pdf) <= limit:
                    # only do the short cut when it returns a frame to avoid
                    # operations on different dataframes in case of series.
                    return kdf

                # Force nullability.
                return_schema = force_decimal_precision_scale(
                    as_nullable_spark_type(kdf._internal.to_internal_spark_frame.schema)
                )

                self_applied = DataFrame(self._kdf._internal.resolved_copy)  # type: DataFrame

                output_func = GroupBy._make_pandas_df_builder_func(
                    self_applied, func, return_schema, retain_index=True
                )
                columns = self_applied._internal.spark_columns
                if should_by_pass:
                    pudf = pandas_udf(
                        output_func, returnType=return_schema, functionType=PandasUDFType.SCALAR
                    )
                    temp_struct_column = verify_temp_column_name(
                        self_applied._internal.spark_frame, "__temp_struct__"
                    )
                    applied = pudf(F.struct(*columns)).alias(temp_struct_column)
                    sdf = self_applied._internal.spark_frame.select(applied)
                    sdf = sdf.selectExpr("%s.*" % temp_struct_column)
                else:
                    applied = []
                    for field in return_schema.fields:
                        applied.append(
                            pandas_udf(
                                pandas_frame_func(output_func, field.name),
                                returnType=field.dataType,
                                functionType=PandasUDFType.SCALAR,
                            )(*columns).alias(field.name)
                        )
                    sdf = self_applied._internal.spark_frame.select(*applied)
                return DataFrame(kdf._internal.with_new_sdf(sdf))
        else:
            return_type = infer_return_type(original_func)
            is_return_series = isinstance(return_type, SeriesType)
            is_return_dataframe = isinstance(return_type, DataFrameType)
            if not is_return_dataframe and not is_return_series:
                raise TypeError(
                    "The given function should specify a frame or series as its type "
                    "hints; however, the return type was %s." % return_sig
                )
            if is_return_series:
                spark_return_type = force_decimal_precision_scale(
                    as_nullable_spark_type(cast(SeriesType, return_type).spark_type)
                )
                return_schema = StructType(
                    [StructField(SPARK_DEFAULT_SERIES_NAME, spark_return_type)]
                )
                output_func = GroupBy._make_pandas_df_builder_func(
                    self._kdf, apply_func, return_schema, retain_index=False
                )

                pudf = pandas_udf(
                    pandas_series_func(output_func, should_by_pass),
                    returnType=spark_return_type,
                    functionType=PandasUDFType.SCALAR,
                )
                columns = self._kdf._internal.spark_columns
                internal = self._kdf._internal.copy(
                    column_labels=[None],
                    data_spark_columns=[
                        (pudf(F.struct(*columns)) if should_by_pass else pudf(*columns)).alias(
                            SPARK_DEFAULT_SERIES_NAME
                        )
                    ],
                    data_dtypes=[cast(SeriesType, return_type).dtype],
                    column_label_names=None,
                )
                return first_series(DataFrame(internal))
            else:
                return_schema = cast(DataFrameType, return_type).spark_type

                self_applied = DataFrame(self._kdf._internal.resolved_copy)

                output_func = GroupBy._make_pandas_df_builder_func(
                    self_applied, func, return_schema, retain_index=False
                )
                columns = self_applied._internal.spark_columns

                if should_by_pass:
                    pudf = pandas_udf(
                        output_func, returnType=return_schema, functionType=PandasUDFType.SCALAR
                    )
                    temp_struct_column = verify_temp_column_name(
                        self_applied._internal.spark_frame, "__temp_struct__"
                    )
                    applied = pudf(F.struct(*columns)).alias(temp_struct_column)
                    sdf = self_applied._internal.spark_frame.select(applied)
                    sdf = sdf.selectExpr("%s.*" % temp_struct_column)
                else:
                    applied = []
                    for field in return_schema.fields:
                        applied.append(
                            pandas_udf(
                                pandas_frame_func(output_func, field.name),
                                returnType=field.dataType,
                                functionType=PandasUDFType.SCALAR,
                            )(*columns).alias(field.name)
                        )
                    sdf = self_applied._internal.spark_frame.select(*applied)
                internal = InternalFrame(
                    spark_frame=sdf,
                    index_spark_columns=None,
                    data_dtypes=cast(DataFrameType, return_type).dtypes,
                )
                return DataFrame(internal)


class KoalasSeriesMethods(object):
    """ Koalas specific features for Series. """

    def __init__(self, series: "Series"):
        self._kser = series

    def transform_batch(self, func, *args, **kwargs) -> "Series":
        """
        Transform the data with the function that takes pandas Series and outputs pandas Series.
        The pandas Series given to the function is of a batch used internally.

        See also `Transform and apply a function
        <https://koalas.readthedocs.io/en/latest/user_guide/transform_apply.html>`_.

        .. note:: the `func` is unable to access to the whole input series. Koalas internally
            splits the input series into multiple batches and calls `func` with each batch multiple
            times. Therefore, operations such as global aggregations are impossible. See the example
            below.

            >>> # This case does not return the length of whole frame but of the batch internally
            ... # used.
            ... def length(pser) -> ks.Series[int]:
            ...     return pd.Series([len(pser)] * len(pser))
            ...
            >>> df = ks.DataFrame({'A': range(1000)})
            >>> df.A.koalas.transform_batch(length)  # doctest: +SKIP
                c0
            0   83
            1   83
            2   83
            ...

        .. note:: this API executes the function once to infer the type which is
            potentially expensive, for instance, when the dataset is created after
            aggregations or sorting.

            To avoid this, specify return type in ``func``, for instance, as below:

            >>> def plus_one(x) -> ks.Series[int]:
            ...     return x + 1

        Parameters
        ----------
        func : function
            Function to apply to each pandas frame.
        *args
            Positional arguments to pass to func.
        **kwargs
            Keyword arguments to pass to func.

        Returns
        -------
        DataFrame

        See Also
        --------
        DataFrame.koalas.apply_batch : Similar but it takes pandas DataFrame as its internal batch.

        Examples
        --------
        >>> df = ks.DataFrame([(1, 2), (3, 4), (5, 6)], columns=['A', 'B'])
        >>> df
           A  B
        0  1  2
        1  3  4
        2  5  6

        >>> def plus_one_func(pser) -> ks.Series[np.int64]:
        ...     return pser + 1
        >>> df.A.koalas.transform_batch(plus_one_func)
        0    2
        1    4
        2    6
        Name: A, dtype: int64

        You can also omit the type hints so Koalas infers the return schema as below:

        >>> df.A.koalas.transform_batch(lambda pser: pser + 1)
        0    2
        1    4
        2    6
        Name: A, dtype: int64

        You can also specify extra arguments.

        >>> def plus_one_func(pser, a, b, c=3) -> ks.Series[np.int64]:
        ...     return pser + a + b + c
        >>> df.A.koalas.transform_batch(plus_one_func, 1, b=2)
        0     7
        1     9
        2    11
        Name: A, dtype: int64

        You can also use ``np.ufunc`` and built-in functions as input.

        >>> df.A.koalas.transform_batch(np.add, 10)
        0    11
        1    13
        2    15
        Name: A, dtype: int64

        >>> (df * -1).A.koalas.transform_batch(abs)
        0    1
        1    3
        2    5
        Name: A, dtype: int64
        """
        assert callable(func), "the first argument should be a callable function."

        return_sig = None
        try:
            spec = inspect.getfullargspec(func)
            return_sig = spec.annotations.get("return", None)
        except TypeError:
            # Falls back to schema inference if it fails to get signature.
            pass

        return_type = None
        if return_sig is not None:
            # Extract the signature arguments from this function.
            sig_return = infer_return_type(func)
            if not isinstance(sig_return, SeriesType):
                raise ValueError(
                    "Expected the return type of this function to be of type column,"
                    " but found type {}".format(sig_return)
                )
            return_type = cast(SeriesType, sig_return)

        return self._transform_batch(lambda c: func(c, *args, **kwargs), return_type)

    def _transform_batch(self, func, return_type: Optional[Union[SeriesType, ScalarType]]):
        from databricks.koalas.groupby import GroupBy
        from databricks.koalas.series import Series, first_series
        from databricks import koalas as ks

        if not isinstance(func, types.FunctionType):
            f = func
            func = lambda *args, **kwargs: f(*args, **kwargs)

        if return_type is None:
            # TODO: In this case, it avoids the shortcut for now (but only infers schema)
            #  because it returns a series from a different DataFrame and it has a different
            #  anchor. We should fix this to allow the shortcut or only allow to infer
            #  schema.
            limit = ks.get_option("compute.shortcut_limit")
            pser = self._kser.head(limit + 1)._to_internal_pandas()
            transformed = pser.transform(func)
            kser = Series(transformed)  # type: Series
            spark_return_type = force_decimal_precision_scale(
                as_nullable_spark_type(kser.spark.data_type)
            )
            dtype = kser.dtype
        else:
            spark_return_type = return_type.spark_type
            dtype = return_type.dtype

        kdf = self._kser.to_frame()
        columns = kdf._internal.spark_column_names

        def pandas_concat(series):
            # The input can only be a DataFrame for struct from Spark 3.0.
            # This works around to make the input as a frame. See SPARK-27240
            pdf = pd.concat(series, axis=1)
            pdf.columns = columns
            return pdf

        def apply_func(pdf):
            return func(first_series(pdf)).to_frame()

        return_schema = StructType([StructField(SPARK_DEFAULT_SERIES_NAME, spark_return_type)])
        output_func = GroupBy._make_pandas_df_builder_func(
            kdf, apply_func, return_schema, retain_index=False
        )

        pudf = pandas_udf(
            lambda *series: first_series(output_func(pandas_concat(series))),
            returnType=spark_return_type,
            functionType=PandasUDFType.SCALAR,
        )

        return self._kser._with_new_scol(
            scol=pudf(*kdf._internal.spark_columns).alias(
                self._kser._internal.spark_column_names[0]
            ),
            dtype=dtype,
        )
