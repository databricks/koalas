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
A base class to be monkey-patched to SparkSession to behave similar to pandas package.
"""
import pandas as pd
from pyspark.sql.types import StructType

from databricks.koalas.dask.compatibility import string_types
from databricks.koalas.metadata import Metadata
from databricks.koalas.series import _col


class SparkSessionPatches(object):
    """
    Methods for :class:`SparkSession`.
    """

    def from_pandas(self, pdf):
        if isinstance(pdf, pd.Series):
            return _col(self.from_pandas(pd.DataFrame(pdf)))
        metadata = Metadata.from_pandas(pdf)
        reset_index = pdf.reset_index()
        reset_index.columns = metadata.all_fields
        df = self.createDataFrame(reset_index)
        df._metadata = metadata
        return df

    def read_csv(self, path, header='infer', names=None, usecols=None,
                 mangle_dupe_cols=True, parse_dates=False, comment=None):
        if mangle_dupe_cols is not True:
            raise ValueError("mangle_dupe_cols can only be `True`: %s" % mangle_dupe_cols)
        if parse_dates is not False:
            raise ValueError("parse_dates can only be `False`: %s" % parse_dates)

        if usecols is not None and not callable(usecols):
            usecols = list(usecols)
        if usecols is None or callable(usecols) or len(usecols) > 0:
            reader = self.read.option("inferSchema", "true")

            if header == 'infer':
                header = 0 if names is None else None
            if header == 0:
                reader.option("header", True)
            elif header is None:
                reader.option("header", False)
            else:
                raise ValueError("Unknown header argument {}".format(header))

            if comment is not None:
                if not isinstance(comment, string_types) or len(comment) != 1:
                    raise ValueError("Only length-1 comment characters supported")
                reader.option("comment", comment)

            df = reader.csv(path)

            if header is None:
                df = df._spark_selectExpr(*["`%s` as `%s`" % (field.name, i)
                                            for i, field in enumerate(df.schema)])
            if names is not None:
                names = list(names)
                if len(set(names)) != len(names):
                    raise ValueError('Found non-unique column index')
                if len(names) != len(df.schema):
                    raise ValueError('Names do not match the number of columns: %d' % len(names))
                df = df._spark_selectExpr(*["`%s` as `%s`" % (field.name, name)
                                            for field, name in zip(df.schema, names)])

            if usecols is not None:
                if callable(usecols):
                    cols = [field.name for field in df.schema if usecols(field.name)]
                    missing = []
                elif all(isinstance(col, int) for col in usecols):
                    cols = [field.name for i, field in enumerate(df.schema) if i in usecols]
                    missing = [col for col in usecols
                               if col >= len(df.schema) or df.schema[col].name not in cols]
                elif all(isinstance(col, string_types) for col in usecols):
                    cols = [field.name for field in df.schema if field.name in usecols]
                    missing = [col for col in usecols if col not in cols]
                else:
                    raise ValueError("'usecols' must either be list-like of all strings, "
                                     "all unicode, all integers or a callable.")
                if len(missing) > 0:
                    raise ValueError('Usecols do not match columns, columns expected but not '
                                     'found: %s' % missing)

                if len(cols) > 0:
                    df = df._spark_select(cols)
                else:
                    df = self.createDataFrame([], schema=StructType())
        else:
            df = self.createDataFrame([], schema=StructType())
        return df

    def read_parquet(self, path, columns=None):
        if columns is not None:
            columns = list(columns)
        if columns is None or len(columns) > 0:
            df = self.read.parquet(path)
            if columns is not None:
                fields = [field.name for field in df.schema]
                cols = [col for col in columns if col in fields]
                if len(cols) > 0:
                    df = df._spark_select(cols)
                else:
                    df = self.createDataFrame([], schema=StructType())
        else:
            df = self.createDataFrame([], schema=StructType())
        return df
