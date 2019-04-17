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

import numpy as np


class _MissingPandasLikeDataFrame(object):

    def add(self, other, axis='columns', level=None, fill_value=None):
        """A stub for the equivalent method to `pd.DataFrame.add()`.

        The method `pd.DataFrame.add()` is not implemented yet.
        """
        raise NotImplementedError("The method `add()` is not implemented yet.")

    def add_prefix(self, prefix):
        """A stub for the equivalent method to `pd.DataFrame.add_prefix()`.

        The method `pd.DataFrame.add_prefix()` is not implemented yet.
        """
        raise NotImplementedError("The method `add_prefix()` is not implemented yet.")

    def add_suffix(self, suffix):
        """A stub for the equivalent method to `pd.DataFrame.add_suffix()`.

        The method `pd.DataFrame.add_suffix()` is not implemented yet.
        """
        raise NotImplementedError("The method `add_suffix()` is not implemented yet.")

    def agg(self, func, axis=0, *args, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.agg()`.

        The method `pd.DataFrame.agg()` is not implemented yet.
        """
        raise NotImplementedError("The method `agg()` is not implemented yet.")

    def aggregate(self, func, axis=0, *args, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.aggregate()`.

        The method `pd.DataFrame.aggregate()` is not implemented yet.
        """
        raise NotImplementedError("The method `aggregate()` is not implemented yet.")

    def align(self, other, join='outer', axis=None, level=None, copy=True, fill_value=None,
              method=None, limit=None, fill_axis=0, broadcast_axis=None):
        """A stub for the equivalent method to `pd.DataFrame.align()`.

        The method `pd.DataFrame.align()` is not implemented yet.
        """
        raise NotImplementedError("The method `align()` is not implemented yet.")

    def all(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.all()`.

        The method `pd.DataFrame.all()` is not implemented yet.
        """
        raise NotImplementedError("The method `all()` is not implemented yet.")

    def any(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.any()`.

        The method `pd.DataFrame.any()` is not implemented yet.
        """
        raise NotImplementedError("The method `any()` is not implemented yet.")

    def append(self, other, ignore_index=False, verify_integrity=False, sort=None):
        """A stub for the equivalent method to `pd.DataFrame.append()`.

        The method `pd.DataFrame.append()` is not implemented yet.
        """
        raise NotImplementedError("The method `append()` is not implemented yet.")

    def apply(self, func, axis=0, broadcast=None, raw=False, reduce=None, result_type=None,
              args=(), **kwds):
        """A stub for the equivalent method to `pd.DataFrame.apply()`.

        The method `pd.DataFrame.apply()` is not implemented yet.
        """
        raise NotImplementedError("The method `apply()` is not implemented yet.")

    def applymap(self, func):
        """A stub for the equivalent method to `pd.DataFrame.applymap()`.

        The method `pd.DataFrame.applymap()` is not implemented yet.
        """
        raise NotImplementedError("The method `applymap()` is not implemented yet.")

    def as_blocks(self, copy=True):
        """A stub for the equivalent method to `pd.DataFrame.as_blocks()`.

        The method `pd.DataFrame.as_blocks()` is not implemented yet.
        """
        raise NotImplementedError("The method `as_blocks()` is not implemented yet.")

    def as_matrix(self, columns=None):
        """A stub for the equivalent method to `pd.DataFrame.as_matrix()`.

        The method `pd.DataFrame.as_matrix()` is not implemented yet.
        """
        raise NotImplementedError("The method `as_matrix()` is not implemented yet.")

    def asfreq(self, freq, method=None, how=None, normalize=False, fill_value=None):
        """A stub for the equivalent method to `pd.DataFrame.asfreq()`.

        The method `pd.DataFrame.asfreq()` is not implemented yet.
        """
        raise NotImplementedError("The method `asfreq()` is not implemented yet.")

    def asof(self, where, subset=None):
        """A stub for the equivalent method to `pd.DataFrame.asof()`.

        The method `pd.DataFrame.asof()` is not implemented yet.
        """
        raise NotImplementedError("The method `asof()` is not implemented yet.")

    def astype(self, dtype, copy=True, errors='raise', **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.astype()`.

        The method `pd.DataFrame.astype()` is not implemented yet.
        """
        raise NotImplementedError("The method `astype()` is not implemented yet.")

    def at_time(self, time, asof=False, axis=None):
        """A stub for the equivalent method to `pd.DataFrame.at_time()`.

        The method `pd.DataFrame.at_time()` is not implemented yet.
        """
        raise NotImplementedError("The method `at_time()` is not implemented yet.")

    def between_time(self, start_time, end_time, include_start=True, include_end=True, axis=None):
        """A stub for the equivalent method to `pd.DataFrame.between_time()`.

        The method `pd.DataFrame.between_time()` is not implemented yet.
        """
        raise NotImplementedError("The method `between_time()` is not implemented yet.")

    def bfill(self, axis=None, inplace=False, limit=None, downcast=None):
        """A stub for the equivalent method to `pd.DataFrame.bfill()`.

        The method `pd.DataFrame.bfill()` is not implemented yet.
        """
        raise NotImplementedError("The method `bfill()` is not implemented yet.")

    def bool(self):
        """A stub for the equivalent method to `pd.DataFrame.bool()`.

        The method `pd.DataFrame.bool()` is not implemented yet.
        """
        raise NotImplementedError("The method `bool()` is not implemented yet.")

    def boxplot(self, column=None, by=None, ax=None, fontsize=None, rot=0, grid=True, figsize=None,
                layout=None, return_type=None, **kwds):
        """A stub for the equivalent method to `pd.DataFrame.boxplot()`.

        The method `pd.DataFrame.boxplot()` is not implemented yet.
        """
        raise NotImplementedError("The method `boxplot()` is not implemented yet.")

    def clip(self, lower=None, upper=None, axis=None, inplace=False, *args, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.clip()`.

        The method `pd.DataFrame.clip()` is not implemented yet.
        """
        raise NotImplementedError("The method `clip()` is not implemented yet.")

    def clip_lower(self, threshold, axis=None, inplace=False):
        """A stub for the equivalent method to `pd.DataFrame.clip_lower()`.

        The method `pd.DataFrame.clip_lower()` is not implemented yet.
        """
        raise NotImplementedError("The method `clip_lower()` is not implemented yet.")

    def clip_upper(self, threshold, axis=None, inplace=False):
        """A stub for the equivalent method to `pd.DataFrame.clip_upper()`.

        The method `pd.DataFrame.clip_upper()` is not implemented yet.
        """
        raise NotImplementedError("The method `clip_upper()` is not implemented yet.")

    def combine(self, other, func, fill_value=None, overwrite=True):
        """A stub for the equivalent method to `pd.DataFrame.combine()`.

        The method `pd.DataFrame.combine()` is not implemented yet.
        """
        raise NotImplementedError("The method `combine()` is not implemented yet.")

    def combine_first(self, other):
        """A stub for the equivalent method to `pd.DataFrame.combine_first()`.

        The method `pd.DataFrame.combine_first()` is not implemented yet.
        """
        raise NotImplementedError("The method `combine_first()` is not implemented yet.")

    def compound(self, axis=None, skipna=None, level=None):
        """A stub for the equivalent method to `pd.DataFrame.compound()`.

        The method `pd.DataFrame.compound()` is not implemented yet.
        """
        raise NotImplementedError("The method `compound()` is not implemented yet.")

    def convert_objects(self, convert_dates=True, convert_numeric=False, convert_timedeltas=True,
                        copy=True):
        """A stub for the equivalent method to `pd.DataFrame.convert_objects()`.

        The method `pd.DataFrame.convert_objects()` is not implemented yet.
        """
        raise NotImplementedError("The method `convert_objects()` is not implemented yet.")

    def corr(self, method='pearson', min_periods=1):
        """A stub for the equivalent method to `pd.DataFrame.corr()`.

        The method `pd.DataFrame.corr()` is not implemented yet.
        """
        raise NotImplementedError("The method `corr()` is not implemented yet.")

    def corrwith(self, other, axis=0, drop=False, method='pearson'):
        """A stub for the equivalent method to `pd.DataFrame.corrwith()`.

        The method `pd.DataFrame.corrwith()` is not implemented yet.
        """
        raise NotImplementedError("The method `corrwith()` is not implemented yet.")

    def cov(self, min_periods=None):
        """A stub for the equivalent method to `pd.DataFrame.cov()`.

        The method `pd.DataFrame.cov()` is not implemented yet.
        """
        raise NotImplementedError("The method `cov()` is not implemented yet.")

    def cummax(self, axis=None, skipna=True, *args, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.cummax()`.

        The method `pd.DataFrame.cummax()` is not implemented yet.
        """
        raise NotImplementedError("The method `cummax()` is not implemented yet.")

    def cummin(self, axis=None, skipna=True, *args, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.cummin()`.

        The method `pd.DataFrame.cummin()` is not implemented yet.
        """
        raise NotImplementedError("The method `cummin()` is not implemented yet.")

    def cumprod(self, axis=None, skipna=True, *args, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.cumprod()`.

        The method `pd.DataFrame.cumprod()` is not implemented yet.
        """
        raise NotImplementedError("The method `cumprod()` is not implemented yet.")

    def cumsum(self, axis=None, skipna=True, *args, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.cumsum()`.

        The method `pd.DataFrame.cumsum()` is not implemented yet.
        """
        raise NotImplementedError("The method `cumsum()` is not implemented yet.")

    def describe(self, percentiles=None, include=None, exclude=None):
        """A stub for the equivalent method to `pd.DataFrame.describe()`.

        The method `pd.DataFrame.describe()` is not implemented yet.
        """
        raise NotImplementedError("The method `describe()` is not implemented yet.")

    def diff(self, periods=1, axis=0):
        """A stub for the equivalent method to `pd.DataFrame.diff()`.

        The method `pd.DataFrame.diff()` is not implemented yet.
        """
        raise NotImplementedError("The method `diff()` is not implemented yet.")

    def div(self, other, axis='columns', level=None, fill_value=None):
        """A stub for the equivalent method to `pd.DataFrame.div()`.

        The method `pd.DataFrame.div()` is not implemented yet.
        """
        raise NotImplementedError("The method `div()` is not implemented yet.")

    def divide(self, other, axis='columns', level=None, fill_value=None):
        """A stub for the equivalent method to `pd.DataFrame.divide()`.

        The method `pd.DataFrame.divide()` is not implemented yet.
        """
        raise NotImplementedError("The method `divide()` is not implemented yet.")

    def dot(self, other):
        """A stub for the equivalent method to `pd.DataFrame.dot()`.

        The method `pd.DataFrame.dot()` is not implemented yet.
        """
        raise NotImplementedError("The method `dot()` is not implemented yet.")

    def drop_duplicates(self, subset=None, keep='first', inplace=False):
        """A stub for the equivalent method to `pd.DataFrame.drop_duplicates()`.

        The method `pd.DataFrame.drop_duplicates()` is not implemented yet.
        """
        raise NotImplementedError("The method `drop_duplicates()` is not implemented yet.")

    def droplevel(self, level, axis=0):
        """A stub for the equivalent method to `pd.DataFrame.droplevel()`.

        The method `pd.DataFrame.droplevel()` is not implemented yet.
        """
        raise NotImplementedError("The method `droplevel()` is not implemented yet.")

    def duplicated(self, subset=None, keep='first'):
        """A stub for the equivalent method to `pd.DataFrame.duplicated()`.

        The method `pd.DataFrame.duplicated()` is not implemented yet.
        """
        raise NotImplementedError("The method `duplicated()` is not implemented yet.")

    def eq(self, other, axis='columns', level=None):
        """A stub for the equivalent method to `pd.DataFrame.eq()`.

        The method `pd.DataFrame.eq()` is not implemented yet.
        """
        raise NotImplementedError("The method `eq()` is not implemented yet.")

    def equals(self, other):
        """A stub for the equivalent method to `pd.DataFrame.equals()`.

        The method `pd.DataFrame.equals()` is not implemented yet.
        """
        raise NotImplementedError("The method `equals()` is not implemented yet.")

    def eval(self, expr, inplace=False, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.eval()`.

        The method `pd.DataFrame.eval()` is not implemented yet.
        """
        raise NotImplementedError("The method `eval()` is not implemented yet.")

    def ewm(self, com=None, span=None, halflife=None, alpha=None, min_periods=0, adjust=True,
            ignore_na=False, axis=0):
        """A stub for the equivalent method to `pd.DataFrame.ewm()`.

        The method `pd.DataFrame.ewm()` is not implemented yet.
        """
        raise NotImplementedError("The method `ewm()` is not implemented yet.")

    def expanding(self, min_periods=1, center=False, axis=0):
        """A stub for the equivalent method to `pd.DataFrame.expanding()`.

        The method `pd.DataFrame.expanding()` is not implemented yet.
        """
        raise NotImplementedError("The method `expanding()` is not implemented yet.")

    def ffill(self, axis=None, inplace=False, limit=None, downcast=None):
        """A stub for the equivalent method to `pd.DataFrame.ffill()`.

        The method `pd.DataFrame.ffill()` is not implemented yet.
        """
        raise NotImplementedError("The method `ffill()` is not implemented yet.")

    def fillna(self, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None,
               **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.fillna()`.

        The method `pd.DataFrame.fillna()` is not implemented yet.
        """
        raise NotImplementedError("The method `fillna()` is not implemented yet.")

    def filter(self, items=None, like=None, regex=None, axis=None):
        """A stub for the equivalent method to `pd.DataFrame.filter()`.

        The method `pd.DataFrame.filter()` is not implemented yet.
        """
        raise NotImplementedError("The method `filter()` is not implemented yet.")

    def first(self, offset):
        """A stub for the equivalent method to `pd.DataFrame.first()`.

        The method `pd.DataFrame.first()` is not implemented yet.
        """
        raise NotImplementedError("The method `first()` is not implemented yet.")

    def first_valid_index(self):
        """A stub for the equivalent method to `pd.DataFrame.first_valid_index()`.

        The method `pd.DataFrame.first_valid_index()` is not implemented yet.
        """
        raise NotImplementedError("The method `first_valid_index()` is not implemented yet.")

    def floordiv(self, other, axis='columns', level=None, fill_value=None):
        """A stub for the equivalent method to `pd.DataFrame.floordiv()`.

        The method `pd.DataFrame.floordiv()` is not implemented yet.
        """
        raise NotImplementedError("The method `floordiv()` is not implemented yet.")

    def ge(self, other, axis='columns', level=None):
        """A stub for the equivalent method to `pd.DataFrame.ge()`.

        The method `pd.DataFrame.ge()` is not implemented yet.
        """
        raise NotImplementedError("The method `ge()` is not implemented yet.")

    def get_dtype_counts(self):
        """A stub for the equivalent method to `pd.DataFrame.get_dtype_counts()`.

        The method `pd.DataFrame.get_dtype_counts()` is not implemented yet.
        """
        raise NotImplementedError("The method `get_dtype_counts()` is not implemented yet.")

    def get_ftype_counts(self):
        """A stub for the equivalent method to `pd.DataFrame.get_ftype_counts()`.

        The method `pd.DataFrame.get_ftype_counts()` is not implemented yet.
        """
        raise NotImplementedError("The method `get_ftype_counts()` is not implemented yet.")

    def get_value(self, index, col, takeable=False):
        """A stub for the equivalent method to `pd.DataFrame.get_value()`.

        The method `pd.DataFrame.get_value()` is not implemented yet.
        """
        raise NotImplementedError("The method `get_value()` is not implemented yet.")

    def get_values(self):
        """A stub for the equivalent method to `pd.DataFrame.get_values()`.

        The method `pd.DataFrame.get_values()` is not implemented yet.
        """
        raise NotImplementedError("The method `get_values()` is not implemented yet.")

    def gt(self, other, axis='columns', level=None):
        """A stub for the equivalent method to `pd.DataFrame.gt()`.

        The method `pd.DataFrame.gt()` is not implemented yet.
        """
        raise NotImplementedError("The method `gt()` is not implemented yet.")

    def hist(data, column=None, by=None, grid=True, xlabelsize=None, xrot=None, ylabelsize=None,
             yrot=None, ax=None, sharex=False, sharey=False, figsize=None, layout=None, bins=10,
             **kwds):
        """A stub for the equivalent method to `pd.DataFrame.hist()`.

        The method `pd.DataFrame.hist()` is not implemented yet.
        """
        raise NotImplementedError("The method `hist()` is not implemented yet.")

    def idxmax(self, axis=0, skipna=True):
        """A stub for the equivalent method to `pd.DataFrame.idxmax()`.

        The method `pd.DataFrame.idxmax()` is not implemented yet.
        """
        raise NotImplementedError("The method `idxmax()` is not implemented yet.")

    def idxmin(self, axis=0, skipna=True):
        """A stub for the equivalent method to `pd.DataFrame.idxmin()`.

        The method `pd.DataFrame.idxmin()` is not implemented yet.
        """
        raise NotImplementedError("The method `idxmin()` is not implemented yet.")

    def infer_objects(self):
        """A stub for the equivalent method to `pd.DataFrame.infer_objects()`.

        The method `pd.DataFrame.infer_objects()` is not implemented yet.
        """
        raise NotImplementedError("The method `infer_objects()` is not implemented yet.")

    def info(self, verbose=None, buf=None, max_cols=None, memory_usage=None, null_counts=None):
        """A stub for the equivalent method to `pd.DataFrame.info()`.

        The method `pd.DataFrame.info()` is not implemented yet.
        """
        raise NotImplementedError("The method `info()` is not implemented yet.")

    def insert(self, loc, column, value, allow_duplicates=False):
        """A stub for the equivalent method to `pd.DataFrame.insert()`.

        The method `pd.DataFrame.insert()` is not implemented yet.
        """
        raise NotImplementedError("The method `insert()` is not implemented yet.")

    def interpolate(self, method='linear', axis=0, limit=None, inplace=False,
                    limit_direction='forward', limit_area=None, downcast=None, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.interpolate()`.

        The method `pd.DataFrame.interpolate()` is not implemented yet.
        """
        raise NotImplementedError("The method `interpolate()` is not implemented yet.")

    def isin(self, values):
        """A stub for the equivalent method to `pd.DataFrame.isin()`.

        The method `pd.DataFrame.isin()` is not implemented yet.
        """
        raise NotImplementedError("The method `isin()` is not implemented yet.")

    def items(self):
        """A stub for the equivalent method to `pd.DataFrame.items()`.

        The method `pd.DataFrame.items()` is not implemented yet.
        """
        raise NotImplementedError("The method `items()` is not implemented yet.")

    def iterrows(self):
        """A stub for the equivalent method to `pd.DataFrame.iterrows()`.

        The method `pd.DataFrame.iterrows()` is not implemented yet.
        """
        raise NotImplementedError("The method `iterrows()` is not implemented yet.")

    def itertuples(self, index=True, name='Pandas'):
        """A stub for the equivalent method to `pd.DataFrame.itertuples()`.

        The method `pd.DataFrame.itertuples()` is not implemented yet.
        """
        raise NotImplementedError("The method `itertuples()` is not implemented yet.")

    def join(self, other, on=None, how='left', lsuffix='', rsuffix='', sort=False):
        """A stub for the equivalent method to `pd.DataFrame.join()`.

        The method `pd.DataFrame.join()` is not implemented yet.
        """
        raise NotImplementedError("The method `join()` is not implemented yet.")

    def keys(self):
        """A stub for the equivalent method to `pd.DataFrame.keys()`.

        The method `pd.DataFrame.keys()` is not implemented yet.
        """
        raise NotImplementedError("The method `keys()` is not implemented yet.")

    def kurt(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.kurt()`.

        The method `pd.DataFrame.kurt()` is not implemented yet.
        """
        raise NotImplementedError("The method `kurt()` is not implemented yet.")

    def kurtosis(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.kurtosis()`.

        The method `pd.DataFrame.kurtosis()` is not implemented yet.
        """
        raise NotImplementedError("The method `kurtosis()` is not implemented yet.")

    def last(self, offset):
        """A stub for the equivalent method to `pd.DataFrame.last()`.

        The method `pd.DataFrame.last()` is not implemented yet.
        """
        raise NotImplementedError("The method `last()` is not implemented yet.")

    def last_valid_index(self):
        """A stub for the equivalent method to `pd.DataFrame.last_valid_index()`.

        The method `pd.DataFrame.last_valid_index()` is not implemented yet.
        """
        raise NotImplementedError("The method `last_valid_index()` is not implemented yet.")

    def le(self, other, axis='columns', level=None):
        """A stub for the equivalent method to `pd.DataFrame.le()`.

        The method `pd.DataFrame.le()` is not implemented yet.
        """
        raise NotImplementedError("The method `le()` is not implemented yet.")

    def lookup(self, row_labels, col_labels):
        """A stub for the equivalent method to `pd.DataFrame.lookup()`.

        The method `pd.DataFrame.lookup()` is not implemented yet.
        """
        raise NotImplementedError("The method `lookup()` is not implemented yet.")

    def lt(self, other, axis='columns', level=None):
        """A stub for the equivalent method to `pd.DataFrame.lt()`.

        The method `pd.DataFrame.lt()` is not implemented yet.
        """
        raise NotImplementedError("The method `lt()` is not implemented yet.")

    def mad(self, axis=None, skipna=None, level=None):
        """A stub for the equivalent method to `pd.DataFrame.mad()`.

        The method `pd.DataFrame.mad()` is not implemented yet.
        """
        raise NotImplementedError("The method `mad()` is not implemented yet.")

    def mask(self, cond, other=np.nan, inplace=False, axis=None, level=None, errors='raise',
             try_cast=False, raise_on_error=None):
        """A stub for the equivalent method to `pd.DataFrame.mask()`.

        The method `pd.DataFrame.mask()` is not implemented yet.
        """
        raise NotImplementedError("The method `mask()` is not implemented yet.")

    def mean(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.mean()`.

        The method `pd.DataFrame.mean()` is not implemented yet.
        """
        raise NotImplementedError("The method `mean()` is not implemented yet.")

    def median(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.median()`.

        The method `pd.DataFrame.median()` is not implemented yet.
        """
        raise NotImplementedError("The method `median()` is not implemented yet.")

    def melt(self, id_vars=None, value_vars=None, var_name=None, value_name='value',
             col_level=None):
        """A stub for the equivalent method to `pd.DataFrame.melt()`.

        The method `pd.DataFrame.melt()` is not implemented yet.
        """
        raise NotImplementedError("The method `melt()` is not implemented yet.")

    def memory_usage(self, index=True, deep=False):
        """A stub for the equivalent method to `pd.DataFrame.memory_usage()`.

        The method `pd.DataFrame.memory_usage()` is not implemented yet.
        """
        raise NotImplementedError("The method `memory_usage()` is not implemented yet.")

    def merge(self, right, how='inner', on=None, left_on=None, right_on=None, left_index=False,
              right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False,
              validate=None):
        """A stub for the equivalent method to `pd.DataFrame.merge()`.

        The method `pd.DataFrame.merge()` is not implemented yet.
        """
        raise NotImplementedError("The method `merge()` is not implemented yet.")

    def min(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.min()`.

        The method `pd.DataFrame.min()` is not implemented yet.
        """
        raise NotImplementedError("The method `min()` is not implemented yet.")

    def mod(self, other, axis='columns', level=None, fill_value=None):
        """A stub for the equivalent method to `pd.DataFrame.mod()`.

        The method `pd.DataFrame.mod()` is not implemented yet.
        """
        raise NotImplementedError("The method `mod()` is not implemented yet.")

    def mode(self, axis=0, numeric_only=False, dropna=True):
        """A stub for the equivalent method to `pd.DataFrame.mode()`.

        The method `pd.DataFrame.mode()` is not implemented yet.
        """
        raise NotImplementedError("The method `mode()` is not implemented yet.")

    def mul(self, other, axis='columns', level=None, fill_value=None):
        """A stub for the equivalent method to `pd.DataFrame.mul()`.

        The method `pd.DataFrame.mul()` is not implemented yet.
        """
        raise NotImplementedError("The method `mul()` is not implemented yet.")

    def multiply(self, other, axis='columns', level=None, fill_value=None):
        """A stub for the equivalent method to `pd.DataFrame.multiply()`.

        The method `pd.DataFrame.multiply()` is not implemented yet.
        """
        raise NotImplementedError("The method `multiply()` is not implemented yet.")

    def ne(self, other, axis='columns', level=None):
        """A stub for the equivalent method to `pd.DataFrame.ne()`.

        The method `pd.DataFrame.ne()` is not implemented yet.
        """
        raise NotImplementedError("The method `ne()` is not implemented yet.")

    def nlargest(self, n, columns, keep='first'):
        """A stub for the equivalent method to `pd.DataFrame.nlargest()`.

        The method `pd.DataFrame.nlargest()` is not implemented yet.
        """
        raise NotImplementedError("The method `nlargest()` is not implemented yet.")

    def nsmallest(self, n, columns, keep='first'):
        """A stub for the equivalent method to `pd.DataFrame.nsmallest()`.

        The method `pd.DataFrame.nsmallest()` is not implemented yet.
        """
        raise NotImplementedError("The method `nsmallest()` is not implemented yet.")

    def nunique(self, axis=0, dropna=True):
        """A stub for the equivalent method to `pd.DataFrame.nunique()`.

        The method `pd.DataFrame.nunique()` is not implemented yet.
        """
        raise NotImplementedError("The method `nunique()` is not implemented yet.")

    def pct_change(self, periods=1, fill_method='pad', limit=None, freq=None, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.pct_change()`.

        The method `pd.DataFrame.pct_change()` is not implemented yet.
        """
        raise NotImplementedError("The method `pct_change()` is not implemented yet.")

    def pivot(self, index=None, columns=None, values=None):
        """A stub for the equivalent method to `pd.DataFrame.pivot()`.

        The method `pd.DataFrame.pivot()` is not implemented yet.
        """
        raise NotImplementedError("The method `pivot()` is not implemented yet.")

    def pivot_table(self, values=None, index=None, columns=None, aggfunc='mean', fill_value=None,
                    margins=False, dropna=True, margins_name='All'):
        """A stub for the equivalent method to `pd.DataFrame.pivot_table()`.

        The method `pd.DataFrame.pivot_table()` is not implemented yet.
        """
        raise NotImplementedError("The method `pivot_table()` is not implemented yet.")

    def pop(self, item):
        """A stub for the equivalent method to `pd.DataFrame.pop()`.

        The method `pd.DataFrame.pop()` is not implemented yet.
        """
        raise NotImplementedError("The method `pop()` is not implemented yet.")

    def pow(self, other, axis='columns', level=None, fill_value=None):
        """A stub for the equivalent method to `pd.DataFrame.pow()`.

        The method `pd.DataFrame.pow()` is not implemented yet.
        """
        raise NotImplementedError("The method `pow()` is not implemented yet.")

    def prod(self, axis=None, skipna=None, level=None, numeric_only=None, min_count=0, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.prod()`.

        The method `pd.DataFrame.prod()` is not implemented yet.
        """
        raise NotImplementedError("The method `prod()` is not implemented yet.")

    def product(self, axis=None, skipna=None, level=None, numeric_only=None, min_count=0, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.product()`.

        The method `pd.DataFrame.product()` is not implemented yet.
        """
        raise NotImplementedError("The method `product()` is not implemented yet.")

    def quantile(self, q=0.5, axis=0, numeric_only=True, interpolation='linear'):
        """A stub for the equivalent method to `pd.DataFrame.quantile()`.

        The method `pd.DataFrame.quantile()` is not implemented yet.
        """
        raise NotImplementedError("The method `quantile()` is not implemented yet.")

    def query(self, expr, inplace=False, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.query()`.

        The method `pd.DataFrame.query()` is not implemented yet.
        """
        raise NotImplementedError("The method `query()` is not implemented yet.")

    def radd(self, other, axis='columns', level=None, fill_value=None):
        """A stub for the equivalent method to `pd.DataFrame.radd()`.

        The method `pd.DataFrame.radd()` is not implemented yet.
        """
        raise NotImplementedError("The method `radd()` is not implemented yet.")

    def rank(self, axis=0, method='average', numeric_only=None, na_option='keep', ascending=True,
             pct=False):
        """A stub for the equivalent method to `pd.DataFrame.rank()`.

        The method `pd.DataFrame.rank()` is not implemented yet.
        """
        raise NotImplementedError("The method `rank()` is not implemented yet.")

    def rdiv(self, other, axis='columns', level=None, fill_value=None):
        """A stub for the equivalent method to `pd.DataFrame.rdiv()`.

        The method `pd.DataFrame.rdiv()` is not implemented yet.
        """
        raise NotImplementedError("The method `rdiv()` is not implemented yet.")

    def reindex(self, labels=None, index=None, columns=None, axis=None, method=None, copy=True,
                level=None, fill_value=np.nan, limit=None, tolerance=None):
        """A stub for the equivalent method to `pd.DataFrame.reindex()`.

        The method `pd.DataFrame.reindex()` is not implemented yet.
        """
        raise NotImplementedError("The method `reindex()` is not implemented yet.")

    def reindex_axis(self, labels, axis=0, method=None, level=None, copy=True, limit=None,
                     fill_value=np.nan):
        """A stub for the equivalent method to `pd.DataFrame.reindex_axis()`.

        The method `pd.DataFrame.reindex_axis()` is not implemented yet.
        """
        raise NotImplementedError("The method `reindex_axis()` is not implemented yet.")

    def reindex_like(self, other, method=None, copy=True, limit=None, tolerance=None):
        """A stub for the equivalent method to `pd.DataFrame.reindex_like()`.

        The method `pd.DataFrame.reindex_like()` is not implemented yet.
        """
        raise NotImplementedError("The method `reindex_like()` is not implemented yet.")

    def rename(self, mapper=None, index=None, columns=None, axis=None, copy=True, inplace=False,
               level=None):
        """A stub for the equivalent method to `pd.DataFrame.rename()`.

        The method `pd.DataFrame.rename()` is not implemented yet.
        """
        raise NotImplementedError("The method `rename()` is not implemented yet.")

    def rename_axis(self, mapper=None, index=None, columns=None, axis=None, copy=True,
                    inplace=False):
        """A stub for the equivalent method to `pd.DataFrame.rename_axis()`.

        The method `pd.DataFrame.rename_axis()` is not implemented yet.
        """
        raise NotImplementedError("The method `rename_axis()` is not implemented yet.")

    def reorder_levels(self, order, axis=0):
        """A stub for the equivalent method to `pd.DataFrame.reorder_levels()`.

        The method `pd.DataFrame.reorder_levels()` is not implemented yet.
        """
        raise NotImplementedError("The method `reorder_levels()` is not implemented yet.")

    def replace(self, to_replace=None, value=None, inplace=False, limit=None, regex=False,
                method='pad'):
        """A stub for the equivalent method to `pd.DataFrame.replace()`.

        The method `pd.DataFrame.replace()` is not implemented yet.
        """
        raise NotImplementedError("The method `replace()` is not implemented yet.")

    def resample(self, rule, how=None, axis=0, fill_method=None, closed=None, label=None,
                 convention='start', kind=None, loffset=None, limit=None, base=0, on=None,
                 level=None):
        """A stub for the equivalent method to `pd.DataFrame.resample()`.

        The method `pd.DataFrame.resample()` is not implemented yet.
        """
        raise NotImplementedError("The method `resample()` is not implemented yet.")

    def rfloordiv(self, other, axis='columns', level=None, fill_value=None):
        """A stub for the equivalent method to `pd.DataFrame.rfloordiv()`.

        The method `pd.DataFrame.rfloordiv()` is not implemented yet.
        """
        raise NotImplementedError("The method `rfloordiv()` is not implemented yet.")

    def rmod(self, other, axis='columns', level=None, fill_value=None):
        """A stub for the equivalent method to `pd.DataFrame.rmod()`.

        The method `pd.DataFrame.rmod()` is not implemented yet.
        """
        raise NotImplementedError("The method `rmod()` is not implemented yet.")

    def rmul(self, other, axis='columns', level=None, fill_value=None):
        """A stub for the equivalent method to `pd.DataFrame.rmul()`.

        The method `pd.DataFrame.rmul()` is not implemented yet.
        """
        raise NotImplementedError("The method `rmul()` is not implemented yet.")

    def rolling(self, window, min_periods=None, center=False, win_type=None, on=None, axis=0,
                closed=None):
        """A stub for the equivalent method to `pd.DataFrame.rolling()`.

        The method `pd.DataFrame.rolling()` is not implemented yet.
        """
        raise NotImplementedError("The method `rolling()` is not implemented yet.")

    def round(self, decimals=0, *args, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.round()`.

        The method `pd.DataFrame.round()` is not implemented yet.
        """
        raise NotImplementedError("The method `round()` is not implemented yet.")

    def rpow(self, other, axis='columns', level=None, fill_value=None):
        """A stub for the equivalent method to `pd.DataFrame.rpow()`.

        The method `pd.DataFrame.rpow()` is not implemented yet.
        """
        raise NotImplementedError("The method `rpow()` is not implemented yet.")

    def rsub(self, other, axis='columns', level=None, fill_value=None):
        """A stub for the equivalent method to `pd.DataFrame.rsub()`.

        The method `pd.DataFrame.rsub()` is not implemented yet.
        """
        raise NotImplementedError("The method `rsub()` is not implemented yet.")

    def rtruediv(self, other, axis='columns', level=None, fill_value=None):
        """A stub for the equivalent method to `pd.DataFrame.rtruediv()`.

        The method `pd.DataFrame.rtruediv()` is not implemented yet.
        """
        raise NotImplementedError("The method `rtruediv()` is not implemented yet.")

    def sample(self, n=None, frac=None, replace=False, weights=None, random_state=None, axis=None):
        """A stub for the equivalent method to `pd.DataFrame.sample()`.

        The method `pd.DataFrame.sample()` is not implemented yet.
        """
        raise NotImplementedError("The method `sample()` is not implemented yet.")

    def select(self, crit, axis=0):
        """A stub for the equivalent method to `pd.DataFrame.select()`.

        The method `pd.DataFrame.select()` is not implemented yet.
        """
        raise NotImplementedError("The method `select()` is not implemented yet.")

    def select_dtypes(self, include=None, exclude=None):
        """A stub for the equivalent method to `pd.DataFrame.select_dtypes()`.

        The method `pd.DataFrame.select_dtypes()` is not implemented yet.
        """
        raise NotImplementedError("The method `select_dtypes()` is not implemented yet.")

    def sem(self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.sem()`.

        The method `pd.DataFrame.sem()` is not implemented yet.
        """
        raise NotImplementedError("The method `sem()` is not implemented yet.")

    def set_axis(self, labels, axis=0, inplace=None):
        """A stub for the equivalent method to `pd.DataFrame.set_axis()`.

        The method `pd.DataFrame.set_axis()` is not implemented yet.
        """
        raise NotImplementedError("The method `set_axis()` is not implemented yet.")

    def set_value(self, index, col, value, takeable=False):
        """A stub for the equivalent method to `pd.DataFrame.set_value()`.

        The method `pd.DataFrame.set_value()` is not implemented yet.
        """
        raise NotImplementedError("The method `set_value()` is not implemented yet.")

    def shift(self, periods=1, freq=None, axis=0, fill_value=None):
        """A stub for the equivalent method to `pd.DataFrame.shift()`.

        The method `pd.DataFrame.shift()` is not implemented yet.
        """
        raise NotImplementedError("The method `shift()` is not implemented yet.")

    def skew(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.skew()`.

        The method `pd.DataFrame.skew()` is not implemented yet.
        """
        raise NotImplementedError("The method `skew()` is not implemented yet.")

    def slice_shift(self, periods=1, axis=0):
        """A stub for the equivalent method to `pd.DataFrame.slice_shift()`.

        The method `pd.DataFrame.slice_shift()` is not implemented yet.
        """
        raise NotImplementedError("The method `slice_shift()` is not implemented yet.")

    def sort_index(self, axis=0, level=None, ascending=True, inplace=False, kind='quicksort',
                   na_position='last', sort_remaining=True, by=None):
        """A stub for the equivalent method to `pd.DataFrame.sort_index()`.

        The method `pd.DataFrame.sort_index()` is not implemented yet.
        """
        raise NotImplementedError("The method `sort_index()` is not implemented yet.")

    def squeeze(self, axis=None):
        """A stub for the equivalent method to `pd.DataFrame.squeeze()`.

        The method `pd.DataFrame.squeeze()` is not implemented yet.
        """
        raise NotImplementedError("The method `squeeze()` is not implemented yet.")

    def stack(self, level=-1, dropna=True):
        """A stub for the equivalent method to `pd.DataFrame.stack()`.

        The method `pd.DataFrame.stack()` is not implemented yet.
        """
        raise NotImplementedError("The method `stack()` is not implemented yet.")

    def std(self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.std()`.

        The method `pd.DataFrame.std()` is not implemented yet.
        """
        raise NotImplementedError("The method `std()` is not implemented yet.")

    def sub(self, other, axis='columns', level=None, fill_value=None):
        """A stub for the equivalent method to `pd.DataFrame.sub()`.

        The method `pd.DataFrame.sub()` is not implemented yet.
        """
        raise NotImplementedError("The method `sub()` is not implemented yet.")

    def subtract(self, other, axis='columns', level=None, fill_value=None):
        """A stub for the equivalent method to `pd.DataFrame.subtract()`.

        The method `pd.DataFrame.subtract()` is not implemented yet.
        """
        raise NotImplementedError("The method `subtract()` is not implemented yet.")

    def sum(self, axis=None, skipna=None, level=None, numeric_only=None, min_count=0, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.sum()`.

        The method `pd.DataFrame.sum()` is not implemented yet.
        """
        raise NotImplementedError("The method `sum()` is not implemented yet.")

    def swapaxes(self, axis1, axis2, copy=True):
        """A stub for the equivalent method to `pd.DataFrame.swapaxes()`.

        The method `pd.DataFrame.swapaxes()` is not implemented yet.
        """
        raise NotImplementedError("The method `swapaxes()` is not implemented yet.")

    def swaplevel(self, i=-2, j=-1, axis=0):
        """A stub for the equivalent method to `pd.DataFrame.swaplevel()`.

        The method `pd.DataFrame.swaplevel()` is not implemented yet.
        """
        raise NotImplementedError("The method `swaplevel()` is not implemented yet.")

    def tail(self, n=5):
        """A stub for the equivalent method to `pd.DataFrame.tail()`.

        The method `pd.DataFrame.tail()` is not implemented yet.
        """
        raise NotImplementedError("The method `tail()` is not implemented yet.")

    def take(self, indices, axis=0, convert=None, is_copy=True, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.take()`.

        The method `pd.DataFrame.take()` is not implemented yet.
        """
        raise NotImplementedError("The method `take()` is not implemented yet.")

    def to_clipboard(self, excel=True, sep=None, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.to_clipboard()`.

        The method `pd.DataFrame.to_clipboard()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_clipboard()` is not implemented yet.")

    def to_csv(self, path_or_buf=None, sep=',', na_rep='', float_format=None, columns=None,
               header=True, index=True, index_label=None, mode='w', encoding=None,
               compression='infer', quoting=None, quotechar='"', line_terminator=None,
               chunksize=None, tupleize_cols=None, date_format=None, doublequote=True,
               escapechar=None, decimal='.'):
        """A stub for the equivalent method to `pd.DataFrame.to_csv()`.

        The method `pd.DataFrame.to_csv()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_csv()` is not implemented yet.")

    def to_dense(self):
        """A stub for the equivalent method to `pd.DataFrame.to_dense()`.

        The method `pd.DataFrame.to_dense()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_dense()` is not implemented yet.")

    def to_dict(self, orient='dict', into=dict):
        """A stub for the equivalent method to `pd.DataFrame.to_dict()`.

        The method `pd.DataFrame.to_dict()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_dict()` is not implemented yet.")

    def to_excel(self, excel_writer, sheet_name='Sheet1', na_rep='', float_format=None,
                 columns=None, header=True, index=True, index_label=None, startrow=0, startcol=0,
                 engine=None, merge_cells=True, encoding=None, inf_rep='inf', verbose=True,
                 freeze_panes=None):
        """A stub for the equivalent method to `pd.DataFrame.to_excel()`.

        The method `pd.DataFrame.to_excel()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_excel()` is not implemented yet.")

    def to_feather(self, fname):
        """A stub for the equivalent method to `pd.DataFrame.to_feather()`.

        The method `pd.DataFrame.to_feather()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_feather()` is not implemented yet.")

    def to_gbq(self, destination_table, project_id=None, chunksize=None, reauth=False,
               if_exists='fail', auth_local_webserver=False, table_schema=None, location=None,
               progress_bar=True, credentials=None, verbose=None, private_key=None):
        """A stub for the equivalent method to `pd.DataFrame.to_gbq()`.

        The method `pd.DataFrame.to_gbq()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_gbq()` is not implemented yet.")

    def to_hdf(self, path_or_buf, key, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.to_hdf()`.

        The method `pd.DataFrame.to_hdf()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_hdf()` is not implemented yet.")

    def to_json(self, path_or_buf=None, orient=None, date_format=None, double_precision=10,
                force_ascii=True, date_unit='ms', default_handler=None, lines=False,
                compression='infer', index=True):
        """A stub for the equivalent method to `pd.DataFrame.to_json()`.

        The method `pd.DataFrame.to_json()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_json()` is not implemented yet.")

    def to_latex(self, buf=None, columns=None, col_space=None, header=True, index=True,
                 na_rep='NaN', formatters=None, float_format=None, sparsify=None, index_names=True,
                 bold_rows=False, column_format=None, longtable=None, escape=None, encoding=None,
                 decimal='.', multicolumn=None, multicolumn_format=None, multirow=None):
        """A stub for the equivalent method to `pd.DataFrame.to_latex()`.

        The method `pd.DataFrame.to_latex()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_latex()` is not implemented yet.")

    def to_msgpack(self, path_or_buf=None, encoding='utf-8', **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.to_msgpack()`.

        The method `pd.DataFrame.to_msgpack()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_msgpack()` is not implemented yet.")

    def to_numpy(self, dtype=None, copy=False):
        """A stub for the equivalent method to `pd.DataFrame.to_numpy()`.

        The method `pd.DataFrame.to_numpy()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_numpy()` is not implemented yet.")

    def to_panel(self):
        """A stub for the equivalent method to `pd.DataFrame.to_panel()`.

        The method `pd.DataFrame.to_panel()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_panel()` is not implemented yet.")

    def to_parquet(self, fname, engine='auto', compression='snappy', index=None,
                   partition_cols=None, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.to_parquet()`.

        The method `pd.DataFrame.to_parquet()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_parquet()` is not implemented yet.")

    def to_period(self, freq=None, axis=0, copy=True):
        """A stub for the equivalent method to `pd.DataFrame.to_period()`.

        The method `pd.DataFrame.to_period()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_period()` is not implemented yet.")

    def to_pickle(self, path, compression='infer', protocol=4):
        """A stub for the equivalent method to `pd.DataFrame.to_pickle()`.

        The method `pd.DataFrame.to_pickle()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_pickle()` is not implemented yet.")

    def to_records(self, index=True, convert_datetime64=None, column_dtypes=None,
                   index_dtypes=None):
        """A stub for the equivalent method to `pd.DataFrame.to_records()`.

        The method `pd.DataFrame.to_records()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_records()` is not implemented yet.")

    def to_sparse(self, fill_value=None, kind='block'):
        """A stub for the equivalent method to `pd.DataFrame.to_sparse()`.

        The method `pd.DataFrame.to_sparse()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_sparse()` is not implemented yet.")

    def to_sql(self, name, con, schema=None, if_exists='fail', index=True, index_label=None,
               chunksize=None, dtype=None, method=None):
        """A stub for the equivalent method to `pd.DataFrame.to_sql()`.

        The method `pd.DataFrame.to_sql()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_sql()` is not implemented yet.")

    def to_stata(self, fname, convert_dates=None, write_index=True, encoding='latin-1',
                 byteorder=None, time_stamp=None, data_label=None, variable_labels=None,
                 version=114, convert_strl=None):
        """A stub for the equivalent method to `pd.DataFrame.to_stata()`.

        The method `pd.DataFrame.to_stata()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_stata()` is not implemented yet.")

    def to_string(self, buf=None, columns=None, col_space=None, header=True, index=True,
                  na_rep='NaN', formatters=None, float_format=None, sparsify=None,
                  index_names=True, justify=None, max_rows=None, max_cols=None,
                  show_dimensions=False, decimal='.', line_width=None):
        """A stub for the equivalent method to `pd.DataFrame.to_string()`.

        The method `pd.DataFrame.to_string()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_string()` is not implemented yet.")

    def to_timestamp(self, freq=None, how='start', axis=0, copy=True):
        """A stub for the equivalent method to `pd.DataFrame.to_timestamp()`.

        The method `pd.DataFrame.to_timestamp()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_timestamp()` is not implemented yet.")

    def to_xarray(self):
        """A stub for the equivalent method to `pd.DataFrame.to_xarray()`.

        The method `pd.DataFrame.to_xarray()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_xarray()` is not implemented yet.")

    def transform(self, func, axis=0, *args, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.transform()`.

        The method `pd.DataFrame.transform()` is not implemented yet.
        """
        raise NotImplementedError("The method `transform()` is not implemented yet.")

    def transpose(self, *args, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.transpose()`.

        The method `pd.DataFrame.transpose()` is not implemented yet.
        """
        raise NotImplementedError("The method `transpose()` is not implemented yet.")

    def truediv(self, other, axis='columns', level=None, fill_value=None):
        """A stub for the equivalent method to `pd.DataFrame.truediv()`.

        The method `pd.DataFrame.truediv()` is not implemented yet.
        """
        raise NotImplementedError("The method `truediv()` is not implemented yet.")

    def truncate(self, before=None, after=None, axis=None, copy=True):
        """A stub for the equivalent method to `pd.DataFrame.truncate()`.

        The method `pd.DataFrame.truncate()` is not implemented yet.
        """
        raise NotImplementedError("The method `truncate()` is not implemented yet.")

    def tshift(self, periods=1, freq=None, axis=0):
        """A stub for the equivalent method to `pd.DataFrame.tshift()`.

        The method `pd.DataFrame.tshift()` is not implemented yet.
        """
        raise NotImplementedError("The method `tshift()` is not implemented yet.")

    def tz_convert(self, tz, axis=0, level=None, copy=True):
        """A stub for the equivalent method to `pd.DataFrame.tz_convert()`.

        The method `pd.DataFrame.tz_convert()` is not implemented yet.
        """
        raise NotImplementedError("The method `tz_convert()` is not implemented yet.")

    def tz_localize(self, tz, axis=0, level=None, copy=True, ambiguous='raise',
                    nonexistent='raise'):
        """A stub for the equivalent method to `pd.DataFrame.tz_localize()`.

        The method `pd.DataFrame.tz_localize()` is not implemented yet.
        """
        raise NotImplementedError("The method `tz_localize()` is not implemented yet.")

    def unstack(self, level=-1, fill_value=None):
        """A stub for the equivalent method to `pd.DataFrame.unstack()`.

        The method `pd.DataFrame.unstack()` is not implemented yet.
        """
        raise NotImplementedError("The method `unstack()` is not implemented yet.")

    def update(self, other, join='left', overwrite=True, filter_func=None, errors='ignore'):
        """A stub for the equivalent method to `pd.DataFrame.update()`.

        The method `pd.DataFrame.update()` is not implemented yet.
        """
        raise NotImplementedError("The method `update()` is not implemented yet.")

    def var(self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs):
        """A stub for the equivalent method to `pd.DataFrame.var()`.

        The method `pd.DataFrame.var()` is not implemented yet.
        """
        raise NotImplementedError("The method `var()` is not implemented yet.")

    def where(self, cond, other=np.nan, inplace=False, axis=None, level=None, errors='raise',
              try_cast=False, raise_on_error=None):
        """A stub for the equivalent method to `pd.DataFrame.where()`.

        The method `pd.DataFrame.where()` is not implemented yet.
        """
        raise NotImplementedError("The method `where()` is not implemented yet.")

    def xs(self, key, axis=0, level=None, drop_level=True):
        """A stub for the equivalent method to `pd.DataFrame.xs()`.

        The method `pd.DataFrame.xs()` is not implemented yet.
        """
        raise NotImplementedError("The method `xs()` is not implemented yet.")
