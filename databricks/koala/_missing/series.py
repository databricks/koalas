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


class _MissingPandasLikeSeries(object):

    def add(self, other, level=None, fill_value=None, axis=0):
        """A stub for the equivalent method to `pd.Series.add()`.

        The method `pd.Series.add()` is not implemented yet.
        """
        raise NotImplementedError("The method `add()` is not implemented yet.")

    def add_prefix(self, prefix):
        """A stub for the equivalent method to `pd.Series.add_prefix()`.

        The method `pd.Series.add_prefix()` is not implemented yet.
        """
        raise NotImplementedError("The method `add_prefix()` is not implemented yet.")

    def add_suffix(self, suffix):
        """A stub for the equivalent method to `pd.Series.add_suffix()`.

        The method `pd.Series.add_suffix()` is not implemented yet.
        """
        raise NotImplementedError("The method `add_suffix()` is not implemented yet.")

    def agg(self, func, axis=0, *args, **kwargs):
        """A stub for the equivalent method to `pd.Series.agg()`.

        The method `pd.Series.agg()` is not implemented yet.
        """
        raise NotImplementedError("The method `agg()` is not implemented yet.")

    def aggregate(self, func, axis=0, *args, **kwargs):
        """A stub for the equivalent method to `pd.Series.aggregate()`.

        The method `pd.Series.aggregate()` is not implemented yet.
        """
        raise NotImplementedError("The method `aggregate()` is not implemented yet.")

    def align(self, other, join='outer', axis=None, level=None, copy=True, fill_value=None,
              method=None, limit=None, fill_axis=0, broadcast_axis=None):
        """A stub for the equivalent method to `pd.Series.align()`.

        The method `pd.Series.align()` is not implemented yet.
        """
        raise NotImplementedError("The method `align()` is not implemented yet.")

    def all(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        """A stub for the equivalent method to `pd.Series.all()`.

        The method `pd.Series.all()` is not implemented yet.
        """
        raise NotImplementedError("The method `all()` is not implemented yet.")

    def any(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        """A stub for the equivalent method to `pd.Series.any()`.

        The method `pd.Series.any()` is not implemented yet.
        """
        raise NotImplementedError("The method `any()` is not implemented yet.")

    def append(self, to_append, ignore_index=False, verify_integrity=False):
        """A stub for the equivalent method to `pd.Series.append()`.

        The method `pd.Series.append()` is not implemented yet.
        """
        raise NotImplementedError("The method `append()` is not implemented yet.")

    def apply(self, func, convert_dtype=True, args=(), **kwds):
        """A stub for the equivalent method to `pd.Series.apply()`.

        The method `pd.Series.apply()` is not implemented yet.
        """
        raise NotImplementedError("The method `apply()` is not implemented yet.")

    def argmax(self, axis=0, skipna=True, *args, **kwargs):
        """A stub for the equivalent method to `pd.Series.argmax()`.

        The method `pd.Series.argmax()` is not implemented yet.
        """
        raise NotImplementedError("The method `argmax()` is not implemented yet.")

    def argmin(self, axis=0, skipna=True, *args, **kwargs):
        """A stub for the equivalent method to `pd.Series.argmin()`.

        The method `pd.Series.argmin()` is not implemented yet.
        """
        raise NotImplementedError("The method `argmin()` is not implemented yet.")

    def argsort(self, axis=0, kind='quicksort', order=None):
        """A stub for the equivalent method to `pd.Series.argsort()`.

        The method `pd.Series.argsort()` is not implemented yet.
        """
        raise NotImplementedError("The method `argsort()` is not implemented yet.")

    def as_blocks(self, copy=True):
        """A stub for the equivalent method to `pd.Series.as_blocks()`.

        The method `pd.Series.as_blocks()` is not implemented yet.
        """
        raise NotImplementedError("The method `as_blocks()` is not implemented yet.")

    def as_matrix(self, columns=None):
        """A stub for the equivalent method to `pd.Series.as_matrix()`.

        The method `pd.Series.as_matrix()` is not implemented yet.
        """
        raise NotImplementedError("The method `as_matrix()` is not implemented yet.")

    def asfreq(self, freq, method=None, how=None, normalize=False, fill_value=None):
        """A stub for the equivalent method to `pd.Series.asfreq()`.

        The method `pd.Series.asfreq()` is not implemented yet.
        """
        raise NotImplementedError("The method `asfreq()` is not implemented yet.")

    def asof(self, where, subset=None):
        """A stub for the equivalent method to `pd.Series.asof()`.

        The method `pd.Series.asof()` is not implemented yet.
        """
        raise NotImplementedError("The method `asof()` is not implemented yet.")

    def at_time(self, time, asof=False, axis=None):
        """A stub for the equivalent method to `pd.Series.at_time()`.

        The method `pd.Series.at_time()` is not implemented yet.
        """
        raise NotImplementedError("The method `at_time()` is not implemented yet.")

    def autocorr(self, lag=1):
        """A stub for the equivalent method to `pd.Series.autocorr()`.

        The method `pd.Series.autocorr()` is not implemented yet.
        """
        raise NotImplementedError("The method `autocorr()` is not implemented yet.")

    def between(self, left, right, inclusive=True):
        """A stub for the equivalent method to `pd.Series.between()`.

        The method `pd.Series.between()` is not implemented yet.
        """
        raise NotImplementedError("The method `between()` is not implemented yet.")

    def between_time(self, start_time, end_time, include_start=True, include_end=True, axis=None):
        """A stub for the equivalent method to `pd.Series.between_time()`.

        The method `pd.Series.between_time()` is not implemented yet.
        """
        raise NotImplementedError("The method `between_time()` is not implemented yet.")

    def bfill(self, axis=None, inplace=False, limit=None, downcast=None):
        """A stub for the equivalent method to `pd.Series.bfill()`.

        The method `pd.Series.bfill()` is not implemented yet.
        """
        raise NotImplementedError("The method `bfill()` is not implemented yet.")

    def bool(self):
        """A stub for the equivalent method to `pd.Series.bool()`.

        The method `pd.Series.bool()` is not implemented yet.
        """
        raise NotImplementedError("The method `bool()` is not implemented yet.")

    def clip(self, lower=None, upper=None, axis=None, inplace=False, *args, **kwargs):
        """A stub for the equivalent method to `pd.Series.clip()`.

        The method `pd.Series.clip()` is not implemented yet.
        """
        raise NotImplementedError("The method `clip()` is not implemented yet.")

    def clip_lower(self, threshold, axis=None, inplace=False):
        """A stub for the equivalent method to `pd.Series.clip_lower()`.

        The method `pd.Series.clip_lower()` is not implemented yet.
        """
        raise NotImplementedError("The method `clip_lower()` is not implemented yet.")

    def clip_upper(self, threshold, axis=None, inplace=False):
        """A stub for the equivalent method to `pd.Series.clip_upper()`.

        The method `pd.Series.clip_upper()` is not implemented yet.
        """
        raise NotImplementedError("The method `clip_upper()` is not implemented yet.")

    def combine(self, other, func, fill_value=None):
        """A stub for the equivalent method to `pd.Series.combine()`.

        The method `pd.Series.combine()` is not implemented yet.
        """
        raise NotImplementedError("The method `combine()` is not implemented yet.")

    def combine_first(self, other):
        """A stub for the equivalent method to `pd.Series.combine_first()`.

        The method `pd.Series.combine_first()` is not implemented yet.
        """
        raise NotImplementedError("The method `combine_first()` is not implemented yet.")

    def compound(self, axis=None, skipna=None, level=None):
        """A stub for the equivalent method to `pd.Series.compound()`.

        The method `pd.Series.compound()` is not implemented yet.
        """
        raise NotImplementedError("The method `compound()` is not implemented yet.")

    def compress(self, condition, *args, **kwargs):
        """A stub for the equivalent method to `pd.Series.compress()`.

        The method `pd.Series.compress()` is not implemented yet.
        """
        raise NotImplementedError("The method `compress()` is not implemented yet.")

    def convert_objects(self, convert_dates=True, convert_numeric=False, convert_timedeltas=True,
                        copy=True):
        """A stub for the equivalent method to `pd.Series.convert_objects()`.

        The method `pd.Series.convert_objects()` is not implemented yet.
        """
        raise NotImplementedError("The method `convert_objects()` is not implemented yet.")

    def copy(self, deep=True):
        """A stub for the equivalent method to `pd.Series.copy()`.

        The method `pd.Series.copy()` is not implemented yet.
        """
        raise NotImplementedError("The method `copy()` is not implemented yet.")

    def corr(self, other, method='pearson', min_periods=None):
        """A stub for the equivalent method to `pd.Series.corr()`.

        The method `pd.Series.corr()` is not implemented yet.
        """
        raise NotImplementedError("The method `corr()` is not implemented yet.")

    def count(self, level=None):
        """A stub for the equivalent method to `pd.Series.count()`.

        The method `pd.Series.count()` is not implemented yet.
        """
        raise NotImplementedError("The method `count()` is not implemented yet.")

    def cov(self, other, min_periods=None):
        """A stub for the equivalent method to `pd.Series.cov()`.

        The method `pd.Series.cov()` is not implemented yet.
        """
        raise NotImplementedError("The method `cov()` is not implemented yet.")

    def cummax(self, axis=None, skipna=True, *args, **kwargs):
        """A stub for the equivalent method to `pd.Series.cummax()`.

        The method `pd.Series.cummax()` is not implemented yet.
        """
        raise NotImplementedError("The method `cummax()` is not implemented yet.")

    def cummin(self, axis=None, skipna=True, *args, **kwargs):
        """A stub for the equivalent method to `pd.Series.cummin()`.

        The method `pd.Series.cummin()` is not implemented yet.
        """
        raise NotImplementedError("The method `cummin()` is not implemented yet.")

    def cumprod(self, axis=None, skipna=True, *args, **kwargs):
        """A stub for the equivalent method to `pd.Series.cumprod()`.

        The method `pd.Series.cumprod()` is not implemented yet.
        """
        raise NotImplementedError("The method `cumprod()` is not implemented yet.")

    def cumsum(self, axis=None, skipna=True, *args, **kwargs):
        """A stub for the equivalent method to `pd.Series.cumsum()`.

        The method `pd.Series.cumsum()` is not implemented yet.
        """
        raise NotImplementedError("The method `cumsum()` is not implemented yet.")

    def describe(self, percentiles=None, include=None, exclude=None):
        """A stub for the equivalent method to `pd.Series.describe()`.

        The method `pd.Series.describe()` is not implemented yet.
        """
        raise NotImplementedError("The method `describe()` is not implemented yet.")

    def diff(self, periods=1):
        """A stub for the equivalent method to `pd.Series.diff()`.

        The method `pd.Series.diff()` is not implemented yet.
        """
        raise NotImplementedError("The method `diff()` is not implemented yet.")

    def div(self, other, level=None, fill_value=None, axis=0):
        """A stub for the equivalent method to `pd.Series.div()`.

        The method `pd.Series.div()` is not implemented yet.
        """
        raise NotImplementedError("The method `div()` is not implemented yet.")

    def divide(self, other, level=None, fill_value=None, axis=0):
        """A stub for the equivalent method to `pd.Series.divide()`.

        The method `pd.Series.divide()` is not implemented yet.
        """
        raise NotImplementedError("The method `divide()` is not implemented yet.")

    def divmod(self, other, level=None, fill_value=None, axis=0):
        """A stub for the equivalent method to `pd.Series.divmod()`.

        The method `pd.Series.divmod()` is not implemented yet.
        """
        raise NotImplementedError("The method `divmod()` is not implemented yet.")

    def dot(self, other):
        """A stub for the equivalent method to `pd.Series.dot()`.

        The method `pd.Series.dot()` is not implemented yet.
        """
        raise NotImplementedError("The method `dot()` is not implemented yet.")

    def drop(self, labels=None, axis=0, index=None, columns=None, level=None, inplace=False,
             errors='raise'):
        """A stub for the equivalent method to `pd.Series.drop()`.

        The method `pd.Series.drop()` is not implemented yet.
        """
        raise NotImplementedError("The method `drop()` is not implemented yet.")

    def drop_duplicates(self, keep='first', inplace=False):
        """A stub for the equivalent method to `pd.Series.drop_duplicates()`.

        The method `pd.Series.drop_duplicates()` is not implemented yet.
        """
        raise NotImplementedError("The method `drop_duplicates()` is not implemented yet.")

    def droplevel(self, level, axis=0):
        """A stub for the equivalent method to `pd.Series.droplevel()`.

        The method `pd.Series.droplevel()` is not implemented yet.
        """
        raise NotImplementedError("The method `droplevel()` is not implemented yet.")

    def duplicated(self, keep='first'):
        """A stub for the equivalent method to `pd.Series.duplicated()`.

        The method `pd.Series.duplicated()` is not implemented yet.
        """
        raise NotImplementedError("The method `duplicated()` is not implemented yet.")

    def eq(self, other, level=None, fill_value=None, axis=0):
        """A stub for the equivalent method to `pd.Series.eq()`.

        The method `pd.Series.eq()` is not implemented yet.
        """
        raise NotImplementedError("The method `eq()` is not implemented yet.")

    def equals(self, other):
        """A stub for the equivalent method to `pd.Series.equals()`.

        The method `pd.Series.equals()` is not implemented yet.
        """
        raise NotImplementedError("The method `equals()` is not implemented yet.")

    def ewm(self, com=None, span=None, halflife=None, alpha=None, min_periods=0, adjust=True,
            ignore_na=False, axis=0):
        """A stub for the equivalent method to `pd.Series.ewm()`.

        The method `pd.Series.ewm()` is not implemented yet.
        """
        raise NotImplementedError("The method `ewm()` is not implemented yet.")

    def expanding(self, min_periods=1, center=False, axis=0):
        """A stub for the equivalent method to `pd.Series.expanding()`.

        The method `pd.Series.expanding()` is not implemented yet.
        """
        raise NotImplementedError("The method `expanding()` is not implemented yet.")

    def factorize(self, sort=False, na_sentinel=-1):
        """A stub for the equivalent method to `pd.Series.factorize()`.

        The method `pd.Series.factorize()` is not implemented yet.
        """
        raise NotImplementedError("The method `factorize()` is not implemented yet.")

    def ffill(self, axis=None, inplace=False, limit=None, downcast=None):
        """A stub for the equivalent method to `pd.Series.ffill()`.

        The method `pd.Series.ffill()` is not implemented yet.
        """
        raise NotImplementedError("The method `ffill()` is not implemented yet.")

    def fillna(self, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None,
               **kwargs):
        """A stub for the equivalent method to `pd.Series.fillna()`.

        The method `pd.Series.fillna()` is not implemented yet.
        """
        raise NotImplementedError("The method `fillna()` is not implemented yet.")

    def filter(self, items=None, like=None, regex=None, axis=None):
        """A stub for the equivalent method to `pd.Series.filter()`.

        The method `pd.Series.filter()` is not implemented yet.
        """
        raise NotImplementedError("The method `filter()` is not implemented yet.")

    def first(self, offset):
        """A stub for the equivalent method to `pd.Series.first()`.

        The method `pd.Series.first()` is not implemented yet.
        """
        raise NotImplementedError("The method `first()` is not implemented yet.")

    def first_valid_index(self):
        """A stub for the equivalent method to `pd.Series.first_valid_index()`.

        The method `pd.Series.first_valid_index()` is not implemented yet.
        """
        raise NotImplementedError("The method `first_valid_index()` is not implemented yet.")

    def floordiv(self, other, level=None, fill_value=None, axis=0):
        """A stub for the equivalent method to `pd.Series.floordiv()`.

        The method `pd.Series.floordiv()` is not implemented yet.
        """
        raise NotImplementedError("The method `floordiv()` is not implemented yet.")

    def ge(self, other, level=None, fill_value=None, axis=0):
        """A stub for the equivalent method to `pd.Series.ge()`.

        The method `pd.Series.ge()` is not implemented yet.
        """
        raise NotImplementedError("The method `ge()` is not implemented yet.")

    def get(self, key, default=None):
        """A stub for the equivalent method to `pd.Series.get()`.

        The method `pd.Series.get()` is not implemented yet.
        """
        raise NotImplementedError("The method `get()` is not implemented yet.")

    def get_dtype_counts(self):
        """A stub for the equivalent method to `pd.Series.get_dtype_counts()`.

        The method `pd.Series.get_dtype_counts()` is not implemented yet.
        """
        raise NotImplementedError("The method `get_dtype_counts()` is not implemented yet.")

    def get_ftype_counts(self):
        """A stub for the equivalent method to `pd.Series.get_ftype_counts()`.

        The method `pd.Series.get_ftype_counts()` is not implemented yet.
        """
        raise NotImplementedError("The method `get_ftype_counts()` is not implemented yet.")

    def get_value(self, label, takeable=False):
        """A stub for the equivalent method to `pd.Series.get_value()`.

        The method `pd.Series.get_value()` is not implemented yet.
        """
        raise NotImplementedError("The method `get_value()` is not implemented yet.")

    def get_values(self):
        """A stub for the equivalent method to `pd.Series.get_values()`.

        The method `pd.Series.get_values()` is not implemented yet.
        """
        raise NotImplementedError("The method `get_values()` is not implemented yet.")

    def groupby(self, by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True,
                squeeze=False, observed=False, **kwargs):
        """A stub for the equivalent method to `pd.Series.groupby()`.

        The method `pd.Series.groupby()` is not implemented yet.
        """
        raise NotImplementedError("The method `groupby()` is not implemented yet.")

    def gt(self, other, level=None, fill_value=None, axis=0):
        """A stub for the equivalent method to `pd.Series.gt()`.

        The method `pd.Series.gt()` is not implemented yet.
        """
        raise NotImplementedError("The method `gt()` is not implemented yet.")

    def hist(self, by=None, ax=None, grid=True, xlabelsize=None, xrot=None, ylabelsize=None,
             yrot=None, figsize=None, bins=10, **kwds):
        """A stub for the equivalent method to `pd.Series.hist()`.

        The method `pd.Series.hist()` is not implemented yet.
        """
        raise NotImplementedError("The method `hist()` is not implemented yet.")

    def idxmax(self, axis=0, skipna=True, *args, **kwargs):
        """A stub for the equivalent method to `pd.Series.idxmax()`.

        The method `pd.Series.idxmax()` is not implemented yet.
        """
        raise NotImplementedError("The method `idxmax()` is not implemented yet.")

    def idxmin(self, axis=0, skipna=True, *args, **kwargs):
        """A stub for the equivalent method to `pd.Series.idxmin()`.

        The method `pd.Series.idxmin()` is not implemented yet.
        """
        raise NotImplementedError("The method `idxmin()` is not implemented yet.")

    def infer_objects(self):
        """A stub for the equivalent method to `pd.Series.infer_objects()`.

        The method `pd.Series.infer_objects()` is not implemented yet.
        """
        raise NotImplementedError("The method `infer_objects()` is not implemented yet.")

    def interpolate(self, method='linear', axis=0, limit=None, inplace=False,
                    limit_direction='forward', limit_area=None, downcast=None, **kwargs):
        """A stub for the equivalent method to `pd.Series.interpolate()`.

        The method `pd.Series.interpolate()` is not implemented yet.
        """
        raise NotImplementedError("The method `interpolate()` is not implemented yet.")

    def isin(self, values):
        """A stub for the equivalent method to `pd.Series.isin()`.

        The method `pd.Series.isin()` is not implemented yet.
        """
        raise NotImplementedError("The method `isin()` is not implemented yet.")

    def item(self):
        """A stub for the equivalent method to `pd.Series.item()`.

        The method `pd.Series.item()` is not implemented yet.
        """
        raise NotImplementedError("The method `item()` is not implemented yet.")

    def items(self):
        """A stub for the equivalent method to `pd.Series.items()`.

        The method `pd.Series.items()` is not implemented yet.
        """
        raise NotImplementedError("The method `items()` is not implemented yet.")

    def iteritems(self):
        """A stub for the equivalent method to `pd.Series.iteritems()`.

        The method `pd.Series.iteritems()` is not implemented yet.
        """
        raise NotImplementedError("The method `iteritems()` is not implemented yet.")

    def keys(self):
        """A stub for the equivalent method to `pd.Series.keys()`.

        The method `pd.Series.keys()` is not implemented yet.
        """
        raise NotImplementedError("The method `keys()` is not implemented yet.")

    def kurt(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """A stub for the equivalent method to `pd.Series.kurt()`.

        The method `pd.Series.kurt()` is not implemented yet.
        """
        raise NotImplementedError("The method `kurt()` is not implemented yet.")

    def kurtosis(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """A stub for the equivalent method to `pd.Series.kurtosis()`.

        The method `pd.Series.kurtosis()` is not implemented yet.
        """
        raise NotImplementedError("The method `kurtosis()` is not implemented yet.")

    def last(self, offset):
        """A stub for the equivalent method to `pd.Series.last()`.

        The method `pd.Series.last()` is not implemented yet.
        """
        raise NotImplementedError("The method `last()` is not implemented yet.")

    def last_valid_index(self):
        """A stub for the equivalent method to `pd.Series.last_valid_index()`.

        The method `pd.Series.last_valid_index()` is not implemented yet.
        """
        raise NotImplementedError("The method `last_valid_index()` is not implemented yet.")

    def le(self, other, level=None, fill_value=None, axis=0):
        """A stub for the equivalent method to `pd.Series.le()`.

        The method `pd.Series.le()` is not implemented yet.
        """
        raise NotImplementedError("The method `le()` is not implemented yet.")

    def lt(self, other, level=None, fill_value=None, axis=0):
        """A stub for the equivalent method to `pd.Series.lt()`.

        The method `pd.Series.lt()` is not implemented yet.
        """
        raise NotImplementedError("The method `lt()` is not implemented yet.")

    def mad(self, axis=None, skipna=None, level=None):
        """A stub for the equivalent method to `pd.Series.mad()`.

        The method `pd.Series.mad()` is not implemented yet.
        """
        raise NotImplementedError("The method `mad()` is not implemented yet.")

    def map(self, arg, na_action=None):
        """A stub for the equivalent method to `pd.Series.map()`.

        The method `pd.Series.map()` is not implemented yet.
        """
        raise NotImplementedError("The method `map()` is not implemented yet.")

    def mask(self, cond, other=np.nan, inplace=False, axis=None, level=None, errors='raise',
             try_cast=False, raise_on_error=None):
        """A stub for the equivalent method to `pd.Series.mask()`.

        The method `pd.Series.mask()` is not implemented yet.
        """
        raise NotImplementedError("The method `mask()` is not implemented yet.")

    def mean(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """A stub for the equivalent method to `pd.Series.mean()`.

        The method `pd.Series.mean()` is not implemented yet.
        """
        raise NotImplementedError("The method `mean()` is not implemented yet.")

    def median(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """A stub for the equivalent method to `pd.Series.median()`.

        The method `pd.Series.median()` is not implemented yet.
        """
        raise NotImplementedError("The method `median()` is not implemented yet.")

    def memory_usage(self, index=True, deep=False):
        """A stub for the equivalent method to `pd.Series.memory_usage()`.

        The method `pd.Series.memory_usage()` is not implemented yet.
        """
        raise NotImplementedError("The method `memory_usage()` is not implemented yet.")

    def min(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """A stub for the equivalent method to `pd.Series.min()`.

        The method `pd.Series.min()` is not implemented yet.
        """
        raise NotImplementedError("The method `min()` is not implemented yet.")

    def mod(self, other, level=None, fill_value=None, axis=0):
        """A stub for the equivalent method to `pd.Series.mod()`.

        The method `pd.Series.mod()` is not implemented yet.
        """
        raise NotImplementedError("The method `mod()` is not implemented yet.")

    def mode(self, dropna=True):
        """A stub for the equivalent method to `pd.Series.mode()`.

        The method `pd.Series.mode()` is not implemented yet.
        """
        raise NotImplementedError("The method `mode()` is not implemented yet.")

    def mul(self, other, level=None, fill_value=None, axis=0):
        """A stub for the equivalent method to `pd.Series.mul()`.

        The method `pd.Series.mul()` is not implemented yet.
        """
        raise NotImplementedError("The method `mul()` is not implemented yet.")

    def multiply(self, other, level=None, fill_value=None, axis=0):
        """A stub for the equivalent method to `pd.Series.multiply()`.

        The method `pd.Series.multiply()` is not implemented yet.
        """
        raise NotImplementedError("The method `multiply()` is not implemented yet.")

    def ne(self, other, level=None, fill_value=None, axis=0):
        """A stub for the equivalent method to `pd.Series.ne()`.

        The method `pd.Series.ne()` is not implemented yet.
        """
        raise NotImplementedError("The method `ne()` is not implemented yet.")

    def nlargest(self, n=5, keep='first'):
        """A stub for the equivalent method to `pd.Series.nlargest()`.

        The method `pd.Series.nlargest()` is not implemented yet.
        """
        raise NotImplementedError("The method `nlargest()` is not implemented yet.")

    def nonzero(self):
        """A stub for the equivalent method to `pd.Series.nonzero()`.

        The method `pd.Series.nonzero()` is not implemented yet.
        """
        raise NotImplementedError("The method `nonzero()` is not implemented yet.")

    def nsmallest(self, n=5, keep='first'):
        """A stub for the equivalent method to `pd.Series.nsmallest()`.

        The method `pd.Series.nsmallest()` is not implemented yet.
        """
        raise NotImplementedError("The method `nsmallest()` is not implemented yet.")

    def nunique(self, dropna=True):
        """A stub for the equivalent method to `pd.Series.nunique()`.

        The method `pd.Series.nunique()` is not implemented yet.
        """
        raise NotImplementedError("The method `nunique()` is not implemented yet.")

    def pct_change(self, periods=1, fill_method='pad', limit=None, freq=None, **kwargs):
        """A stub for the equivalent method to `pd.Series.pct_change()`.

        The method `pd.Series.pct_change()` is not implemented yet.
        """
        raise NotImplementedError("The method `pct_change()` is not implemented yet.")

    def pipe(self, func, *args, **kwargs):
        """A stub for the equivalent method to `pd.Series.pipe()`.

        The method `pd.Series.pipe()` is not implemented yet.
        """
        raise NotImplementedError("The method `pipe()` is not implemented yet.")

    def pop(self, item):
        """A stub for the equivalent method to `pd.Series.pop()`.

        The method `pd.Series.pop()` is not implemented yet.
        """
        raise NotImplementedError("The method `pop()` is not implemented yet.")

    def pow(self, other, level=None, fill_value=None, axis=0):
        """A stub for the equivalent method to `pd.Series.pow()`.

        The method `pd.Series.pow()` is not implemented yet.
        """
        raise NotImplementedError("The method `pow()` is not implemented yet.")

    def prod(self, axis=None, skipna=None, level=None, numeric_only=None, min_count=0, **kwargs):
        """A stub for the equivalent method to `pd.Series.prod()`.

        The method `pd.Series.prod()` is not implemented yet.
        """
        raise NotImplementedError("The method `prod()` is not implemented yet.")

    def product(self, axis=None, skipna=None, level=None, numeric_only=None, min_count=0, **kwargs):
        """A stub for the equivalent method to `pd.Series.product()`.

        The method `pd.Series.product()` is not implemented yet.
        """
        raise NotImplementedError("The method `product()` is not implemented yet.")

    def ptp(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """A stub for the equivalent method to `pd.Series.ptp()`.

        The method `pd.Series.ptp()` is not implemented yet.
        """
        raise NotImplementedError("The method `ptp()` is not implemented yet.")

    def put(self, *args, **kwargs):
        """A stub for the equivalent method to `pd.Series.put()`.

        The method `pd.Series.put()` is not implemented yet.
        """
        raise NotImplementedError("The method `put()` is not implemented yet.")

    def quantile(self, q=0.5, interpolation='linear'):
        """A stub for the equivalent method to `pd.Series.quantile()`.

        The method `pd.Series.quantile()` is not implemented yet.
        """
        raise NotImplementedError("The method `quantile()` is not implemented yet.")

    def radd(self, other, level=None, fill_value=None, axis=0):
        """A stub for the equivalent method to `pd.Series.radd()`.

        The method `pd.Series.radd()` is not implemented yet.
        """
        raise NotImplementedError("The method `radd()` is not implemented yet.")

    def rank(self, axis=0, method='average', numeric_only=None, na_option='keep', ascending=True,
             pct=False):
        """A stub for the equivalent method to `pd.Series.rank()`.

        The method `pd.Series.rank()` is not implemented yet.
        """
        raise NotImplementedError("The method `rank()` is not implemented yet.")

    def ravel(self, order='C'):
        """A stub for the equivalent method to `pd.Series.ravel()`.

        The method `pd.Series.ravel()` is not implemented yet.
        """
        raise NotImplementedError("The method `ravel()` is not implemented yet.")

    def rdiv(self, other, level=None, fill_value=None, axis=0):
        """A stub for the equivalent method to `pd.Series.rdiv()`.

        The method `pd.Series.rdiv()` is not implemented yet.
        """
        raise NotImplementedError("The method `rdiv()` is not implemented yet.")

    def rdivmod(self, other, level=None, fill_value=None, axis=0):
        """A stub for the equivalent method to `pd.Series.rdivmod()`.

        The method `pd.Series.rdivmod()` is not implemented yet.
        """
        raise NotImplementedError("The method `rdivmod()` is not implemented yet.")

    def reindex(self, index=None, **kwargs):
        """A stub for the equivalent method to `pd.Series.reindex()`.

        The method `pd.Series.reindex()` is not implemented yet.
        """
        raise NotImplementedError("The method `reindex()` is not implemented yet.")

    def reindex_axis(self, labels, axis=0, **kwargs):
        """A stub for the equivalent method to `pd.Series.reindex_axis()`.

        The method `pd.Series.reindex_axis()` is not implemented yet.
        """
        raise NotImplementedError("The method `reindex_axis()` is not implemented yet.")

    def reindex_like(self, other, method=None, copy=True, limit=None, tolerance=None):
        """A stub for the equivalent method to `pd.Series.reindex_like()`.

        The method `pd.Series.reindex_like()` is not implemented yet.
        """
        raise NotImplementedError("The method `reindex_like()` is not implemented yet.")

    def rename_axis(self, mapper=None, index=None, columns=None, axis=None, copy=True,
                    inplace=False):
        """A stub for the equivalent method to `pd.Series.rename_axis()`.

        The method `pd.Series.rename_axis()` is not implemented yet.
        """
        raise NotImplementedError("The method `rename_axis()` is not implemented yet.")

    def reorder_levels(self, order):
        """A stub for the equivalent method to `pd.Series.reorder_levels()`.

        The method `pd.Series.reorder_levels()` is not implemented yet.
        """
        raise NotImplementedError("The method `reorder_levels()` is not implemented yet.")

    def repeat(self, repeats, axis=None):
        """A stub for the equivalent method to `pd.Series.repeat()`.

        The method `pd.Series.repeat()` is not implemented yet.
        """
        raise NotImplementedError("The method `repeat()` is not implemented yet.")

    def replace(self, to_replace=None, value=None, inplace=False, limit=None, regex=False,
                method='pad'):
        """A stub for the equivalent method to `pd.Series.replace()`.

        The method `pd.Series.replace()` is not implemented yet.
        """
        raise NotImplementedError("The method `replace()` is not implemented yet.")

    def resample(self, rule, how=None, axis=0, fill_method=None, closed=None, label=None,
                 convention='start', kind=None, loffset=None, limit=None, base=0, on=None,
                 level=None):
        """A stub for the equivalent method to `pd.Series.resample()`.

        The method `pd.Series.resample()` is not implemented yet.
        """
        raise NotImplementedError("The method `resample()` is not implemented yet.")

    def rfloordiv(self, other, level=None, fill_value=None, axis=0):
        """A stub for the equivalent method to `pd.Series.rfloordiv()`.

        The method `pd.Series.rfloordiv()` is not implemented yet.
        """
        raise NotImplementedError("The method `rfloordiv()` is not implemented yet.")

    def rmod(self, other, level=None, fill_value=None, axis=0):
        """A stub for the equivalent method to `pd.Series.rmod()`.

        The method `pd.Series.rmod()` is not implemented yet.
        """
        raise NotImplementedError("The method `rmod()` is not implemented yet.")

    def rmul(self, other, level=None, fill_value=None, axis=0):
        """A stub for the equivalent method to `pd.Series.rmul()`.

        The method `pd.Series.rmul()` is not implemented yet.
        """
        raise NotImplementedError("The method `rmul()` is not implemented yet.")

    def rolling(self, window, min_periods=None, center=False, win_type=None, on=None, axis=0,
                closed=None):
        """A stub for the equivalent method to `pd.Series.rolling()`.

        The method `pd.Series.rolling()` is not implemented yet.
        """
        raise NotImplementedError("The method `rolling()` is not implemented yet.")

    def round(self, decimals=0, *args, **kwargs):
        """A stub for the equivalent method to `pd.Series.round()`.

        The method `pd.Series.round()` is not implemented yet.
        """
        raise NotImplementedError("The method `round()` is not implemented yet.")

    def rpow(self, other, level=None, fill_value=None, axis=0):
        """A stub for the equivalent method to `pd.Series.rpow()`.

        The method `pd.Series.rpow()` is not implemented yet.
        """
        raise NotImplementedError("The method `rpow()` is not implemented yet.")

    def rsub(self, other, level=None, fill_value=None, axis=0):
        """A stub for the equivalent method to `pd.Series.rsub()`.

        The method `pd.Series.rsub()` is not implemented yet.
        """
        raise NotImplementedError("The method `rsub()` is not implemented yet.")

    def rtruediv(self, other, level=None, fill_value=None, axis=0):
        """A stub for the equivalent method to `pd.Series.rtruediv()`.

        The method `pd.Series.rtruediv()` is not implemented yet.
        """
        raise NotImplementedError("The method `rtruediv()` is not implemented yet.")

    def sample(self, n=None, frac=None, replace=False, weights=None, random_state=None, axis=None):
        """A stub for the equivalent method to `pd.Series.sample()`.

        The method `pd.Series.sample()` is not implemented yet.
        """
        raise NotImplementedError("The method `sample()` is not implemented yet.")

    def searchsorted(self, value, side='left', sorter=None):
        """A stub for the equivalent method to `pd.Series.searchsorted()`.

        The method `pd.Series.searchsorted()` is not implemented yet.
        """
        raise NotImplementedError("The method `searchsorted()` is not implemented yet.")

    def select(self, crit, axis=0):
        """A stub for the equivalent method to `pd.Series.select()`.

        The method `pd.Series.select()` is not implemented yet.
        """
        raise NotImplementedError("The method `select()` is not implemented yet.")

    def sem(self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs):
        """A stub for the equivalent method to `pd.Series.sem()`.

        The method `pd.Series.sem()` is not implemented yet.
        """
        raise NotImplementedError("The method `sem()` is not implemented yet.")

    def set_axis(self, labels, axis=0, inplace=None):
        """A stub for the equivalent method to `pd.Series.set_axis()`.

        The method `pd.Series.set_axis()` is not implemented yet.
        """
        raise NotImplementedError("The method `set_axis()` is not implemented yet.")

    def set_value(self, label, value, takeable=False):
        """A stub for the equivalent method to `pd.Series.set_value()`.

        The method `pd.Series.set_value()` is not implemented yet.
        """
        raise NotImplementedError("The method `set_value()` is not implemented yet.")

    def shift(self, periods=1, freq=None, axis=0, fill_value=None):
        """A stub for the equivalent method to `pd.Series.shift()`.

        The method `pd.Series.shift()` is not implemented yet.
        """
        raise NotImplementedError("The method `shift()` is not implemented yet.")

    def skew(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """A stub for the equivalent method to `pd.Series.skew()`.

        The method `pd.Series.skew()` is not implemented yet.
        """
        raise NotImplementedError("The method `skew()` is not implemented yet.")

    def slice_shift(self, periods=1, axis=0):
        """A stub for the equivalent method to `pd.Series.slice_shift()`.

        The method `pd.Series.slice_shift()` is not implemented yet.
        """
        raise NotImplementedError("The method `slice_shift()` is not implemented yet.")

    def sort_index(self, axis=0, level=None, ascending=True, inplace=False, kind='quicksort',
                   na_position='last', sort_remaining=True):
        """A stub for the equivalent method to `pd.Series.sort_index()`.

        The method `pd.Series.sort_index()` is not implemented yet.
        """
        raise NotImplementedError("The method `sort_index()` is not implemented yet.")

    def sort_values(self, axis=0, ascending=True, inplace=False, kind='quicksort',
                    na_position='last'):
        """A stub for the equivalent method to `pd.Series.sort_values()`.

        The method `pd.Series.sort_values()` is not implemented yet.
        """
        raise NotImplementedError("The method `sort_values()` is not implemented yet.")

    def squeeze(self, axis=None):
        """A stub for the equivalent method to `pd.Series.squeeze()`.

        The method `pd.Series.squeeze()` is not implemented yet.
        """
        raise NotImplementedError("The method `squeeze()` is not implemented yet.")

    def std(self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs):
        """A stub for the equivalent method to `pd.Series.std()`.

        The method `pd.Series.std()` is not implemented yet.
        """
        raise NotImplementedError("The method `std()` is not implemented yet.")

    def sub(self, other, level=None, fill_value=None, axis=0):
        """A stub for the equivalent method to `pd.Series.sub()`.

        The method `pd.Series.sub()` is not implemented yet.
        """
        raise NotImplementedError("The method `sub()` is not implemented yet.")

    def subtract(self, other, level=None, fill_value=None, axis=0):
        """A stub for the equivalent method to `pd.Series.subtract()`.

        The method `pd.Series.subtract()` is not implemented yet.
        """
        raise NotImplementedError("The method `subtract()` is not implemented yet.")

    def sum(self, axis=None, skipna=None, level=None, numeric_only=None, min_count=0, **kwargs):
        """A stub for the equivalent method to `pd.Series.sum()`.

        The method `pd.Series.sum()` is not implemented yet.
        """
        raise NotImplementedError("The method `sum()` is not implemented yet.")

    def swapaxes(self, axis1, axis2, copy=True):
        """A stub for the equivalent method to `pd.Series.swapaxes()`.

        The method `pd.Series.swapaxes()` is not implemented yet.
        """
        raise NotImplementedError("The method `swapaxes()` is not implemented yet.")

    def swaplevel(self, i=-2, j=-1, copy=True):
        """A stub for the equivalent method to `pd.Series.swaplevel()`.

        The method `pd.Series.swaplevel()` is not implemented yet.
        """
        raise NotImplementedError("The method `swaplevel()` is not implemented yet.")

    def tail(self, n=5):
        """A stub for the equivalent method to `pd.Series.tail()`.

        The method `pd.Series.tail()` is not implemented yet.
        """
        raise NotImplementedError("The method `tail()` is not implemented yet.")

    def take(self, indices, axis=0, convert=None, is_copy=True, **kwargs):
        """A stub for the equivalent method to `pd.Series.take()`.

        The method `pd.Series.take()` is not implemented yet.
        """
        raise NotImplementedError("The method `take()` is not implemented yet.")

    def to_clipboard(self, excel=True, sep=None, **kwargs):
        """A stub for the equivalent method to `pd.Series.to_clipboard()`.

        The method `pd.Series.to_clipboard()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_clipboard()` is not implemented yet.")

    def to_csv(self, *args, **kwargs):
        """A stub for the equivalent method to `pd.Series.to_csv()`.

        The method `pd.Series.to_csv()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_csv()` is not implemented yet.")

    def to_dense(self):
        """A stub for the equivalent method to `pd.Series.to_dense()`.

        The method `pd.Series.to_dense()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_dense()` is not implemented yet.")

    def to_dict(self, into=dict):
        """A stub for the equivalent method to `pd.Series.to_dict()`.

        The method `pd.Series.to_dict()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_dict()` is not implemented yet.")

    def to_excel(self, excel_writer, sheet_name='Sheet1', na_rep='', float_format=None,
                 columns=None, header=True, index=True, index_label=None, startrow=0, startcol=0,
                 engine=None, merge_cells=True, encoding=None, inf_rep='inf', verbose=True,
                 freeze_panes=None):
        """A stub for the equivalent method to `pd.Series.to_excel()`.

        The method `pd.Series.to_excel()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_excel()` is not implemented yet.")

    def to_frame(self, name=None):
        """A stub for the equivalent method to `pd.Series.to_frame()`.

        The method `pd.Series.to_frame()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_frame()` is not implemented yet.")

    def to_hdf(self, path_or_buf, key, **kwargs):
        """A stub for the equivalent method to `pd.Series.to_hdf()`.

        The method `pd.Series.to_hdf()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_hdf()` is not implemented yet.")

    def to_json(self, path_or_buf=None, orient=None, date_format=None, double_precision=10,
                force_ascii=True, date_unit='ms', default_handler=None, lines=False,
                compression='infer', index=True):
        """A stub for the equivalent method to `pd.Series.to_json()`.

        The method `pd.Series.to_json()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_json()` is not implemented yet.")

    def to_latex(self, buf=None, columns=None, col_space=None, header=True, index=True,
                 na_rep='NaN', formatters=None, float_format=None, sparsify=None, index_names=True,
                 bold_rows=False, column_format=None, longtable=None, escape=None, encoding=None,
                 decimal='.', multicolumn=None, multicolumn_format=None, multirow=None):
        """A stub for the equivalent method to `pd.Series.to_latex()`.

        The method `pd.Series.to_latex()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_latex()` is not implemented yet.")

    def to_list(self):
        """A stub for the equivalent method to `pd.Series.to_list()`.

        The method `pd.Series.to_list()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_list()` is not implemented yet.")

    def to_msgpack(self, path_or_buf=None, encoding='utf-8', **kwargs):
        """A stub for the equivalent method to `pd.Series.to_msgpack()`.

        The method `pd.Series.to_msgpack()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_msgpack()` is not implemented yet.")

    def to_numpy(self, dtype=None, copy=False):
        """A stub for the equivalent method to `pd.Series.to_numpy()`.

        The method `pd.Series.to_numpy()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_numpy()` is not implemented yet.")

    def to_period(self, freq=None, copy=True):
        """A stub for the equivalent method to `pd.Series.to_period()`.

        The method `pd.Series.to_period()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_period()` is not implemented yet.")

    def to_pickle(self, path, compression='infer', protocol=4):
        """A stub for the equivalent method to `pd.Series.to_pickle()`.

        The method `pd.Series.to_pickle()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_pickle()` is not implemented yet.")

    def to_sparse(self, kind='block', fill_value=None):
        """A stub for the equivalent method to `pd.Series.to_sparse()`.

        The method `pd.Series.to_sparse()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_sparse()` is not implemented yet.")

    def to_sql(self, name, con, schema=None, if_exists='fail', index=True, index_label=None,
               chunksize=None, dtype=None, method=None):
        """A stub for the equivalent method to `pd.Series.to_sql()`.

        The method `pd.Series.to_sql()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_sql()` is not implemented yet.")

    def to_string(self, buf=None, na_rep='NaN', float_format=None, header=True, index=True,
                  length=False, dtype=False, name=False, max_rows=None):
        """A stub for the equivalent method to `pd.Series.to_string()`.

        The method `pd.Series.to_string()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_string()` is not implemented yet.")

    def to_timestamp(self, freq=None, how='start', copy=True):
        """A stub for the equivalent method to `pd.Series.to_timestamp()`.

        The method `pd.Series.to_timestamp()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_timestamp()` is not implemented yet.")

    def to_xarray(self):
        """A stub for the equivalent method to `pd.Series.to_xarray()`.

        The method `pd.Series.to_xarray()` is not implemented yet.
        """
        raise NotImplementedError("The method `to_xarray()` is not implemented yet.")

    def tolist(self):
        """A stub for the equivalent method to `pd.Series.tolist()`.

        The method `pd.Series.tolist()` is not implemented yet.
        """
        raise NotImplementedError("The method `tolist()` is not implemented yet.")

    def transform(self, func, axis=0, *args, **kwargs):
        """A stub for the equivalent method to `pd.Series.transform()`.

        The method `pd.Series.transform()` is not implemented yet.
        """
        raise NotImplementedError("The method `transform()` is not implemented yet.")

    def transpose(self, *args, **kwargs):
        """A stub for the equivalent method to `pd.Series.transpose()`.

        The method `pd.Series.transpose()` is not implemented yet.
        """
        raise NotImplementedError("The method `transpose()` is not implemented yet.")

    def truediv(self, other, level=None, fill_value=None, axis=0):
        """A stub for the equivalent method to `pd.Series.truediv()`.

        The method `pd.Series.truediv()` is not implemented yet.
        """
        raise NotImplementedError("The method `truediv()` is not implemented yet.")

    def truncate(self, before=None, after=None, axis=None, copy=True):
        """A stub for the equivalent method to `pd.Series.truncate()`.

        The method `pd.Series.truncate()` is not implemented yet.
        """
        raise NotImplementedError("The method `truncate()` is not implemented yet.")

    def tshift(self, periods=1, freq=None, axis=0):
        """A stub for the equivalent method to `pd.Series.tshift()`.

        The method `pd.Series.tshift()` is not implemented yet.
        """
        raise NotImplementedError("The method `tshift()` is not implemented yet.")

    def tz_convert(self, tz, axis=0, level=None, copy=True):
        """A stub for the equivalent method to `pd.Series.tz_convert()`.

        The method `pd.Series.tz_convert()` is not implemented yet.
        """
        raise NotImplementedError("The method `tz_convert()` is not implemented yet.")

    def tz_localize(self, tz, axis=0, level=None, copy=True, ambiguous='raise',
                    nonexistent='raise'):
        """A stub for the equivalent method to `pd.Series.tz_localize()`.

        The method `pd.Series.tz_localize()` is not implemented yet.
        """
        raise NotImplementedError("The method `tz_localize()` is not implemented yet.")

    def unstack(self, level=-1, fill_value=None):
        """A stub for the equivalent method to `pd.Series.unstack()`.

        The method `pd.Series.unstack()` is not implemented yet.
        """
        raise NotImplementedError("The method `unstack()` is not implemented yet.")

    def update(self, other):
        """A stub for the equivalent method to `pd.Series.update()`.

        The method `pd.Series.update()` is not implemented yet.
        """
        raise NotImplementedError("The method `update()` is not implemented yet.")

    def valid(self, inplace=False, **kwargs):
        """A stub for the equivalent method to `pd.Series.valid()`.

        The method `pd.Series.valid()` is not implemented yet.
        """
        raise NotImplementedError("The method `valid()` is not implemented yet.")

    def var(self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs):
        """A stub for the equivalent method to `pd.Series.var()`.

        The method `pd.Series.var()` is not implemented yet.
        """
        raise NotImplementedError("The method `var()` is not implemented yet.")

    def view(self, dtype=None):
        """A stub for the equivalent method to `pd.Series.view()`.

        The method `pd.Series.view()` is not implemented yet.
        """
        raise NotImplementedError("The method `view()` is not implemented yet.")

    def where(self, cond, other=np.nan, inplace=False, axis=None, level=None, errors='raise',
              try_cast=False, raise_on_error=None):
        """A stub for the equivalent method to `pd.Series.where()`.

        The method `pd.Series.where()` is not implemented yet.
        """
        raise NotImplementedError("The method `where()` is not implemented yet.")

    def xs(self, key, axis=0, level=None, drop_level=True):
        """A stub for the equivalent method to `pd.Series.xs()`.

        The method `pd.Series.xs()` is not implemented yet.
        """
        raise NotImplementedError("The method `xs()` is not implemented yet.")
