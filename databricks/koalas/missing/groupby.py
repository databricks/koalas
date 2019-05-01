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

from databricks.koalas.missing import _unsupported_function


def unsupported_function(method_name):
    return _unsupported_function(class_name='pd.groupby.GroupBy', method_name=method_name)


class _MissingPandasLikeDataFrameGroupBy(object):

    all = unsupported_function('all')
    any = unsupported_function('any')
    apply = unsupported_function('apply')
    backfill = unsupported_function('backfill')
    bfill = unsupported_function('bfill')
    boxplot = unsupported_function('boxplot')
    cumcount = unsupported_function('cumcount')
    cummax = unsupported_function('cummax')
    cummin = unsupported_function('cummin')
    cumprod = unsupported_function('cumprod')
    cumsum = unsupported_function('cumsum')
    describe = unsupported_function('describe')
    expanding = unsupported_function('expanding')
    ffill = unsupported_function('ffill')
    filter = unsupported_function('filter')
    get_group = unsupported_function('get_group')
    head = unsupported_function('head')
    median = unsupported_function('median')
    ngroup = unsupported_function('ngroup')
    nth = unsupported_function('nth')
    nunique = unsupported_function('nunique')
    ohlc = unsupported_function('ohlc')
    pad = unsupported_function('pad')
    pct_change = unsupported_function('pct_change')
    pipe = unsupported_function('pipe')
    prod = unsupported_function('prod')
    rank = unsupported_function('rank')
    resample = unsupported_function('resample')
    rolling = unsupported_function('rolling')
    sem = unsupported_function('sem')
    shift = unsupported_function('shift')
    size = unsupported_function('size')
    tail = unsupported_function('tail')
    transform = unsupported_function('transform')


class _MissingPandasLikeSeriesGroupBy(object):

    # TODO: agg and aggregate should be implemented.
    # agg = unsupported_function('agg')
    # aggregate = unsupported_function('aggregate')
    all = unsupported_function('all')
    any = unsupported_function('any')
    apply = unsupported_function('apply')
    backfill = unsupported_function('backfill')
    bfill = unsupported_function('bfill')
    cumcount = unsupported_function('cumcount')
    cummax = unsupported_function('cummax')
    cummin = unsupported_function('cummin')
    cumprod = unsupported_function('cumprod')
    cumsum = unsupported_function('cumsum')
    describe = unsupported_function('describe')
    expanding = unsupported_function('expanding')
    ffill = unsupported_function('ffill')
    filter = unsupported_function('filter')
    get_group = unsupported_function('get_group')
    head = unsupported_function('head')
    median = unsupported_function('median')
    ngroup = unsupported_function('ngroup')
    nth = unsupported_function('nth')
    nunique = unsupported_function('nunique')
    ohlc = unsupported_function('ohlc')
    pad = unsupported_function('pad')
    pct_change = unsupported_function('pct_change')
    pipe = unsupported_function('pipe')
    prod = unsupported_function('prod')
    rank = unsupported_function('rank')
    resample = unsupported_function('resample')
    rolling = unsupported_function('rolling')
    sem = unsupported_function('sem')
    shift = unsupported_function('shift')
    size = unsupported_function('size')
    tail = unsupported_function('tail')
    transform = unsupported_function('transform')
    value_counts = unsupported_function('value_counts')
