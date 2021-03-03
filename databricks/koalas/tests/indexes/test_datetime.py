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

from distutils.version import LooseVersion

import pandas as pd

import databricks.koalas as ks

from databricks.koalas.testing.utils import ReusedSQLTestCase, TestUtils


class DatetimeIndexTest(ReusedSQLTestCase, TestUtils):
    @property
    def fixed_freqs(self):
        return [
            "D",
            "H",
            "T",  # min
            "S",
            "L",  # ms
            "U",  # us
            # 'N' not supported
        ]

    @property
    def non_fixed_freqs(self):
        return ["W", "Q"]

    @property
    def pidxs(self):
        return [
            pd.DatetimeIndex([0]),
            pd.DatetimeIndex(["2004-01-01", "2002-12-31", "2000-04-01"]),
        ] + [
            pd.date_range("2000-01-01", periods=3, freq=freq)
            for freq in (self.fixed_freqs + self.non_fixed_freqs)
        ]

    @property
    def kidxs(self):
        return [ks.from_pandas(pidx) for pidx in self.pidxs]

    @property
    def idx_pairs(self):
        return list(zip(self.kidxs, self.pidxs))

    def test_properties(self):
        for kidx, pidx in self.idx_pairs:
            self.assert_eq(kidx.year, pidx.year)
            self.assert_eq(kidx.month, pidx.month)
            self.assert_eq(kidx.day, pidx.day)
            self.assert_eq(kidx.hour, pidx.hour)
            self.assert_eq(kidx.minute, pidx.minute)
            self.assert_eq(kidx.second, pidx.second)
            self.assert_eq(kidx.microsecond, pidx.microsecond)
            self.assert_eq(kidx.week, pidx.week)
            self.assert_eq(kidx.weekofyear, pidx.weekofyear)
            self.assert_eq(kidx.dayofweek, pidx.dayofweek)
            self.assert_eq(kidx.weekday, pidx.weekday)
            self.assert_eq(kidx.dayofyear, pidx.dayofyear)
            self.assert_eq(kidx.quarter, pidx.quarter)
            self.assert_eq(kidx.daysinmonth, pidx.daysinmonth)
            self.assert_eq(kidx.days_in_month, pidx.days_in_month)
            self.assert_eq(kidx.is_month_start, pd.Index(pidx.is_month_start))
            self.assert_eq(kidx.is_month_end, pd.Index(pidx.is_month_end))
            self.assert_eq(kidx.is_quarter_start, pd.Index(pidx.is_quarter_start))
            self.assert_eq(kidx.is_quarter_end, pd.Index(pidx.is_quarter_end))
            self.assert_eq(kidx.is_year_start, pd.Index(pidx.is_year_start))
            self.assert_eq(kidx.is_year_end, pd.Index(pidx.is_year_end))
            self.assert_eq(kidx.is_leap_year, pd.Index(pidx.is_leap_year))

            if LooseVersion(pd.__version__) >= LooseVersion("1.2.0"):
                self.assert_eq(kidx.day_of_year, pidx.day_of_year)
                self.assert_eq(kidx.day_of_week, pidx.day_of_week)
