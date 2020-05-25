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

import datetime
from decimal import Decimal
from distutils.version import LooseVersion

import numpy as np
import pandas as pd

from databricks import koalas as ks
from databricks.koalas.testing.utils import ReusedSQLTestCase


class ReshapeTest(ReusedSQLTestCase):
    def test_get_dummies(self):
        for pdf_or_ps in [
            pd.Series([1, 1, 1, 2, 2, 1, 3, 4]),
            # pd.Series([1, 1, 1, 2, 2, 1, 3, 4], dtype='category'),
            # pd.Series(pd.Categorical([1, 1, 1, 2, 2, 1, 3, 4],
            #                          categories=[4, 3, 2, 1])),
            pd.DataFrame(
                {
                    "a": [1, 2, 3, 4, 4, 3, 2, 1],
                    # 'b': pd.Categorical(list('abcdabcd')),
                    "b": list("abcdabcd"),
                }
            ),
        ]:
            kdf_or_kser = ks.from_pandas(pdf_or_ps)

            self.assert_eq(ks.get_dummies(kdf_or_kser), pd.get_dummies(pdf_or_ps), almost=True)

        kser = ks.Series([1, 1, 1, 2, 2, 1, 3, 4])
        with self.assertRaisesRegex(
            NotImplementedError, "get_dummies currently does not support sparse"
        ):
            ks.get_dummies(kser, sparse=True)

    def test_get_dummies_object(self):
        pdf = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 4, 3, 2, 1],
                # 'a': pd.Categorical([1, 2, 3, 4, 4, 3, 2, 1]),
                "b": list("abcdabcd"),
                # 'c': pd.Categorical(list('abcdabcd')),
                "c": list("abcdabcd"),
            }
        )
        kdf = ks.from_pandas(pdf)

        # Explicitly exclude object columns
        self.assert_eq(
            ks.get_dummies(kdf, columns=["a", "c"]),
            pd.get_dummies(pdf, columns=["a", "c"]),
            almost=True,
        )

        self.assert_eq(ks.get_dummies(kdf), pd.get_dummies(pdf), almost=True)
        self.assert_eq(ks.get_dummies(kdf.b), pd.get_dummies(pdf.b), almost=True)
        self.assert_eq(
            ks.get_dummies(kdf, columns=["b"]), pd.get_dummies(pdf, columns=["b"]), almost=True
        )

    def test_get_dummies_date_datetime(self):
        pdf = pd.DataFrame(
            {
                "d": [
                    datetime.date(2019, 1, 1),
                    datetime.date(2019, 1, 2),
                    datetime.date(2019, 1, 1),
                ],
                "dt": [
                    datetime.datetime(2019, 1, 1, 0, 0, 0),
                    datetime.datetime(2019, 1, 1, 0, 0, 1),
                    datetime.datetime(2019, 1, 1, 0, 0, 0),
                ],
            }
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(ks.get_dummies(kdf), pd.get_dummies(pdf), almost=True)
        self.assert_eq(ks.get_dummies(kdf.d), pd.get_dummies(pdf.d), almost=True)
        self.assert_eq(ks.get_dummies(kdf.dt), pd.get_dummies(pdf.dt), almost=True)

    def test_get_dummies_boolean(self):
        pdf = pd.DataFrame({"b": [True, False, True]})
        kdf = ks.from_pandas(pdf)

        self.assert_eq(ks.get_dummies(kdf), pd.get_dummies(pdf), almost=True)
        self.assert_eq(ks.get_dummies(kdf.b), pd.get_dummies(pdf.b), almost=True)

    def test_get_dummies_decimal(self):
        pdf = pd.DataFrame({"d": [Decimal(1.0), Decimal(2.0), Decimal(1)]})
        kdf = ks.from_pandas(pdf)

        self.assert_eq(ks.get_dummies(kdf), pd.get_dummies(pdf), almost=True)
        self.assert_eq(ks.get_dummies(kdf.d), pd.get_dummies(pdf.d), almost=True)

    def test_get_dummies_kwargs(self):
        # pser = pd.Series([1, 1, 1, 2, 2, 1, 3, 4], dtype='category')
        pser = pd.Series([1, 1, 1, 2, 2, 1, 3, 4])
        kser = ks.from_pandas(pser)
        self.assert_eq(
            ks.get_dummies(kser, prefix="X", prefix_sep="-"),
            pd.get_dummies(pser, prefix="X", prefix_sep="-"),
            almost=True,
        )

        self.assert_eq(
            ks.get_dummies(kser, drop_first=True),
            pd.get_dummies(pser, drop_first=True),
            almost=True,
        )

        # nan
        # pser = pd.Series([1, 1, 1, 2, np.nan, 3, np.nan, 5], dtype='category')
        pser = pd.Series([1, 1, 1, 2, np.nan, 3, np.nan, 5])
        kser = ks.from_pandas(pser)
        self.assert_eq(ks.get_dummies(kser), pd.get_dummies(pser), almost=True)

        # dummy_na
        self.assert_eq(
            ks.get_dummies(kser, dummy_na=True), pd.get_dummies(pser, dummy_na=True), almost=True
        )

    def test_get_dummies_prefix(self):
        pdf = pd.DataFrame({"A": ["a", "b", "a"], "B": ["b", "a", "c"], "D": [0, 0, 1],})
        kdf = ks.from_pandas(pdf)

        self.assert_eq(
            ks.get_dummies(kdf, prefix=["foo", "bar"]),
            pd.get_dummies(pdf, prefix=["foo", "bar"]),
            almost=True,
        )

        self.assert_eq(
            ks.get_dummies(kdf, prefix=["foo"], columns=["B"]),
            pd.get_dummies(pdf, prefix=["foo"], columns=["B"]),
            almost=True,
        )

        self.assert_eq(
            ks.get_dummies(kdf, prefix={"A": "foo", "B": "bar"}),
            pd.get_dummies(pdf, prefix={"A": "foo", "B": "bar"}),
            almost=True,
        )

        self.assert_eq(
            ks.get_dummies(kdf, prefix={"B": "foo", "A": "bar"}),
            pd.get_dummies(pdf, prefix={"B": "foo", "A": "bar"}),
            almost=True,
        )

        self.assert_eq(
            ks.get_dummies(kdf, prefix={"A": "foo", "B": "bar"}, columns=["A", "B"]),
            pd.get_dummies(pdf, prefix={"A": "foo", "B": "bar"}, columns=["A", "B"]),
            almost=True,
        )

        with self.assertRaisesRegex(NotImplementedError, "string types"):
            ks.get_dummies(kdf, prefix="foo")
        with self.assertRaisesRegex(ValueError, "Length of 'prefix' \\(1\\) .* \\(2\\)"):
            ks.get_dummies(kdf, prefix=["foo"])
        with self.assertRaisesRegex(ValueError, "Length of 'prefix' \\(2\\) .* \\(1\\)"):
            ks.get_dummies(kdf, prefix=["foo", "bar"], columns=["B"])

        pser = pd.Series([1, 1, 1, 2, 2, 1, 3, 4], name="A")
        kser = ks.from_pandas(pser)

        self.assert_eq(
            ks.get_dummies(kser, prefix="foo"), pd.get_dummies(pser, prefix="foo"), almost=True
        )

        # columns are ignored.
        self.assert_eq(
            ks.get_dummies(kser, prefix=["foo"], columns=["B"]),
            pd.get_dummies(pser, prefix=["foo"], columns=["B"]),
            almost=True,
        )

    def test_get_dummies_dtype(self):
        pdf = pd.DataFrame(
            {
                # "A": pd.Categorical(['a', 'b', 'a'], categories=['a', 'b', 'c']),
                "A": ["a", "b", "a"],
                "B": [0, 0, 1],
            }
        )
        kdf = ks.from_pandas(pdf)

        if LooseVersion("0.23.0") <= LooseVersion(pd.__version__):
            exp = pd.get_dummies(pdf, dtype="float64")
        else:
            exp = pd.get_dummies(pdf)
            exp = exp.astype({"A_a": "float64", "A_b": "float64"})
        res = ks.get_dummies(kdf, dtype="float64")
        self.assert_eq(res, exp, almost=True)

    def test_get_dummies_multiindex_columns(self):
        pdf = pd.DataFrame(
            {
                ("x", "a", "1"): [1, 2, 3, 4, 4, 3, 2, 1],
                ("x", "b", "2"): list("abcdabcd"),
                ("y", "c", "3"): list("abcdabcd"),
            }
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(ks.get_dummies(kdf), pd.get_dummies(pdf), almost=True)
        self.assert_eq(
            ks.get_dummies(kdf, columns=[("y", "c", "3"), ("x", "a", "1")]),
            pd.get_dummies(pdf, columns=[("y", "c", "3"), ("x", "a", "1")]),
            almost=True,
        )
        self.assert_eq(
            ks.get_dummies(kdf, columns=["x"]), pd.get_dummies(pdf, columns=["x"]), almost=True
        )
        self.assert_eq(
            ks.get_dummies(kdf, columns=("x", "a")),
            pd.get_dummies(pdf, columns=("x", "a")),
            almost=True,
        )
        self.assert_eq(
            ks.get_dummies(kdf, columns=["x"]), pd.get_dummies(pdf, columns=["x"]), almost=True
        )

        self.assertRaises(KeyError, lambda: ks.get_dummies(kdf, columns=["z"]))
        self.assertRaises(KeyError, lambda: ks.get_dummies(kdf, columns=("x", "c")))
        self.assertRaises(ValueError, lambda: ks.get_dummies(kdf, columns=[("x",), "c"]))
        self.assertRaises(TypeError, lambda: ks.get_dummies(kdf, columns="x"))
