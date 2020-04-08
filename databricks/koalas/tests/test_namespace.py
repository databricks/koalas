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
import itertools

import pandas as pd

from databricks import koalas as ks
from databricks.koalas.testing.utils import ReusedSQLTestCase, SQLTestUtils


class NamespaceTest(ReusedSQLTestCase, SQLTestUtils):
    def test_from_pandas(self):
        pdf = pd.DataFrame({"year": [2015, 2016], "month": [2, 3], "day": [4, 5]})
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf, pdf)

        pser = pdf.year
        kser = ks.from_pandas(pser)

        self.assert_eq(kser, pser)

        pidx = pdf.index
        kidx = ks.from_pandas(pidx)

        self.assert_eq(kidx, pidx)

        pmidx = pdf.set_index("year", append=True).index
        kmidx = ks.from_pandas(pmidx)

        self.assert_eq(kmidx, pmidx)

        expected_error_message = "Unknown data type: {}".format(type(kidx))
        with self.assertRaisesRegex(ValueError, expected_error_message):
            ks.from_pandas(kidx)

    def test_to_datetime(self):
        pdf = pd.DataFrame({"year": [2015, 2016], "month": [2, 3], "day": [4, 5]})
        kdf = ks.from_pandas(pdf)
        dict_from_pdf = pdf.to_dict()

        self.assert_eq(pd.to_datetime(pdf), ks.to_datetime(kdf))
        self.assert_eq(pd.to_datetime(dict_from_pdf), ks.to_datetime(dict_from_pdf))

        self.assert_eq(pd.to_datetime(1490195805, unit="s"), ks.to_datetime(1490195805, unit="s"))
        self.assert_eq(
            pd.to_datetime(1490195805433502912, unit="ns"),
            ks.to_datetime(1490195805433502912, unit="ns"),
        )

        self.assert_eq(
            pd.to_datetime([1, 2, 3], unit="D", origin=pd.Timestamp("1960-01-01")),
            ks.to_datetime([1, 2, 3], unit="D", origin=pd.Timestamp("1960-01-01")),
        )

    def test_concat(self):
        pdf = pd.DataFrame({"A": [0, 2, 4], "B": [1, 3, 5]})
        kdf = ks.from_pandas(pdf)

        self.assert_eq(ks.concat([kdf, kdf.reset_index()]), pd.concat([pdf, pdf.reset_index()]))

        self.assert_eq(
            ks.concat([kdf, kdf[["A"]]], ignore_index=True),
            pd.concat([pdf, pdf[["A"]]], ignore_index=True),
        )

        self.assert_eq(
            ks.concat([kdf, kdf[["A"]]], join="inner"), pd.concat([pdf, pdf[["A"]]], join="inner")
        )

        self.assertRaisesRegex(TypeError, "first argument must be", lambda: ks.concat(kdf))
        self.assertRaisesRegex(TypeError, "cannot concatenate object", lambda: ks.concat([kdf, 1]))

        kdf2 = kdf.set_index("B", append=True)
        self.assertRaisesRegex(
            ValueError, "Index type and names should be same", lambda: ks.concat([kdf, kdf2])
        )

        self.assertRaisesRegex(ValueError, "No objects to concatenate", lambda: ks.concat([]))

        self.assertRaisesRegex(ValueError, "All objects passed", lambda: ks.concat([None, None]))

        pdf3 = pdf.copy()
        kdf3 = kdf.copy()

        columns = pd.MultiIndex.from_tuples([("X", "A"), ("X", "B")])
        pdf3.columns = columns
        kdf3.columns = columns

        self.assert_eq(ks.concat([kdf3, kdf3.reset_index()]), pd.concat([pdf3, pdf3.reset_index()]))

        self.assert_eq(
            ks.concat([kdf3, kdf3[[("X", "A")]]], ignore_index=True),
            pd.concat([pdf3, pdf3[[("X", "A")]]], ignore_index=True),
        )

        self.assert_eq(
            ks.concat([kdf3, kdf3[[("X", "A")]]], join="inner"),
            pd.concat([pdf3, pdf3[[("X", "A")]]], join="inner"),
        )

        self.assertRaisesRegex(
            ValueError,
            "MultiIndex columns should have the same levels",
            lambda: ks.concat([kdf, kdf3]),
        )

        pdf4 = pd.DataFrame({"A": [0, 2, 4], "B": [1, 3, 5], "C": [10, 20, 30]})
        kdf4 = ks.from_pandas(pdf4)
        self.assertRaisesRegex(
            ValueError,
            r"Only can inner \(intersect\) or outer \(union\) join the other axis.",
            lambda: ks.concat([kdf, kdf4], join=""),
        )

        self.assertRaisesRegex(
            ValueError,
            r"Only can inner \(intersect\) or outer \(union\) join the other axis.",
            lambda: ks.concat([kdf, kdf4], join="", axis=1),
        )

        self.assertRaisesRegex(
            ValueError,
            r"Only can inner \(intersect\) or outer \(union\) join the other axis.",
            lambda: ks.concat([kdf.A, kdf4.B], join="", axis=1),
        )

        self.assertRaisesRegex(
            ValueError,
            r"Labels have to be unique; however, got duplicated labels \['A'\].",
            lambda: ks.concat([kdf.A, kdf4.A], join="inner", axis=1),
        )

    def test_concat_column_axis(self):
        pdf1 = pd.DataFrame({"A": [0, 2, 4], "B": [1, 3, 5]}, index=[1, 2, 3])
        pdf2 = pd.DataFrame({"C": [1, 2, 3], "D": [4, 5, 6]}, index=[1, 3, 5])
        kdf1 = ks.from_pandas(pdf1)
        kdf2 = ks.from_pandas(pdf2)

        kdf3 = kdf1.copy()
        kdf4 = kdf2.copy()
        pdf3 = pdf1.copy()
        pdf4 = pdf2.copy()

        columns = pd.MultiIndex.from_tuples([("X", "A"), ("X", "B")])
        pdf3.columns = columns
        kdf3.columns = columns

        columns = pd.MultiIndex.from_tuples([("X", "C"), ("X", "D")])
        pdf4.columns = columns
        kdf4.columns = columns

        pdf5 = pd.DataFrame({"A": [0, 2, 4], "B": [1, 3, 5]}, index=[1, 2, 3])
        pdf6 = pd.DataFrame({"C": [1, 2, 3]}, index=[1, 3, 5])
        kdf5 = ks.from_pandas(pdf5)
        kdf6 = ks.from_pandas(pdf6)

        ignore_indexes = [True, False]
        joins = ["inner", "outer"]

        objs = [
            ([kdf1.A, kdf2.C], [pdf1.A, pdf2.C]),
            ([kdf1, kdf2.C], [pdf1, pdf2.C]),
            ([kdf1.A, kdf2], [pdf1.A, pdf2]),
            ([kdf1.A, kdf2.C], [pdf1.A, pdf2.C]),
            ([kdf1.A, kdf1.A.rename("B")], [pdf1.A, pdf1.A.rename("B")]),
            ([kdf3[("X", "A")], kdf4[("X", "C")]], [pdf3[("X", "A")], pdf4[("X", "C")]]),
            ([kdf3, kdf4[("X", "C")]], [pdf3, pdf4[("X", "C")]]),
            ([kdf3[("X", "A")], kdf4], [pdf3[("X", "A")], pdf4]),
            ([kdf3, kdf4], [pdf3, pdf4]),
            ([kdf3[("X", "A")], kdf3[("X", "B")]], [pdf3[("X", "A")], pdf3[("X", "B")]],),
            (
                [kdf3[("X", "A")], kdf3[("X", "B")].rename("ABC")],
                [pdf3[("X", "A")], pdf3[("X", "B")].rename("ABC")],
            ),
            (
                [kdf3[("X", "A")].rename("ABC"), kdf3[("X", "B")]],
                [pdf3[("X", "A")].rename("ABC"), pdf3[("X", "B")]],
            ),
            ([kdf5, kdf6], [pdf5, pdf6]),
            ([kdf6, kdf5], [pdf6, pdf5]),
        ]

        for ignore_index, join in itertools.product(ignore_indexes, joins):
            for obj in objs:
                kdfs, pdfs = obj
                with self.subTest(ignore_index=ignore_index, join=join, objs=obj):
                    actual = ks.concat(kdfs, axis=1, ignore_index=ignore_index, join=join)
                    expected = pd.concat(pdfs, axis=1, ignore_index=ignore_index, join=join)
                    self.assert_eq(
                        repr(actual.sort_values(list(actual.columns)).reset_index(drop=True)),
                        repr(expected.sort_values(list(expected.columns)).reset_index(drop=True)),
                    )

    # test dataframes equality with broadcast hint.
    def test_broadcast(self):
        kdf = ks.DataFrame(
            {"key": ["K0", "K1", "K2", "K3"], "A": ["A0", "A1", "A2", "A3"]}, columns=["key", "A"]
        )
        self.assert_eq(kdf, ks.broadcast(kdf))

        kser = ks.Series([1, 2, 3])
        expected_error_message = "Invalid type : expected DataFrame got {}".format(type(kser))
        with self.assertRaisesRegex(ValueError, expected_error_message):
            ks.broadcast(kser)
