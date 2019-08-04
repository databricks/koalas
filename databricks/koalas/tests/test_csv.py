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

from contextlib import contextmanager
from distutils.version import LooseVersion

import pandas as pd

from databricks import koalas
from databricks.koalas.testing.utils import ReusedSQLTestCase, TestUtils


def normalize_text(s):
    return '\n'.join(map(str.strip, s.strip().split('\n')))


class CsvTest(ReusedSQLTestCase, TestUtils):

    @property
    def csv_text(self):
        return normalize_text(
            """
            name,amount
            Alice,100
            Bob,-200
            Charlie,300
            Dennis,400
            Edith,-500
            Frank,600
            Alice,200
            Frank,-200
            Bob,600
            Alice,400
            Frank,200
            Alice,300
            Edith,600
            """)

    @property
    def csv_text_2(self):
        return normalize_text(
            """
            A,B
            item1,1
            item2,1,2
            item3,1,2,3,4
            item4,1
            """)

    @property
    def csv_text_with_comments(self):
        return normalize_text(
            """
            # header
            %s
            # comment
            Alice,400
            Edith,600
            # footer
            """ % self.csv_text)

    @contextmanager
    def csv_file(self, csv):
        with self.temp_file() as tmp:
            with open(tmp, 'w') as f:
                f.write(csv)
            yield tmp

    def test_read_csv(self):
        with self.csv_file(self.csv_text) as fn:

            def check(header='infer', names=None, usecols=None):
                expected = pd.read_csv(fn, header=header, names=names, usecols=usecols)
                actual = koalas.read_csv(fn, header=header, names=names, usecols=usecols)
                self.assertPandasAlmostEqual(expected, actual.toPandas())

            check()
            check(header=None)
            check(header=0)
            check(names=['n', 'a'])
            check(header=0, names=['n', 'a'])
            check(usecols=[1])
            check(usecols=[1, 0])
            check(usecols=['amount'])
            check(usecols=['amount', 'name'])
            if LooseVersion("0.20.0") <= LooseVersion(pd.__version__):
                check(usecols=lambda x: x == 'amount')
            check(usecols=[])
            check(usecols=[1, 1])
            check(usecols=['amount', 'amount'])
            if LooseVersion("0.20.0") <= LooseVersion(pd.__version__):
                check(usecols=lambda x: x == 'a')
            check(names=['n', 'a'], usecols=['a'])

            # check with pyspark patch.
            expected = pd.read_csv(fn)
            actual = koalas.read_csv(fn)
            self.assertPandasAlmostEqual(expected, actual.toPandas())

            self.assertRaisesRegex(ValueError, 'non-unique',
                                   lambda: koalas.read_csv(fn, names=['n', 'n']))
            self.assertRaisesRegex(ValueError, 'does not match the number.*3',
                                   lambda: koalas.read_csv(fn, names=['n', 'a', 'b']))
            self.assertRaisesRegex(ValueError, 'does not match the number.*3',
                                   lambda: koalas.read_csv(fn, header=0, names=['n', 'a', 'b']))
            self.assertRaisesRegex(ValueError, 'Usecols do not match.*3',
                                   lambda: koalas.read_csv(fn, usecols=[1, 3]))
            self.assertRaisesRegex(ValueError, 'Usecols do not match.*col',
                                   lambda: koalas.read_csv(fn, usecols=['amount', 'col']))

    def test_read_with_spark_schema(self):
        with self.csv_file(self.csv_text_2) as fn:
            actual = koalas.read_csv(fn, names="A string, B string, C long, D long, E long")
            expected = pd.read_csv(fn, names=['A', 'B', 'C', 'D', 'E'])
            self.assertEqual(repr(expected), repr(actual))

    def test_read_csv_with_comment(self):
        with self.csv_file(self.csv_text_with_comments) as fn:
            expected = pd.read_csv(fn, comment='#')
            actual = koalas.read_csv(fn, comment='#')
            self.assertPandasAlmostEqual(expected, actual.toPandas())

            self.assertRaisesRegex(ValueError, 'Only length-1 comment characters supported',
                                   lambda: koalas.read_csv(fn, comment='').show())
            self.assertRaisesRegex(ValueError, 'Only length-1 comment characters supported',
                                   lambda: koalas.read_csv(fn, comment='##').show())
            self.assertRaisesRegex(ValueError, 'Only length-1 comment characters supported',
                                   lambda: koalas.read_csv(fn, comment=1))
            self.assertRaisesRegex(ValueError, 'Only length-1 comment characters supported',
                                   lambda: koalas.read_csv(fn, comment=[1]))

    def test_read_csv_with_mangle_dupe_cols(self):
        self.assertRaisesRegex(ValueError, 'mangle_dupe_cols',
                               lambda: koalas.read_csv('path', mangle_dupe_cols=False))

    def test_read_csv_with_parse_dates(self):
        self.assertRaisesRegex(ValueError, 'parse_dates',
                               lambda: koalas.read_csv('path', parse_dates=True))
