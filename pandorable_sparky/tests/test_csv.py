from contextlib import contextmanager
import unittest

import pandas as pd
import pandorable_sparky
import pyspark

from pandorable_sparky.testing.utils import ReusedSQLTestCase, TestUtils


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
                actual = pyspark.read_csv(fn, header=header, names=names, usecols=usecols)
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
            check(usecols=lambda x: x == 'amount')
            check(usecols=[])
            check(usecols=[1, 1])
            check(usecols=['amount', 'amount'])
            check(usecols=lambda x: x == 'a')
            check(names=['n', 'a'], usecols=['a'])

            self.assertRaisesRegex(ValueError, 'non-unique',
                                   lambda: pyspark.read_csv(fn, names=['n', 'n']))
            self.assertRaisesRegex(ValueError, 'Names do not match.*3',
                                   lambda: pyspark.read_csv(fn, names=['n', 'a', 'b']))
            self.assertRaisesRegex(ValueError, 'Names do not match.*3',
                                   lambda: pyspark.read_csv(fn, header=0, names=['n', 'a', 'b']))
            self.assertRaisesRegex(ValueError, 'Usecols do not match.*3',
                                   lambda: pyspark.read_csv(fn, usecols=[1, 3]))
            self.assertRaisesRegex(ValueError, 'Usecols do not match.*col',
                                   lambda: pyspark.read_csv(fn, usecols=['amount', 'col']))

    def test_read_csv_with_comment(self):
        with self.csv_file(self.csv_text_with_comments) as fn:
            expected = pd.read_csv(fn, comment='#')
            actual = pyspark.read_csv(fn, comment='#')
            self.assertPandasAlmostEqual(expected, actual.toPandas())

            self.assertRaisesRegex(ValueError, 'Only length-1 comment characters supported',
                                   lambda: pyspark.read_csv(fn, comment='').show())
            self.assertRaisesRegex(ValueError, 'Only length-1 comment characters supported',
                                   lambda: pyspark.read_csv(fn, comment='##').show())
            self.assertRaisesRegex(ValueError, 'Only length-1 comment characters supported',
                                   lambda: pyspark.read_csv(fn, comment=1))
            self.assertRaisesRegex(ValueError, 'Only length-1 comment characters supported',
                                   lambda: pyspark.read_csv(fn, comment=[1]))


if __name__ == "__main__":
    from pandorable_sparky.tests.test_csv import *

    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
