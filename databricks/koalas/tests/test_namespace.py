import pandas as pd
from pyspark.sql import SQLContext

from databricks import koalas
from databricks.koalas.testing.utils import ReusedSQLTestCase, SQLTestUtils


class StatsTest(ReusedSQLTestCase, SQLTestUtils):
    def test_sql(self):
        kdf = koalas.DataFrame({'A': [1, 2, 3]})
        expected_output = kdf.to_pandas()
        expected_output.insert(0, '__index_level_0__', [0, 1, 2])

        query_without_dataframe = "select * from range(10) where id > 7"
        query_result = koalas.sql(query_without_dataframe)
        self.assert_eq(query_result, pd.DataFrame({'id': [8, 9]}))

        lowercase_query = "select * from kdf"
        query_result = koalas.sql(lowercase_query)
        self.assert_eq(query_result, expected_output)
        # Also make sure the temporary table was dropped after query execution
        sql = SQLContext(self.spark)
        table_names = sql.tableNames()
        self.assertNotIn('kdf', table_names)

        uppercase_query = "SELECT * FROM kdf"
        query_result = koalas.sql(uppercase_query)
        self.assert_eq(query_result, expected_output)

        query_with_tautological_where_condition = "select * from kdf where 1=1"
        query_result = koalas.sql(query_with_tautological_where_condition)
        self.assert_eq(query_result, expected_output)

        query_with_real_where_condition = "select * from kdf where A > 1"
        query_result = koalas.sql(query_with_real_where_condition)
        self.assert_eq(query_result, expected_output.query("A > 1").reset_index(drop=True))
