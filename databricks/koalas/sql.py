import _string
from typing import Dict, Any, Set, Optional
import pymysql
import inspect

from pyspark.sql import SparkSession

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
from databricks.koalas.utils import default_session
from databricks.koalas.frame import DataFrame
from databricks.koalas.series import Series


__all__ = ["sql"]

from builtins import globals as builtin_globals
from builtins import locals as builtin_locals


def sql(query: str, globals=None, locals=None, **kwargs) -> DataFrame:
    """
    Execute a SQL query and return the result as a Koalas DataFrame.

    Parameters
    ----------
    query : str
        the SQL query

    globals : the dictionary of global variables, if explicitly set by the user

    locals : the dictionary of local variables, if explicitly set by the user

    kwargs : other variables that the user may want to set manually.

    Returns
    -------
    DataFrame

    Notes
    -----

    The index is not preserved. The SQL syntax in general is too flexible to
    guarantee that the index can be preserved, so it is simply dropped.
    If you need to preserver any index, you must reset the index.

    >>> sql("SELECT * from {df}", df=ks.DataFrame({"x": [1,2]}, index=["a", "b"]))
       x
    0  1
    1  2

    The reductions return a DataFrame with a single row. This behaviour may
    change in the future and return a pandas Series.

    >>> sql("SELECT count(*) from {df}", df=ks.DataFrame({"x": [1,2]}))
       count(1)
    0         2

    In addition to the locals, globals and parameters, the function will also attempt
    to determine if the program currently runs in an IPython (or Jupyter) environment
    and to import the variables from this environment. The variables have the same
    precedence as globals. This behaviour cannot be changed.

    Arbitrary statements are not supported for the {} expressions.

    Examples
    --------

    Calling a built-in SQL function.

    >>> ks.sql("select * from range(10) where id > 7")
       id
    0   8
    1   9

    A query that depends on a local variable, a dataframe, and a parameter:

    >>> mydf = ks.DataFrame({"x": range(5)})
    >>> x = range(4)
    >>> sql("SELECT * from {mydf} m WHERE m.x IN {x} AND m.x < {mymax}", mymax=2)
       x
    0  0
    1  1

    Queries can also be arbitrarily nested in functions:

    >>> def statement():
    ...     mydf2 = ks.DataFrame({"x": range(2)})
    ...     return sql("SELECT * from {mydf2}")
    >>> statement()
       x
    0  0
    1  1

    """
    if globals is None:
        globals = _get_ipython_scope()
    _globals = builtin_globals() if globals is None else dict(globals)
    _locals = builtin_locals() if locals is None else dict(locals)
    # The default choice is the globals
    _dict = dict(_globals)
    # The vars:
    _scope = _get_local_scope()
    _dict.update(_scope)
    # Then the locals
    _dict.update(_locals)
    # Highest order of precedence is the locals
    _dict.update(kwargs)
    return SQLProcessor(_dict, query, default_session()).execute()


def _get_local_scope():
    # Get 2 scopes above (_get_local_scope -> sql -> ...) to capture the vars there.
    try:
        return inspect.stack()[2][0].f_locals
    except Exception as e:
        return {}


def _get_ipython_scope():
    """
    Tries to extract the dictionary of variables if the program is running
    in an IPython notebook environment.
    """
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell.user_ns
    except Exception as e:
        return None


class SQLProcessor(object):

    def __init__(self, scope: Dict[str, Any], statement: str, session: SparkSession):
        self._scope = scope
        self._statement = statement
        # All the temporary views created when executing this statement
        self._temp_views = set()  # type: Set[str]
        self._cached_vars = {}  # type: Dict[str, Any]
        self._normalized_statement = None  # type: Optional[str]
        self._session = session

    def execute(self) -> DataFrame:
        blocks = _string.formatter_parser(self._statement)
        # TODO: use a string builder
        res = ""
        try:
            for (pre, inner, _, _) in blocks:
                var_next = "" if inner is None else self._convert(inner)
                res = res + pre + var_next
            self._normalized_statement = res
            sdf = self._session.sql(self._normalized_statement)
        except Exception as e:
            # Simply propagate PySpark exceptions
            raise e
        finally:
            for v in self._temp_views:
                self._session.catalog.dropTempView(v)
        return DataFrame(sdf)

    def _convert(self, key) -> Any:
        # Already cached?
        if key in self._cached_vars:
            return self._cached_vars[key]
        # Analyze:
        if key not in self._scope:
            raise ValueError("The key {} in the SQL statement was not found in global,"
                             " local or parameters variables".format(key))
        var = self._scope[key]
        fillin = self._convert_var(var)
        self._cached_vars[key] = fillin
        return fillin

    def _convert_var(self, var) -> Any:
        """
        Converts a python object into a string that is legal SQL.
        """
        if isinstance(var, (int, float)):
            return str(var)
        if isinstance(var, Series):
            return self._convert_var(var.to_dataframe())
        if isinstance(var, DataFrame):
            df_id = "koalas_" + str(id(var))
            if df_id not in self._temp_views:
                sdf = var.to_spark()
                # We should not capture extra index.
                # TODO: document this
                sdf = sdf[list(var.columns)]
                sdf.createOrReplaceTempView(df_id)
            return df_id
        if isinstance(var, str):
            return pymysql.escape_string(str)
        if isinstance(var, list):
            return "(" + ", ".join([self._convert_var(v) for v in var]) + ")"
        if isinstance(var, (tuple, range)):
            return self._convert_var(list(var))
        raise ValueError("Cannot understand value of type {}: {}".format(type(var), str(var)))
