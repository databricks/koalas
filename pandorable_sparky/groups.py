"""
A wrapper for GroupedData to behave similar to pandas.
"""
import sys

if sys.version > '3':
    basestring = unicode = str


class PandasLikeGroupBy(object):
    """
    Extends Spark's group data to do more flexible operations.
    """

    def __init__(self, df, sgd, cols):
        self._groupdata = sgd
        self._df = df
        # cols can either be:
        # none -> all the dataframe
        # a single element (string) to select a single col (and return a series)
        # a list of cols / strings for all the returns (and return a DF)
        self._cols = cols
        self._schema = _current_schema(df, cols)  # type: StructType

    def __getitem__(self, key):
        # TODO: handle deeper cases. Right now, it will break with nested columns.
        if not isinstance(key, list):
            l = [key]
        else:
            l = key
        fnames = set(self._schema.fieldNames())
        for k in l:
            if k not in fnames:
                raise ValueError("{} not a field. Possible values are {}".format(k, fnames))
        return PandasLikeGroupBy(self._df, self._groupdata, key)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(e)

    def aggregate(self, func_or_funcs, *args, **kwargs):
        """Compute aggregates and returns the result as a :class:`DataFrame`.

        The available aggregate functions can be built-in aggregation functions, such as `avg`,
        `max`, `min`, `sum`, `count`.

        :param func_or_funcs: a dict mapping from column name (string) to aggregate functions
                              (string).
        """
        if not isinstance(func_or_funcs, dict) or \
            not all(isinstance(key, basestring) and isinstance(value, basestring)
                    for key, value in func_or_funcs.items()):
            raise ValueError("aggs must be a dict mapping from column name (string) to aggregate "
                             "functions (string).")
        df = self._groupdata.agg(func_or_funcs)

        reorder = ['%s(%s)' % (value, key) for key, value in iter(func_or_funcs.items())]
        df = df._spark_select(reorder)
        df.columns = [key for key in iter(func_or_funcs.keys())]

        return df

    agg = aggregate

    def count(self):
        return self._groupdata.count()

    def sum(self):
        # Just do sums for the numerical types
        cols = self._cols or []
        df = self._groupdata.sum(*cols)
        if self._return_df():
            return df
        else:
            # The first column is the index
            return df[df.columns[-1]]

    def _return_df(self):
        return self._cols is None or isinstance(self._cols, list)


def _current_schema(df, cols):
    if not cols:
        return df.schema
    if isinstance(cols, list):
        return df.select(*cols).schema
    else:
        col = cols
        return df.select(col).schema
