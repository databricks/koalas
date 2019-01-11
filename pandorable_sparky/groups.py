
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
