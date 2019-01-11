import pyspark.sql.dataframe as df
import pyspark.sql.column as col
import pyspark.sql.functions as F
import pyspark
from decorator import decorator
import types
import logging

from .structures import *
from . import namespace

logger = logging.getLogger('spark')

_TOUCHED_TEST = "_pandas_updated"


def patch_spark():
    """
    This function monkey patches Spark to make PySpark's behavior similar to Pandas.

    See the readme documentation for an exhaustive list of the changes performed by this function.

    Once this function is called, the behavior cannot be reverted.
    """
    # Directly patching the base does not work because DataFrame inherits from object
    # (known python limitation)
    # NormalDF = pyspark.sql.dataframe.DataFrame
    # PatchedDF = type("DataFrame0", (PandasLikeDataFrame, object), dict(NormalDF.__dict__))
    # pyspark.sql.dataframe.DataFrame = PatchedDF
    # pyspark.sql.DataFrame = PatchedDF

    # Just going to update the dictionary
    _inject(df.DataFrame, PandasLikeDataFrame)
    _inject(df.Column, PandasLikeSeries)
    # Override in all cases these methods to prevent any dispatching.
    df.Column.__repr__ = PandasLikeSeries.__repr__
    df.Column.__str__ = PandasLikeSeries.__str__
    # Replace the creation of the operators in columns
    _wrap_operators()
    # Wrap all the functions in the standard libraries
    _wrap_functions()
    # Inject a few useful functions.
    pyspark.read_csv = namespace.read_csv
    pyspark.to_datetime = namespace.to_datetime


@decorator
def wrap_column_function(f, *args, **kwargs):
    # Call the function first
    # print("wrap_column_function:calling {} on args={}, kwargs={}".format(f, args, kwargs))
    res = f(*args, **kwargs)
    if isinstance(res, col.Column):
        # print("res is a column")
        # Need to track where this column is coming from
        all_inputs = list(args) + list(kwargs.values())

        def ref_df(x):
            if isinstance(x, df.DataFrame):
                return x
            if isinstance(x, df.Column):
                if hasattr(x, "_spark_ref_dataframe"):
                    return x._spark_ref_dataframe
                else:
                    logger.warning("Found a column without reference: {}".format(str(x)))
            return None
        all_col_inputs = [ref_df(c) for c in all_inputs]
        # print("wrap_column_function:all_col_inputs", all_col_inputs)
        all_df_inputs = list(dict([(id(f), f) for f in all_col_inputs if f]).items())
        # print("wrap_column_function:all_df_inputs", all_df_inputs)
        if len(all_df_inputs) > 1:
            logger.warning("Too many anchors to conclude")
        elif not all_df_inputs:
            logger.warning("Could not find anchors")
        else:
            (_, df_ref) = all_df_inputs[0]
            res._spark_ref_dataframe = df_ref
        return res


def _wrap_operators():
    attrs = ["__neg__", "__add__", "__sub__", "__mul__", "__div__", "__truediv__", "__mod__",
             "__eq__", "__ne__", "__lt__", "__le__", "__ge__", "__gt__"]
    if hasattr(col.Column, _TOUCHED_TEST):
        return
    for attr in attrs:
        oldfun = getattr(col.Column, attr)
        fun = wrap_column_function(oldfun)
        setattr(col.Column, attr, fun)
    setattr(col.Column, _TOUCHED_TEST, "")


def _wrap_functions():
    all_funs = F.__all__
    if hasattr(F, _TOUCHED_TEST):
        return
    for fname in all_funs:
        if fname in ('pandas_udf',):
            continue
        oldfun = getattr(F, fname)
        if isinstance(oldfun, types.FunctionType):
            fun = wrap_column_function(oldfun)
            setattr(F, fname, fun)
    setattr(F, _TOUCHED_TEST, "")


def _inject(target_type, inject_type):
    # Make sure to resolve the base classes too.
    mro = list(inject_type.__mro__)
    mro.reverse()
    print(mro)
    # Keep a duplicate of all the existing methods:
    setattr(target_type, "_spark_getattr", target_type.__getattr__)
    setattr(target_type, "_spark_getitem", target_type.__getitem__)
    for (key, fun) in list(target_type.__dict__.items()):
        # Skip the system attributes
        if key.startswith("__") or key.startswith("_spark_"):
            continue
        setattr(target_type, "_spark_" + key, fun)

    # Inject all the methods from the hierarchy:
    setattr(target_type, "__getattr__", inject_type.__getattr__)
    setattr(target_type, "__getitem__", inject_type.__getitem__)
    for attr in ["__iter__", "__len__", "__invert__", "__setitem__"]:
        if hasattr(inject_type, attr):
            setattr(target_type, attr, inject_type.__dict__[attr])
    for t in mro:
        if t == object:
            continue
        for (key, fun) in list(t.__dict__.items()):
            # Skip the system attributes
            if key.startswith("__") or key.startswith("_spark_"):
                continue
            setattr(target_type, key, fun)
