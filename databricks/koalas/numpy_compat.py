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
from collections import OrderedDict
from typing import Callable, Any

import numpy as np
from pyspark.sql import functions as F, Column
from pyspark.sql.types import DoubleType, LongType, BooleanType


unary_np_spark_mappings = OrderedDict(
    {
        "abs": F.abs,
        "absolute": F.abs,
        "arccos": F.acos,
        "arccosh": F.pandas_udf(lambda s: np.arccosh(s), DoubleType()),
        "arcsin": F.asin,
        "arcsinh": F.pandas_udf(lambda s: np.arcsinh(s), DoubleType()),
        "arctan": F.atan,
        "arctanh": F.pandas_udf(lambda s: np.arctanh(s), DoubleType()),
        "bitwise_not": F.bitwiseNOT,
        "cbrt": F.cbrt,
        "ceil": F.ceil,
        # It requires complex type which Koalas does not support yet
        "conj": lambda _: NotImplemented,
        "conjugate": lambda _: NotImplemented,  # It requires complex type
        "cos": F.cos,
        "cosh": F.pandas_udf(lambda s: np.cosh(s), DoubleType()),
        "deg2rad": F.pandas_udf(lambda s: np.deg2rad(s), DoubleType()),
        "degrees": F.degrees,
        "exp": F.exp,
        "exp2": F.pandas_udf(lambda s: np.exp2(s), DoubleType()),
        "expm1": F.expm1,
        "fabs": F.pandas_udf(lambda s: np.fabs(s), DoubleType()),
        "floor": F.floor,
        "frexp": lambda _: NotImplemented,  # 'frexp' output lengths become different
        # and it cannot be supported via pandas UDF.
        "invert": F.pandas_udf(lambda s: np.invert(s), DoubleType()),
        "isfinite": lambda c: c != float("inf"),
        "isinf": lambda c: c == float("inf"),
        "isnan": F.isnan,
        "isnat": lambda c: NotImplemented,  # Koalas and PySpark does not have Nat concept.
        "log": F.log,
        "log10": F.log10,
        "log1p": F.log1p,
        "log2": F.pandas_udf(lambda s: np.log2(s), DoubleType()),
        "logical_not": lambda c: ~(c.cast(BooleanType())),
        "matmul": lambda _: NotImplemented,  # Can return a NumPy array in pandas.
        "negative": lambda c: c * -1,
        "positive": lambda c: c,
        "rad2deg": F.pandas_udf(lambda s: np.rad2deg(s), DoubleType()),
        "radians": F.radians,
        "reciprocal": F.pandas_udf(lambda s: np.reciprocal(s), DoubleType()),
        "rint": F.pandas_udf(lambda s: np.rint(s), DoubleType()),
        "sign": lambda c: F.when(c == 0, 0).when(c < 0, -1).otherwise(1),
        "signbit": lambda c: F.when(c < 0, True).otherwise(False),
        "sin": F.sin,
        "sinh": F.pandas_udf(lambda s: np.sinh(s), DoubleType()),
        "spacing": F.pandas_udf(lambda s: np.spacing(s), DoubleType()),
        "sqrt": F.sqrt,
        "square": F.pandas_udf(lambda s: np.square(s), DoubleType()),
        "tan": F.tan,
        "tanh": F.pandas_udf(lambda s: np.tanh(s), DoubleType()),
        "trunc": F.pandas_udf(lambda s: np.trunc(s), DoubleType()),
    }
)

binary_np_spark_mappings = OrderedDict(
    {
        "arctan2": F.atan2,
        "bitwise_and": lambda c1, c2: c1.bitwiseAND(c2),
        "bitwise_or": lambda c1, c2: c1.bitwiseOR(c2),
        "bitwise_xor": lambda c1, c2: c1.bitwiseXOR(c2),
        "copysign": F.pandas_udf(lambda s1, s2: np.copysign(s1, s2), DoubleType()),
        "float_power": F.pandas_udf(lambda s1, s2: np.float_power(s1, s2), DoubleType()),
        "floor_divide": F.pandas_udf(lambda s1, s2: np.floor_divide(s1, s2), DoubleType()),
        "fmax": F.pandas_udf(lambda s1, s2: np.fmax(s1, s2), DoubleType()),
        "fmin": F.pandas_udf(lambda s1, s2: np.fmin(s1, s2), DoubleType()),
        "fmod": F.pandas_udf(lambda s1, s2: np.fmod(s1, s2), DoubleType()),
        "gcd": F.pandas_udf(lambda s1, s2: np.gcd(s1, s2), DoubleType()),
        "heaviside": F.pandas_udf(lambda s1, s2: np.heaviside(s1, s2), DoubleType()),
        "hypot": F.hypot,
        "lcm": F.pandas_udf(lambda s1, s2: np.lcm(s1, s2), DoubleType()),
        "ldexp": F.pandas_udf(lambda s1, s2: np.ldexp(s1, s2), DoubleType()),
        "left_shift": F.pandas_udf(lambda s1, s2: np.left_shift(s1, s2), LongType()),
        "logaddexp": F.pandas_udf(lambda s1, s2: np.logaddexp(s1, s2), DoubleType()),
        "logaddexp2": F.pandas_udf(lambda s1, s2: np.logaddexp2(s1, s2), DoubleType()),
        "logical_and": lambda c1, c2: c1.cast(BooleanType()) & c2.cast(BooleanType()),
        "logical_or": lambda c1, c2: c1.cast(BooleanType()) | c2.cast(BooleanType()),
        "logical_xor": lambda c1, c2: (
            # mimics xor by logical operators.
            (c1.cast(BooleanType()) | c2.cast(BooleanType()))
            & (~(c1.cast(BooleanType())) | ~(c2.cast(BooleanType())))
        ),
        "maximum": F.greatest,
        "minimum": F.least,
        "modf": F.pandas_udf(lambda s1, s2: np.modf(s1, s2), DoubleType()),
        "nextafter": F.pandas_udf(lambda s1, s2: np.nextafter(s1, s2), DoubleType()),
        "right_shift": F.pandas_udf(lambda s1, s2: np.right_shift(s1, s2), LongType()),
    }
)


# Copied from pandas.
# See also https://docs.scipy.org/doc/numpy/reference/arrays.classes.html#standard-array-subclasses
def maybe_dispatch_ufunc_to_dunder_op(
    ser_or_index, ufunc: Callable, method: str, *inputs, **kwargs: Any
):
    special = {
        "add",
        "sub",
        "mul",
        "pow",
        "mod",
        "floordiv",
        "truediv",
        "divmod",
        "eq",
        "ne",
        "lt",
        "gt",
        "le",
        "ge",
        "remainder",
        "matmul",
    }
    aliases = {
        "absolute": "abs",  # TODO: Koalas Series and Index should implement __abs__.
        "multiply": "mul",
        "floor_divide": "floordiv",
        "true_divide": "truediv",
        "power": "pow",
        "remainder": "mod",
        "divide": "div",
        "equal": "eq",
        "not_equal": "ne",
        "less": "lt",
        "less_equal": "le",
        "greater": "gt",
        "greater_equal": "ge",
    }

    # For op(., Array) -> Array.__r{op}__
    flipped = {
        "lt": "__gt__",
        "le": "__ge__",
        "gt": "__lt__",
        "ge": "__le__",
        "eq": "__eq__",
        "ne": "__ne__",
    }

    op_name = ufunc.__name__
    op_name = aliases.get(op_name, op_name)

    def not_implemented(*args, **kwargs):
        return NotImplemented

    if method == "__call__" and op_name in special and kwargs.get("out") is None:
        if isinstance(inputs[0], type(ser_or_index)):
            name = "__{}__".format(op_name)
            return getattr(ser_or_index, name, not_implemented)(inputs[1])
        else:
            name = flipped.get(op_name, "__r{}__".format(op_name))
            return getattr(ser_or_index, name, not_implemented)(inputs[0])
    else:
        return NotImplemented


# See also https://docs.scipy.org/doc/numpy/reference/arrays.classes.html#standard-array-subclasses
def maybe_dispatch_ufunc_to_spark_func(
    ser_or_index, ufunc: Callable, method: str, *inputs, **kwargs: Any
):
    from databricks.koalas.base import _column_op

    op_name = ufunc.__name__

    if (
        method == "__call__"
        and (op_name in unary_np_spark_mappings or op_name in binary_np_spark_mappings)
        and kwargs.get("out") is None
    ):

        np_spark_map_func = unary_np_spark_mappings.get(op_name) or binary_np_spark_mappings.get(
            op_name
        )

        def convert_arguments(*args):
            args = [  # type: ignore
                F.lit(inp) if not isinstance(inp, Column) else inp for inp in args
            ]  # type: ignore
            return np_spark_map_func(*args)

        return _column_op(convert_arguments)(*inputs)  # type: ignore
    else:
        return NotImplemented
