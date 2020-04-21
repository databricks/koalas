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
import numpy as np
import pandas as pd
from numpy import *
from pandas import *
from inspect import getfullargspec


def resolve_string_type_hint(tpe):
    import databricks.koalas as ks
    from databricks.koalas import DataFrame, Series

    locs = {"ks": ks, "DataFrame": DataFrame, "Series": Series}
    # This is a hack to resolve the forward reference string.
    exec("def func() -> %s: pass\narg_spec = getfullargspec(func)" % tpe, globals(), locs)
    return locs["arg_spec"].annotations.get("return", None)
