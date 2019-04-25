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

import pandas as pd
import numpy as np
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation


def corr(kdf, method='pearson'):
    """
    The correlation matrix of all the numerical columns of this dataframe.

    Only accepts scalar numerical values for now.

    :param kdf: the koalas dataframe.
    :param method: {'pearson', 'spearman'}
                   * pearson : standard correlation coefficient
                   * spearman : Spearman rank correlation
    :return: :class:`pandas.DataFrame`
    """
    assert method in ('pearson', 'spearman'), method
    ndf, fields = to_numeric_df(kdf)
    corr = Correlation.corr(ndf, "_1", method)
    pcorr = corr.toPandas()
    arr = pcorr.iloc[0, 0].toArray()
    arr = pd.DataFrame(arr)
    arr.columns = fields
    arr = arr.set_index(pd.Index(fields))
    return arr


def to_numeric_df(kdf):
    """
    Takes a dataframe and turns it into a dataframe containing a single numerical
    vector of doubles. This dataframe has a single field called '_1'.

    TODO: index is not preserved currently
    :param df:
    :return: a pair of dataframe, list of strings (the name of the columns
             that were converted to numerical types)
    """
    # TODO, it should be more robust.
    accepted_types = {np.dtype(dt) for dt in [np.int8, np.int16, np.int32, np.int64,
                                              np.float32, np.float64, np.bool_]}
    numeric_fields = [fname for fname in kdf._metadata.column_fields
                      if kdf[fname].dtype in accepted_types]
    numeric_df = kdf._sdf.select(*numeric_fields)
    va = VectorAssembler(inputCols=numeric_fields, outputCol="_1", handleInvalid="keep")
    v = va.transform(numeric_df).select("_1")
    return v, numeric_fields
