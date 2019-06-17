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

"""
MLflow-related functions to load models and apply them to Koalas dataframes.
"""
from mlflow.pyfunc import load_pyfunc, spark_udf
from pyspark.sql.types import DataType
import pyspark.sql.functions as F
import pandas as pd
import numpy as np
from typing import Any

from databricks.koalas.utils import lazy_property, default_session
from databricks.koalas import Series, DataFrame
from databricks.koalas.typedef import as_spark_type

__all__ = ["PythonModelWrapper", "load_model"]


class PythonModelWrapper(object):
    """
    A wrapper around MLflow's Python object model.

    This wrapper acts as a predictor on koalas

    """
    def __init__(self, path, run_id, return_type_hint):
        self._path = path  # type: str
        self._run_id = run_id  # type: str
        self._return_type_hint = return_type_hint

    @lazy_property
    def _return_type(self) -> DataType:
        hint = self._return_type_hint
        # The logic is simple for now, because it corresponds to the default
        # case: continuous predictions
        # TODO: do something smarter, for example when there is a sklearn.Classifier (it should
        # return an integer or a categorical)
        # We can do the same for pytorch/tensorflow/keras models by looking at the output types.
        # However, this is probably better done in mlflow than here.
        if hint == 'infer' or not hint:
            hint = np.float64
        return as_spark_type(hint)

    @lazy_property
    def _model(self) -> Any:
        """
        The return object has to follow the API of mlflow.pyfunc.PythonModel.
        """
        return load_pyfunc(self._path, self._run_id)

    @lazy_property
    def _model_udf(self):
        spark = default_session()
        return spark_udf(spark, self._path, self._run_id, result_type=self._return_type)

    def __str__(self):
        return "PythonModelWrapper({})".format(str(self._model))

    def __repr__(self):
        return "PythonModelWrapper({})".format(repr(self._model))

    def predict(self, data):
        """
        Returns a prediction on the data.

        If the data is a koalas DataFrame, the return is a Koalas Series.

        If the data is a pandas Dataframe, the return is the expected output of the underlying
        pyfunc object (typically a pandas Series or a numpy array).
        """
        if isinstance(data, pd.DataFrame):
            return self._model.predict(data)
        if isinstance(data, DataFrame):
            cols = [data._sdf[n] for n in data.columns]
            return_col = self._model_udf(*cols)
            # TODO: the columns should be named according to the mlflow spec
            # However, this is only possible with spark >= 3.0
            # s = F.struct(*data.columns)
            # return_col = self._model_udf(s)
            return Series(data._internal.copy(scol=return_col), anchor=data)


def load_model(path, run_id=None, predict_type='infer') -> PythonModelWrapper:
    """
    Loads an MLflow model into an wrapper that can be used both for pandas and Koalas DataFrame.

    Parameters
    ----------
    path : str
        The path of the model, as logged when calling 'mlflow.log_model' or 'mlflow.save_model'
    run_id : str
        The id of the run. See MLflow runs documentation for more details.
    predict_type : a python basic type, a numpy basic type, a Spark type or 'infer'.
       This is the return type that is expected when calling the predict function of the model.
       If 'infer' is specified, the wrapper will attempt to determine automatically the return type
       based on the model type.

    Returns
    -------
    PythonModelWrapper
        A wrapper around MLflow PythonModel objects. This wrapper is expected to adhere to the
        interface of mlflow.pyfunc.PythonModel.

    Examples
    --------
    Here is a full example that creates a model with scikit-learn and saves the model with
     MLflow. The model is then loaded as a predictor that can be applied on a Koalas
     Dataframe.

    We first initialize our MLflow environment:

    >>> from mlflow.tracking import MlflowClient, set_tracking_uri
    >>> import mlflow.sklearn
    >>> from tempfile import mkdtemp
    >>> d = mkdtemp("koalas_mlflow")
    >>> set_tracking_uri("file:%s"%d)
    >>> client = MlflowClient()
    >>> exp = mlflow.create_experiment("my_experiment")
    >>> mlflow.set_experiment("my_experiment")

    We aim at learning this numerical function using a simple linear regressor.

    >>> from sklearn.linear_model import LinearRegression
    >>> train = pd.DataFrame({"x1": np.arange(8), "x2": np.arange(8)**2,
    ...                       "y": np.log(2 + np.arange(8))})
    >>> train_x = train[["x1", "x2"]]
    >>> train_y = train[["y"]]
    >>> with mlflow.start_run():
    ...     lr = LinearRegression()
    ...     lr.fit(train_x, train_y)
    ...     mlflow.sklearn.log_model(lr, "model")
    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

    Now that our model is logged using MLflow, we load it back and apply it on a Koalas dataframe:

    >>> from databricks.koalas.mlflow import load_model
    >>> run_info = client.list_run_infos(exp)[-1]
    >>> model = load_model("model", run_id = run_info.run_uuid)
    >>> prediction_df = ks.DataFrame({"x1": [2.0], "x2": [4.0]})
    >>> prediction_df["prediction"] = model.predict(prediction_df)
    >>> prediction_df
        x1   x2  prediction
    0  2.0  4.0    1.355551

    The model also works on pandas DataFrames as expected:

    >>> model.predict(prediction_df[["x1", "x2"]].toPandas())
    array([[1.35555142]])

    Notes
    -----
    Currently, the model prediction can only be merged back with the existing dataframe.
    Other columns have to be manually joined.
    For example, this code will not work:

    >>> df = ks.DataFrame({"x1": [2.0], "x2": [3.0], "z": [-1]})
    >>> features = df[["x1", "x2"]]
    >>> y = model.predict(features)
    >>> # Works:
    >>> features["y"] = y   # doctest: +SKIP
    >>> # Will fail with a message about dataframes not aligned.
    >>> df["y"] = y   # doctest: +SKIP

    A current workaround is to use the .merge() function, using the feature values
    as merging keys.

    >>> features['y'] = y
    >>> everything = df.merge(features, on=['x1', 'x2'])
    >>> everything
        x1   x2  z         y
    0  2.0  3.0 -1  1.376932
    """
    return PythonModelWrapper(path, run_id, predict_type)
