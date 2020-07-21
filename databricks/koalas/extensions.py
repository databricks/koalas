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
import warnings


class CachedAccessor:
    """
    Custom property-like object.

    A descriptor for caching accessors:

    Parameters
    ----------
    name : str
        Namespace that accessor's methods, properties, etc will be accessed under, e.g. "foo" for a
        dataframe accessor yields the accessor ``df.foo``
    accessor: cls
        Class with the extension methods.

    Notes
    -----
    For accessor, the class's __init__ method assumes that you are registering an accessor for one
    of ``Series``, ``DataFrame``, or ``Index``.

    This object is not meant to be instantiated directly. Instead, use register_dataframe_accessor,
    register_series_accessor, or register_index_accessor.

    The Koalas accessor is modified based on pandas.core.accessor.
    """

    def __init__(self, name, accessor):
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        if obj is None:
            return self._accessor
        accessor_obj = self._accessor(obj)
        setattr(obj, self._name, accessor_obj)
        return accessor_obj


def _register_accessor(name, cls):
    """
    Register a custom accessor on {klass} objects.

    Parameters
    ----------
    name : str
        Name under which the accessor should be registered. A warning is issued if this name
        conflicts with a preexisting attribute.

    Returns
    -------
    callable
        A class decorator.

    See Also
    --------
    register_dataframe_accessor: Register a custom accessor on DataFrame objects
    register_series_accessor: Register a custom accessor on Series objects
    register_index_accessor: Register a custom accessor on Index objects

    Notes
    -----
    When accessed, your accessor will be initialiazed with the Koalas object the user is interacting
    with. The code signature must be:

    .. code-block:: python

        def __init__(self, koalas_obj):
            # constructor logic
        ...

    In the pandas API, if data passed to your accessor has an incorrect dtype, it's recommended to
    raise an ``AttributeError`` for consistency purposes. In Koalas, ``ValueError`` is more
    frequently used to annotate when a value's datatype is unexpected for a given method/function.

    Ultimately, you can structure this however you like, but Koalas would likely do something like
    this:

    >>> ks.Series(['a', 'b']).dt
    ...
    Traceback (most recent call last):
        ...
    ValueError: Cannot call DatetimeMethods on type StringType

    Note: This function is not meant to be used directly - instead, use register_dataframe_accessor,
    register_series_accessor, or register_index_accessor.
    """

    def decorator(accessor):
        if hasattr(cls, name):
            msg = "registration of accessor {0} under name {1} for type {2} is overriding \
                a preexisting attribute with the same name.".format(
                accessor, name, cls
            )

            warnings.warn(
                msg, UserWarning, stacklevel=2,
            )
        setattr(cls, name, CachedAccessor(name, accessor))
        return accessor

    return decorator


def register_dataframe_accessor(name):
    """
    Register a custom accessor with a DataFrame

    Parameters
    ----------
    name : str
        name used when calling the accessor after its registered

    Returns
    -------
    callable
        A class decorator.

    See Also
    --------
    register_series_accessor: Register a custom accessor on Series objects
    register_index_accessor: Register a custom accessor on Index objects

    Notes
    -----
    When accessed, your accessor will be initialiazed with the Koalas object the user is interacting
    with. The accessor's init method should always ingest the object being accessed. See the
    examples for the init signature.

    In the pandas API, if data passed to your accessor has an incorrect dtype, it's recommended to
    raise an ``AttributeError`` for consistency purposes. In Koalas, ``ValueError`` is more
    frequently used to annotate when a value's datatype is unexpected for a given method/function.

    Ultimately, you can structure this however you like, but Koalas would likely do something like
    this:

    >>> ks.Series(['a', 'b']).dt
    ...
    Traceback (most recent call last):
        ...
    ValueError: Cannot call DatetimeMethods on type StringType

    Examples
    --------
    In your library code::

        from databricks.koalas.extensions import register_dataframe_accessor

        @register_dataframe_accessor("geo")
        class GeoAccessor:

            def __init__(self, koalas_obj):
                self._obj = koalas_obj
                # other constructor logic

            @property
            def center(self):
                # return the geographic center point of this DataFrame
                lat = self._obj.latitude
                lon = self._obj.longitude
                return (float(lon.mean()), float(lat.mean()))

            def plot(self):
                # plot this array's data on a map
                pass

    Then, in an ipython session::

        >>> ## Import if the accessor is in the other file.
        >>> # from my_ext_lib import GeoAccessor
        >>> kdf = ks.DataFrame({"longitude": np.linspace(0,10),
        ...                     "latitude": np.linspace(0, 20)})
        >>> kdf.geo.center  # doctest: +SKIP
        (5.0, 10.0)

        >>> kdf.geo.plot()  # doctest: +SKIP
    """
    from databricks.koalas import DataFrame

    return _register_accessor(name, DataFrame)


def register_series_accessor(name):
    """
    Register a custom accessor with a Series object

    Parameters
    ----------
    name : str
        name used when calling the accessor after its registered

    Returns
    -------
    callable
        A class decorator.

    See Also
    --------
    register_dataframe_accessor: Register a custom accessor on DataFrame objects
    register_index_accessor: Register a custom accessor on Index objects

    Notes
    -----
    When accessed, your accessor will be initialiazed with the Koalas object the user is interacting
    with. The code signature must be::

        def __init__(self, koalas_obj):
            # constructor logic
        ...

    In the pandas API, if data passed to your accessor has an incorrect dtype, it's recommended to
    raise an ``AttributeError`` for consistency purposes. In Koalas, ``ValueError`` is more
    frequently used to annotate when a value's datatype is unexpected for a given method/function.

    Ultimately, you can structure this however you like, but Koalas would likely do something like
    this:

    >>> ks.Series(['a', 'b']).dt
    ...
    Traceback (most recent call last):
        ...
    ValueError: Cannot call DatetimeMethods on type StringType

    Examples
    --------
    In your library code::

        from databricks.koalas.extensions import register_series_accessor

        @register_series_accessor("geo")
        class GeoAccessor:

            def __init__(self, koalas_obj):
                self._obj = koalas_obj

            @property
            def is_valid(self):
                # boolean check to see if series contains valid geometry
                return True

    Then, in an ipython session::

        >>> ## Import if the accessor is in the other file.
        >>> # from my_ext_lib import GeoAccessor
        >>> kdf = ks.DataFrame({"longitude": np.linspace(0,10),
        ...                     "latitude": np.linspace(0, 20)})
        >>> kdf.longitude.geo.is_valid  # doctest: +SKIP
        True
    """
    from databricks.koalas import Series

    return _register_accessor(name, Series)


def register_index_accessor(name):
    """
    Register a custom accessor with an Index

    Parameters
    ----------
    name : str
        name used when calling the accessor after its registered

    Returns
    -------
    callable
        A class decorator.

    See Also
    --------
    register_dataframe_accessor: Register a custom accessor on DataFrame objects
    register_series_accessor: Register a custom accessor on Series objects

    Notes
    -----
    When accessed, your accessor will be initialiazed with the Koalas object the user is interacting
    with. The code signature must be::

        def __init__(self, koalas_obj):
            # constructor logic
        ...

    In the pandas API, if data passed to your accessor has an incorrect dtype, it's recommended to
    raise an ``AttributeError`` for consistency purposes. In Koalas, ``ValueError`` is more
    frequently used to annotate when a value's datatype is unexpected for a given method/function.

    Ultimately, you can structure this however you like, but Koalas would likely do something like
    this:

    >>> ks.Series(['a', 'b']).dt
    ...
    Traceback (most recent call last):
        ...
    ValueError: Cannot call DatetimeMethods on type StringType

    Examples
    --------
    In your library code::

        from databricks.koalas.extensions import register_index_accessor

        @register_index_accessor("foo")
        class CustomAccessor:

            def __init__(self, koalas_obj):
                self._obj = koalas_obj
                self.item = "baz"

            @property
            def bar(self):
                # return item value
                return self.item

    Then, in an ipython session::

        >>> ## Import if the accessor is in the other file.
        >>> # from my_ext_lib import CustomAccessor
        >>> kdf = ks.DataFrame({"longitude": np.linspace(0,10),
        ...                     "latitude": np.linspace(0, 20)})
        >>> kdf.index.foo.bar  # doctest: +SKIP
        'baz'
    """
    from databricks.koalas import Index

    return _register_accessor(name, Index)
