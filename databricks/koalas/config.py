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
Infrastructure of configuration for Koalas.
"""
import json
from typing import Dict, Union, Any

from pyspark._globals import _NoValue, _NoValueType

from databricks.koalas.utils import default_session


__all__ = ['get_option', 'set_option', 'reset_option']


# dict to store registered options and their default values (key -> default).
_registered_options = {
    # This sets the maximum number of rows koalas should output when printing out various output.
    # For example, this value determines whether the repr() for a dataframe prints out fully or
    # just a truncated repr.
    "display.max_rows": 1000,  # TODO: None should support unlimited.
    "compute.max_rows": 1000,  # TODO: None should support unlimited.
}  # type: Dict[str, Any]


_key_format = 'koalas.{}'.format


class OptionError(AttributeError, KeyError):
    pass


def get_option(key: str, default: Union[str, _NoValueType] = _NoValue) -> Any:
    """
    Retrieves the value of the specified option.

    Parameters
    ----------
    key : str
        The key which should match a single option.
    default : object
        The default value if the option is not set yet. The value should be JSON serializable.

    Returns
    -------
    result : the value of the option

    Raises
    ------
    OptionError : if no such option exists and the default is not provided
    """
    _check_option(key, default)
    if default is _NoValue:
        default = _registered_options[key]
    return json.loads(default_session().conf.get(_key_format(key), default=json.dumps(default)))


def set_option(key: str, value: Any) -> None:
    """
    Sets the value of the specified option.

    Parameters
    ----------
    key : str
        The key which should match a single option.
    value : object
        New value of option. The value should be JSON serializable.

    Returns
    -------
    None
    """
    _check_option(key, value)
    default_session().conf.set(_key_format(key), json.dumps(value))


def reset_option(key: str) -> None:
    """
    Reset one option to their default value.

    Pass "all" as argument to reset all options.

    Parameters
    ----------
    key : str
        If specified only option will be reset.

    Returns
    -------
    None
    """
    _check_option(key)
    default_session().conf.unset(_key_format(key))


def _check_option(key: str, value: Union[str, _NoValueType] = _NoValue) -> None:
    if key not in _registered_options:
        raise OptionError(
            "No such option: '{}'. Available options are [{}]".format(
                key, ", ".join(list(_registered_options.keys()))))

    if value is None:
        return  # None is allowed for all types.
    if value is not _NoValue and not isinstance(value, type(_registered_options[key])):
        raise TypeError("The configuration value for '%s' was %s; however, %s is expected." % (
            key, type(value), type(_registered_options[key])))
