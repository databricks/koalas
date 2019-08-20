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
from typing import Dict, Union

from pyspark._globals import _NoValue, _NoValueType

from databricks.koalas.utils import default_session


__all__ = ['get_option', 'set_option', 'reset_option']


# dict to store registered options and their default values (key -> default).
_registered_options = {}  # type: Dict[str, str]


_key_format = 'koalas.{}'.format


class OptionError(AttributeError, KeyError):
    pass


def get_option(key: str, default: Union[str, _NoValueType] = _NoValue) -> str:
    """
    Retrieves the value of the specified option.

    Parameters
    ----------
    key : str
        The key which should match a single option.

    default : str
        The default value if the option is not set yet.

    Returns
    -------
    result : the value of the option

    Raises
    ------
    OptionError : if no such option exists and the default is not provided
    """
    _check_option_key(key)
    if default is _NoValue:
        default = _registered_options[key]
    return default_session().conf.get(_key_format(key), default=default)


def set_option(key: str, value: str) -> None:
    """
    Sets the value of the specified option.

    Parameters
    ----------
    key : str
        The key which should match a single option.
    value : object
        New value of option.

    Returns
    -------
    None
    """
    _check_option_key(key)
    default_session().conf.set(_key_format(key), value)


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
    _check_option_key(key)
    default_session().conf.unset(_key_format(key))


def _check_option_key(key: str) -> None:
    if key not in _registered_options:
        raise OptionError("No such key: '{}'".format(key))
