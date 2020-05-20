#!/usr/bin/env python
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
A script to generate the missing function stubs. Before running this,
make sure you install koalas from the current checkout by running:
pip install -e .
"""

import inspect

import pandas as pd

from databricks.koalas.frame import DataFrame
from databricks.koalas.groupby import DataFrameGroupBy, SeriesGroupBy
from databricks.koalas.indexes import Index, MultiIndex
from databricks.koalas.series import Series


INDENT_LEN = 4
LINE_LEN_LIMIT = 100


def inspect_missing_functions(original_type, target_type):
    """
    Find functions which exist in original_type but not in target_type,
    or the signature is modified.

    :return: the tuple of the missing function name and its signature,
             and the name of the functions the signature of which is different
             and its original and modified signature.
    """
    missing = []
    deprecated = []
    modified = []

    for name, func in inspect.getmembers(original_type, inspect.isfunction):
        # Skip the private attributes
        if name.startswith('_'):
            continue

        original_signature = inspect.signature(func, follow_wrapped=True)

        if hasattr(target_type, name):
            f = getattr(target_type, name)
            if inspect.isfunction(f):
                target_signature = inspect.signature(f)
                if str(original_signature) != str(target_signature):
                    modified.append((name, original_signature, target_signature))
                continue

        docstring = func.__doc__
        # Use line break and indent to only cover deprecated method, not deprecated parameters
        if docstring and ('\n        .. deprecated::' in docstring):
            deprecated.append((name, original_signature))
        else:
            missing.append((name, original_signature))

    return missing, deprecated, modified


def format_arguments(arguments, prefix_len, suffix_len):
    """Format arguments not to break pydocstyle.

    :param arguments: the argument list
    :param prefix_len: the prefix length when the argument string needs line break
    :param suffix_len: the suffix length to check the line length exceeds the limit
    :return: the formatted argument string
    """
    lines = ['']

    def append_arg(arg):
        if prefix_len + len(lines[-1]) + len(', ') + len(arg) + suffix_len > LINE_LEN_LIMIT:
            lines.append('')
            append_arg(arg)
        else:
            if len(lines[-1]) > 0:
                arg = ', {}'.format(arg)
            lines[-1] += arg

    for arg in arguments:
        append_arg(arg)

    return (',\n' + (' ' * prefix_len)).join(lines)


def format_method_arguments(name, signature):
    """Format the method arguments from its name and signature.

    :return: the formatted argument string
    """
    arguments = []

    for param in signature.parameters.values():
        if param.default is not inspect.Signature.empty and isinstance(param.default, type):
            arguments.append('{}={}'.format(param.name, param.default.__name__))
        elif param.default is not inspect.Signature.empty and repr(param.default) == 'nan':
            arguments.append('{}={}'.format(param.name, 'np.nan'))
        else:
            arguments.append(str(param))

    prefix_len = INDENT_LEN + len('def {}('.format(name))
    suffix_len = len('):')
    return format_arguments(arguments, prefix_len, suffix_len)


def format_derived_from(original_type, unavailable_arguments, signature):
    """Format `@derived_from` decorator.

    :param original_type: the original type to be derived
    :param unavailable_arguments: the arguments Koalas does not support yet
    :param signature: the method signature
    :return: the formatted `@derived_from` decorator
    """
    if len(unavailable_arguments) == 0:
        return '@derived_from(pd.{})'.format(original_type.__name__)

    arguments = []

    for arg in unavailable_arguments:
        param = signature.parameters[arg]
        if param.default == inspect.Parameter.empty or \
                param.kind == inspect.Parameter.VAR_POSITIONAL or \
                param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        arguments.append(repr(arg))

    prefix = '@derived_from(pd.{}, ua_args=['.format(original_type.__name__)
    suffix = '])'
    prefix_len = INDENT_LEN + len(prefix)
    suffix_len = len(suffix)
    return '{}{}{}'.format(prefix, format_arguments(arguments, prefix_len, suffix_len), suffix)


def format_raise_errors(original_type, name, unavailable_arguments, signature):
    """
    Format raise error statements for unavailable arguments when specified the different value
    from the default value.

    :return: the formatted raise error statements
    """
    raise_errors = ''

    for arg in unavailable_arguments:
        param = signature.parameters[arg]
        if param.default == inspect.Parameter.empty or \
                param.kind == inspect.Parameter.VAR_POSITIONAL or \
                param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        if repr(param.default) == 'nan':
            not_equal = 'not np.isnan({})'.format(arg)
        elif isinstance(param.default, type):
            not_equal = '{} is not {}'.format(arg, param.default.__name__)
        elif param.default is None or \
                param.default is True or param.default is False:
            not_equal = '{} is not {}'.format(arg, repr(param.default))
        else:
            not_equal = '{} != {}'.format(arg, repr(param.default))

        raise_error_prefix = 'raise PandasNotImplementedError('
        raise_error_suffix = ')'
        arguments = format_arguments(
            arguments=["class_name='pd.{}'".format(original_type.__name__),
                       "method_name='{}'".format(name),
                       "arg_name='{}'".format(arg)],
            prefix_len=(INDENT_LEN * 3 + len(raise_error_prefix)),
            suffix_len=len(raise_error_suffix))
        raise_errors += ("""
        if {0}:
            {1}{2}{3}""".format(not_equal, raise_error_prefix, arguments, raise_error_suffix))

    return raise_errors


def make_missing_function(original_type, name, signature):
    """Make a missing functions stub.

    :return: the stub definition for the missing function
    """
    arguments = format_method_arguments(name, signature)
    error_argument = format_arguments(
        arguments=["class_name='pd.{}'".format(original_type.__name__),
                   "method_name='{}'".format(name)],
        prefix_len=(8 + len('raise PandasNotImplementedError(')),
        suffix_len=len(')'))

    return ("""
    def {0}({1}):
        \"""A stub for the equivalent method to `pd.{2}.{0}()`.

        The method `pd.{2}.{0}()` is not implemented yet.
        \"""
        raise PandasNotImplementedError({3})"""
            .format(name, arguments, original_type.__name__, error_argument))


def make_modified_function_def(original_type, name, original, target):
    """Make the modified function definition.

    :return: the definition for the modified function
    """
    arguments = format_method_arguments(name, original)
    argument_names = set(target.parameters)
    unavailable_arguments = [p for p in original.parameters if p not in argument_names]
    derived_from = format_derived_from(original_type, unavailable_arguments, original)
    raise_error = format_raise_errors(original_type, name, unavailable_arguments, original)
    return ("""
    {0}
    def {1}({2}):{3}""".format(derived_from, name, arguments, raise_error))


def _main():
    for original_type, target_type in [(pd.DataFrame, DataFrame),
                                       (pd.Series, Series),
                                       (pd.core.groupby.DataFrameGroupBy, DataFrameGroupBy),
                                       (pd.core.groupby.SeriesGroupBy, SeriesGroupBy),
                                       (pd.Index, Index),
                                       (pd.MultiIndex, MultiIndex)]:
        missing, deprecated, modified = inspect_missing_functions(original_type, target_type)

        print('MISSING functions for {}'.format(original_type.__name__))
        for name, signature in missing:
            # print(make_missing_function(original_type, name, signature))
            print("""    {0} = unsupported_function('{0}')""".format(name))

        print()
        print('DEPRECATED functions for {}'.format(original_type.__name__))
        for name, signature in deprecated:
            print("""    {0} = unsupported_function('{0}', deprecated=True)""".format(name))

        print()
        print('MODIFIED functions for {}'.format(original_type.__name__))
        for name, original, target in modified:
            print(make_modified_function_def(original_type, name, original, target))
        print()


if __name__ == '__main__':
    _main()
