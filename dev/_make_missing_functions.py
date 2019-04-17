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

import inspect

import pandas as pd

from databricks.koala.frame import PandasLikeDataFrame
from databricks.koala.series import PandasLikeSeries


def inspect_missing_functions(original_type, target_type):
    missings = []
    updatings = []

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
                    updatings.append((name, original_signature, target_signature))
                continue

        missings.append((name, original_signature))

    return missings, updatings


def format_arguments(name, signature):
    args_indent = 4 * 2 + len(name) + 1  # 2 tabs and len(name) + '('
    arguments = ['']

    def append_arg(arg):
        if args_indent + len(arguments[-1]) + len(arg) + 4 > 100:  # 4 = len(', ') + len('):')
            arguments.append('')
            append_arg(arg)
        else:
            if len(arguments[-1]) > 0:
                arg = ', {}'.format(arg)
            arguments[-1] += arg

    for param in signature.parameters.values():
        if param.default is not inspect.Signature.empty and isinstance(param.default, type):
            append_arg('{}={}'.format(param.name, param.default.__name__))
        elif param.default is not inspect.Signature.empty and repr(param.default) == 'nan':
            append_arg('{}={}'.format(param.name, 'np.nan'))
        else:
            append_arg(str(param))

    return (',\n' + (' ' * args_indent)).join(arguments)


def format_ua_args(original_type, missing_arguments, signature):
    if len(missing_arguments) == 0:
        return ''

    args_indent = 4 + len('@derived_from(pd.{}, ua_args=['.format(original_type.__name__))
    arguments = ['']

    def append_arg(arg):
        if args_indent + len(arguments[-1]) + len(arg) + 4 > 100:  # 4 = len(', ') + len('])')
            arguments.append('')
            append_arg(arg)
        else:
            if len(arguments[-1]) > 0:
                arg = ', {}'.format(arg)
            arguments[-1] += arg

    for arg in missing_arguments:
        parameter = signature.parameters[arg]
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL or \
                parameter.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        append_arg(repr(arg))

    return ', ua_args=[' + (',\n' + (' ' * args_indent)).join(arguments) + ']'


def format_raise_errors(name, missing_arguments, signature):
    raise_errors = ''

    for arg in missing_arguments:
        param = signature.parameters[arg]
        if param.kind == inspect.Parameter.VAR_POSITIONAL or \
           param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        if repr(param.default) == 'nan':
            non_equal = 'not np.isnan({})'.format(arg)
        elif isinstance(param.default, type) or param.default is None or \
                param.default is True or param.default is False:
            non_equal = '{} is not {}'.format(arg, repr(param.default))
        else:
            non_equal = '{} != {}'.format(arg, repr(param.default))
        raise_errors += ("""
        if {0}:
            raise NotImplementedError("{1} currently does not support {2}")"""
                         .format(non_equal, name, arg))

    return raise_errors


def _main():
    for original_type, target_type in [(pd.DataFrame, PandasLikeDataFrame),
                                       (pd.Series, PandasLikeSeries)]:
        missings, updatings = inspect_missing_functions(original_type, target_type)

        print('MISSING: {}'.format(original_type.__name__))
        for name, signature in missings:
            arguments = format_arguments(name, signature)
            print("""
    def {0}({1}):
        \"""A stub for the equivalent method to `pd.{2}.{0}()`.

        The method `pd.{2}.{0}()` is not implemented yet.
        \"""
        raise NotImplementedError("The method `{0}()` is not implemented yet.")"""
                  .format(name, arguments, original_type.__name__))

        print('UPDATING: {}'.format(original_type.__name__))
        for name, original, target in updatings:
            arguments = format_arguments(name, original)
            argument_names = set(target.parameters)
            missing_arguments = [p for p in original.parameters if p not in argument_names]
            ua_args = format_ua_args(original_type, missing_arguments, original)
            raise_error = format_raise_errors(name, missing_arguments, original)

            print("""
    @derived_from(pd.{0}{1})
    def {2}({3}):{4}""".format(original_type.__name__, ua_args, name, arguments, raise_error))


if __name__ == '__main__':
    _main()
