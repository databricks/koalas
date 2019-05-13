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
This module has methods to hijack from Pandas'.
"""
import pandas.io.formats.format as fmt


def to_string(self, buf=None, *args, **kwargs):
    """
    Virtually same as `pandas.DataFrame.to_string` except that we add
    "showing only the first ..." for its footer.
    """
    formatter = fmt.DataFrameFormatter(frame=self, buf=buf, *args, **kwargs)

    if formatter.should_show_dimensions:
        footer = ("\n\n[showing only the first {nrows} rows x {ncols} columns]"
                  .format(nrows=len(formatter.frame), ncols=len(formatter.frame.columns)))
        formatter.show_dimensions = False
    else:
        footer = ""

    formatter.to_string()
    formatter.buf.write(footer)

    if buf is None:
        result = formatter.buf.getvalue()
        return result


def to_html(self, buf=None, columns=None, col_space=None, header=True,
            index=True, na_rep='NaN', formatters=None, float_format=None,
            sparsify=None, index_names=True, justify=None, bold_rows=True,
            classes=None, escape=True, max_rows=None, max_cols=None,
            show_dimensions=False, notebook=False, decimal='.',
            border=None, table_id=None):
    """
    Virtually same as `pandas.DataFrame.to_html` except that we add
    "showing only the first ..." for its footer.
    """
    if (justify is not None and
            justify not in fmt._VALID_JUSTIFY_PARAMETERS):
        raise ValueError("Invalid value for justify parameter")

    formatter = fmt.DataFrameFormatter(self, buf=buf, columns=columns,
                                       col_space=col_space, na_rep=na_rep,
                                       formatters=formatters,
                                       float_format=float_format,
                                       sparsify=sparsify, justify=justify,
                                       index_names=index_names,
                                       header=header, index=index,
                                       bold_rows=bold_rows, escape=escape,
                                       max_rows=max_rows,
                                       max_cols=max_cols,
                                       show_dimensions=show_dimensions,
                                       decimal=decimal, table_id=table_id)

    def _to_html(inner_self, c=None, n=False, b=None):
        from pandas.io.formats.html import HTMLFormatter
        html_renderer = HTMLFormatter(inner_self, classes=c,
                                      max_cols=inner_self.max_cols,
                                      notebook=n,
                                      border=b,
                                      table_id=inner_self.table_id)

        by = chr(215)
        footer = ('<p>showing only the first {rows} rows {by} {cols} columns</p>'
                  .format(rows=len(html_renderer.frame),
                          by=by,
                          cols=len(html_renderer.frame.columns)))

        old_write = html_renderer.write

        def write(s, indent=0):
            if s.startswith("<p>"):
                # this is the footer.
                old_write(footer, indent=indent)
            else:
                old_write(s, indent=indent)

        html_renderer.write = write

        if hasattr(inner_self.buf, 'write'):
            html_renderer.write_result(inner_self.buf)
        elif isinstance(inner_self.buf, str):
            with open(inner_self.buf, 'w') as f:
                html_renderer.write_result(f)
        else:
            raise TypeError('buf is not a file name and it has no write '
                            ' method')

    formatter.to_html = lambda *a, **kw: _to_html(formatter, *a, **kw)
    formatter.to_html(classes, notebook, border)

    if buf is None:
        return formatter.buf.getvalue()
