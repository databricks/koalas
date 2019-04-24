import functools
from databricks.koalas.dask.compatibility import get_named_args


def derived_from(original_klass, version=None, ua_args=[]):
    """Decorator to attach original class's docstring to the wrapped method.

    Parameters
    ----------
    original_klass: type
        Original class which the method is derived from
    version : str
        Original package version which supports the wrapped method
    ua_args : list
        List of keywords which Dask doesn't support. Keywords existing in
        original but not in Dask will automatically be added.
    """
    def wrapper(method):
        method_name = method.__name__

        try:
            # do not use wraps here, as it hides keyword arguments displayed
            # in the doc
            original_method = getattr(original_klass, method_name)
            doc = original_method.__doc__
            if doc is None:
                doc = ''

            try:
                method_args = get_named_args(method)
                original_args = get_named_args(original_method)
                not_supported = [m for m in original_args if m not in method_args]
            except ValueError:
                not_supported = []

            if len(ua_args) > 0:
                not_supported.extend(ua_args)

            if len(not_supported) > 0:
                note = ("\n        Notes\n        -----\n"
                        "        Koalas doesn't support the following argument(s).\n\n")
                args = ''.join(['        * {0}\n'.format(a) for a in not_supported])
                doc = doc + note + args
            doc = skip_doctest(doc)
            doc = extra_titles(doc)

            method.__doc__ = doc
            return method

        except AttributeError:
            module_name = original_klass.__module__.split('.')[0]

            @functools.wraps(method)
            def wrapped(*args, **kwargs):
                msg = "Base package doesn't support '{0}'.".format(method_name)
                if version is not None:
                    msg2 = " Use {0} {1} or later to use this method."
                    msg += msg2.format(module_name, version)
                raise NotImplementedError(msg)
            return wrapped
    return wrapper


def _skip_doctest(line):
    # NumPy docstring contains cursor and comment only example
    stripped = line.strip()
    if stripped == '>>>' or stripped.startswith('>>> #'):
        return stripped
    elif '>>>' in stripped and '+SKIP' not in stripped:
        return line + '  # doctest: +SKIP'
    else:
        return line


def _pandas_to_koalas_in_doctest(line):
    return line.replace("pd", "ks")


def skip_doctest(doc):
    if doc is None:
        return ''
    return '\n'.join(
        [_pandas_to_koalas_in_doctest(_skip_doctest(line)) for line in doc.split('\n')])


def extra_titles(doc):
    lines = doc.split('\n')
    titles = {i: lines[i].strip() for i in range(len(lines) - 1)
              if lines[i + 1] and all(c == '-' for c in lines[i + 1].strip())}

    seen = set()
    for i, title in sorted(titles.items()):
        if title in seen:
            new_title = 'Extra ' + title
            lines[i] = lines[i].replace(title, new_title)
            lines[i + 1] = lines[i + 1].replace('-' * len(title),
                                                '-' * len(new_title))
        else:
            seen.add(title)

    return '\n'.join(lines)
