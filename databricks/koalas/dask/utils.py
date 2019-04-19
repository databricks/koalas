import functools
from databricks.koalas.dask.compatibility import get_named_args

_method_cache = {}


class methodcaller(object):
    """
    Return a callable object that calls the given method on its operand.

    Unlike the builtin `operator.methodcaller`, instances of this class are
    serializable
    """

    __slots__ = ('method',)
    func = property(lambda self: self.method)  # For `funcname` to work

    def __new__(cls, method):
        if method in _method_cache:
            return _method_cache[method]
        self = object.__new__(cls)
        self.method = method
        _method_cache[method] = self
        return self

    def __call__(self, obj, *args, **kwargs):
        return getattr(obj, self.method)(*args, **kwargs)

    def __reduce__(self):
        return (methodcaller, (self.method,))

    def __str__(self):
        return "<%s: %s>" % (self.__class__.__name__, self.method)

    __repr__ = __str__


class MethodCache(object):
    """Attribute access on this object returns a methodcaller for that
    attribute.

    Examples
    --------
    >>> a = [1, 3, 3]
    >>> M.count(a, 3) == a.count(3)
    True
    """
    __getattr__ = staticmethod(methodcaller)
    __dir__ = lambda self: list(_method_cache)


M = MethodCache()


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


def skip_doctest(doc):
    if doc is None:
        return ''
    return '\n'.join([_skip_doctest(line) for line in doc.split('\n')])


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
