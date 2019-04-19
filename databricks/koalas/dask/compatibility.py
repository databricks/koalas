import inspect

string_types = (str,)


def get_named_args(func):
    """Get all non ``*args/**kwargs`` arguments for a function"""
    s = inspect.signature(func)
    return [n for n, p in s.parameters.items()
            if p.kind == p.POSITIONAL_OR_KEYWORD]
