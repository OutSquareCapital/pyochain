from collections.abc import Callable, Collection
from functools import wraps


def no_doctest[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to mark functions that should skip doctest checks.

    This is type-checking only and has no runtime effect.

    Can also be marked in docstrings like so:

    @no_doctest

    Args:
        func (Callable[P, R]): The function to mark.

    Returns:
        Callable[P, R]: The original function, unmodified.
    """

    @wraps(func)
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return func(*args, **kwargs)

    return _wrapper


@no_doctest
def get_repr(data: Collection[object]) -> str:
    from pprint import pformat

    def _repr_inner(data: Collection[object]) -> str:
        return pformat(data, sort_dicts=False)[1:-1]

    match data:
        case set() | frozenset():
            return _repr_inner(tuple(data))
        case _:
            match len(data):
                case 0:
                    return ""
                case _:
                    return _repr_inner(data)
