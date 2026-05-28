from __future__ import annotations

from collections.abc import Callable, Collection
from collections.abc import Set as AbstractSet
from typing import overload


@overload
def no_doctest[T](obj: type[T], /) -> type[T]: ...
@overload
def no_doctest[**P, R](obj: Callable[P, R], /) -> Callable[P, R]: ...
def no_doctest(obj: object, /) -> object:
    """Decorator to mark classes/functions that should skip doctest checks.

    This decorator has zero runtime effect.

    Args:
        obj (object): The object to mark.

    Returns:
        object: the same object, unchanged
    """
    return obj


@no_doctest
def get_repr(data: Collection[object]) -> str:
    from pprint import pformat

    def _repr_inner(data: Collection[object]) -> str:
        return pformat(data, sort_dicts=False)[1:-1]

    match data:
        case AbstractSet():
            return _repr_inner(tuple(data))
        case _:
            match len(data):
                case 0:
                    return ""
                case _:
                    return _repr_inner(data)
