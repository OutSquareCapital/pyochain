from __future__ import annotations

from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    from collections.abc import Callable


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
