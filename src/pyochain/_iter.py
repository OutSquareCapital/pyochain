from __future__ import annotations

from typing import TYPE_CHECKING, override

from .abc import PyoIterator

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

BOOL_SENTINEL = object()
"""Sentinel object used in `Iter::__bool__` to check for emptiness without consuming elements."""


class Iter[T](PyoIterator[T]):
    """Concrete implementation for `abc::PyoIterator`.

    Can be instantiated from any `Iterable` (like lists, sets, generators, etc.) efficiently (it only calls the builtin `iter()` on the input).

    As such, creating an `Iter` from an `Iterator` is virtually free.

    Tip:
        `Iter::__iter__()` returns the underlying wrapped `Iterator`, hence native speed is kept.

        i.e `Iter([...]).map(f).collect(list)` is as fast as `list(map(f, [...]))`.

    Args:
        data (Iterable[T]): Any object that can be iterated over.

    See Also:
        [`abc::PyoIterator`][PyoIterator]: The abstract base class that `Iter` implements.

    Example:
        ```python
        >>> from pyochain import Iter, Seq
        >>>
        >>> data = (0, 1, 2, 3, 4)
        >>> Iter(data).collect(Seq)
        Seq(0, 1, 2, 3, 4)
        >>> iterator = Iter(data)
        >>> # First we have a tuple iterator
        >>> iterator._inner.__class__.__name__
        'tuple_iterator'
        >>> # Now we have a map object
        >>> mapped = iterator.map(lambda x: x * 2)
        >>> mapped._inner.__class__.__name__
        'map'
        >>> # We collect it, by default into a Seq
        >>> mapped.collect(Seq)
        Seq(0, 2, 4, 6, 8)
        >>> # iterator is now exhausted
        >>> iterator.collect(Seq)
        Seq()

        ```
        You can also easily create an `Iter` from a generator expression:
        ```python
        >>> from pyochain import Iter
        >>> gen_expr = (x * x for x in range(5))
        >>> Iter(gen_expr).collect(Seq)
        Seq(0, 1, 4, 9, 16)

        ```
        Or from a generator function:
        ```python
        >>> from pyochain import Iter
        >>> def gen_func():
        ...     for x in range(5):
        ...         yield x * x
        >>>
        >>> Iter(gen_func()).collect(Seq)
        Seq(0, 1, 4, 9, 16)

        ```
    """

    _inner: Iterator[T]
    __slots__ = ("_inner",)  # pyright: ignore[reportUnannotatedClassAttribute]

    def __init__(self, data: Iterable[T]) -> None:
        self._inner = iter(data)

    @override
    def __iter__(self) -> Iterator[T]:
        return self._inner

    @override
    def __next__(self) -> T:
        return next(self._inner)

    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._inner.__repr__()})"
