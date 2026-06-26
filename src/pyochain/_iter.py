from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Self, override

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

    def __bool__(self) -> bool:
        """Check if the `Iterator` has at least one element (mutates **self**).

        After calling this, the `Iterator` still contains all elements.

        Returns:
            bool: True if the `Iterator` has at least one element, False otherwise.

        Example:
            ```python
            >>> from pyochain import Iter, Seq
            >>> it = Iter((1, 2, 3))
            >>> bool(it)
            True
            >>> it.collect(Seq)  # All elements still available
            Seq(1, 2, 3)

            ```
        """
        match next(self._inner, BOOL_SENTINEL):
            case sentinel if sentinel is BOOL_SENTINEL:
                return False
            case some_val:
                self._inner = itertools.chain((some_val,), self._inner)  # pyright: ignore[reportAttributeAccessIssue]
                return True

    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._inner.__repr__()})"

    @classmethod
    def from_ref(cls, other: Self) -> Self:
        """Create an independent lazy copy from another `Iter`.

        Both the original and the returned `Iter` can be consumed independently, in a lazy manner.

        Note:
            Values consumed by one iterator remain in the shared buffer until the other iterator consumes them too.

            This is the unavoidable cost of having two independent iterators over the same source.

            However, once both iterators have passed a value, it's freed from memory.

        See Also:
            - [`Iter::cloned`][cloned] which is the instance method version of this function.

        Args:
            other (Self): An `Iter` instance to copy.

        Returns:
            Self: A new `Iter` instance that is independent from the original.

        Example:
            ```python
            >>> from pyochain import Iter, Seq
            >>> original = Iter((1, 2, 3))
            >>> copy = Iter.from_ref(original)
            >>> copy.map(lambda x: x * 2).collect(Seq)
            Seq(2, 4, 6)
            >>> original.next()
            Some(1)

            ```
        """
        it1, it2 = itertools.tee(other._inner)
        other._inner = it1
        return cls(it2)

    def cloned(self) -> PyoIterator[T]:
        """Clone the `Iter` into a new independent `Iter` using `itertools.tee`.

        After calling this method, the original `Iter` will continue to yield elements independently of the cloned one.

        Note:
            Values consumed by one iterator remain in the shared buffer until the other iterator consumes them too.

            This is the unavoidable cost of having two independent iterators over the same source.

            However, once both iterators have passed a value, it's freed from memory.

        Returns:
            PyoIterator[T]: A new independent cloned iterator.

        Example:
            ```python
            >>> from pyochain import Iter, Seq
            >>> it = Iter((1, 2, 3))
            >>> cloned = it.cloned()
            >>> cloned.collect(Seq)
            Seq(1, 2, 3)
            >>> it.collect(Seq)
            Seq(1, 2, 3)

            ```
        """
        it1, it2 = itertools.tee(self._inner)
        self._inner = it1
        return self._from_iterable(it2)
