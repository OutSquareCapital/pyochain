from __future__ import annotations

from abc import ABC
from collections.abc import Callable, MutableSequence
from typing import TYPE_CHECKING

from .. import (
    _tools as tls,  # pyright: ignore[reportMissingModuleSource, reportPrivateUsage]
)
from ._iterator import PyoSequence  # pyright: ignore[reportMissingModuleSource]

if TYPE_CHECKING:
    from ._iterator import PyoIterator  # pyright: ignore[reportMissingModuleSource]


class PyoMutableSequence[T](PyoSequence[T], MutableSequence[T], ABC):
    """Extends `PyoSequence[T]` and `collections.abc.MutableSequence[T]`.

    This ABC is the base class for mutable sequence types in pyochain, such as `Vec`.

    Any concrete subclass must implement the required `MutableSequence` dunder methods:

    - `__getitem__`
    - `__setitem__`
    - `__delitem__`
    - `__len__`
    - `insert`

    This class notably provides various methods inspired from Rust's `Vec` type, which provides memory-efficient in-place operations.

    They are slower than simple `.extend()`, slices and `clear()` calls, but avoids all intermediate allocations, making them suitable for large collections where memory usage is a concern.
    """

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute]

    def retain(self, predicate: Callable[[T], bool]) -> None:
        """Retains only the elements specified by the *predicate*.

        In other words, remove all elements for which the *predicate* function returns `False`.

        This is similar to filtering, but operates in place, visiting each element exactly once in forward order.

        Compared to `.iter().filter(predicate).collect(Seq)`, this avoids creating a new collection.

        The order of the retained elements is preserved.

        Args:
            predicate (Callable[[T], bool]): A function that returns `True` for elements to keep and `False` for elements to remove.

        Example:
            ```python
            >>> from pyochain import Vec, Seq
            >>> vec = Vec((1, 2, 3, 4))
            >>> vec.retain(lambda x: x % 2 == 0)
            >>> vec
            Vec(2, 4)

            ```
            External state may be used to decide which elements to keep.

            ```python
            >>> vec = Vec((1, 2, 3, 4, 5))
            >>> keep = Seq((False, True, True, False, True)).iter()
            >>> vec.retain(lambda _: next(keep))
            >>> vec
            Vec(2, 3, 5)

            ```
        """
        return tls.retain(self, predicate)

    def truncate(self, length: int) -> None:
        """Shortens the `MutableSequence`, keeping the first *length* elements and dropping the rest.

        If *length* is greater or equal to the `MutableSequence` current `__len__()`, this has no effect.

        `Vec::drain` can emulate `Vec::truncate`, but causes the excess elements to be returned instead of dropped.

        This is equivalent to `del seq[length:]`.

        Args:
            length (int): The length to truncate the `MutableSequence` to.

        Example:
            ```python
            >>> from pyochain import Vec
            >>> # Truncating a five element vector to two elements:
            >>> vec = Vec((1, 2, 3, 4, 5))
            >>> vec.truncate(2)
            >>> vec
            Vec(1, 2)

            ```
            No truncation occurs when len is greater than the `MutableSequence` current length:
            ```python
            >>> from pyochain import Vec
            >>> vec = Vec((1, 2, 3))
            >>> vec.truncate(8)
            >>> vec
            Vec(1, 2, 3)

            ```
            Truncating when len == 0 is equivalent to calling the clear method.
            ```python
            >>> from pyochain import Vec
            >>> vec = Vec((1, 2, 3))
            >>> vec.truncate(0)
            >>> vec
            Vec()

            ```
        """
        del self[length:]

    def extract_if(
        self, predicate: Callable[[T], bool], start: int = 0, end: int | None = None
    ) -> PyoIterator[T]:
        """Creates an `Iter` which uses a *predicate* to determine if an element in `Self` should be removed.

        If the *predicate* returns `True`, the element is removed from `Self` and yielded.

        If the *predicate* returns `False`, the element remains in `Self` and will not be yielded.

        You can specify a range for the extraction.

        If the returned `Iterator` is not exhausted, e.g. because it is dropped without iterating or the iteration short-circuits, then the remaining elements will be retained.

        Args:
            predicate (Callable[[T], bool]): A function that takes an element and returns `True` if it should be extracted, or `False` if it should be retained.
            start (int): The starting index of the range to consider for extraction. Defaults to `0`.
            end (int | None): The ending index of the range to consider for extraction. Defaults to `None`, which means the end of `Self`.

        Returns:
            PyoIterator[T]: An `Iterator` that yields the extracted elements.

        Example:
            ```python
            >>> from pyochain import Vec
            >>> data = (1, 2, 3, 4, 5)
            >>> vec = Vec(data)
            >>> extracted = vec.extract_if(lambda x: x % 2 == 0).collect(Vec)
            >>> extracted
            Vec(2, 4)
            >>> vec
            Vec(1, 3, 5)
            >>> # Extracting with a range
            >>> vec = Vec(data)
            >>> extracted = vec.extract_if(
            ...     lambda x: x % 2 == 0, start=1, end=4
            ... ).collect(Vec)
            >>> extracted
            Vec(2, 4)
            >>> vec
            Vec(1, 3, 5)

            ```
        """
        return tls.Iter(tls.ExtractIf(self, predicate, start, end))

    def drain(self, start: int | None = None, end: int | None = None) -> PyoIterator[T]:
        """Removes the subslice indicated by the given *start* and *end* from the `Vec`, returning an `Iterator` over the removed subslice.

        If the `Iterator` is dropped before being fully consumed, it drops the remaining removed elements.

        Args:
            start (int | None): Starting index of the subslice to drain. Defaults to `0` if `None`.
            end (int | None): Ending index of the subslice to drain. Defaults to `len(self)` if `None`.

        Returns:
            PyoIterator[T]: An `Iterator` over the drained elements.

        Example:
            ```python
            >>> from pyochain import Vec
            >>> v = Vec.from_ref([1, 2, 3])
            >>> u = v.drain(1).collect(Vec)
            >>> v
            Vec(1)
            >>> u
            Vec(2, 3)

            ```
            Fully consuming the `Iterator` removes all drained elements
            ```python
            >>> from pyochain import Vec
            >>> v = Vec.from_ref([1, 2, 3])
            >>> _ = v.drain().collect(Vec)
            >>> v
            Vec()

            ```
            Deleting the `Iterator` will also remove all drained elements.
            ```python
            >>> from pyochain import Vec
            >>> vec = Vec.from_ref([1, 2, 3])
            >>> iterator = vec.drain()
            >>> del iterator
            >>> vec
            Vec()

            ```
        """
        return tls.Iter(tls.Drain(self, start, end))
