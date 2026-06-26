from __future__ import annotations

from abc import ABC
from collections.abc import Callable, MutableSequence, Reversible, Sequence
from typing import TYPE_CHECKING, overload

from .. import _tools as tls  # pyright: ignore[reportMissingModuleSource]
from ..rs import NONE, Option, Some
from ._collection import PyoCollection

if TYPE_CHECKING:
    from ._iterator import PyoIterator  # pyright: ignore[reportMissingModuleSource]


class PyoReversible[T](Reversible[T], ABC):
    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute]

    def rev(self) -> PyoIterator[T]:
        """Return an `Iterator` with the elements of the `Sequence` in reverse order.

        Returns:
            PyoIterator[T]: An `Iterator` with the elements in reverse order.

        Example:
            ```python
            >>> from pyochain import Seq, Range
            >>> Seq((1, 2, 3)).rev().collect(Seq)
            Seq(3, 2, 1)
            >>> Range(0, 5).rev().collect(Seq)
            Seq(4, 3, 2, 1, 0)

            ```
        """
        from .._iter import Iter

        return Iter(reversed(self))


class PyoSequence[T](PyoCollection[T], PyoReversible[T], Sequence[T], ABC):
    """Extends `PyoCollection[T]` and `collections.abc.Sequence[T]`.

    Is the shared ABC for concrete sequences: `Seq`, `Range` and `Vec`.

    Any concrete subclass must implement the required `Sequence` dunder methods:

    - `__getitem__`
    - `__len__`
    - `__contains__`
    - `__iter__`

    Example:
        ```python
        >>> from pyochain.abc import PyoSequence
        >>> from pyochain import Seq
        >>>
        >>> class MySeq(PyoSequence[int]):
        ...     def __init__(self, data: list[int]):
        ...         self._data = data
        ...
        ...     def __getitem__(self, index: int) -> int:
        ...         return self._data[index]
        ...
        ...     def __len__(self) -> int:
        ...         return len(self._data)
        ...
        ...     def __contains__(self, item: int) -> bool:
        ...         return item in self._data
        ...
        ...     def __iter__(self) -> Iterator[int]:
        ...         return iter(self._data)
        >>>
        >>> my_seq = MySeq([10, 20, 30])
        >>> my_seq.first()
        10
        >>> my_seq.rev().collect(Seq)
        Seq(30, 20, 10)

        ```
    """

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute]

    def first(self) -> T:
        """Return the first element of the `Sequence`.

        Returns:
            T: The first element of the `Sequence`.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> from pyochain.collections import StableSet
            >>> data = Seq((1, 2))
            >>> data.first()
            1
            >>> # With an Iterator, the equivalent would be:
            >>> data.iter().next().unwrap()
            1

            ```
        """
        return self[0]

    def last(self) -> T:
        """Return the last element of the `Sequence`.

        This is similar to `my_sequence[-1]`.

        Returns:
            T: The last element of the `Sequence`.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> Seq((1, 2, 3)).last()
            3

            ```
        """
        return self[-1]

    @overload
    def get(self, index: int) -> Option[T]: ...
    @overload
    def get(self, index: slice) -> Option[Sequence[T]]: ...
    def get(self, index: int | slice) -> Option[T] | Option[Sequence[T]]:
        """Return the element at the specified index as `Some(value)`, or `None` if the index is out of bounds.

        Args:
            index (int | slice): The index or slice of the element to retrieve.

        Returns:
            Option[T] | Option[Sequence[T]]: `Some(value)` if the index is valid, otherwise `None`.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> data = Seq((10, 20, 30))
            >>> data.get(1)
            Some(20)
            >>> data.get(5)
            NONE

            ```
        """
        try:
            # pyrefly: ignore [bad-return]
            return Some(self[index])  # pyright: ignore[reportReturnType]
        except IndexError:
            return NONE


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
        from .._iter import Iter

        return Iter(tls.ExtractIf(self, predicate, start, end))

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
        from .._iter import Iter

        return Iter(tls.Drain(self, start, end))
