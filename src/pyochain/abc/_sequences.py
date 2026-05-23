from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Iterator, MutableSequence, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, overload, override

from .. import _tools as tls  # pyright: ignore[reportMissingModuleSource]
from ..rs import NONE, Option, Some
from ._collection import PyoCollection

if TYPE_CHECKING:
    from .._iter import Iter


@dataclass(slots=True)
class DrainIterator[T](Iterator[T]):
    """An `Iterator` that drains elements from a `Vec` within a specified range.

    This class is not supposed to be used directly. Use `Vec.drain()` instead to obtain an `Iter` wrapper around it.

    See `Vec.drain()` for details.
    """

    _vec: MutableSequence[T]
    _idx: int
    _end_idx: int

    @override
    def __iter__(self) -> Self:
        return self

    @override
    def __next__(self) -> T:
        if self._idx >= self._end_idx:
            raise StopIteration
        val = self._vec.pop(self._idx)
        self._end_idx -= 1
        return val

    # TODO: replace this by del slice, benchmark it
    def __del__(self) -> None:
        pop = self._vec.pop
        while self._idx < self._end_idx:
            _ = pop(self._idx)
            self._end_idx -= 1


class PyoSequence[T](PyoCollection[T], Sequence[T], ABC):
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
        >>> my_seq.rev().collect()
        Seq(30, 20, 10)

        ```
    """

    # pyrefly: ignore [implicit-any-attribute]
    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute]

    @override
    def first(self) -> T:
        return self[0]

    @override
    def second(self) -> T:
        return self[1]

    @override
    def last(self) -> T:
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

    def rev(self) -> Iter[T]:
        """Return an `Iterator` with the elements of the `Sequence` in reverse order.

        Returns:
            Iter[T]: An `Iterator` with the elements in reverse order.

        Example:
            ```python
            >>> from pyochain import Seq, Range
            >>> Seq((1, 2, 3)).rev().collect()
            Seq(3, 2, 1)
            >>> Range(0, 5).rev().collect()
            Seq(4, 3, 2, 1, 0)

            ```
        """
        from .._iter import Iter

        return Iter(reversed(self))


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

    # pyrefly: ignore [implicit-any-attribute]
    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute]

    def into_iter(self) -> Iter[T]:
        """Creates an `Iterator` that consumes the `MutableSequence`, leaving it empty.

        Each element is extracted from `self`, yielded, and removed from `self` in a single step.

        Returns:
            Iter[T]: An `Iterator` that consumes the `MutableSequence`.

        Example:
            ```python
            >>> from pyochain import Vec
            >>> vec = Vec((1, 2, 3))
            >>> vec.into_iter().collect()
            Seq(1, 2, 3)
            >>> vec
            Vec()

            ```
            This can be used to efficiently consume collections that you know you won't need anymore.
            ```python
            >>> from pyochain import Vec
            >>> txt = "Paris.London.New York.Tokyo.Berlin"
            >>> splitted = txt.split(".")
            >>> vec = Vec.from_ref(splitted)
            >>> words = vec.into_iter().join(", ")
            >>> words
            'Paris, London, New York, Tokyo, Berlin'
            >>> vec
            Vec()
            >>> splitted
            []

            ```
            Note that the `Iterator` will stop once the `MutableSequence` original length have been iterated over, regardless of the current state of the `MutableSequence`.

            As such, if you append or insert elements before exhausting the `Iterator`, this will influence his behavior.

            For example, if you append elements at the end, this will simply lead them to not be yielded nor removed by the `Iterator`:
            ```python
            >>> from pyochain import Vec
            >>> vec = Vec((1, 2, 3))
            >>> iterator = vec.into_iter()
            >>> vec.append(4)
            >>> iterator.collect()
            Seq(1, 2, 3)
            >>> vec
            Vec(4)

            ```
            On the other hand, if you insert elements at the beginning, this will lead to the original last elements to not be yielded:
            ```python
            >>> from pyochain import Vec
            >>> vec = Vec((1, 2, 3))
            >>> iterator = vec.into_iter()
            >>> vec.insert(0, 20)
            >>> iterator.collect()
            Seq(20, 1, 2)
            >>> vec
            Vec(3)

            ```

        """
        from .._iter import Iter

        pop = self.pop
        return Iter(pop(0) for _ in range(len(self)))

    def retain(self, predicate: Callable[[T], bool]) -> None:
        """Retains only the elements specified by the *predicate*.

        In other words, remove all elements e for which the *predicate* function returns `False`.

        This method operates in place, visiting each element exactly once in forward order, and preserves the order of the retained elements.

        Note:
            This is similar to filtering, but operates in place without allocating a new collection once collected.

            For example `new_list = list(filter(predicate, my_list))` followed by `my_list.clear()` would allocate a new collection before clearing the original, resulting in higher peak memory usage.

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

    # TODO: replace this by del slice, benchmark it
    def truncate(self, length: int) -> None:
        """Shortens the `MutableSequence`, keeping the first *length* elements and dropping the rest.

        If *length* is greater or equal to the `MutableSequence` current `__len__()`, this has no effect.

        The `Vec::drain` method can emulate `Vec::truncate`, but causes the excess elements to be returned instead of dropped.

        Note:
            This is equivalent to `del seq[length:]`, except that it won't create an intermediate slice object.

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
        pop = self.pop
        for _ in range(len(self) - length):
            _ = pop()

    # NOTE: Rust does not support MutableSequence ATM. We either need to find a new implementation, or wait until MutableSequence is supported to implement this method.
    def extend_move(self, other: Self | list[T]) -> None:
        """Moves all the elements of *other* into *self*, leaving *other* empty.

        This is equivalent to `extend(other)` followed by `other.clear()`, but avoids intermediate allocations by moving elements one at a time.

        Each element is extracted from **other**, appended to **self**, and removed from **other** in a single step.

        Args:
            other (Self | list[T]): The other `MutableSequence` to move elements from.

        Example:
            ```python
            >>> from pyochain import Vec
            >>> v1 = Vec((1, 2, 3))
            >>> v2 = Vec((4, 5, 6))
            >>> v1.extend_move(v2)
            >>> v1
            Vec(1, 2, 3, 4, 5, 6)
            >>> v2
            Vec()

            ```
            If we compare to extend

            ```python
            >>> v1 = Vec((1, 2, 3))
            >>> v2 = Vec((4, 5, 6))
            >>> v1.extend(v2)
            >>> v1
            Vec(1, 2, 3, 4, 5, 6)
            >>> # At this point v2 is still intact,
            >>> # meaning that we have a full intermediate copy of v2 in memory,
            >>> # which is not the case with extend_move
            >>> v2
            Vec(4, 5, 6)
            >>> v2.clear()
            >>> v2
            Vec()

            ```
        """
        pop = other.pop
        self.extend(pop(0) for _ in range(len(other)))

    def extract_if(
        self, predicate: Callable[[T], bool], start: int = 0, end: int | None = None
    ) -> Iter[T]:
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
            Iter[T]: An `Iter` that yields the extracted elements.

        Example:
            ```python
            >>> from pyochain import Vec
            >>> data = (1, 2, 3, 4, 5)
            >>> vec = Vec(data)
            >>> extracted = vec.extract_if(lambda x: x % 2 == 0).collect()
            >>> extracted
            Seq(2, 4)
            >>> vec
            Vec(1, 3, 5)
            >>> # Extracting with a range
            >>> vec = Vec(data)
            >>> extracted = vec.extract_if(
            ...     lambda x: x % 2 == 0, start=1, end=4
            ... ).collect()
            >>> extracted
            Seq(2, 4)
            >>> vec
            Vec(1, 3, 5)

            ```
        """
        from .._iter import Iter

        def _extract_if_gen() -> Iterator[T]:
            effective_end = end if end is not None else len(self)
            i = start
            pop = self.pop
            while i < effective_end and i < len(self):
                if predicate(self[i]):
                    yield pop(i)
                    effective_end -= 1
                else:
                    i += 1

        return Iter(_extract_if_gen())

    def drain(self, start: int | None = None, end: int | None = None) -> Iter[T]:
        """Removes the subslice indicated by the given *start* and *end* from the `Vec`, returning an `Iterator` over the removed subslice.

        If the `Iterator` is dropped before being fully consumed, it drops the remaining removed elements.

        Args:
            start (int | None): Starting index of the subslice to drain. Defaults to `0` if `None`.
            end (int | None): Ending index of the subslice to drain. Defaults to `len(self)` if `None`.

        Returns:
            Iter[T]: An `Iterator` over the drained elements.

        Example:
            ```python
            >>> from pyochain import Vec
            >>> v = Vec.from_ref([1, 2, 3])
            >>> u = v.drain(1).collect()
            >>> v
            Vec(1)
            >>> u
            Seq(2, 3)

            ```
            Fully consuming the `Iterator` removes all drained elements
            ```python
            >>> from pyochain import Vec
            >>> v = Vec.from_ref([1, 2, 3])
            >>> _ = v.drain().collect()
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

        return Iter(DrainIterator(self, start or 0, end or len(self)))
