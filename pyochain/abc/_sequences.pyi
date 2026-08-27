from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator, MutableSequence, Sequence
from typing import Any, Protocol, Self, overload, override, runtime_checkable

from pyochain import Option
from pyochain.abc import PyoCollection, PyoIterable, PyoIterator

@runtime_checkable
class PyoReversible[T](PyoIterable[T], Protocol):
    @abstractmethod
    def __reversed__(self) -> Iterator[T]: ...
    def rev(self) -> PyoIterator[T]:
        """Return an `Iterator` with the elements of the `Sequence` in reverse order.

        Returns:
            PyoIterator[T]: An `Iterator` with the elements in reverse order.

        Example:
            ```python
            from pyochain import Seq, Range

            assert Seq(1, 2, 3).rev().collect(Seq) == Seq(3, 2, 1)
            assert Range(5).rev().collect(Seq) == Seq(4, 3, 2, 1, 0)
            ```
        """

# For some reason, `Sequence` is not a Protocol, so we have to "fake" heritance from it to make typing work.
# pyrefly: ignore [implicit-abstract-class]
class PyoSequence[T](PyoReversible[T], PyoCollection[T], Sequence[T]):  # pyright: ignore[reportImplicitAbstractClass]
    """Extends `PyoCollection[T]` and `collections.abc.Sequence[T]`.

    Is the shared ABC for concrete sequences: `Seq`, `Range` and `Vec`.

    Any concrete subclass must implement the required `Sequence` dunder methods:

    - `__getitem__`
    - `__len__`
    - `__contains__`
    - `__iter__`

    Example:
        ```python
        from collections.abc import Iterator
        from pyochain.abc import PyoSequence
        from pyochain import Seq, Some

        class MySeq(PyoSequence[int]):
            def __init__(self, data: list[int]):
                self._data = data
            def __getitem__(self, index: int) -> int:
                return self._data[index]

            def __len__(self) -> int:
                return len(self._data)

            def __contains__(self, item: int) -> bool:
                return item in self._data

            def __iter__(self) -> Iterator[int]:
                return iter(self._data)

        my_seq = MySeq([10, 20, 30])
        assert my_seq.first() == 10
        assert my_seq.get(2) == Some(30)
        ```
    """
    @overload
    @abstractmethod
    def __getitem__(self, index: int, /) -> T: ...
    @overload
    @abstractmethod
    def __getitem__(self, index: slice[int | None], /) -> Sequence[T]: ...
    @override
    def __contains__(self, value: object, /) -> bool: ...
    @override
    def __iter__(self) -> Iterator[T]: ...
    @override
    def __reversed__(self) -> Iterator[T]: ...
    @override
    def index(self, value: Any, start: int = 0, stop: int = ..., /) -> int: ...  # pyright: ignore[reportAny]
    @override
    def count(self, value: Any, /) -> int: ...  # pyright: ignore[reportAny]
    def first(self) -> T:
        """Return the first element of the `Sequence`.

        Returns:
            T: The first element of the `Sequence`.

        Example:
            ```python
            from pyochain import Seq
            from pyochain.collections import StableSet

            data = Seq(1, 2)
            assert data.first() == 1
            # With an Iterator, the equivalent would be:
            assert data.iter().next().unwrap() == 1
            ```
        """

    def last(self) -> T:
        """Return the last element of the `Sequence`.

        This is similar to `my_sequence[-1]`.

        Returns:
            T: The last element of the `Sequence`.

        Example:
            ```python
            from pyochain import Seq

            assert Seq(1, 2, 3).last() == 3
            ```
        """

    @overload
    def get(self, index: int) -> Option[T]: ...
    @overload
    def get(self, index: slice[int | None]) -> Option[Sequence[T]]: ...
    def get(self, index: int | slice[int | None]) -> Option[T] | Option[Sequence[T]]:
        """Return the element at the specified index as `Some(value)`, or `None` if the index is out of bounds.

        Args:
            index (int | slice[int | None]): The index or slice of the element to retrieve.

        Returns:
            Option[T] | Option[Sequence[T]]: `Some(value)` if the index is valid, otherwise `None`.

        Example:
            ```python
            from pyochain import Seq, Some

            data = Seq(10, 20, 30)
            assert data.get(1) == Some(20)
            assert data.get(5).is_none()
            ```
        """

class PyoMutableSequence[T](PyoSequence[T], MutableSequence[T]):  # pyright: ignore[reportImplicitAbstractClass]
    """Extends `PyoSequence[T]` and `collections.abc.MutableSequence[T]`.

    This ABC is the base class for mutable sequence types in pyochain, such as `Vec`.

    This class notably provides various methods inspired from Rust's `Vec` type, which provides memory-efficient in-place operations.

    Any concrete subclass must implement the required `MutableSequence` dunder methods:

    - `__getitem__`
    - `__setitem__`
    - `__delitem__`
    - `__len__`
    - `insert`
    """

    @overload
    @abstractmethod
    def __getitem__(self, index: int, /) -> T: ...
    @overload
    @abstractmethod
    def __getitem__(self, index: slice[int | None], /) -> MutableSequence[T]: ...
    @overload
    @abstractmethod
    def __setitem__(self, index: int, value: T, /) -> None: ...
    @overload
    @abstractmethod
    def __setitem__(self, index: slice[int | None], value: Iterable[T], /) -> None: ...
    @abstractmethod
    @override
    def __setitem__(
        self, index: int | slice[int | None], value: T | Iterable[T], /
    ) -> None: ...
    @overload
    @abstractmethod
    def __delitem__(self, index: int, /) -> None: ...
    @overload
    @abstractmethod
    def __delitem__(self, index: slice[int | None], /) -> None: ...
    @abstractmethod
    @override
    def __delitem__(self, index: int | slice[int | None], /) -> None: ...
    @override
    def __iadd__(self, values: Iterable[T], /) -> Self: ...
    @abstractmethod
    @override
    def insert(self, index: int, value: T, /) -> None: ...
    @override
    def append(self, value: T, /) -> None: ...
    @override
    def clear(self) -> None: ...
    @override
    def extend(self, values: Iterable[T], /) -> None:
        """Extend a `MutableSequence` with the contents of an `Iterable`.

        `Iterable::__iter__` returns an `Iterator` that produces a series of values, and a `MutableSequence` can also be thought of as a series of values.

        This method bridges this gap, allowing you to *extend* a `MutableSequence` by including the contents of that `Iterable`.

        Args:
            values (Iterable[T]): An `Iterable` of values to extend the `MutableSequence` with.

        Example:
            ```python
            from pyochain import Vec

            a = Vec(1, 2)
            b = Vec(3, 4)
            a.extend(b)
            assert a == Vec(1, 2, 3, 4)

            # extend and collect can be considered two sides of the same coin:
            collected = Range(5).iter().collect(Vec)
            extended = Vec()
            extended.extend(collected)
            assert collected == extended
            ```
        """
    @override
    def pop(self, index: int = -1, /) -> T: ...
    @override
    def remove(self, value: T, /) -> None: ...
    @override
    def reverse(self) -> None: ...
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
            from pyochain import Vec, Seq

            vec = Vec(1, 2, 3, 4)
            assert vec.retain(lambda x: x % 2 == 0) is None
            assert vec == Vec(2, 4)
            ```
            External state may be used to decide which elements to keep.

            ```python
            vec = Vec(1, 2, 3, 4, 5)
            keep = Seq(False, True, True, False, True).iter()
            vec.retain(lambda _: next(keep))
            assert vec == Vec(2, 3, 5)
            ```
        """

    def truncate(self, length: int) -> None:
        """Shortens the `MutableSequence`, keeping the first *length* elements and dropping the rest.

        If *length* is greater or equal to the `MutableSequence` current `__len__()`, this has no effect.

        `Vec::drain` can emulate `Vec::truncate`, but causes the excess elements to be returned instead of dropped.

        This is equivalent to `del seq[length:]`.

        Args:
            length (int): The length to truncate the `MutableSequence` to.

        Example:
            ```python
            from pyochain import Vec

            # Truncating a five element vector to two elements:
            vec = Vec(1, 2, 3, 4, 5)
            vec.truncate(2)
            assert vec == Vec(1, 2)
            ```
            No truncation occurs when len is greater than the `MutableSequence` current length:
            ```python
            vec = Vec(1, 2, 3)
            vec.truncate(8)
            assert vec == Vec(1, 2, 3)
            ```
            Truncating when len == 0 is equivalent to calling the clear method.
            ```python
            vec = Vec(1, 2, 3)
            vec.truncate(0)
            assert vec.is_empty()
            ```
        """

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
            from pyochain import Vec

            data = (1, 2, 3, 4, 5)
            vec = Vec(data)
            extracted = vec.extract_if(lambda x: x % 2 == 0).collect(Vec)
            assert extracted == Vec(2, 4)
            assert vec == Vec(1, 3, 5)
            # Extracting with a range
            vec = Vec(data)
            extracted = vec.extract_if(lambda x: x % 2 == 0, 1, 4).collect(Vec)
            assert extracted == Vec(2, 4)
            assert vec == Vec(1, 3, 5)
            ```
        """

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
            from pyochain import Vec

            v = Vec(1, 2, 3)
            u = v.drain(1).collect(Vec)
            assert v == Vec(
                1,
            )
            assert u == Vec(2, 3)
            ```
            Fully consuming the `Iterator` removes all drained elements
            ```python
            v = Vec(1, 2, 3)
            v.drain().collect(Vec)
            assert v.is_empty()
            ```
            Deleting the `Iterator` will also remove all drained elements.
            ```python
            vec = Vec(1, 2, 3)
            iterator = vec.drain()
            del iterator
            assert vec.is_empty()
            ```
        """
