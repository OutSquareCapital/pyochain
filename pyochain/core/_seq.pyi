from collections.abc import Iterable, Iterator
from typing import Any, Self, SupportsIndex, final, overload, override

from pyochain.abc import PyoSequence

type IntoSeq[T] = Seq[T] | tuple[T, ...]

@final
class Seq[T](PyoSequence[T]):
    """Represent an in memory `Sequence`.

    Implements the `Sequence` Protocol from `collections.abc`, as well as `PyoSequence`.

    This class is notably the default return type of [`Iter::collect`][Iter.collect].

    The underlying data structure is an immutable `tuple`, hence the memory efficiency is better than a [`Vec`][Vec].

    Tip:
        `Seq(tuple)` is preferred over `Seq(list)` as this is a no-copy operation (Python optimizes `tuple` creation from another `tuple`).

        If you have an existing `list`, consider using [`Vec`][Vec] instead to avoid unnecessary copying.

        If you need immediate iteration anyway, you can directly use [`Iter`][Iter] instead.

    """
    @overload
    def __new__(cls) -> Self: ...
    @overload
    def __new__(cls, data: Iterable[T], /) -> Self: ...
    @overload
    def __new__(cls, data: T, /, *more: T) -> Self: ...
    def __new__(cls, data: Iterable[T] | T = (), /, *more: T) -> Self:
        """Create a new `Seq` instance.

        If no arguments are provided, an empty `Seq` is created.

        Passing a `tuple` or another `Seq` will not copy the underlying data.

        Args:
            data (Iterable[T] | T | None): Initial data to populate the `Seq` with. Defaults to `()`.
            *more (T): Additional elements to include in the `Seq`.

        Returns:
            Self: A new `Seq` instance.

        Example:
            ```python
            from pyochain import Seq

            py_tuple = (1, 2, 3)
            # Create a Seq from an iterable
            assert Seq(iter(py_tuple)) == py_tuple

            # Create a Seq from individual elements
            assert Seq(1, 2, 3) == py_tuple

            # Create a Seq from a tuple without copying
            seq3 = Seq(py_tuple)
            assert id(seq3[0]) == id(py_tuple[0])

            # Create an empty Seq
            assert Seq() == Seq([]) == Seq(()) == ()
            ```
        """
    @override
    def __iter__(self) -> Iterator[T]: ...
    @override
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: SupportsIndex, /) -> T: ...
    @overload
    def __getitem__(self, key: slice[SupportsIndex | None], /) -> Seq[T]: ...
    @override
    def __getitem__(
        self, index: SupportsIndex | slice[SupportsIndex | None]
    ) -> T | Seq[T]: ...
    @override
    def __hash__(self) -> int: ...
    def __add__[O](self, value: IntoSeq[O], /) -> Seq[T | O]: ...
    @override
    def __eq__(self, other: object) -> bool: ...
    def __lt__[S](self: Seq[S], value: IntoSeq[S], /) -> bool:
        """Return True if *self* is less than value, False otherwise.

        Raises `TypeError` if value is not a `Seq` or a `tuple`.

        Args:
            value (IntoSeq[S]): The value to compare against. Can be a `Seq` or a `tuple`.

        Returns:
            bool: True if *self* is less than value, False otherwise.

        Example:
            ```python
            from pyochain import Seq, Err, Ok

            s1 = Seq(1, 2, 3)
            s2 = Seq(1, 2, 4)
            assert s1 < s2
            assert not s1 < (1, 2, 3)
            try:
                res = Ok(s1 < [1, 2, 3])
            except TypeError as e:
                res = Err(e)
            assert res.is_err()
            assert res.map_err(lambda e: isinstance(e, TypeError)).unwrap_err()
            ```
        """
    def __le__[S](self: Seq[S], value: IntoSeq[S], /) -> bool: ...
    def __gt__[S](self: Seq[S], value: IntoSeq[S], /) -> bool: ...
    def __ge__[S](self: Seq[S], value: IntoSeq[S], /) -> bool: ...
    def __mul__(self, value: SupportsIndex, /) -> Seq[T]: ...
    def __rmul__(self, value: SupportsIndex, /) -> Seq[T]: ...
    @override
    def __reversed__(self) -> Iterator[T]: ...
    @override
    def count(self, value: Any, /) -> int: ...  # pyright: ignore[reportAny]
    @override
    def index(
        self,
        value: Any,  # pyright: ignore[reportAny]
        start: SupportsIndex = 0,
        stop: SupportsIndex = ...,
        /,
    ) -> int: ...
    def repeat(self, n: int) -> Self:
        """Repeat the `Seq` **n** times and return a new `Seq`.

        This is equivalent to `tuple_1 * n` for standard tuples.

        Args:
            n (int): The number of times to repeat the elements.

        Returns:
            Self: The new `Seq` after repetition.

        Example:
            ```python
            from pyochain import Seq

            s = Seq(1, 2, 3)
            assert s.repeat(2) == Seq(1, 2, 3, 1, 2, 3)
            ```
        """
    def concat[O](self, other: IntoSeq[O]) -> Seq[T | O]:
        """Concatenate another `Seq` or `tuple` to **self** and return a new `Seq`.

        This is equivalent to `tuple_1 + tuple_2` for standard tuples.

        Args:
            other (IntoSeq[O]): The other `Seq` to concatenate.

        Returns:
            Seq[T | O]: The new `Seq` after concatenation.

        Example:
            ```python
            from pyochain import Seq

            s1 = Seq(1, 2, 3)
            s2 = (4, 5, 6)  # Can also concatenate a standard tuple
            s3 = s1.concat(s2)
            assert s3 == Seq(1, 2, 3, 4, 5, 6)
            ```
        """
