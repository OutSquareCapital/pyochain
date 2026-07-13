from collections.abc import Iterable, Iterator, Sequence
from typing import Final, Self, SupportsIndex, final, overload, override

from pyochain.abc import PyoSequence

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

    Args:
        data (Iterable[T]): The data to initialize the Seq with.

    Example:
        ```python
        >>> from pyochain import Seq
        >>> Seq(())
        Seq()
        >>> t = (1, 2, 3)
        >>> seq = Seq(t)
        >>> seq
        Seq(1, 2, 3)
        >>> seq_2 = Seq(seq.inner)
        >>> # No copy is made when creating seq_2 from seq.inner, they reference the same underlying tuple.
        >>> is_no_copy = (
        ...     seq.inner is seq_2.inner
        ...     and seq.inner is t
        ...     and seq_2.inner is t
        ...     and tuple(seq.inner) is t
        ... )
        >>> is_no_copy
        True
        >>> # However, creating a new Seq from seq (not using .inner) will be a copy operation.
        >>> Seq(seq).inner is seq.inner
        False

        ```
    """

    inner: Final[tuple[T, ...]]

    def __init__(self, data: Iterable[T]) -> None: ...
    @override
    def __iter__(self) -> Iterator[T]: ...
    @override
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: SupportsIndex, /) -> T: ...
    @overload
    def __getitem__(self, key: slice[SupportsIndex | None], /) -> tuple[T, ...]: ...
    @override
    def __getitem__(
        self, index: SupportsIndex | slice[SupportsIndex | None]
    ) -> T | Sequence[T]: ...
    @override
    def __eq__(self, other: object) -> bool: ...
    @override
    def __hash__(self) -> int: ...
    def repeat(self, n: int) -> Self:
        """Repeat the `Seq` **n** times and return a new `Seq`.

        This is equivalent to `tuple_1 * n` for standard tuples.

        Args:
            n (int): The number of times to repeat the elements.

        Returns:
            Self: The new `Seq` after repetition.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> s = Seq((1, 2, 3))
            >>> s.repeat(2)
            Seq(1, 2, 3, 1, 2, 3)

            ```
        """

    @override
    def __reversed__(self) -> Iterator[T]: ...
    def concat[O](self, other: tuple[O, ...] | Seq[O]) -> Seq[T | O]:
        """Concatenate another `Seq` or `tuple` to **self** and return a new `Seq`.

        This is equivalent to `tuple_1 + tuple_2` for standard tuples.

        Args:
            other (tuple[O, ...] | Seq[O]): The other `Seq` to concatenate.

        Returns:
            Seq[T | O]: The new `Seq` after concatenation.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> s1 = Seq((1, 2, 3))
            >>> s2 = (4, 5, 6)  # Can also concatenate a standard tuple
            >>> s3 = s1.concat(s2)
            >>> s3
            Seq(1, 2, 3, 4, 5, 6)

            ```
        """
