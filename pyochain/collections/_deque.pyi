from collections import deque
from collections.abc import Iterable, Iterator
from typing import Final, Self, SupportsIndex, final, overload, override

from pyochain.abc import PyoMutableSequence

type IntoDeque[T] = deque[T] | Deque[T]

@final
class Deque[T](PyoMutableSequence[T]):
    """Returns a new `Deque` object initialized left-to-right (using append()) from `data`.

    Deques are a generalization of stacks and queues (the name is pronounced “deck” and is short for “double-ended queue”).

    Deques support thread-safe, memory efficient appends and pops from either side of the deque with approximately the same O(1) performance in either direction.

    Though list objects support similar operations, they are optimized for fast fixed-length operations and incur O(n) memory movement costs for pop(0) and insert(0, v) operations which change both the size and position of the underlying data representation.

    If `max_length` is not specified or is `None`, `Deque`s may grow to an arbitrary length.

    Otherwise, the `Deque` is bounded to the specified maximum length.

    Once a bounded length `Deque` is full, when new items are added, a corresponding number of items are discarded from the opposite end.

    Bounded length `Deque`s provide functionality similar to the tail filter in Unix.

    They are also useful for tracking transactions and other pools of data where only the most recent activity is of interest.

    Args:
        data (Iterable[T]): the elements to initialize the `Deque` with. If not specified, the `Deque` is initialized empty.
        max_length (int | None): the maximum length of the `Deque`. If not specified or None, the `Deque` is unbounded.

    Note:
        Adapted from Python Software Foundation documentation for `collections.deque`.

        Copyright (c) 2001-2024 PSF. Used under PSF License.

        See https://docs.python.org/3/library/collections.html#collections.deque for more details.
    """

    max_length: Final[int | None]

    @overload
    def __init__(self, *, max_length: int | None = None) -> None: ...
    @overload
    def __init__(self, data: Iterable[T], max_length: int | None = None) -> None: ...
    def __init__(
        self, data: Iterable[T] = (), max_length: int | None = None
    ) -> None: ...
    @staticmethod
    def from_ref[T1](data: deque[T1]) -> Deque[T1]:
        """Create a `Deque` from a reference to an existing `deque`.

        This method wraps the provided `deque` without copying it, allowing for efficient object instanciation.

        This is the recommended way to create a `Deque` from foreign functions that return `deque` objects.

        Warning:
            Since the `Deque` directly references the original `deque`, any modifications made to the `Deque` will also affect the original `deque`, and vice versa.

        Args:
            data (deque[T1]): The `deque` to wrap.

        Returns:
            Deque[T1]: A new `Deque` instance.

        Example:
            ```python
            >>> from pyochain.collections import Deque
            >>> from collections import deque
            >>>
            >>> original = deque([1, 2, 3])
            >>> deque_obj = Deque.from_ref(original)
            >>> deque_obj
            Deque([1, 2, 3])
            >>> original.append(4)
            >>> deque_obj
            Deque([1, 2, 3, 4])

            ```
        """

    @override
    def __iter__(self) -> Iterator[T]:
        """Return an iterator over the elements in the deque."""

    def __copy__(self) -> Deque[T]:
        """Return a shallow copy of a deque."""

    @override
    def __len__(self) -> int:
        """Return len(self)."""

    @override
    # pyrefly: ignore [bad-override]
    def __getitem__(self, key: SupportsIndex, /) -> T:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return self[key]."""

    @override
    # pyrefly: ignore [bad-override]
    def __setitem__(self, key: SupportsIndex, value: T, /) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Set self[key] to value."""

    @override
    # pyrefly: ignore [bad-override]
    def __delitem__(self, key: SupportsIndex, /) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Delete self[key]."""

    @override
    def __contains__(self, key: object, /) -> bool:
        """Return bool(key in self)."""

    @override
    def __iadd__(self, value: Iterable[T], /) -> Deque[T]:
        """Implement self+=value.

        Args:
            value (Iterable[T]): The values to add to the right end of the `Deque`.

        Returns:
            Deque[T]: The modified `Deque` instance after in-place addition.
        """

    def __add__(self, value: IntoDeque[T], /) -> Deque[T]:
        """Return self+value."""
    def __mul__(self, value: int, /) -> Deque[T]:
        """Return self*value."""
    def __rmul__(self, value: int, /) -> Deque[T]: ...
    def __imul__(self, value: int, /) -> Deque[T]:
        """Implement self*=value.

        Args:
            value (int): The number of times to repeat the `Deque`.

        Returns:
            Deque[T]: The modified `Deque` instance after in-place multiplication.
        """

    def __lt__(self, value: IntoDeque[T], /) -> bool:
        """Return self<value."""

    def __le__(self, value: IntoDeque[T], /) -> bool:
        """Return self<=value."""

    def __gt__(self, value: IntoDeque[T], /) -> bool:
        """Return self>value."""
    def __ge__(self, value: IntoDeque[T], /) -> bool:
        """Return self>=value."""

    @override
    def __eq__(self, value: object, /) -> bool:
        """Return self==value."""

    @override
    def append(self, x: T, /) -> None: ...
    def append_left(self, x: T, /) -> None:
        """Add an element to the left side of the deque.

        Args:
            x (T): The element to add to the left side of the `Deque`.

        Examples:
            ```python
            >>> from pyochain.collections import Deque
            >>> d = Deque([1, 2, 3])
            >>> d.append_left(0)
            >>> d
            Deque([0, 1, 2, 3])

            ```
        """

    @override
    def extend(self, iterable: Iterable[T]) -> None: ...
    @override
    def clear(self) -> None: ...
    def copy(self) -> Deque[T]:
        """Return a shallow copy of a deque.

        Returns:
            Deque[T]: A new `Deque` instance that is a shallow copy of the original.

        Example:
            ```python
            >>> from pyochain.collections import Deque
            >>> d = Deque([1, 2, 3])
            >>> d_copy = d.copy()
            >>> d_copy
            Deque([1, 2, 3])
            >>> d.append(4)
            >>> d
            Deque([1, 2, 3, 4])
            >>> d_copy
            Deque([1, 2, 3])

            ```
        """

    def extend_left(self, iterable: Iterable[T], /) -> None:
        """Extend the left side of the deque with elements from the iterable.

        Args:
            iterable (Iterable[T]): The elements to add to the left side of the `Deque`.

        Examples:
            ```python
            >>> from pyochain.collections import Deque
            >>> d = Deque([1, 2, 3])
            >>> d.extend_left([0, -1])
            >>> d
            Deque([-1, 0, 1, 2, 3])

            ```
        """

    def pop_left(self) -> T:
        """Remove and return the leftmost element.

        Returns:
            T: The leftmost element of the deque.

        Examples:
            ```python
            >>> from pyochain.collections import Deque
            >>> d = Deque([1, 2, 3])
            >>> d.pop_left()
            1
            >>> d
            Deque([2, 3])

            ```
        """
    def rotate(self, n: int = 1, /) -> Self:
        """Rotate the deque n steps.

        Args:
            n (int): The number of steps to rotate the deque. If n is positive, rotates right. if negative, rotates left.

        Returns:
            Self: The modified `Deque` instance after rotation.

        Examples:
            ```python
            >>> from pyochain.collections import Deque
            >>> d = Deque([1, 2, 3, 4, 5])
            >>> d.rotate(2)
            Deque([4, 5, 1, 2, 3])
            >>> d.rotate(-3)
            Deque([2, 3, 4, 5, 1])

            ```
        """

    @override
    def insert(self, index: int, value: T) -> None:
        """Insert value before index."""
    @override
    def count(self, x: T, /) -> int: ...
    @override
    def index(self, x: T, start: int = 0, stop: int = ..., /) -> int: ...
    @override
    # pyrefly: ignore [bad-override]
    def pop(self) -> T: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def remove(self, value: T, /) -> None: ...
