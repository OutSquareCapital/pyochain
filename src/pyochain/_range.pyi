from collections.abc import Iterator
from typing import Final, SupportsIndex, final, overload, override

from pyochain.abc import PyoSequence

@final
class Range(PyoSequence[int]):
    """A wrapper around the built-in `range` type that implements the `PyoSequence` protocol.

    `start` must be specified, unlike the built-in type, but everything else is the same.

    Args:
        start (int): The starting value of the range (inclusive).
        stop (int): The ending value of the range (exclusive).
        step (int, optional): The step size between values in the range. Defaults to 1.

    Example:
        ```python
        >>> from pyochain import Range, Dict, Seq
        >>>
        >>> r = Range(1, 6, 2)
        >>> r
        Range(1, 6, 2)
        >>> r.iter().collect(Seq)
        Seq(1, 3, 5)
        >>> r.rev().collect(Seq)
        Seq(5, 3, 1)
        >>> names = ("alice", "bob", "CHARLIE", "dave")
        >>> indexed_names = (
        ...     Range(0, 100)
        ...     .iter()
        ...     .zip(names)
        ...     .map_star(lambda i, n: (i, n.title()))
        ...     .collect(Dict)
        ... )
        >>> indexed_names
        Dict(0: 'Alice', 1: 'Bob', 2: 'Charlie', 3: 'Dave')

        ```
    """

    inner: Final[range]

    def __init__(self, start: int, stop: int, step: int = 1) -> None: ...
    @override
    def __iter__(self) -> Iterator[int]: ...
    @override
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: SupportsIndex, /) -> int: ...
    @overload
    def __getitem__(self, key: slice[SupportsIndex | None], /) -> range: ...
    @override
    def __getitem__(
        self, index: SupportsIndex | slice[SupportsIndex | None]
    ) -> int | range: ...
    @override
    def __reversed__(self) -> Iterator[int]: ...
