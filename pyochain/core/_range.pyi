from collections.abc import Iterator
from typing import Self, SupportsIndex, final, overload, override

from pyochain.abc import PyoSequence

@final
class Range(PyoSequence[int]):
    """A wrapper around the built-in `range` type that implements the `PyoSequence` protocol.

    Behaves identically to Python's built-in range type

    Args:
        *args: can be passed as:
            - stop (int): Range from 0 to `stop` (exclusive).
            - start (int), stop (int): Range from `start`(inclusive) to `stop`(exclusive).
            - start (int), stop (int), step (int): Range from `start` to `stop` with `step`.

    Example:
        ```python
        from pyochain import Range, Dict, Seq

        r = Range(1, 6, 2)
        assert r == Range(1, 6, 2)
        assert r.iter().collect(Seq) == Seq((1, 3, 5))
        assert r.rev().collect(Seq) == Seq((5, 3, 1))
        names = ("alice", "bob", "CHARLIE", "dave")
        indexed_names = (
            Range(100)
            .iter()
            .zip(names)
            .map_star(lambda i, n: (i, n.title()))
            .collect(Dict)
        )
        assert indexed_names == Dict({0: "Alice", 1: "Bob", 2: "Charlie", 3: "Dave"})
        ```
    """

    @overload
    def __new__(cls, stop: int, /) -> Self: ...
    @overload
    def __new__(cls, start: int, stop: int, step: int = 1, /) -> Self: ...
    def __new__(cls, *args: int) -> Self: ...
    @override
    def __iter__(self) -> Iterator[int]: ...
    @override
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: SupportsIndex, /) -> int: ...
    @overload
    def __getitem__(self, key: slice[SupportsIndex | None], /) -> Self: ...
    @override
    def __getitem__(
        self, index: SupportsIndex | slice[SupportsIndex | None]
    ) -> int | Self: ...
    @override
    def __reversed__(self) -> Iterator[int]: ...
