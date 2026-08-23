from collections.abc import Iterator
from typing import Self, SupportsIndex, final, overload, override

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
        from pyochain import Range, Dict, Seq

        r = Range(1, 6, 2)
        assert r == Range(1, 6, 2)
        assert r.iter().collect(Seq) == Seq(1, 3, 5)
        assert r.rev().collect(Seq) == Seq(5, 3, 1)
        names = ("alice", "bob", "CHARLIE", "dave")
        indexed_names = (
            Range(0, 100)
            .iter()
            .zip(names)
            .map_star(lambda i, n: (n.title(), i))
            .collect(Dict)
        )
        assert indexed_names == Dict(Alice=0, Bob=1, Charlie=2, Dave=3)
        ```
    """

    def __init__(self, start: int, stop: int, step: int = 1) -> None: ...
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
