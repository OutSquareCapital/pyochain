from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, overload, override

from .abc import PyoSequence


class Range(PyoSequence[int]):
    """A wrapper around the built-in `range` type that implements the `PyoSequence` protocol.

    `start` must be specified, unlike the built-in type, but everything else is the same.

    Args:
        start (int): The starting value of the range (inclusive).
        stop (int): The ending value of the range (exclusive).
        step (int, optional): The step size between values in the range. Defaults to 1.

    Example:
        ```python
        >>> from pyochain import Range, Dict
        >>> r = Range(1, 6, 2)
        >>> r
        Range(1, 6, 2)
        >>> r.iter().collect()
        Seq(1, 3, 5)
        >>> r.rev().collect()
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

    _inner: range
    __slots__ = ("_inner",)  # pyright: ignore[reportIncompatibleUnannotatedOverride, reportUnannotatedClassAttribute]

    def __init__(self, start: int, stop: int, step: int = 1) -> None:
        self._inner = range(start, stop, step)

    @override
    def __iter__(self) -> Iterator[int]:
        return iter(self._inner)

    @override
    def __len__(self) -> int:
        return len(self._inner)

    @overload
    def __getitem__(self, index: int) -> int: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[int]: ...
    @override
    def __getitem__(self, index: int | slice[Any, Any, Any]) -> int | Sequence[int]:  # pyright: ignore[reportExplicitAny]
        return self._inner.__getitem__(index)

    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._inner.start}, {self._inner.stop}, {self._inner.step})"
