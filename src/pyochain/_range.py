from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, overload, override

from .traits import PyoSequence


class Range(PyoSequence[int]):
    """A wrapper around the built-in `range` type that implements the `PyoSequence` protocol.

    `start` must be specified, unlike the built-in type, but everything else is the same.
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
