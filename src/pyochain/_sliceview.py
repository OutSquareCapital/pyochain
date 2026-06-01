"""sliceview — Zero-copy slice views for Python sequences."""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    MutableSequence,
    Sequence,
)
from dataclasses import dataclass
from typing import Self, SupportsIndex, overload, override

from ._utils import no_doctest
from .abc import PyoSequence


@dataclass(slots=True)
class _OpenRange:
    start: int
    step: int

    @no_doctest
    def resolve(self, b_len: int) -> range:
        """Return a concrete range clamped to the current base length."""
        stop = b_len if self.step > 0 else -1
        return range(self.start, stop, self.step)


# TODO: See if it make sense to separate mutable vs immutable slices
# TODO: See if collections should have dedicated slice views methods


class SliceView[T](PyoSequence[T]):  # noqa: PLW1641
    """A zero-copy, composable slice view over any `collections::abc::Sequence`.

    A `SliceView` presents a live window into an existing sequence:

    - reads and writes reflect the underlying sequence
    - view-to-view slicing composes in O(1)
    - no data is copied unless explicitly requested.

    Any object that implements `__len__` and `__getitem__` with integer indices is accepted

    Credits:
        - Original code and idea by @julianofischer in https://github.com/julianofischer/sliceview
        - Generically typed version by @hwelch-fle in https://github.com/hwelch-fle/sliceview which is what was used as the basis for this implementation.

        No major changes besides linter/type-checker/docstring related-changes were made, besides the name (titled `SliceView` here instead of `sliceview` in the original repos).

        And of course the pyochain integration with `PyoSequence`.

    Args:
        base (Sequence[T]): The underlying sequence.
        start (slice |int | None): Starting index of the view (inclusive). Defaults to 0.
        stop (int | None): Ending index of the view (exclusive). Has no effect if **start** is a `slice`.
        step (int | None): Step size for the view. Has no effect if **start** is a `slice`.

    Examples:
        ```python
        >>> from pyochain import SliceView
        >>> sv = SliceView([0, 1, 2, 3, 4, 5])
        >>> sv[1:4].iter().collect()
        Seq(1, 2, 3)
        >>> sv[::2].iter().collect()
        Seq(0, 2, 4)
        >>> sv2 = sv[1:][::2]  # composed — O(1), no copy
        >>> sv2.iter().collect()
        Seq(1, 3, 5)

        ```
    """

    _base: Sequence[T] | MutableSequence[T]
    _range: range | _OpenRange

    __slots__ = ("_base", "_range")  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]

    @overload
    def __init__(self, base: Sequence[T]) -> None: ...
    @overload
    def __init__(self, base: Sequence[T], start: slice) -> None: ...
    @overload
    def __init__(
        self,
        base: Sequence[T],
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> None: ...

    def __init__(
        self,
        base: Sequence[T],
        start: slice | int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> None:
        self._base = base
        if isinstance(start, slice):
            sl: slice = start
        else:
            sl = slice(start, stop, step)

        i_start, i_stop, i_step = sl.indices(len(base))

        # If the original stop was None, store an open sentinel so that the
        # view grows when elements are appended to the base.
        if sl.stop is None:  # pyright: ignore[reportAny]
            self._range = _OpenRange(i_start, i_step)
        else:
            self._range = range(i_start, i_stop, i_step)

    @classmethod
    def _from_range(cls, base: Sequence[T], r: range) -> SliceView[T]:
        sv = cls.__new__(cls)
        sv._base = base
        sv._range = r
        return sv

    @override
    def __iter__(self) -> Iterator[T]:
        base = self._base
        for i in self._current_range():
            yield base[i]

    @override
    def __contains__(self, item: object) -> bool:
        return any(item == el for el in self)

    @override
    def __reversed__(self) -> Iterator[T]:
        base = self._base
        return (base[i] for i in reversed(self._current_range()))

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Sequence):
            return len(self) == len(other) and all(
                a == b for a, b in zip(self, other, strict=False)
            )
        return False

    @override
    def __repr__(self) -> str:
        cr = self._current_range()
        name = self.__class__.__name__
        return f"{name}({self._base!r})[{cr.start}:{cr.stop}:{cr.step}]"

    @override
    def __len__(self) -> int:
        return len(self._current_range())

    @overload
    def __getitem__(self, index: SupportsIndex) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> SliceView[T]: ...
    @override
    def __getitem__(self, index: SupportsIndex | slice) -> SliceView[T] | T:
        if isinstance(index, slice):
            # Compose slices using Python's range slicing — O(1), exact.
            sub = self._current_range()[index]
            return self.__class__._from_range(self._base, sub)

        cr = self._current_range()
        length = len(cr)
        index = index.__index__()
        if index < 0:
            index += length
        if not (0 <= index < length):
            msg = "sliceview index out of range"
            raise IndexError(msg)
        return self._base[cr[index]]

    @overload
    def __setitem__(self, index: SupportsIndex, value: T) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[T]) -> None: ...
    def __setitem__(self, index: slice | SupportsIndex, value: T | Iterable[T]) -> None:
        match self._base, index:
            case MutableSequence(), slice():
                tr = self._current_range()[index]
                if abs(tr.step) != 1:
                    values: tuple[T, ...] = tuple(value)  # pyright: ignore[reportArgumentType, reportUnknownVariableType]
                    if len(values) != len(tr):
                        msg = f"attempt to assign sequence of size {len(values)} to slice of size {len(tr)}"
                        raise ValueError(msg)
                    for i, v in zip(tr, values, strict=False):
                        self._base[i] = v
                    return
                self._base[slice(tr.start, tr.stop, tr.step)] = value  # pyright: ignore[reportCallIssue, reportArgumentType]
                return
            case MutableSequence(), SupportsIndex():
                cr = self._current_range()
                length = len(cr)
                index = index.__index__()
                if index < 0:
                    index += length
                if not (0 <= index < length):
                    msg = "SliceView index out of range"
                    raise IndexError(msg)
                self._base[cr[index]] = value  # pyright: ignore[reportArgumentType, reportCallIssue]
            case _:
                msg = f"underlying sequence of type '{self._base.__class__}' has no '__setitem__'"
                raise TypeError(msg)

    def _current_range(self) -> range:
        """Return a concrete ``range`` clamped to the current base length."""
        r = self._range
        if isinstance(r, _OpenRange):
            return r.resolve(len(self._base))
        return r

    def advance(self, n: int) -> Self:
        """Shift the view's window forward by *n* index positions in-place.

        Args:
            n (int): Positions to advance (negative to retreat).

        Returns:
            Self: the view with its window advanced.

        Examples:
            This can be useful for sliding windows:

            ```python
            >>> from pyochain import SliceView, Range
            >>> data = Range(0, 10).iter().collect()
            >>> sv = SliceView(data, 0, 3)
            >>> sv.iter().collect()
            Seq(0, 1, 2)
            >>> sv.advance(3)
            SliceView(Seq(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))[3:6:1]
            >>> sv.iter().collect()
            Seq(3, 4, 5)

            ```
        """
        b_len = len(self._base)
        cr = self._current_range()
        new_start = max(0, min(cr.start + n, b_len))
        delta = new_start - cr.start
        new_stop = max(0, min(cr.stop + delta, b_len))
        self._range = range(new_start, new_stop, cr.step)
        return self

    @property
    def base(self) -> Sequence[T] | MutableSequence[T]:
        """The underlying sequence this view points into.

        Returns:
            Sequence[T] | MutableSequence[T]: The underlying sequence.

        Examples:
            ```python
            >>> from pyochain import SliceView
            >>> data = [1, 2, 3, 4, 5]
            >>> sv = SliceView(data)[1:4]
            >>> sv.base is data
            True

            ```
        """
        return self._base
