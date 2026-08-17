"""sliceview — Zero-copy slice views for Python sequences."""

from collections.abc import Iterable, Iterator, MutableSequence, Sequence
from typing import Final, Self, SupportsIndex, overload, override

from pyochain.abc import PyoSequence

class SliceView[T](PyoSequence[T]):
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

    Examples:
        ```python
        from pyochain import SliceView, Seq

        sv = SliceView([0, 1, 2, 3, 4, 5])
        assert sv[1:4].iter().collect(Seq) == Seq([1, 2, 3])
        assert sv[::2].iter().collect(Seq) == Seq([0, 2, 4])

        sv2 = sv[1:][::2]  # composed — O(1), no copy
        assert sv2.iter().collect(Seq) == Seq([1, 3, 5])
        ```
    """

    inner: Final[Sequence[T] | MutableSequence[T]]
    """Final[Sequence[T] | MutableSequence[T]]: The underlying sequence that this view is based on."""

    @overload
    def __new__(cls, base: Sequence[T]) -> Self: ...
    @overload
    def __new__(cls, base: Sequence[T], start: slice) -> Self: ...
    @overload
    def __new__(
        cls,
        base: Sequence[T],
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> Self: ...
    def __new__(
        cls,
        base: Sequence[T],
        start: slice | int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> Self:
        """Create a new `SliceView` over the given sequence.

        Args:
            base (Sequence[T]): The underlying sequence.
            start (slice |int | None): Starting index of the view (inclusive). Defaults to 0.
            stop (int | None): Ending index of the view (exclusive). Has no effect if **start** is a `slice`.
            step (int | None): Step size for the view. Has no effect if **start** is a `slice`.

        Returns:
            Self: A new `SliceView` instance.
        """
    @override
    def __iter__(self) -> Iterator[T]: ...
    @override
    def __contains__(self, item: object) -> bool: ...
    @override
    def __reversed__(self) -> Iterator[T]: ...
    @override
    def __eq__(self, other: object) -> bool: ...
    @override
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, index: SupportsIndex) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> SliceView[T]: ...
    @override
    def __getitem__(self, index: SupportsIndex | slice) -> SliceView[T] | T: ...
    @overload
    def __setitem__(self, index: SupportsIndex, value: T) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[T]) -> None: ...
    def __setitem__(
        self, index: slice | SupportsIndex, value: T | Iterable[T]
    ) -> None: ...
    def advance(self, n: int) -> Self:
        """Shift the view's window forward by *n* index positions in-place.

        Args:
            n (int): Positions to advance (negative to retreat).

        Returns:
            Self: the view with its window advanced.

        Examples:
            This can be useful for sliding windows:

            ```python
            from pyochain import SliceView, Range, Seq

            data = Range(0, 10).iter().collect(Seq)
            sv = SliceView(data, 0, 3)
            assert sv.iter().collect(Seq) == Seq([0, 1, 2])

            sv.advance(3)
            assert sv.iter().collect(Seq) == Seq([3, 4, 5])
            ```
        """
