"""Additional benchmarks not presents in the original `sortedcontainers` package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

from pyochain import Iter, Range, Seq

from ._utils import (
    LST_PARAMS,
    SC_SIZES,
    SEQ_PARAMS,
    List,
    SortedSequence,
    run_pedantic,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from .._utils import BenchFixture

type SliceFn = Callable[[int], slice]

SLICES: Final[Seq[object]] = (
    Iter[tuple[SliceFn, str]](
        (lambda _: slice(None), "full"),
        (lambda n: slice(n // 2, n // 2 + n // 16), "partial"),
        (lambda n: slice(n - n // 16, None), "to_end"),
        (lambda n: slice(n // 4, 3 * n // 4), "large"),
        (lambda n: slice(n // 4, None), "large_to_end"),
        (lambda n: slice(n // 2 + n // 16, n // 2, -1), "backward"),
        (lambda _: slice(None, None, 10), "step"),
        (lambda _: slice(None, None, -10), "backward_step"),
    )
    .map_star(lambda f, name: pytest.param(f, id=name))
    .collect(Seq)
)
"""One slice per path of `ops::Slice` and `ForwardSlice`, built from the size of the collection."""
NUM: Final[int] = 3
"""Number of times to multiply the list in `test_mul` and `test_imul`."""


@pytest.mark.parametrize("slc", SLICES)
@SEQ_PARAMS
@SC_SIZES
def test_getitem_slice(
    benchmark: BenchFixture, cls: type[SortedSequence], size: int, slc: SliceFn
) -> None:
    seq = cls(Range(size))
    idx = slc(size)
    _ = benchmark(lambda: seq[idx])


@pytest.mark.parametrize("slc", SLICES)
@SEQ_PARAMS
@SC_SIZES
def test_delitem_slice(
    benchmark: BenchFixture, cls: type[SortedSequence], size: int, slc: SliceFn
) -> None:
    idx = slc(size)

    def f(seq: SortedSequence) -> None:
        del seq[idx]

    run_pedantic(benchmark, cls, size, f)


@LST_PARAMS
@SC_SIZES
def test_mul(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    lst = cls(Range(size))
    _ = benchmark(lambda: lst * NUM)


@LST_PARAMS
@SC_SIZES
def test_imul(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    def f(lst: List) -> None:
        lst *= NUM

    run_pedantic(benchmark, cls, size, f)
