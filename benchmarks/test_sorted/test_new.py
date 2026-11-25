"""Additional benchmarks not presents in the original `sortedcontainers` package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

from benchmarks._utils import SIZE_PARAMS
from pyochain import Iter, Range, Seq

from ._utils import (
    LST_PARAMS,
    SC_SIZES,
    SEQ_PARAMS,
    List,
    SortedCollection,
    SortedSequence,
    run_pedantic,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyochain.abc import PyoIterable

    from .._utils import BenchFixture

type SliceFn = Callable[[int], slice]
type InputFn = Callable[[int], PyoIterable[int]]

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


def partition(size: int) -> Seq[int]:
    return (
        Range(NUM)
        .rev()
        .flat_map(lambda p: Range(p * size // NUM, (p + 1) * size // NUM))
        .collect(Seq)
    )


SORTED_INPUTS: Final[Seq[object]] = (
    Iter[tuple[InputFn, str]](
        (lambda size: Seq(Range(size)), "sorted"), (partition, "partitioned")
    )
    .map_star(lambda f, name: pytest.param(f, id=name))
    .collect(Seq)
)
"""Inputs already sorted as a whole, or by `NUM` disjoint sorted partitions, concatenated from the last one to the first one."""


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


@pytest.mark.parametrize("inputs", SORTED_INPUTS)
@SEQ_PARAMS
@SIZE_PARAMS
def test_init_sorted(
    benchmark: BenchFixture, cls: type[SortedCollection], size: int, inputs: InputFn
) -> None:
    values = inputs(size)
    # pyrefly: ignore [bad-argument-type, incompatible-overload-residual]
    assert len(benchmark(cls, values)) == size  # pyright: ignore[reportCallIssue]
