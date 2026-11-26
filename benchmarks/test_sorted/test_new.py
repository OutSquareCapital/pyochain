"""Additional benchmarks not presents in the original `sortedcontainers` package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest
from _pytest.mark import ParameterSet

from benchmarks._utils import SIZE_PARAMS
from pyochain import Iter, Range, Seq
from pyochain.abc import PyoIterator

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


def into_inputs[T](inputs: PyoIterator[tuple[T, str]]) -> Seq[ParameterSet]:
    return inputs.map_star(lambda f, name: pytest.param(f, id=name)).collect(Seq)


SLICES: Final[Seq[object]] = Iter[tuple[SliceFn, str]](
    (lambda _: slice(None), "full"),
    (lambda n: slice(n // 2, n // 2 + n // 16), "partial"),
    (lambda n: slice(n - n // 16, None), "to_end"),
    (lambda n: slice(n // 4, 3 * n // 4), "large"),
    (lambda n: slice(n // 4, None), "large_to_end"),
    (lambda n: slice(n // 2 + n // 16, n // 2, -1), "backward"),
    (lambda _: slice(None, None, 10), "step"),
    (lambda _: slice(None, None, -10), "backward_step"),
).pipe(into_inputs)
"""One slice per path of `ops::Slice` and `ForwardSlice`, built from the size of the collection."""
NUM: Final[int] = 3
"""Number of times to multiply the list in `test_mul` and `test_imul`."""
GROUP_LEN: Final[int] = 10
"""Number of equal items per group in the `grouped` inputs of `test_mul`."""


def partition(size: int) -> Seq[int]:
    return (
        Range(NUM)
        .rev()
        .flat_map(lambda p: Range(p * size // NUM, (p + 1) * size // NUM))
        .collect(Seq)
    )


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


@pytest.mark.parametrize(
    "inputs",
    Iter[tuple[InputFn, str]](
        (Range, "distinct"),
        (
            lambda size: Range(size).iter().map(lambda x: x // GROUP_LEN).collect(Seq),
            "grouped",
        ),
    ).pipe(into_inputs),
)
@LST_PARAMS
@SC_SIZES
def test_mul(
    benchmark: BenchFixture, cls: type[List], size: int, inputs: InputFn
) -> None:
    lst = cls(inputs(size))
    _ = benchmark(lambda: lst * NUM)


@LST_PARAMS
@SC_SIZES
def test_imul(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    def f(lst: List) -> None:
        lst *= NUM

    run_pedantic(benchmark, cls, size, f)


@pytest.mark.parametrize(
    "inputs",
    Iter[tuple[InputFn, str]](
        (lambda size: Seq(Range(size)), "sorted"), (partition, "partitioned")
    ).pipe(into_inputs),
)
@SEQ_PARAMS
@SIZE_PARAMS
def test_init_sorted(
    benchmark: BenchFixture, cls: type[SortedCollection], size: int, inputs: InputFn
) -> None:
    values = inputs(size)
    assert len(benchmark(cls, values)) == size  # pyright: ignore[reportCallIssue]
