from __future__ import annotations

from pyochain import Range

from .._utils import SIZE_PARAMS, BenchFixture, consume
from ._utils import SC_SIZES, SEQ_PARAMS, InitFn, SeqConfig, SortedSequence, shuffled


@SEQ_PARAMS
@SIZE_PARAMS
def test_init(
    benchmark: BenchFixture, cls: InitFn[int, SortedSequence], size: int
) -> None:
    assert benchmark(cls, shuffled(size))


@SEQ_PARAMS
@SIZE_PARAMS
def test_iter(benchmark: BenchFixture, cls: type[SortedSequence], size: int) -> None:
    lst = SeqConfig(benchmark, cls, size).new_from_size()
    benchmark(consume, lst)


@SEQ_PARAMS
@SC_SIZES
def test_remove(benchmark: BenchFixture, cls: type[SortedSequence], size: int) -> None:
    values = shuffled(size, 100)

    def f(lst: SortedSequence) -> None:
        values.iter().for_each(lst.remove)

    SeqConfig(benchmark, cls, size).run_pedantic(f)


@SEQ_PARAMS
@SC_SIZES
def test_pop(benchmark: BenchFixture, cls: type[SortedSequence], size: int) -> None:
    values = Range(size // 100)

    def f(lst: SortedSequence) -> None:
        values.iter().for_each(lambda _: lst.pop())

    SeqConfig(benchmark, cls, size).run_pedantic(f)
