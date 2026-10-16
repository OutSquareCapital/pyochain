from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING, Final

import pytest

from pyochain import Iter, Range, Seq

from .._utils import SIZE_PARAMS, identity
from . import methods
from ._utils import (
    CLS_PARAMS,
    RANDOM,
    BenchConfig,
    InitFn,
    KeyList,
    List,
    calls_nb,
    shuffled,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from _pytest.mark.structures import MarkDecorator

    from .._utils import BenchFixture
WIDTH_PARAMS: Final[MarkDecorator] = pytest.mark.parametrize(
    "width_fn",
    Iter[tuple[Callable[[int], int], str]](
        (lambda _: 1, "single"),
        (lambda n: n // 10, "10pct"),
        (lambda n: n // 4, "25pct"),
        (identity, "full"),
    )
    .map_star(lambda fn, name: pytest.param(fn, id=name))
    .collect(tuple),
)


# NOTE: mostly test the "empty maxes" path in `update` method.
@CLS_PARAMS
@SIZE_PARAMS
def test_init(benchmark: BenchFixture, cls: InitFn[List], size: int) -> None:
    assert benchmark(cls, shuffled(size))


@CLS_PARAMS
@SIZE_PARAMS
def test_init_by_key(benchmark: BenchFixture, cls: type[KeyList], size: int) -> None:
    data = shuffled(size)
    fn: Callable[[int], int] = lambda x: x + 1
    assert benchmark(methods.key_init(cls), data, fn)


@CLS_PARAMS
@SIZE_PARAMS
def test_contains(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    data = shuffled(size)
    sl = cls(range(size))
    assert benchmark(lambda: data.iter().map(sl.__contains__).last())


@CLS_PARAMS
@SIZE_PARAMS
def test_getitem(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    data = shuffled(size)
    sl = cls(range(size))
    assert benchmark(lambda: data.iter().map(sl.__getitem__).last()) == data.last()


@CLS_PARAMS
@SIZE_PARAMS
def test_getitem_slice(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    data = shuffled(size)
    sl = cls(range(size))
    assert benchmark(lambda: data.iter().map(lambda i: sl[i : i + 1]).last())


@CLS_PARAMS
@SIZE_PARAMS
def test_bisect_left(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    data = shuffled(size)
    sl = cls(range(size))
    assert benchmark(lambda: data.iter().map(sl.bisect_left).last()) == data.last()


@CLS_PARAMS
@SIZE_PARAMS
def test_bisect_right(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    data = shuffled(size)
    sl = cls(range(size))
    assert benchmark(lambda: data.iter().map(sl.bisect_right).last())


@CLS_PARAMS
@SIZE_PARAMS
def test_iter(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    def f(obj: Iterable[int]) -> None:
        for _ in obj:
            pass

    benchmark(f, cls(range(size)))


@CLS_PARAMS
@SIZE_PARAMS
def test_iter_collect(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    _ = benchmark(tuple, cls(range(size)))


@CLS_PARAMS
@SIZE_PARAMS
def test_iter_rev_collect(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    def f(lst: Sequence[int]) -> Sequence[int]:
        return tuple(reversed(lst))

    _ = benchmark(f, cls(Range(size)))


@CLS_PARAMS
@SIZE_PARAMS
def test_count(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    data = shuffled(size)
    sl = cls(range(size))
    assert benchmark(lambda: data.iter().map(lambda i: sl.count(i) == 1).last())


@CLS_PARAMS
@SIZE_PARAMS
def test_index(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    data = shuffled(size)
    sl = cls(range(size))
    assert benchmark(lambda: data.iter().map(lambda i: sl.index(i) == i).last())


@CLS_PARAMS
@SIZE_PARAMS
def test_islice(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    sl = cls(range(size))
    assert benchmark(lambda: tuple(sl.islice(stop=size - 1)))


@CLS_PARAMS
@SIZE_PARAMS
def test_irange(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    sl = cls(range(size))
    maximum = size - 1
    assert benchmark(lambda: tuple(sl.irange(maximum=maximum)))


@CLS_PARAMS
@SIZE_PARAMS
def test_update_large[L: List](
    benchmark: BenchFixture, cls: type[L], size: int
) -> None:
    data = shuffled(size)
    method = methods.update(cls)
    benchmark(lambda: method(cls([1]), data))


@CLS_PARAMS
@SIZE_PARAMS
def test_update_small[L: List](
    benchmark: BenchFixture, cls: type[L], size: int
) -> None:
    k = size // 8
    values = shuffled(k)
    config = BenchConfig(benchmark, cls, size)
    method = methods.update(cls)
    config.run_pedantic(lambda sl: method(sl, values))


@CLS_PARAMS
@SIZE_PARAMS
def test_add(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    values = _values_sample(size)
    method = methods.add(cls)

    def f(sl: List) -> None:
        for val in values:
            method(sl, val)

    BenchConfig(benchmark, cls, size).run_pedantic(f)


@CLS_PARAMS
@SIZE_PARAMS
def test_remove(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    values = _values_sample(size)

    def f(sl: List) -> None:
        for val in values:
            sl.remove(val)

    BenchConfig(benchmark, cls, size).run_pedantic(f)


@CLS_PARAMS
@SIZE_PARAMS
def test_pop(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    calls = range(calls_nb(size))

    def f(sl: List) -> None:
        for _ in calls:
            _ = sl.pop()

    BenchConfig(benchmark, cls, size).run_pedantic(f)


@CLS_PARAMS
@SIZE_PARAMS
def test_delitem(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    positions = _del_positions(size, calls_nb(size), 1)

    def f(sl: List) -> None:
        for pos in positions:
            del sl[pos]

    BenchConfig(benchmark, cls, size).run_pedantic(f)


@CLS_PARAMS
@SIZE_PARAMS
def test_delslice(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    width = max(1, size // 10)
    pos = _del_positions(size, 1, width).first()

    def f(sl: List) -> None:
        del sl[pos : pos + width]

    BenchConfig(benchmark, cls, size).run_pedantic(f)


@cache
def _del_positions(size: int, count: int, width: int) -> Seq[int]:
    return (
        Range(count)
        .iter()
        .map(lambda i: RANDOM.randrange(size - (i + 1) * width + 1))
        .collect(Seq)
    )


def _values_sample(size: int) -> Seq[int]:
    return shuffled(size).iter().take(calls_nb(size)).collect(Seq)
