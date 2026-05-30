from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from pyochain import Iter, Null, Option, Range, Seq, Some, Vec
from pyochain.abc import PyoSequence

from ._utils import SIZES, Sizes

if TYPE_CHECKING:
    from ._utils import BenchFixture


@pytest.mark.benchmark(group="filter_map")
@pytest.mark.parametrize("size", [64, 256, 1024, 4096])
def test_filter_map(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size)
    assert benchmark(_filter_map, data) == size - 2


def _filter_map(data: Range) -> int:
    return data.iter().filter_map(lambda i: Some(i) if i % 2 == 0 else Null()).last()


@pytest.mark.parametrize("size", [64, 256, 1024, 4096])
def test_filter_map_star(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size).iter().enumerate().collect()
    assert benchmark(_filter_map_star, data) == size - 2


def _filter_map_star(data: Seq[tuple[int, int]]) -> int:
    return (
        data
        .iter()
        .filter_map_star(lambda x, y: Some((x, y)) if x % 2 == 0 else Null())
        .last()[0]
    )


@pytest.mark.benchmark(group="try_collect")
@pytest.mark.parametrize("size", SIZES)
def test_try_collect(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size)
    assert benchmark(_try_collect, data, size).is_none()


def _try_collect(data: Range, size: int) -> Option[Vec[int]]:
    return data.iter().map(lambda x: Some(x) if x < size - 1 else Null()).try_collect()


def _identity[T](x: T) -> T:
    return x


def _with_args(*args: int) -> tuple[int, ...]:
    return args


def _with_kwargs(**kwargs: int) -> dict[str, int]:
    return kwargs


def _with_args_and_kwargs(
    *args: int, **kwargs: int
) -> tuple[tuple[int, ...], dict[str, int]]:
    return args, kwargs


def _for_each(data: Range) -> None:
    data.iter().for_each(_identity)


def _for_each_args(data: Range) -> None:
    data.iter().for_each(_with_args, 1, 2, 3)


def _for_each_kwargs(data: Range) -> None:
    data.iter().for_each(lambda x: _with_kwargs(a=x, b=2, c=3))


def _for_each_args_and_kwargs(data: Range) -> None:
    data.iter().for_each(_with_args_and_kwargs, 1, 2, 3, a=4, b=5, c=6)


type ForEachFn[T] = Callable[[PyoSequence[T]], None]


@pytest.mark.benchmark(group="for_each")
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize(
    "fn",
    [
        pytest.param(_for_each, id="for_each"),
        pytest.param(_for_each_args, id="for_each_args"),
        pytest.param(_for_each_kwargs, id="for_each_kwargs"),
        pytest.param(_for_each_args_and_kwargs, id="for_each_args_and_kwargs"),
    ],
)
def test_for_each(benchmark: BenchFixture, fn: ForEachFn[int], size: int) -> None:
    data = Range(0, size)
    assert benchmark(fn, data) is None


@pytest.mark.benchmark(group="for_each_star")
@pytest.mark.parametrize("size", SIZES)
def test_for_each_star(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size).iter().map(lambda i: (i, i * 2, i * 3)).collect()
    assert benchmark(_for_each_star, data) is None


def _for_each_star(data: Seq[tuple[int, int, int]]) -> None:
    data.iter().for_each_star(_with_args)


@pytest.mark.benchmark(group="for_each_star_args")
@pytest.mark.parametrize("size", SIZES)
def test_for_each_star_args(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size).iter().map(lambda i: (i, i * 2, i * 3)).collect()
    assert benchmark(_for_each_star_args, data) is None


def _for_each_star_args(data: Seq[tuple[int, int, int]]) -> None:
    data.iter().for_each_star(_with_args, 1, 2, 3)


@pytest.mark.benchmark(group="for_each_star_kwargs")
@pytest.mark.parametrize("size", SIZES)
def test_for_each_star_kwargs(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size).iter().map(lambda i: (i, i * 2, i * 3)).collect()
    assert benchmark(_for_each_star_kwargs, data) is None


def _for_each_star_kwargs(data: Seq[tuple[int, int, int]]) -> None:
    def fn(_a: int, _b: int, _c: int, _d: int, _e: int, **_kwargs: int) -> None:
        pass

    data.iter().for_each_star(fn, 1, 2, d=4, e=5)


@pytest.mark.benchmark(group="intersperse")
@pytest.mark.parametrize("size", SIZES)
def test_intersperse(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size)
    assert benchmark(_intersperse, data).last() is not None


def _intersperse(data: Range) -> Seq[int]:
    return data.iter().intersperse(1).collect()


@pytest.mark.benchmark(group="map_juxt")
@pytest.mark.parametrize("size", [1, 2, 4, 8, 16, 32, 64])
def test_map_juxt(benchmark: BenchFixture, size: int) -> None:

    data = Range(0, 4096)
    funcs = Range(0, size).iter().map(_create_fn).collect()
    assert benchmark(_map_juxt, data, funcs).last() is not None


def _map_juxt(data: Range, funcs: Seq[Callable[[int], int]]) -> Seq[tuple[int, ...]]:
    return data.iter().map_juxt(*funcs).collect()


def _create_fn(i: int) -> Callable[[int], int]:
    def fn(x: int) -> int:
        return x * i

    return fn


@pytest.mark.parametrize("size", SIZES)
def test_scan(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size)
    assert benchmark(_scan, data).last() is not None


def _scan(data: Range) -> Seq[int]:
    return data.iter().scan(0, lambda acc, x: Some(acc + x)).collect()


@pytest.mark.parametrize("size", SIZES)
def test_map_while(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size)
    assert benchmark(_map_while, data, size) is not None


def _map_while(data: Range, size: int) -> int:
    return data.iter().map_while(lambda x: Some(x) if x < size else Null()).last()


def _from_fn() -> int:
    return Iter.from_fn(lambda: Some(0)).take(Sizes.SIZE_4096).last()


def _from_fn_args() -> int:
    def fn(a: int, b: int, c: int) -> Option[int]:
        return Some(a + b + c)

    return Iter.from_fn(fn, 1, 2, 3).take(Sizes.SIZE_4096).last()


def _from_fn_kwargs() -> int:
    def fn(a: int, b: int, c: int) -> Option[int]:
        return Some(a + b + c)

    return Iter.from_fn(fn, a=1, b=2, c=3).take(Sizes.SIZE_4096).last()


def _from_fn_args_and_kwargs() -> int:
    def fn(a: int, b: int, c: int, d: int, e: int) -> Option[int]:
        return Some(a + b + c + d + e)

    return Iter.from_fn(fn, 1, 2, 3, d=4, e=5).take(Sizes.SIZE_4096).last()


@pytest.mark.parametrize(
    "fn",
    [
        pytest.param(_from_fn, id="from_fn"),
        pytest.param(_from_fn_args, id="from_fn_args"),
        pytest.param(_from_fn_kwargs, id="from_fn_kwargs"),
        pytest.param(_from_fn_args_and_kwargs, id="from_fn_args_and_kwargs"),
    ],
)
def test_from_fn(benchmark: BenchFixture, fn: Callable[[], int]) -> None:
    assert benchmark(fn) is not None


@pytest.mark.parametrize("size", SIZES)
def test_successors(benchmark: BenchFixture, size: int) -> None:
    assert benchmark(_successors, size) == size


def _successors(size: int) -> int:
    def f(x: int) -> Option[int]:
        return Some(x + 1) if x < size else Null()

    return Iter.successors(Some(0), f).last()
