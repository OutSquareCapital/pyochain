from __future__ import annotations

import operator
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from pyochain import NONE, Iter, Null, Option, Range, Seq, Some, Vec
from pyochain.abc import PyoIterable, PyoSequence

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
    data = Range(0, size).iter().enumerate().collect(Seq)
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


def _with_kwargs(**kwargs: int) -> dict[str, int]:
    return kwargs


def _for_each_args_and_kwargs(data: Range) -> None:
    data.iter().for_each(_with_args_and_kwargs, 1, 2, 3, a=4, b=5, c=6)


type ForEachFn[T] = Callable[[PyoSequence[T]], None]


@pytest.mark.benchmark(group="for_each")
@pytest.mark.parametrize(
    "fn",
    [
        pytest.param(_for_each, id="for_each"),
        pytest.param(_for_each_args, id="for_each_args"),
        pytest.param(_for_each_kwargs, id="for_each_kwargs"),
        pytest.param(_for_each_args_and_kwargs, id="for_each_args_and_kwargs"),
    ],
)
def test_for_each(benchmark: BenchFixture, fn: ForEachFn[int]) -> None:
    data = Range(0, 1000)
    assert benchmark(fn, data) is None


@pytest.mark.benchmark(group="for_each_star")
@pytest.mark.parametrize("size", SIZES)
def test_for_each_star(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size).iter().map(lambda i: (i, i * 2, i * 3)).collect(Seq)
    assert benchmark(_for_each_star, data) is None


def _for_each_star(data: Seq[tuple[int, int, int]]) -> None:
    data.iter().for_each_star(_with_args)


@pytest.mark.benchmark(group="for_each_star_args")
@pytest.mark.parametrize("size", SIZES)
def test_for_each_star_args(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size).iter().map(lambda i: (i, i * 2, i * 3)).collect(Seq)
    assert benchmark(_for_each_star_args, data) is None


def _for_each_star_args(data: Seq[tuple[int, int, int]]) -> None:
    data.iter().for_each_star(_with_args, 1, 2, 3)


@pytest.mark.benchmark(group="for_each_star_kwargs")
@pytest.mark.parametrize("size", SIZES)
def test_for_each_star_kwargs(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size).iter().map(lambda i: (i, i * 2, i * 3)).collect(Seq)
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
    return data.iter().intersperse(1).collect(Seq)


@pytest.mark.benchmark(group="map_juxt")
@pytest.mark.parametrize("size", [1, 2, 4, 8, 16, 32, 64])
def test_map_juxt(benchmark: BenchFixture, size: int) -> None:

    data = Range(0, 4096)
    funcs = Range(0, size).iter().map(_create_fn).collect(Seq)
    assert benchmark(_map_juxt, data, funcs).last() is not None


def _map_juxt(data: Range, funcs: Seq[Callable[[int], int]]) -> Seq[tuple[int, ...]]:
    return data.iter().map_juxt(*funcs).collect(Seq)


def _create_fn(i: int) -> Callable[[int], int]:
    def fn(x: int) -> int:
        return x * i

    return fn


@pytest.mark.parametrize("size", SIZES)
def test_scan(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size)
    assert benchmark(_scan, data).last() is not None


def _scan(data: Range) -> Seq[int]:
    return data.iter().scan(0, lambda acc, x: Some(acc + x)).collect(Seq)


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


def test_all(benchmark: BenchFixture) -> None:
    data = Range(0, 20_000)
    assert benchmark(_all, data) is True


def _all(data: Range) -> bool:
    return data.iter().all(lambda x: x < 20_000)


@pytest.mark.parametrize("size", SIZES)
def test_all_no_closure(benchmark: BenchFixture, size: int) -> None:
    data = Range(1, size)
    assert benchmark(_all_no_closure, data) is True


def _all_no_closure(data: Range) -> bool:
    return data.iter().all()


def test_any(benchmark: BenchFixture) -> None:
    data = Range(0, 20_000)
    assert benchmark(_any, data) is True


def _any(data: Range) -> bool:
    return data.iter().any(lambda x: x == 19_999)


def test_bool(benchmark: BenchFixture) -> None:
    data = Seq((1, 2, 3))
    assert benchmark(lambda: bool(data.iter())) is True


def test_reduce(benchmark: BenchFixture) -> None:
    data = Range(0, 20_000)
    assert benchmark(_reduce, data) == 19999 * 20000 // 2


def _reduce(data: Range) -> int:
    return data.iter().reduce(operator.add)


@pytest.mark.parametrize("size", SIZES)
def test_find(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size)
    assert benchmark(_find, data) == Some(size - 1)


def _find(data: Range) -> Option[int]:
    return data.iter().find(lambda x: x == data.last())


@pytest.mark.parametrize("size", [1, 2, 4096])
def test_with_position(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size)
    assert benchmark(_with_position, data) is not None


def _with_position(data: Range) -> str:
    return data.iter().with_position().last()[0]


def test_fold_star(benchmark: BenchFixture) -> None:
    data = Range(0, 4096).iter().enumerate().collect(Seq)
    assert benchmark(_fold_star, data) is not None


def _fold_star(data: Seq[tuple[int, int]]) -> int:
    return data.iter().fold_star(0, lambda acc, x, y: acc + x + y)


def test_fold_star_args_and_kwargs(benchmark: BenchFixture) -> None:
    data = Range(0, 4096).iter().enumerate().collect(Seq)
    assert benchmark(_fold_star_args_and_kwargs, data) is not None


def _fold_star_args_and_kwargs(data: Seq[tuple[int, int]]) -> int:
    def f(acc: int, x: int, y: int, offset: int, additional: int) -> int:
        return acc + x + y + offset + additional

    return data.iter().fold_star(0, f, 10, additional=5)


def test_nth(benchmark: BenchFixture) -> None:
    size = 1_000_000
    data = Range(0, size)
    assert benchmark(_nth, data, size + 1).is_none()


def _nth(data: Range, n: int) -> Option[int]:
    return data.iter().nth(n)


@pytest.mark.parametrize("size", SIZES)
def test_arg_min(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size).iter().map(lambda x: size - x).collect(Seq)
    assert benchmark(_arg_min, data) == size - 1


def _arg_min(data: Seq[int]) -> int:
    return data.iter().arg_min()


@pytest.mark.parametrize("size", SIZES)
def test_arg_max(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size).iter().map(lambda x: size + x).collect(Seq)
    assert benchmark(_arg_max, data) == size - 1


def _arg_max(data: Seq[int]) -> int:
    return data.iter().arg_max()


@pytest.mark.parametrize("size", SIZES)
def test_arg_max_by(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size).iter().enumerate().collect(Seq)
    assert benchmark(_arg_max_by, data) == size - 1


def _arg_max_by(data: Seq[tuple[int, int]]) -> int:
    return data.iter().arg_max_by(operator.itemgetter(1))


@pytest.mark.parametrize("size", SIZES)
def test_arg_min_by(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size).iter().map(lambda x: (x, size - x)).collect(Seq)
    assert benchmark(_arg_min_by, data) == size - 1


def _arg_min_by(data: Seq[tuple[int, int]]) -> int:
    return data.iter().arg_min_by(operator.itemgetter(1))


@pytest.mark.parametrize("size", SIZES)
def test_unpack_into(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size)
    assert benchmark(_unpack_into, data) == size * (size - 1) // 2


def _unpack_into(data: Range) -> int:
    def func(*args: int) -> int:
        return sum(args)

    return data.iter().unpack_into(func)


@pytest.mark.parametrize("size", SIZES)
def test_zip_longest(benchmark: BenchFixture, size: int) -> None:
    data1 = Range(0, size)
    data2 = Range(0, size // 2)
    assert benchmark(_zip_longest, data1, data2) == (Some(size - 1), NONE)


def _zip_longest(data1: Range, data2: Range) -> tuple[Option[int], Option[int]]:
    return data1.iter().zip_longest(data2).last()


@pytest.mark.parametrize("size", SIZES)
def test_unzip(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size).iter().enumerate().collect(Seq)
    expected = size - 1
    assert benchmark(_unzip, data) == (expected, expected)


def _unzip(data: Seq[tuple[int, int]]) -> tuple[int, int]:
    left, right = data.iter().unzip()
    return left.last(), right.last()


@pytest.mark.parametrize("size", SIZES)
def test_all_equal(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size).iter().map(lambda _: 1).collect(Vec)
    data.append(2)
    assert benchmark(_all_equal, data) is False


def _all_equal(data: PyoIterable[int]) -> bool:
    return data.iter().all_equal()
