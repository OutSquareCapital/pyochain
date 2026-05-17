from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from pyochain import Err, Ok, Range, Seq

if TYPE_CHECKING:
    from ._utils import BenchFixture, BenchFn


def _pyochain_filter_map(size: int) -> Seq[int]:
    return (
        Range(0, size)
        .iter()
        .map(lambda x: x * 2)
        .filter(lambda x: x % 3 == 0)
        .collect()
    )


def _python_filter_map(size: int) -> tuple[int, ...]:
    return tuple(
        filter(
            lambda x: x % 3 == 0,
            map(lambda x: x * 2, range(size)),
        )
    )


@pytest.mark.benchmark(group="filter_map")
@pytest.mark.parametrize(
    "fn",
    [
        pytest.param(_python_filter_map, id="python"),
        pytest.param(_pyochain_filter_map, id="pyochain"),
    ],
)
@pytest.mark.parametrize("size", [10, 100, 500])
def test_filter_map(benchmark: BenchFixture, fn: BenchFn, size: int) -> None:
    result = benchmark(fn, size)
    assert result


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


def _for_each_star(data: Seq[tuple[int, int, int]]) -> None:
    data.iter().for_each_star(_with_args)


def _try_find_some(data: Range) -> None:
    _ = data.iter().try_find(lambda value: Ok(value == 9))


def _try_find_err(data: Range) -> None:
    _ = data.iter().try_find(
        lambda value: Err(value) if value == 9 else Ok(value == -1)
    )


def _try_fold_ok(data: Range) -> None:
    _ = data.iter().try_fold(0, lambda accumulator, value: Ok(accumulator + value))


def _try_reduce_ok(data: Range) -> None:
    _ = data.iter().try_reduce(lambda left, right: Ok(left + right))


def _try_reduce_err(data: Range) -> None:
    _ = data.iter().try_reduce(
        lambda left, right: Err(right) if right == 9 else Ok(left + right)
    )


type ForEachFn = Callable[[Range], None]


@pytest.mark.benchmark(group="for_each")
@pytest.mark.parametrize("size", [10, 100, 500])
@pytest.mark.parametrize(
    "fn",
    [
        pytest.param(_for_each, id="for_each"),
        pytest.param(_for_each_args, id="for_each_args"),
        pytest.param(_for_each_kwargs, id="for_each_kwargs"),
        pytest.param(_for_each_args_and_kwargs, id="for_each_args_and_kwargs"),
    ],
)
def test_for_each(benchmark: BenchFixture, fn: ForEachFn, size: int) -> None:
    data = Range(0, size)
    assert benchmark(fn, data) is None


@pytest.mark.benchmark(group="for_each")
@pytest.mark.parametrize("size", [10, 100, 500])
def test_for_each_star(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size).iter().map(lambda i: (i, i * 2, i * 3)).collect()
    assert benchmark(_for_each_star, data) is None


@pytest.mark.benchmark(group="try_find")
@pytest.mark.parametrize(
    "fn",
    [
        pytest.param(_try_find_some, id="some"),
        pytest.param(_try_find_err, id="err"),
    ],
)
@pytest.mark.parametrize("size", [10, 100, 500])
def test_try_find(benchmark: BenchFixture, fn: ForEachFn, size: int) -> None:
    data = Range(0, size)
    assert benchmark(fn, data) is None


@pytest.mark.benchmark(group="try_fold")
@pytest.mark.parametrize("size", [10, 100, 500])
def test_try_fold_ok(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size)
    assert benchmark(_try_fold_ok, data) is None


@pytest.mark.benchmark(group="try_reduce")
@pytest.mark.parametrize(
    "fn",
    [
        pytest.param(_try_reduce_ok, id="ok"),
        pytest.param(_try_reduce_err, id="err"),
    ],
)
@pytest.mark.parametrize("size", [10, 100, 500])
def test_try_reduce(benchmark: BenchFixture, fn: ForEachFn, size: int) -> None:
    data = Range(0, size)
    assert benchmark(fn, data) is None
