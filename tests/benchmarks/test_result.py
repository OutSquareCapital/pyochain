from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from pyochain import NONE, Err, Ok, Result, Some

from ._utils import VariantGroups

if TYPE_CHECKING:
    from ._utils import BenchFixture


type BenchCall = Callable[[], object]


@pytest.mark.benchmark(group=VariantGroups.MAP.value)
def test_map_without_kwargs(benchmark: BenchFixture) -> None:
    def double(value: int) -> int:
        return value * 2

    def fn() -> None:
        _ = Ok(10).map(double)

    assert benchmark(fn) is None


@pytest.mark.benchmark(group=VariantGroups.MAP.value)
def test_map_with_kwargs(benchmark: BenchFixture) -> None:
    def scale(value: int, *, factor: int, offset: int) -> int:
        return value * factor + offset

    def fn() -> None:
        _ = Ok(10).map(scale, factor=3, offset=1)

    assert benchmark(fn) is None


def test_map_star(benchmark: BenchFixture) -> None:
    def combine(_a: int, _b: int, _c: int) -> int:
        return 1

    def fn() -> Result[int, int]:
        return Ok((1, 2, 3)).map_star(combine)

    assert benchmark(fn).unwrap() == 1


@pytest.mark.benchmark(group=VariantGroups.AND_THEN.value)
def test_and_then_with_kwargs(benchmark: BenchFixture) -> None:
    def keep_if_at_least(value: int, *, minimum: int) -> Result[int, int]:
        return Ok(value) if value >= minimum else Err(value)

    def fn() -> None:
        _ = Ok(10).and_then(keep_if_at_least, minimum=5)

    assert benchmark(fn) is None


@pytest.mark.benchmark(group=VariantGroups.MATCH.value)
def test_match_case(benchmark: BenchFixture) -> None:
    def describe(opt: Result[int, int]) -> int:
        match opt:
            case Ok(_):
                return 1
            case Err(_):
                return 0

    assert benchmark(describe, Ok(10)) == 1


@pytest.mark.benchmark(group="result_convert")
@pytest.mark.parametrize(
    "fn",
    [
        pytest.param(Ok(10).ok, id="ok_to_option"),
        pytest.param(Err(10).err, id="err_to_option"),
    ],
)
def test_convert(benchmark: BenchFixture, fn: BenchCall) -> None:
    def run() -> None:
        _ = fn()

    assert benchmark(run) is None


@pytest.mark.benchmark(group="result_flatten")
@pytest.mark.parametrize(
    "fn",
    [
        pytest.param(Ok(Ok(10)).flatten, id="ok"),
        pytest.param(Ok(Err(10)).flatten, id="err"),
    ],
)
def test_flatten(benchmark: BenchFixture, fn: BenchCall) -> None:
    def run() -> None:
        _ = fn()

    assert benchmark(run) is None


@pytest.mark.benchmark(group="result_transpose")
@pytest.mark.parametrize(
    "fn",
    [
        pytest.param(Ok(Some(10)).transpose, id="some"),  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownArgumentType]
        pytest.param(Ok(NONE).transpose, id="none"),
        pytest.param(Err(10).transpose, id="err"),
    ],
)
def test_transpose(benchmark: BenchFixture, fn: BenchCall) -> None:
    def run() -> None:
        _ = fn()

    assert benchmark(run) is None


@pytest.mark.benchmark(group="result_swap")
@pytest.mark.parametrize(
    "fn",
    [
        pytest.param(Ok(10).swap, id="ok"),
        pytest.param(Err(10).swap, id="err"),
    ],
)
def test_swap(benchmark: BenchFixture, fn: BenchCall) -> None:
    def run() -> None:
        _ = fn()

    assert benchmark(run) is None
