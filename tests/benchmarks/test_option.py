from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from pyochain import NONE, Err, Null, Ok, Option, Range, Some, option

from ._utils import SIZES, VariantGroups

if TYPE_CHECKING:
    from ._utils import BenchFixture


type BenchCall = Callable[[], object]
type BenchCallWithInt = Callable[[int], object]


@pytest.mark.benchmark(group=VariantGroups.CREATE.value)
def test_create_some(benchmark: BenchFixture) -> None:
    assert benchmark(option, 10) == Some(10)


@pytest.mark.benchmark(group=VariantGroups.CREATE.value)
def test_create_none(benchmark: BenchFixture) -> None:
    assert benchmark(option, None) == NONE


@pytest.mark.benchmark(group=VariantGroups.MAP.value)
def test_map_without_kwargs(benchmark: BenchFixture) -> None:
    def double(value: int) -> int:
        return value * 2

    assert benchmark(Some(10).map, double) == Some(20)


@pytest.mark.benchmark(group=VariantGroups.MAP.value)
def test_map_with_kwargs(benchmark: BenchFixture) -> None:
    def scale(value: int, *, factor: int, offset: int) -> int:
        return value * factor + offset

    assert benchmark(Some(10).map, scale, factor=3, offset=1) == Some(31)


@pytest.mark.benchmark(group=VariantGroups.AND_THEN.value)
def test_and_then_with_kwargs(benchmark: BenchFixture) -> None:
    def keep_if_at_least(value: int, *, minimum: int) -> Option[int]:
        return Some(value) if value >= minimum else NONE

    assert benchmark(Some(10).and_then, keep_if_at_least, minimum=5) == Some(10)


@pytest.mark.benchmark(group=VariantGroups.MATCH.value)
def test_match_case_some(benchmark: BenchFixture) -> None:
    def describe(opt: Option[int]) -> int:
        match opt:
            case Some(_):
                return 1
            case Null():
                return 0

    assert benchmark(describe, Some(10)) == 1


@pytest.mark.benchmark(group="option_convert")
@pytest.mark.parametrize(
    "fn",
    [
        pytest.param(Some(10).ok_or, id="some"),
        pytest.param(NONE.ok_or, id="none"),
    ],
)
def test_ok_or(benchmark: BenchFixture, fn: BenchCallWithInt) -> None:
    def run() -> None:
        _ = fn(0)

    assert benchmark(run) is None


@pytest.mark.benchmark(group="option_flatten")
def test_flatten_nested_some(benchmark: BenchFixture) -> None:
    assert benchmark(Some(Some(10)).flatten) == Some(10)


@pytest.mark.benchmark(group="option_flatten")
def test_flatten_nested_none(benchmark: BenchFixture) -> None:
    assert benchmark(Some(NONE).flatten) == NONE


@pytest.mark.benchmark(group="option_transpose")
@pytest.mark.parametrize(
    "fn",
    [
        pytest.param(Some(Ok(10)).transpose, id="ok"),  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownArgumentType]
        pytest.param(Some(Err(10)).transpose, id="err"),
        pytest.param(NONE.transpose, id="none"),
    ],
)
def test_transpose(benchmark: BenchFixture, fn: BenchCall) -> None:
    def run() -> None:
        _ = fn()

    assert benchmark(run) is None


def test_call_none(benchmark: BenchFixture) -> None:
    data = Range(0, 100_000)

    def fn() -> Option[int]:
        return data.iter().map(lambda _: NONE).last()

    assert benchmark(fn) is NONE


@pytest.mark.parametrize("size", SIZES)
def test_iter_some(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size)
    opt = Some(0)
    assert benchmark(_iter, data, opt).is_some()


@pytest.mark.parametrize("size", SIZES)
def test_iter_none(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size)
    opt = NONE
    assert benchmark(_iter, data, opt).is_none()


def _iter(data: Range, opt: Option[int]) -> Option[int]:
    for _ in data.iter():
        _ = opt.iter()
    return opt.iter().next()
