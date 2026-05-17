from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pyochain import NONE, Err, Ok, Option, Some

from ._utils import VariantGroups

if TYPE_CHECKING:
    from ._utils import BenchFixture


@pytest.mark.benchmark(group=VariantGroups.CREATE.value)
def test_option_create_some(benchmark: BenchFixture) -> None:
    assert benchmark(Option, 10) == Some(10)


@pytest.mark.benchmark(group=VariantGroups.CREATE.value)
def test_option_create_none(benchmark: BenchFixture) -> None:
    assert benchmark(Option, None) == NONE


@pytest.mark.benchmark(group=VariantGroups.MAP.value)
def test_option_map_without_kwargs(benchmark: BenchFixture) -> None:
    def double(value: int) -> int:
        return value * 2

    assert benchmark(Some(10).map, double) == Some(20)


@pytest.mark.benchmark(group=VariantGroups.MAP.value)
def test_option_map_with_kwargs(benchmark: BenchFixture) -> None:
    def scale(value: int, *, factor: int, offset: int) -> int:
        return value * factor + offset

    assert benchmark(Some(10).map, scale, factor=3, offset=1) == Some(31)


@pytest.mark.benchmark(group=VariantGroups.AND_THEN.value)
def test_option_and_then_with_kwargs(benchmark: BenchFixture) -> None:
    def keep_if_at_least(value: int, *, minimum: int) -> Option[int]:
        return Some(value) if value >= minimum else NONE

    assert benchmark(Some(10).and_then, keep_if_at_least, minimum=5) == Some(10)


@pytest.mark.benchmark(group=VariantGroups.MATCH.value)
def test_option_match_case_some(benchmark: BenchFixture) -> None:
    def describe(opt: Option[int]) -> int:
        match opt:
            case Some(_):
                return 1
            case _:
                return 0

    assert benchmark(describe, Some(10)) == 1


@pytest.mark.benchmark(group="option_convert")
def test_option_ok_or_some(benchmark: BenchFixture) -> None:
    assert benchmark(Some(10).ok_or, 0) == Ok(10)


@pytest.mark.benchmark(group="option_convert")
def test_option_ok_or_none(benchmark: BenchFixture) -> None:
    assert benchmark(NONE.ok_or, 0) == Err(0)


@pytest.mark.benchmark(group="option_flatten")
def test_option_flatten_nested_some(benchmark: BenchFixture) -> None:
    assert benchmark(Some(Some(10)).flatten) == Some(10)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]


@pytest.mark.benchmark(group="option_flatten")
def test_option_flatten_nested_none(benchmark: BenchFixture) -> None:
    assert benchmark(Some(NONE).flatten) == NONE  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]


@pytest.mark.benchmark(group="option_transpose")
def test_option_transpose_ok(benchmark: BenchFixture) -> None:
    assert benchmark(Some(Ok(10)).transpose) == Ok(Some(10))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]


@pytest.mark.benchmark(group="option_transpose")
def test_option_transpose_err(benchmark: BenchFixture) -> None:
    assert benchmark(Some(Err(10)).transpose) == Err(10)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]


@pytest.mark.benchmark(group="option_transpose")
def test_option_transpose_none(benchmark: BenchFixture) -> None:
    assert benchmark(NONE.transpose) == Ok(NONE)
