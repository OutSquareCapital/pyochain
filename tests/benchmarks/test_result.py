from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pyochain import NONE, Err, Ok, Result, Some

from ._utils import VariantGroups

if TYPE_CHECKING:
    from ._utils import BenchFixture


@pytest.mark.benchmark(group=VariantGroups.MAP.value)
def test_option_map_without_kwargs(benchmark: BenchFixture) -> None:
    def double(value: int) -> int:
        return value * 2

    assert benchmark(Ok(10).map, double) == Ok(20)


@pytest.mark.benchmark(group=VariantGroups.MAP.value)
def test_option_map_with_kwargs(benchmark: BenchFixture) -> None:
    def scale(value: int, *, factor: int, offset: int) -> int:
        return value * factor + offset

    assert benchmark(Ok(10).map, scale, factor=3, offset=1) == Ok(31)


@pytest.mark.benchmark(group=VariantGroups.AND_THEN.value)
def test_option_and_then_with_kwargs(benchmark: BenchFixture) -> None:
    def keep_if_at_least(value: int, *, minimum: int) -> Result[int, int]:
        return Ok(value) if value >= minimum else Err(value)

    assert benchmark(Ok(10).and_then, keep_if_at_least, minimum=5) == Ok(10)


@pytest.mark.benchmark(group=VariantGroups.MATCH.value)
def test_option_match_case_some(benchmark: BenchFixture) -> None:
    def describe(opt: Result[int, int]) -> int:
        match opt:
            case Ok(_):
                return 1
            case Err(_):
                return 0

    assert benchmark(describe, Ok(10)) == 1


@pytest.mark.benchmark(group="result_convert")
def test_result_ok_to_option(benchmark: BenchFixture) -> None:
    assert benchmark(Ok(10).ok) == Some(10)


@pytest.mark.benchmark(group="result_convert")
def test_result_err_to_option(benchmark: BenchFixture) -> None:
    assert benchmark(Err(10).err) == Some(10)


@pytest.mark.benchmark(group="result_flatten")
def test_result_flatten_ok(benchmark: BenchFixture) -> None:
    assert benchmark(Ok(Ok(10)).flatten) == Ok(10)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]


@pytest.mark.benchmark(group="result_flatten")
def test_result_flatten_err(benchmark: BenchFixture) -> None:
    assert benchmark(Ok(Err(10)).flatten) == Err(10)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]


@pytest.mark.benchmark(group="result_transpose")
def test_result_transpose_some(benchmark: BenchFixture) -> None:
    assert benchmark(Ok(Some(10)).transpose) == Some(Ok(10))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]


@pytest.mark.benchmark(group="result_transpose")
def test_result_transpose_none(benchmark: BenchFixture) -> None:
    assert benchmark(Ok(NONE).transpose) == NONE  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]


@pytest.mark.benchmark(group="result_transpose")
def test_result_transpose_err(benchmark: BenchFixture) -> None:
    assert benchmark(Err(10).transpose) == Some(Err(10))


@pytest.mark.benchmark(group="result_swap")
def test_result_swap_ok(benchmark: BenchFixture) -> None:
    assert benchmark(Ok(10).swap) == Err(10)


@pytest.mark.benchmark(group="result_swap")
def test_result_swap_err(benchmark: BenchFixture) -> None:
    assert benchmark(Err(10).swap) == Ok(10)
