from __future__ import annotations

from enum import StrEnum, auto
from typing import TYPE_CHECKING

import pytest

from pyochain import NONE, Option, Some

if TYPE_CHECKING:
    from ._utils import BenchFixture


class OptionGroups(StrEnum):
    CREATE = auto()
    MAP = auto()
    AND_THEN = auto()
    MATCH = auto()


@pytest.mark.benchmark(group=OptionGroups.CREATE.value)
def test_option_create_some(benchmark: BenchFixture) -> None:
    assert benchmark(Option, 10) == Some(10)


@pytest.mark.benchmark(group=OptionGroups.CREATE.value)
def test_option_create_none(benchmark: BenchFixture) -> None:
    assert benchmark(Option, None) == NONE


@pytest.mark.benchmark(group=OptionGroups.MAP.value)
def test_option_map_without_kwargs(benchmark: BenchFixture) -> None:
    def double(value: int) -> int:
        return value * 2

    assert benchmark(Some(10).map, double) == Some(20)


@pytest.mark.benchmark(group=OptionGroups.MAP.value)
def test_option_map_with_kwargs(benchmark: BenchFixture) -> None:
    def scale(value: int, *, factor: int, offset: int) -> int:
        return value * factor + offset

    assert benchmark(Some(10).map, scale, factor=3, offset=1) == Some(31)


@pytest.mark.benchmark(group=OptionGroups.AND_THEN.value)
def test_option_and_then_with_kwargs(benchmark: BenchFixture) -> None:
    def keep_if_at_least(value: int, *, minimum: int) -> Option[int]:
        return Some(value) if value >= minimum else NONE

    assert benchmark(Some(10).and_then, keep_if_at_least, minimum=5) == Some(10)


@pytest.mark.benchmark(group=OptionGroups.MATCH.value)
def test_option_match_case_some(benchmark: BenchFixture) -> None:
    def describe(opt: Option[int]) -> int:
        match opt:
            case Some(_):
                return 1
            case _:
                return 0

    assert benchmark(describe, Some(10)) == 1
