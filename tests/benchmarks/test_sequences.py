from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pyochain import Option, Range, Vec

from ._utils import SIZES

if TYPE_CHECKING:
    from pyochain.abc import PyoReversible, PyoSequence

    from ._utils import BenchFixture


@pytest.mark.parametrize("size", SIZES)
def test_rev(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size)
    assert benchmark(_rev, data) == 0


def _rev(data: PyoReversible[int]) -> int:
    return data.rev().last()


@pytest.mark.parametrize("size", SIZES)
def test_first(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size).pipe(Vec)
    assert benchmark(_first, data) == 0


def _first(data: PyoSequence[int]) -> int:
    for _ in data:
        _ = data.first()
    return data.first()


@pytest.mark.parametrize("size", SIZES)
def test_last(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size).pipe(Vec)
    assert benchmark(_last, data) == size - 1


def _last(data: PyoSequence[int]) -> int:
    for _ in data:
        _ = data.last()
    return data.last()


@pytest.mark.parametrize("size", SIZES)
def test_get_some(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size).pipe(Vec)
    assert benchmark(_get, data, size - 1).is_some()


@pytest.mark.parametrize("size", SIZES)
def test_get_none(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size)
    assert benchmark(_get, data, size).is_none()


def _get(data: PyoSequence[int], idx: int) -> Option[int]:
    for _ in data:
        _ = data.get(idx)
    return data.get(idx)
