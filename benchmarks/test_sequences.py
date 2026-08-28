from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import pytest

from pyochain import Range, Seq, Set, SetMut, Vec
from pyochain.abc import PyoCollection
from pyochain.collections import Deque, StableSet

from ._utils import SIZES

if TYPE_CHECKING:
    from pyochain import Option
    from pyochain.abc import PyoReversible, PyoSequence

    from ._utils import BenchFixture


@pytest.mark.parametrize("size", SIZES)
def test_rev(benchmark: BenchFixture, size: int) -> None:
    data = Range(size)
    assert benchmark(_rev, data) == 0


def _rev(data: PyoReversible[int]) -> int:
    return data.rev().last()


@pytest.mark.parametrize("size", SIZES)
def test_first(benchmark: BenchFixture, size: int) -> None:
    data = Range(size).pipe(Seq)
    assert benchmark(_first, data) == 0


def _first(data: PyoSequence[int]) -> int:
    for _ in data:
        _ = data.first()
    return data.first()


@pytest.mark.parametrize("size", SIZES)
def test_last(benchmark: BenchFixture, size: int) -> None:
    data = Range(size).pipe(Seq)
    assert benchmark(_last, data) == size - 1


def _last(data: PyoSequence[int]) -> int:
    for _ in data:
        _ = data.last()
    return data.last()


@pytest.mark.parametrize("size", SIZES)
def test_get_some(benchmark: BenchFixture, size: int) -> None:
    data = Range(size).pipe(Seq)
    assert benchmark(_get, data, size - 1).is_some()


@pytest.mark.parametrize("size", SIZES)
def test_get_none(benchmark: BenchFixture, size: int) -> None:
    data = Range(size)
    assert benchmark(_get, data, size).is_none()


def _get(data: PyoSequence[int], idx: int) -> Option[int]:
    for _ in data:
        _ = data.get(idx)
    return data.get(idx)


type SeqFn[T] = Callable[[Iterable[T]], PyoCollection[T]]


@pytest.mark.parametrize("obj", (Vec, Seq, Set, SetMut, StableSet, Deque))
@pytest.mark.parametrize("size", SIZES)
def test_init(benchmark: BenchFixture, obj: SeqFn[int], size: int) -> None:
    data = range(size)

    _ = benchmark(lambda: obj(data))


@pytest.mark.parametrize("size", SIZES)
def test_init_range(benchmark: BenchFixture, size: int) -> None:
    data = Range(size)
    assert benchmark(_init_range, size).first() == data.first()


def _init_range(size: int) -> Range:
    for _ in SIZES[size]:
        _ = Range(size)
    return Range(size)


@pytest.mark.parametrize("size", SIZES)
def test_concat(benchmark: BenchFixture, size: int) -> None:
    data = Range(size).pipe(Seq)
    assert benchmark(_concat, data, size).last() == data.last()


def _concat(data: Seq[int], size: int) -> Seq[int]:
    for _ in SIZES[size]:
        _ = data.concat(data)
    return data.concat(data)
