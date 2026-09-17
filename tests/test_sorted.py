from __future__ import annotations

from collections.abc import Iterable, Sequence
from collections.abc import Set as AbstractSet
from operator import neg
from typing import TYPE_CHECKING

import pytest

from pyochain import Range
from pyochain.collections import SortedKeySet, SortedSet

if TYPE_CHECKING:
    from collections.abc import Callable

    from _pytest.mark import ParameterSet

    from pyochain.collections._sorted import BaseSortedSet
type UpdateFn = Callable[[SortedSet[int], *tuple[Iterable[int]]], None]
type UpdateFn1 = Callable[[SortedSet[int], Iterable[int]], None]
type SortedSetFactory = Callable[[], BaseSortedSet[int]]
DATA = Range(5)
PY_EMPTY = set[int]()
PY_DATA_SET = set[int](DATA)
DATA_SET = SortedSet(DATA)
DATA_KEY_SET = SortedKeySet(neg, DATA)
VALUES_PARAMS = pytest.mark.parametrize(
    "data",
    ([DATA], [DATA, DATA], [DATA_SET], [DATA_SET, DATA_SET]),
)


@VALUES_PARAMS
@pytest.mark.parametrize(
    "fn", (SortedSet[int].update, SortedSet[int].intersection_update)
)
def test_update_deadlock(data: Iterable[Iterable[int]], fn: UpdateFn) -> None:
    a = SortedSet(DATA)
    fn(a, *data)
    assert a == DATA_SET


@VALUES_PARAMS
def test_diff_update_deadlock(data: Sequence[Iterable[int]]) -> None:
    a = SortedSet(DATA)
    a.difference_update(*data)
    assert a.is_empty()


@pytest.mark.parametrize("data", (DATA, DATA_SET))
def test_symmetric_diff_update_deadlock(data: Iterable[int]) -> None:
    a = SortedSet(DATA)
    a.symmetric_difference_update(data)
    assert a.is_empty()


@pytest.mark.parametrize("data", (DATA_SET, SortedKeySet(neg, DATA)))
def test_set_bisect(data: BaseSortedSet[int]) -> None:
    assert data.bisect_left(2) == 2
    assert data.bisect_right(2) == 3


def _param(fn: UpdateFn1, expected: AbstractSet[int]) -> ParameterSet:
    return pytest.param(fn, expected, id=fn.__name__)


@pytest.mark.parametrize(
    ("method", "expected"),
    (
        _param(SortedSet[int].difference_update, PY_EMPTY),
        _param(SortedSet[int].symmetric_difference_update, PY_EMPTY),
        _param(SortedSet[int].update, PY_DATA_SET),
        _param(SortedSet[int].intersection_update, PY_DATA_SET),
    ),
)
def test_update_semantics(method: UpdateFn1, expected: AbstractSet[int]) -> None:
    a = SortedSet(DATA)
    method(a, a)
    assert a == expected
