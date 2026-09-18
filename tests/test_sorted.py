from __future__ import annotations

from collections.abc import Iterable
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
type UpdateFn1[T] = Callable[[SortedSet[int], Iterable[int]], T]
type SortedSetFactory = Callable[[], BaseSortedSet[int]]
DATA = Range(5)
PY_EMPTY = set[int]()
PY_DATA_SET = set[int](DATA)
DATA_SET = SortedSet(DATA)
DATA_KEY_SET = SortedKeySet(neg, DATA)


@pytest.mark.parametrize("data", (DATA_SET, SortedKeySet(neg, DATA)))
def test_set_bisect(data: BaseSortedSet[int]) -> None:
    assert data.bisect_left(2) == 2
    assert data.bisect_right(2) == 3


def _param[T](fn: UpdateFn1[T], expected: T) -> ParameterSet:
    return pytest.param(fn, expected, id=fn.__name__)


@pytest.mark.parametrize(
    ("method", "expected"),
    (
        _param(SortedSet[int].difference, PY_EMPTY),
        _param(SortedSet[int].__sub__, PY_EMPTY),
        _param(SortedSet[int].symmetric_difference, PY_EMPTY),
        _param(SortedSet[int].__xor__, PY_EMPTY),
        _param(SortedSet[int].union, PY_DATA_SET),
        _param(SortedSet[int].__or__, PY_DATA_SET),
        _param(SortedSet[int].intersection, PY_DATA_SET),
        _param(SortedSet[int].__and__, PY_DATA_SET),
        _param(SortedSet[int].is_disjoint, True),
        _param(SortedSet[int].is_subset, True),
        _param(SortedSet[int].is_superset, True),
    ),
)
def test_deadlock(
    method: UpdateFn1[AbstractSet[int]], expected: AbstractSet[int]
) -> None:
    a = SortedSet(DATA)
    assert method(a, a) == expected


@pytest.mark.parametrize(
    ("method", "expected"),
    (
        _param(SortedSet[int].difference_update, PY_EMPTY),
        _param(SortedSet[int].__isub__, PY_EMPTY),
        _param(SortedSet[int].symmetric_difference_update, PY_EMPTY),
        _param(SortedSet[int].__ixor__, PY_EMPTY),
        _param(SortedSet[int].update, PY_DATA_SET),
        _param(SortedSet[int].__ior__, PY_DATA_SET),
        _param(SortedSet[int].intersection_update, PY_DATA_SET),
        _param(SortedSet[int].__iand__, PY_DATA_SET),
    ),
)
def test_deadlock_mut[T](method: UpdateFn1[object], expected: object) -> None:
    a = SortedSet(DATA)
    _ = method(a, a)
    assert a == expected
