from __future__ import annotations

from collections.abc import Iterable, Sequence
from operator import neg
from typing import TYPE_CHECKING

import pytest

from pyochain import Range
from pyochain.collections import SortedKeySet, SortedSet

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyochain.collections._sorted import BaseSortedSet
type UpdateFn = Callable[[SortedSet[int], *tuple[Iterable[int]]], None]
DATA = Range(5)
DATA_SET = SortedSet(DATA)

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
