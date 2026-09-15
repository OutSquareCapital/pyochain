from __future__ import annotations

from operator import neg
from typing import TYPE_CHECKING

import pytest

from pyochain import Range
from pyochain.collections import SortedKeySet, SortedSet

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyochain.collections._sorted import BaseSortedSet


@pytest.mark.parametrize(
    "fn", (SortedSet[int].update, SortedSet[int].difference_update)
)
def test_update_set_deadlock(
    fn: Callable[[SortedSet[int], SortedSet[int]], None],
) -> None:
    data = Range(3)
    a = SortedSet(data)
    assert fn(a, a) == a


@pytest.mark.parametrize("data", (SortedSet(range(5)), SortedKeySet(neg, range(5))))
def test_set_bisect(data: BaseSortedSet[int]) -> None:
    assert data.bisect_left(2) == 2
    assert data.bisect_right(2) == 3
