from __future__ import annotations

import pytest
import sortedcontainers as sc

from pyochain.collections import SortedKeyList, SortedList
from pyochain.collections._sorted import check_sorted_key_list, check_sorted_list  # ruff: ignore[import-private-name]

from ._utils import LIST_CLASSES, List


@LIST_CLASSES
def test_mut_empty_collection(cls: type[List]) -> None:
    x = cls()
    # On a slice, it's a no-op
    del x[0:1]
    # On an index, it raises an IndexError
    with pytest.raises(IndexError):
        del x[-1]
    match x:
        case sc.SortedList():
            _try_setitem(x, NotImplementedError)
        case _:
            _try_setitem(x, IndexError)


def _try_setitem(x: List, err: type[Exception]) -> None:
    with pytest.raises(err):
        x[-1] = 0


_SLICE_VALS = (-6, -4, -2, 0, 2, 4, 6)
_SLICE_STEPS = (-3, -2, -1, 1, 2, 3)


@pytest.mark.parametrize("start", _SLICE_VALS)
@pytest.mark.parametrize("stop", _SLICE_VALS)
@pytest.mark.parametrize("step", _SLICE_STEPS)
def test_delitem_slice_big(start: int, stop: int, step: int) -> None:
    r = range(10)
    slt = SortedList(r)
    _check_lst(slt, list(r), slice(start, stop, step))
    check_sorted_list(slt)


@pytest.mark.parametrize("start", _SLICE_VALS)
@pytest.mark.parametrize("stop", _SLICE_VALS)
@pytest.mark.parametrize("step", _SLICE_STEPS)
def test_delitem_slice_big_key(start: int, stop: int, step: int) -> None:
    r = range(10)
    slt = SortedKeyList(lambda x: x, r)
    _check_lst(slt, list(r), slice(start, stop, step))
    check_sorted_key_list(slt)


def _check_lst(
    slt: SortedList[int] | SortedKeyList[int, int], lst: list[int], pyslice: slice
) -> None:
    slt.reset(4)
    del slt[pyslice]
    del lst[pyslice]
    assert slt == lst
