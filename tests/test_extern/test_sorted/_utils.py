from __future__ import annotations

import sys
import traceback
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyochain.collections import SortedDict, SortedKeyList, SortedList, SortedSet
    from pyochain.rs import InnerKeyLists, InnerSorted


def check_sorted_key_list(lst: SortedKeyList[Any, Any]) -> None:  # ruff:ignore[complex-structure, too-many-branches]
    """Check invariants of sorted-key list.

    Runtime complexity: `O(n)`

    """
    data = lst.inner
    try:  # ruff:ignore[too-many-statements-in-try-clause]
        assert data.load >= 4
        assert data.maxes.len() == data.lists.len() == data.keys.len()
        assert data.len == data.lists.iter().map(lambda sublist: sublist.len()).sum()

        # Check all sublists are sorted.

        for sublist in data.keys:
            for pos in range(1, sublist.len()):
                assert sublist[pos - 1] <= sublist[pos]

        # Check beginning/end of sublists are sorted.

        for pos in range(1, data.keys.len()):
            assert data.keys[pos - 1][-1] <= data.keys[pos][0]

        # Check _keys matches _key mapped to _lists.

        for val_sublist, key_sublist in data.lists.iter().zip(data.keys):
            assert val_sublist.len() == key_sublist.len()
            for val, key in val_sublist.iter().zip(key_sublist):  # pyright: ignore[reportAny]
                assert data.key(val) == key

        # Check _maxes index is the last value of each sublist.

        for pos in range(data.maxes.len()):
            assert data.maxes[pos] == data.keys[pos][-1]

        # Check sublist lengths are less than double load-factor.

        double = data.load << 1
        assert data.lists.iter().all(lambda sublist: sublist.len() <= double)

        # Check sublist lengths are greater than half load-factor for all
        # but the last sublist.

        half = data.load >> 1
        for pos in range(data.lists.len() - 1):
            assert data.lists[pos].len() >= half

        if data.idx:
            assert data.len == data.idx[0]
            assert len(data.idx) == data.offset + data.lists.len()

            # Check index leaf nodes equal length of sublists.

            for pos in range(data.lists.len()):
                leaf = data.idx[data.offset + pos]
                assert leaf == len(data.lists[pos])

            # Check index branch nodes are the sum of their children.

            for pos in range(data.offset):
                child = (pos << 1) + 1
                if child >= len(data.idx):
                    assert data.idx[pos] == 0
                elif child + 1 == len(data.idx):
                    assert data.idx[pos] == data.idx[child]
                else:
                    child_sum = data.idx[child] + data.idx[child + 1]
                    assert child_sum == data.idx[pos]
    except:
        traceback.print_exc(file=sys.stdout)
        _show_key_list(data)
        raise


def check_sorted_list(lst: SortedList[Any]) -> None:  # ruff:ignore[complex-structure]

    data = lst.inner
    try:  # ruff:ignore[too-many-statements-in-try-clause]
        assert data.load >= 4
        assert data.maxes.len() == data.lists.len()
        assert data.len == data.lists.iter().map(lambda sublist: sublist.len()).sum()

        # Check all sublists are sorted.

        for sublist in data.lists:
            for pos in range(1, sublist.len()):
                assert sublist[pos - 1] <= sublist[pos]

        # Check beginning/end of sublists are sorted.

        for pos in range(1, data.lists.len()):
            assert data.lists[pos - 1][-1] <= data.lists[pos][0]

        # Check _maxes index is the last value of each sublist.

        for pos in range(data.maxes.len()):
            assert data.maxes[pos] == data.lists[pos][-1]

        # Check sublist lengths are less than double load-factor.

        double = data.load << 1
        assert data.lists.iter().all(lambda sublist: sublist.len() <= double)

        # Check sublist lengths are greater than half load-factor for all
        # but the last sublist.

        half = data.load >> 1
        for pos in range(data.lists.len() - 1):
            assert data.lists[pos].len() >= half

        if data.idx:
            assert data.len == data.idx[0]
            assert len(data.idx) == data.offset + data.lists.len()

            # Check index leaf nodes equal length of sublists.

            for pos in range(data.lists.len()):
                leaf = data.idx[data.offset + pos]
                assert leaf == data.lists[pos].len()

            # Check index branch nodes are the sum of their children.

            for pos in range(data.offset):
                child = (pos << 1) + 1
                if child >= len(data.idx):
                    assert data.idx[pos] == 0
                elif child + 1 == len(data.idx):
                    assert data.idx[pos] == data.idx[child]
                else:
                    child_sum = data.idx[child] + data.idx[child + 1]
                    assert child_sum == data.idx[pos]
    except:
        traceback.print_exc(file=sys.stdout)
        _show_list(data)
        raise


def _show_list(data: InnerSorted[Any, Any]) -> None:
    print("len", data.len)
    print("load", data.load)
    print("offset", data.offset)
    print("len_index", len(data.idx))
    print("index", data.idx)
    print("len_maxes", data.maxes.len())
    print("maxes", data.maxes)
    print("len_lists", data.lists.len())
    print("lists", data.lists)


def _show_key_list(data: InnerKeyLists[Any, Any, Any]) -> None:
    _show_list(data)
    print("len_keys", data.keys.len())
    print("keys", data.keys)


def check_sorted_set(data: SortedSet[Any]) -> None:
    set_ = data._set  # pyright: ignore[reportPrivateUsage]
    list_ = data._list  # pyright: ignore[reportPrivateUsage]
    check_sorted_list(list_)
    assert len(set_) == list_.len()
    assert list_.iter().all(lambda value: value in set_)  # pyright: ignore[reportAny]


def check_sorted_dict(data: SortedDict[Any, Any]) -> None:
    list_ = data._list  # pyright: ignore[reportPrivateUsage]
    check_sorted_list(list_)
    assert data.len() == list_.len()
    assert list_.iter().all(data.contains)
