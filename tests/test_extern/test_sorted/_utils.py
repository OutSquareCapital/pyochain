from __future__ import annotations

import sys
import traceback
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyochain.collections import SortedDict, SortedKeyList, SortedList, SortedSet


def check_sorted_key_list(lst: SortedKeyList[Any, Any]) -> None:  # ruff:ignore[complex-structure, too-many-branches]
    """Check invariants of sorted-key list.

    Runtime complexity: `O(n)`

    """
    data = lst.inner
    try:  # ruff:ignore[too-many-statements-in-try-clause]
        assert data.load >= 4
        assert len(data.maxes) == len(data.lists) == len(data.keys)
        assert data.len == sum(len(sublist) for sublist in data.lists)

        # Check all sublists are sorted.

        for sublist in data.keys:
            for pos in range(1, len(sublist)):
                assert sublist[pos - 1] <= sublist[pos]

        # Check beginning/end of sublists are sorted.

        for pos in range(1, len(data.keys)):
            assert data.keys[pos - 1][-1] <= data.keys[pos][0]

        # Check _keys matches _key mapped to _lists.

        for val_sublist, key_sublist in zip(
            data.lists,
            data.keys,
            strict=False,
        ):
            assert len(val_sublist) == len(key_sublist)
            for val, key in zip(val_sublist, key_sublist, strict=False):  # pyright: ignore[reportAny]
                assert data.key(val) == key

        # Check _maxes index is the last value of each sublist.

        for pos in range(len(data.maxes)):
            assert data.maxes[pos] == data.keys[pos][-1]

        # Check sublist lengths are less than double load-factor.

        double = data.load << 1
        assert all(len(sublist) <= double for sublist in data.lists)

        # Check sublist lengths are greater than half load-factor for all
        # but the last sublist.

        half = data.load >> 1
        for pos in range(len(data.lists) - 1):
            assert len(data.lists[pos]) >= half

        if data.idx:
            assert data.len == data.idx[0]
            assert len(data.idx) == data.offset + len(data.lists)

            # Check index leaf nodes equal length of sublists.

            for pos in range(len(data.lists)):
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
        print("len", data.len)
        print("load", data.load)
        print("offset", data.offset)
        print("len_index", len(data.idx))
        print("index", data.idx)
        print("len_maxes", len(data.maxes))
        print("maxes", data.maxes)
        print("len_keys", len(data.keys))
        print("keys", data.keys)
        print("len_lists", len(data.lists))
        print("lists", data.lists)
        raise


def check_sorted_set(data: SortedSet[Any]) -> None:
    set_ = data._set  # pyright: ignore[reportPrivateUsage]
    list_ = data._list  # pyright: ignore[reportPrivateUsage]
    check_sorted_list(list_)
    assert len(set_) == len(list_)
    assert all(value in set_ for value in list_)  # pyright: ignore[reportAny]


def check_sorted_dict(data: SortedDict[Any, Any]) -> None:
    list_ = data._list  # pyright: ignore[reportPrivateUsage]
    check_sorted_list(list_)
    assert len(data) == len(list_)
    assert all(key in data for key in list_)  # pyright: ignore[reportAny]


def check_sorted_list(lst: SortedList[Any]) -> None:  # ruff:ignore[complex-structure]

    data = lst.inner
    try:  # ruff:ignore[too-many-statements-in-try-clause]
        assert data.load >= 4
        assert len(data.maxes) == len(data.lists)
        assert data.len == sum(len(sublist) for sublist in data.lists)

        # Check all sublists are sorted.

        for sublist in data.lists:
            for pos in range(1, len(sublist)):
                assert sublist[pos - 1] <= sublist[pos]

        # Check beginning/end of sublists are sorted.

        for pos in range(1, len(data.lists)):
            assert data.lists[pos - 1][-1] <= data.lists[pos][0]

        # Check _maxes index is the last value of each sublist.

        for pos in range(len(data.maxes)):
            assert data.maxes[pos] == data.lists[pos][-1]

        # Check sublist lengths are less than double load-factor.

        double = data.load << 1
        assert all(len(sublist) <= double for sublist in data.lists)

        # Check sublist lengths are greater than half load-factor for all
        # but the last sublist.

        half = data.load >> 1
        for pos in range(len(data.lists) - 1):
            assert len(data.lists[pos]) >= half

        if data.idx:
            assert data.len == data.idx[0]
            assert len(data.idx) == data.offset + len(data.lists)

            # Check index leaf nodes equal length of sublists.

            for pos in range(len(data.lists)):
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
        print("len", data.len)
        print("load", data.load)
        print("offset", data.offset)
        print("len_index", len(data.idx))
        print("index", data.idx)
        print("len_maxes", len(data.maxes))
        print("maxes", data.maxes)
        print("len_lists", len(data.lists))
        print("lists", data.lists)
        raise
