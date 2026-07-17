import sys
import traceback
from typing import Any

from pyochain.collections import SortedDict, SortedKeyList, SortedList, SortedSet


def check_sorted_key_list(data: SortedKeyList[Any, Any]) -> None:  # ruff:ignore[complex-structure, too-many-branches]
    """Check invariants of sorted-key list.

    Runtime complexity: `O(n)`

    """
    try:  # ruff:ignore[too-many-statements-in-try-clause]
        assert data._load >= 4  # pyright: ignore[reportPrivateUsage]
        assert len(data._maxes) == len(data._lists) == len(data._keys)  # pyright: ignore[reportPrivateUsage]
        assert data._len == sum(len(sublist) for sublist in data._lists)  # pyright: ignore[reportPrivateUsage]

        # Check all sublists are sorted.

        for sublist in data._keys:  # pyright: ignore[reportPrivateUsage]
            for pos in range(1, len(sublist)):
                assert sublist[pos - 1] <= sublist[pos]

        # Check beginning/end of sublists are sorted.

        for pos in range(1, len(data._keys)):  # pyright: ignore[reportPrivateUsage]
            assert data._keys[pos - 1][-1] <= data._keys[pos][0]  # pyright: ignore[reportPrivateUsage]

        # Check _keys matches _key mapped to _lists.

        for val_sublist, key_sublist in zip(data._lists, data._keys, strict=False):  # pyright: ignore[reportPrivateUsage]
            assert len(val_sublist) == len(key_sublist)
            for val, key in zip(val_sublist, key_sublist, strict=False):  # pyright: ignore[reportAny]
                assert data._key(val) == key  # pyright: ignore[reportPrivateUsage]

        # Check _maxes index is the last value of each sublist.

        for pos in range(len(data._maxes)):  # pyright: ignore[reportPrivateUsage]
            assert data._maxes[pos] == data._keys[pos][-1]  # pyright: ignore[reportPrivateUsage]

        # Check sublist lengths are less than double load-factor.

        double = data._load << 1  # pyright: ignore[reportPrivateUsage]
        assert all(len(sublist) <= double for sublist in data._lists)  # pyright: ignore[reportPrivateUsage]

        # Check sublist lengths are greater than half load-factor for all
        # but the last sublist.

        half = data._load >> 1  # pyright: ignore[reportPrivateUsage]
        for pos in range(len(data._lists) - 1):  # pyright: ignore[reportPrivateUsage]
            assert len(data._lists[pos]) >= half  # pyright: ignore[reportPrivateUsage]

        if data._index:  # pyright: ignore[reportPrivateUsage]
            assert data._len == data._index[0]  # pyright: ignore[reportPrivateUsage]
            assert len(data._index) == data._offset + len(data._lists)  # pyright: ignore[reportPrivateUsage]

            # Check index leaf nodes equal length of sublists.

            for pos in range(len(data._lists)):  # pyright: ignore[reportPrivateUsage]
                leaf = data._index[data._offset + pos]  # pyright: ignore[reportPrivateUsage]
                assert leaf == len(data._lists[pos])  # pyright: ignore[reportPrivateUsage]

            # Check index branch nodes are the sum of their children.

            for pos in range(data._offset):  # pyright: ignore[reportPrivateUsage]
                child = (pos << 1) + 1
                if child >= len(data._index):  # pyright: ignore[reportPrivateUsage]
                    assert data._index[pos] == 0  # pyright: ignore[reportPrivateUsage]
                elif child + 1 == len(data._index):  # pyright: ignore[reportPrivateUsage]
                    assert data._index[pos] == data._index[child]  # pyright: ignore[reportPrivateUsage]
                else:
                    child_sum = data._index[child] + data._index[child + 1]  # pyright: ignore[reportPrivateUsage]
                    assert child_sum == data._index[pos]  # pyright: ignore[reportPrivateUsage]
    except:
        traceback.print_exc(file=sys.stdout)
        print("len", data._len)  # pyright: ignore[reportPrivateUsage]
        print("load", data._load)  # pyright: ignore[reportPrivateUsage]
        print("offset", data._offset)  # pyright: ignore[reportPrivateUsage]
        print("len_index", len(data._index))  # pyright: ignore[reportPrivateUsage]
        print("index", data._index)  # pyright: ignore[reportPrivateUsage]
        print("len_maxes", len(data._maxes))  # pyright: ignore[reportPrivateUsage]
        print("maxes", data._maxes)  # pyright: ignore[reportPrivateUsage]
        print("len_keys", len(data._keys))  # pyright: ignore[reportPrivateUsage]
        print("keys", data._keys)  # pyright: ignore[reportPrivateUsage]
        print("len_lists", len(data._lists))  # pyright: ignore[reportPrivateUsage]
        print("lists", data._lists)  # pyright: ignore[reportPrivateUsage]
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


def check_sorted_list(data: SortedList[Any]) -> None:  # ruff:ignore[complex-structure]
    try:  # ruff:ignore[too-many-statements-in-try-clause]
        assert data._load >= 4  # pyright: ignore[reportPrivateUsage]
        assert len(data._maxes) == len(data._lists)  # pyright: ignore[reportPrivateUsage]
        assert data._len == sum(len(sublist) for sublist in data._lists)  # pyright: ignore[reportPrivateUsage]

        # Check all sublists are sorted.

        for sublist in data._lists:  # pyright: ignore[reportPrivateUsage]
            for pos in range(1, len(sublist)):
                assert sublist[pos - 1] <= sublist[pos]

        # Check beginning/end of sublists are sorted.

        for pos in range(1, len(data._lists)):  # pyright: ignore[reportPrivateUsage]
            assert data._lists[pos - 1][-1] <= data._lists[pos][0]  # pyright: ignore[reportPrivateUsage]

        # Check _maxes index is the last value of each sublist.

        for pos in range(len(data._maxes)):  # pyright: ignore[reportPrivateUsage]
            assert data._maxes[pos] == data._lists[pos][-1]  # pyright: ignore[reportPrivateUsage]

        # Check sublist lengths are less than double load-factor.

        double = data._load << 1  # pyright: ignore[reportPrivateUsage]
        assert all(len(sublist) <= double for sublist in data._lists)  # pyright: ignore[reportPrivateUsage]

        # Check sublist lengths are greater than half load-factor for all
        # but the last sublist.

        half = data._load >> 1  # pyright: ignore[reportPrivateUsage]
        for pos in range(len(data._lists) - 1):  # pyright: ignore[reportPrivateUsage]
            assert len(data._lists[pos]) >= half  # pyright: ignore[reportPrivateUsage]

        if data._index:  # pyright: ignore[reportPrivateUsage]
            assert data._len == data._index[0]  # pyright: ignore[reportPrivateUsage]
            assert len(data._index) == data._offset + len(data._lists)  # pyright: ignore[reportPrivateUsage]

            # Check index leaf nodes equal length of sublists.

            for pos in range(len(data._lists)):  # pyright: ignore[reportPrivateUsage]
                leaf = data._index[data._offset + pos]  # pyright: ignore[reportPrivateUsage]
                assert leaf == len(data._lists[pos])  # pyright: ignore[reportPrivateUsage]

            # Check index branch nodes are the sum of their children.

            for pos in range(data._offset):  # pyright: ignore[reportPrivateUsage]
                child = (pos << 1) + 1
                if child >= len(data._index):  # pyright: ignore[reportPrivateUsage]
                    assert data._index[pos] == 0  # pyright: ignore[reportPrivateUsage]
                elif child + 1 == len(data._index):  # pyright: ignore[reportPrivateUsage]
                    assert data._index[pos] == data._index[child]  # pyright: ignore[reportPrivateUsage]
                else:
                    child_sum = data._index[child] + data._index[child + 1]  # pyright: ignore[reportPrivateUsage]
                    assert child_sum == data._index[pos]  # pyright: ignore[reportPrivateUsage]
    except:
        traceback.print_exc(file=sys.stdout)
        print("len", data._len)  # pyright: ignore[reportPrivateUsage]
        print("load", data._load)  # pyright: ignore[reportPrivateUsage]
        print("offset", data._offset)  # pyright: ignore[reportPrivateUsage]
        print("len_index", len(data._index))  # pyright: ignore[reportPrivateUsage]
        print("index", data._index)  # pyright: ignore[reportPrivateUsage]
        print("len_maxes", len(data._maxes))  # pyright: ignore[reportPrivateUsage]
        print("maxes", data._maxes)  # pyright: ignore[reportPrivateUsage]
        print("len_lists", len(data._lists))  # pyright: ignore[reportPrivateUsage]
        print("lists", data._lists)  # pyright: ignore[reportPrivateUsage]
        raise
