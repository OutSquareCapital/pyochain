import sys
import traceback
from typing import Any

from pyochain.collections import SortedDict, SortedKeyList, SortedList, SortedSet


def check_sorted_list(data: SortedList[Any]) -> None:  # noqa: C901
    try:
        assert data._load >= 4
        assert len(data._maxes) == len(data._lists)
        assert data._len == sum(len(sublist) for sublist in data._lists)

        # Check all sublists are sorted.

        for sublist in data._lists:
            for pos in range(1, len(sublist)):
                assert sublist[pos - 1] <= sublist[pos]

        # Check beginning/end of sublists are sorted.

        for pos in range(1, len(data._lists)):
            assert data._lists[pos - 1][-1] <= data._lists[pos][0]

        # Check _maxes index is the last value of each sublist.

        for pos in range(len(data._maxes)):
            assert data._maxes[pos] == data._lists[pos][-1]

        # Check sublist lengths are less than double load-factor.

        double = data._load << 1
        assert all(len(sublist) <= double for sublist in data._lists)

        # Check sublist lengths are greater than half load-factor for all
        # but the last sublist.

        half = data._load >> 1
        for pos in range(len(data._lists) - 1):
            assert len(data._lists[pos]) >= half

        if data._index:
            assert data._len == data._index[0]
            assert len(data._index) == data._offset + len(data._lists)

            # Check index leaf nodes equal length of sublists.

            for pos in range(len(data._lists)):
                leaf = data._index[data._offset + pos]
                assert leaf == len(data._lists[pos])

            # Check index branch nodes are the sum of their children.

            for pos in range(data._offset):
                child = (pos << 1) + 1
                if child >= len(data._index):
                    assert data._index[pos] == 0
                elif child + 1 == len(data._index):
                    assert data._index[pos] == data._index[child]
                else:
                    child_sum = data._index[child] + data._index[child + 1]
                    assert child_sum == data._index[pos]
    except:
        traceback.print_exc(file=sys.stdout)
        print("len", data._len)
        print("load", data._load)
        print("offset", data._offset)
        print("len_index", len(data._index))
        print("index", data._index)
        print("len_maxes", len(data._maxes))
        print("maxes", data._maxes)
        print("len_lists", len(data._lists))
        print("lists", data._lists)
        raise


def check_sorted_key_list(data: SortedKeyList[Any, Any]) -> None:  # noqa: C901, PLR0912
    """Check invariants of sorted-key list.

    Runtime complexity: `O(n)`

    """
    try:
        assert data._load >= 4
        assert len(data._maxes) == len(data._lists) == len(data._keys)
        assert data._len == sum(len(sublist) for sublist in data._lists)

        # Check all sublists are sorted.

        for sublist in data._keys:
            for pos in range(1, len(sublist)):
                assert sublist[pos - 1] <= sublist[pos]

        # Check beginning/end of sublists are sorted.

        for pos in range(1, len(data._keys)):
            assert data._keys[pos - 1][-1] <= data._keys[pos][0]

        # Check _keys matches _key mapped to _lists.

        for val_sublist, key_sublist in zip(data._lists, data._keys, strict=False):
            assert len(val_sublist) == len(key_sublist)
            for val, key in zip(val_sublist, key_sublist, strict=False):
                assert data._key(val) == key

        # Check _maxes index is the last value of each sublist.

        for pos in range(len(data._maxes)):
            assert data._maxes[pos] == data._keys[pos][-1]

        # Check sublist lengths are less than double load-factor.

        double = data._load << 1
        assert all(len(sublist) <= double for sublist in data._lists)

        # Check sublist lengths are greater than half load-factor for all
        # but the last sublist.

        half = data._load >> 1
        for pos in range(len(data._lists) - 1):
            assert len(data._lists[pos]) >= half

        if data._index:
            assert data._len == data._index[0]
            assert len(data._index) == data._offset + len(data._lists)

            # Check index leaf nodes equal length of sublists.

            for pos in range(len(data._lists)):
                leaf = data._index[data._offset + pos]
                assert leaf == len(data._lists[pos])

            # Check index branch nodes are the sum of their children.

            for pos in range(data._offset):
                child = (pos << 1) + 1
                if child >= len(data._index):
                    assert data._index[pos] == 0
                elif child + 1 == len(data._index):
                    assert data._index[pos] == data._index[child]
                else:
                    child_sum = data._index[child] + data._index[child + 1]
                    assert child_sum == data._index[pos]
    except:
        traceback.print_exc(file=sys.stdout)
        print("len", data._len)
        print("load", data._load)
        print("offset", data._offset)
        print("len_index", len(data._index))
        print("index", data._index)
        print("len_maxes", len(data._maxes))
        print("maxes", data._maxes)
        print("len_keys", len(data._keys))
        print("keys", data._keys)
        print("len_lists", len(data._lists))
        print("lists", data._lists)
        raise


def check_sorted_set(data: SortedSet[Any]) -> None:
    set_ = data._set  # pyright: ignore[reportPrivateUsage]
    list_ = data._list  # pyright: ignore[reportPrivateUsage]
    check_sorted_list(list_)
    assert len(set_) == len(list_)
    assert all(value in set_ for value in list_)


def check_sorted_dict(data: SortedDict[Any, Any]) -> None:
    list_ = data._list
    check_sorted_list(list_)
    assert len(data) == len(list_)
    assert all(key in data for key in list_)
