"""Adapted from `sortedcontainers` test suite.

Original source:
https://github.com/grantjenks/python-sortedcontainers/blob/master/tests/test_coverage_sorteddict.py
"""

import gc
import operator
import platform
import string
from collections.abc import ItemsView, KeysView, Mapping
from typing import override

import pytest

from pyochain._types import SupportsHashableAndRichComparison
from pyochain.collections import SortedDict, SortedKeyDict

from ._utils import check_sorted_dict


def modulo(value: int) -> int:
    return value % 10


def test_init() -> None:
    temp = SortedDict[str, int]()
    check_sorted_dict(temp)


def test_init_key() -> None:
    temp = SortedKeyDict[int, int, int](key=operator.neg)
    assert temp.key == operator.neg
    check_sorted_dict(temp)


def test_init_args() -> None:
    temp = SortedDict([("a", 1), ("b", 2)])
    assert len(temp) == 2
    assert temp["a"] == 1
    assert temp["b"] == 2
    check_sorted_dict(temp)


def test_init_kwargs() -> None:
    temp = SortedDict[str, int](a=1, b=2)
    assert len(temp) == 2
    assert temp["a"] == 1
    assert temp["b"] == 2
    check_sorted_dict(temp)


def test_clear() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict(mapping)
    assert len(temp) == 26
    assert list(temp.items()) == mapping
    temp.clear()
    assert len(temp) == 0


def test_contains() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict(mapping)
    assert all((val in temp) for val in string.ascii_lowercase)


def test_delitem() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict(mapping)
    del temp["a"]
    check_sorted_dict(temp)


def test_getitem() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict(mapping)
    assert all((temp[val] == pos) for pos, val in enumerate(string.ascii_lowercase))


def test_eq() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp1 = SortedDict(mapping)
    temp2 = SortedDict(mapping)
    assert temp1 == temp2
    assert temp1 == temp2
    temp2["a"] = 100
    assert temp1 != temp2
    assert temp1 != temp2
    del temp2["a"]
    assert temp1 != temp2
    assert temp1 != temp2
    temp2["zz"] = 0
    assert temp1 != temp2
    assert temp1 != temp2


def test_iter() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict(mapping)
    assert all(
        lhs == rhs for lhs, rhs in zip(temp, string.ascii_lowercase, strict=False)
    )


def test_iter_key() -> None:
    temp = SortedKeyDict(((val, val) for val in range(100)), key=operator.neg)
    temp.reset(7)
    assert all(lhs == rhs for lhs, rhs in zip(temp, reversed(range(100)), strict=False))


def test_reversed() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict(mapping)
    assert all(
        lhs == rhs
        for lhs, rhs in zip(
            reversed(temp), reversed(string.ascii_lowercase), strict=False
        )
    )


def test_reversed_key() -> None:
    temp = SortedKeyDict(((val, val) for val in range(100)), key=modulo)
    temp.reset(7)
    values = sorted(range(100), key=modulo)
    assert all(
        lhs == rhs for lhs, rhs in zip(reversed(temp), reversed(values), strict=False)
    )


def test_islice() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict(mapping)
    temp.reset(7)

    for start in range(30):
        for stop in range(30):
            assert list(temp.islice(start, stop)) == list(
                string.ascii_lowercase[start:stop]
            )


def test_irange() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict(mapping)
    temp.reset(7)
    for start in range(26):
        for stop in range(start + 1, 26):
            result = list(string.ascii_lowercase[start:stop])
            assert list(temp.irange(result[0], result[-1])) == result


def test_irange_key() -> None:
    temp = SortedKeyDict(((val, val) for val in range(100)), key=modulo)
    temp.reset(7)
    values = sorted(range(100), key=modulo)

    for start in range(10):
        for stop in range(start, 10):
            result = list(temp.irange_key(start, stop))
            assert result == values[(start * 10) : ((stop + 1) * 10)]


def test_len() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict(mapping)
    assert len(temp) == 26


def test_setitem() -> None:
    temp = SortedDict[str, int]()

    for pos, key in enumerate(string.ascii_lowercase):
        temp[key] = pos
        check_sorted_dict(temp)

    assert len(temp) == 26

    for pos, key in enumerate(string.ascii_lowercase):
        temp[key] = pos
        check_sorted_dict(temp)

    assert len(temp) == 26


def test_copy() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict(mapping)
    dup = temp.copy()
    assert len(temp) == 26
    assert len(dup) == 26
    dup.clear()
    assert len(temp) == 26
    assert len(dup) == 0


def test_copy_copy() -> None:
    import copy

    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict(mapping)
    dup = copy.copy(temp)
    assert len(temp) == 26
    assert len(dup) == 26
    dup.clear()
    assert len(temp) == 26
    assert len(dup) == 0


def test_fromkeys() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict.fromkeys(mapping, 1)
    assert all(temp[key] == 1 for key in temp)


def test_get() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict(mapping)
    assert temp.get("a") == 0
    assert temp.get("A", -1) == -1


def test_items() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict(mapping)
    assert list(temp.items()) == mapping


def test_keys() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict(mapping)
    assert list(temp.keys()) == [key for key, _ in mapping]


def test_values() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict(mapping)
    assert list(temp.values()) == [pos for _, pos in mapping]


def test_notgiven() -> None:
    assert repr(SortedDict._SortedDict__not_given) == "<not-given>"  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]


def test_pop() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict(mapping)
    assert temp.pop("a") == 0
    assert temp.pop("a", -1) == -1


def test_pop2() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict(mapping)
    with pytest.raises(KeyError):
        _ = temp.pop("A")


def test_popitem() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict(mapping)
    assert temp.popitem() == ("z", 25)


def test_popitem2() -> None:
    temp = SortedDict[str, str]()
    with pytest.raises(KeyError):
        _ = temp.popitem()


def test_popitem3() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict(mapping)
    assert temp.popitem(index=0) == ("a", 0)


def test_peekitem() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict(mapping)
    assert temp.peekitem() == ("z", 25)
    assert temp.peekitem(0) == ("a", 0)
    assert temp.peekitem(index=4) == ("e", 4)


def test_peekitem2() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict(mapping)
    with pytest.raises(IndexError):
        _ = temp.peekitem(index=100)


def test_setdefault() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict(mapping)
    assert temp.setdefault("a", -1) == 0
    assert temp["a"] == 0
    assert temp.setdefault("A", -1) == -1


def test_update() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict[str, int]()
    temp.update()
    temp.update(mapping)
    temp.update(dict(mapping))
    temp.update(mapping[5:7])
    assert list(temp.items()) == mapping


def test_update2() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict[str, int]()
    temp.update(**dict(mapping))
    assert list(temp.items()) == mapping


def test_repr() -> None:
    temp = SortedDict({"alice": 3, "bob": 1, "carol": 2, "dave": 4})
    assert repr(temp) == "SortedDict({'alice': 3, 'bob': 1, 'carol': 2, 'dave': 4})"


class Identity:
    def __call__[T](self, value: T) -> T:
        return value

    @override
    def __repr__(self) -> str:
        return "identity"


def test_repr_recursion() -> None:
    temp = SortedKeyDict({"alice": 3, "bob": 1, "carol": 2, "dave": 4}, key=Identity())
    temp["bob"] = temp  # pyright: ignore[reportArgumentType]
    assert (
        repr(temp)
        == "SortedKeyDict(identity, {'alice': 3, 'bob': ..., 'carol': 2, 'dave': 4})"
    )


def test_repr_subclass() -> None:
    class CustomSortedDict[K: SupportsHashableAndRichComparison, V](SortedDict[K, V]):
        pass

    temp = CustomSortedDict({"alice": 3, "bob": 1, "carol": 2, "dave": 4})
    assert (
        repr(temp) == "CustomSortedDict({'alice': 3, 'bob': 1, 'carol': 2, 'dave': 4})"
    )


def test_index() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict(mapping)
    assert temp.index("a") == 0
    assert temp.index("f", 3, -3) == 5


def test_index_key() -> None:
    temp = SortedKeyDict(((val, val) for val in range(100)), key=operator.neg)
    temp.reset(7)
    assert all(temp.index(val) == (99 - val) for val in range(100))


def test_bisect() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict(mapping)
    assert temp.bisect_left("a") == 0
    assert temp.bisect_right("f") == 6


def test_bisect_key() -> None:
    temp = SortedKeyDict(((val, val) for val in range(100)), key=modulo)
    temp.reset(7)
    assert all(temp.bisect_right(val) == ((val % 10) + 1) * 10 for val in range(100))
    assert all(temp.bisect_left(val) == (val % 10) * 10 for val in range(100))


def test_bisect_key2() -> None:
    temp = SortedKeyDict(((val, val) for val in range(100)), key=modulo)
    temp.reset(7)
    assert all(temp.bisect_key_right(val) == ((val % 10) + 1) * 10 for val in range(10))
    assert all(temp.bisect_key_left(val) == (val % 10) * 10 for val in range(10))


def test_keysview() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict(mapping[:13])
    keys = temp.keys()

    assert len(keys) == 13
    assert "a" in keys
    assert list(keys) == [val for val, _ in mapping[:13]]
    assert keys[0] == "a"
    assert list(reversed(keys)) == list(reversed(string.ascii_lowercase[:13]))
    assert keys.index("f") == 5
    assert keys.count("m") == 1
    assert keys.count("0") == 0
    assert keys.isdisjoint(["1", "2", "3"])

    temp.update(mapping[13:])

    assert len(keys) == 26
    assert "z" in keys
    assert list(keys) == [val for val, _ in mapping]

    that = dict(mapping)

    that_keys = get_keysview(that)

    assert keys == that_keys
    assert keys == that_keys
    assert not (keys < that_keys)
    assert not (keys > that_keys)
    assert keys <= that_keys
    assert keys >= that_keys

    assert list(keys & that_keys) == [val for val, _ in mapping]
    assert list(keys | that_keys) == [val for val, _ in mapping]
    assert list(keys - that_keys) == []
    assert list(keys ^ that_keys) == []

    keys = SortedDict(mapping[:2]).keys()
    assert repr(keys) == "SortedKeysView(SortedDict({'a': 0, 'b': 1}))"


def get_keysview[K, V](dic: Mapping[K, V]) -> KeysView[K]:
    return dic.keys()


def test_valuesview() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict(mapping[:13])
    values = temp.values()

    assert len(values) == 13
    assert 0 in values
    assert list(values) == [pos for _, pos in mapping[:13]]
    assert values[0] == 0
    assert values[-3:] == [10, 11, 12]
    assert list(reversed(values)) == list(reversed(range(13)))
    assert values.index(5) == 5
    assert values.count(10) == 1

    temp.update(mapping[13:])

    assert len(values) == 26
    assert 25 in values
    assert list(values) == [pos for _, pos in mapping]

    values = SortedDict(mapping[:2]).values()
    assert repr(values) == "SortedValuesView(SortedDict({'a': 0, 'b': 1}))"


def test_values_view_index() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict(mapping[:13])
    values = temp.values()
    with pytest.raises(ValueError):
        _ = values.index(100)


def test_itemsview() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict(mapping[:13])
    items = temp.items()

    assert len(items) == 13
    assert ("a", 0) in items
    assert list(items) == mapping[:13]
    assert items[0] == ("a", 0)
    assert items[-3:] == [("k", 10), ("l", 11), ("m", 12)]
    assert list(reversed(items)) == list(reversed(mapping[:13]))
    assert items.index(("f", 5)) == 5
    assert items.count(("m", 12)) == 1
    assert items.isdisjoint([("0", 26), ("1", 27)])
    assert not items.isdisjoint([("a", 0), ("b", 1)])

    temp.update(mapping[13:])

    assert len(items) == 26
    assert ("z", 25) in items
    assert list(items) == mapping

    that = dict(mapping)
    that_items = get_itemsview(that)

    assert items == that_items
    assert items == that_items
    assert not (items < that_items)
    assert not (items > that_items)
    assert items <= that_items
    assert items >= that_items

    assert list(items & that_items) == mapping
    assert list(items | that_items) == mapping
    assert list(items - that_items) == []
    assert list(items ^ that_items) == []

    items = SortedDict(mapping[:2]).items()
    assert repr(items) == "SortedItemsView(SortedDict({'a': 0, 'b': 1}))"


def get_itemsview[K, V](dic: Mapping[K, V]) -> ItemsView[K, V]:
    return dic.items()


def test_items_view_index() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp = SortedDict(mapping[:13])
    items = temp.items()
    with pytest.raises(ValueError):
        _ = items.index(("f", 100))


def test_pickle() -> None:
    import pickle

    alpha = SortedKeyDict(
        zip(range(10000), range(10000), strict=False), key=operator.neg
    )
    alpha.reset(500)
    beta: SortedKeyDict[int, int, int] = pickle.loads(pickle.dumps(alpha))  # pyright: ignore[reportAny]
    assert alpha == beta
    assert alpha._key == beta._key  # pyright: ignore[reportPrivateUsage]


if platform.python_implementation() == "CPython":

    def test_ref_counts() -> None:
        start_count = len(gc.get_objects())
        temp = SortedDict[float, float]()
        init_count = len(gc.get_objects())
        assert init_count > start_count
        del temp
        del_count = len(gc.get_objects())
        assert start_count == del_count


def test_or() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp1 = SortedDict(mapping[:13])
    temp2 = SortedDict(mapping[13:])
    temp3 = temp1 | temp2
    assert temp3 == dict(mapping)


def test_ror() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp1 = dict(mapping[:13])
    temp2 = SortedDict(mapping[13:])
    temp3 = temp1 | temp2
    assert temp3 == dict(mapping)


def test_ior() -> None:
    mapping = [(val, pos) for pos, val in enumerate(string.ascii_lowercase)]
    temp1 = SortedDict(mapping[:13])
    temp2 = SortedDict(mapping[13:])
    temp1 |= temp2
    assert temp1 == dict(mapping)
