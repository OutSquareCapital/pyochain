"""Adapted from `sortedcontainers` test suite.

Copyright 2014-2024 Grant Jenks — Licensed under the Apache License 2.0

Original source:
https://github.com/grantjenks/python-sortedcontainers/blob/master/tests/test_coverage_sortedkeylist_modulo.py
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pytest

from pyochain.abc import PyoMutableSequence
from pyochain.collections import SortedKeyList, SortedList
from pyochain.collections._sorted import (  # ruff: ignore[import-private-name]
    assert_sorted_list_empty,
    check_sorted_key_list,
)

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison


def modulo(val: float) -> float:
    return val % 10


def test_init() -> None:
    slt = SortedKeyList(key=modulo)
    check_sorted_key_list(slt)

    slt = SortedKeyList(key=modulo)
    slt.reset(10000)
    assert slt.load == 10000
    check_sorted_key_list(slt)

    slt = SortedKeyList(range(100), key=modulo)
    assert all(
        tup[0] == tup[1]
        for tup in zip(slt, sorted(range(100), key=modulo), strict=False)
    )

    slt.clear()
    assert_sorted_list_empty(slt)
    assert isinstance(slt, PyoMutableSequence)
    assert not isinstance(slt, SortedList)
    assert isinstance(slt, SortedKeyList)

    check_sorted_key_list(slt)


def test_new() -> None:
    slt = SortedKeyList(iter(range(1000)), key=modulo)
    assert slt == sorted(range(1000), key=modulo)
    check_sorted_key_list(slt)
    assert isinstance(slt, PyoMutableSequence)
    # NOTE: We diverge from original sortedcontainers behavior here. SortedList is NOT a parent class of SortedKeyList anymore.
    assert not isinstance(slt, SortedList)
    assert isinstance(slt, SortedKeyList)
    assert slt.__class__ is SortedKeyList


def test_key() -> None:
    slt = SortedKeyList(range(100), key=lambda val: val % 10)
    check_sorted_key_list(slt)

    values = sorted(range(100), key=lambda val: (val % 10, val))
    assert slt == values
    assert all(val in slt for val in range(100))


def test_key2() -> None:
    class Incomparable:
        pass

    a = Incomparable()
    b = Incomparable()
    slt = SortedKeyList[Incomparable, int](key=lambda _: 1)
    slt.add(a)
    slt.add(b)
    assert slt == [a, b]


def test_add() -> None:
    random.seed(0)
    slt = SortedKeyList(key=modulo)
    for val in range(1000):
        slt.add(val)
    check_sorted_key_list(slt)

    slt = SortedKeyList(key=modulo)
    for val in range(1000, 0, -1):
        slt.add(val)
    check_sorted_key_list(slt)

    slt = SortedKeyList(key=modulo)
    for _ in range(1000):
        slt.add(random.random())
    check_sorted_key_list(slt)


def test_update() -> None:
    slt = SortedKeyList(key=modulo)

    slt.update(range(1000))
    assert all(
        tup[0] == tup[1]
        for tup in zip(slt, sorted(range(1000), key=modulo), strict=False)
    )
    assert len(slt) == 1000
    check_sorted_key_list(slt)

    slt.update(range(100))
    assert len(slt) == 1100
    check_sorted_key_list(slt)


def test_update_order_consistency() -> None:
    setup = [10, 20, 30]
    slt1 = SortedKeyList(setup, key=modulo)
    slt2 = SortedKeyList(setup, key=modulo)
    addition = [40, 50, 60]
    for value in addition:
        slt1.add(value)
    slt2.update(addition)
    assert slt1 == slt2


def test_contains() -> None:
    slt = SortedKeyList(key=modulo)
    slt.reset(7)

    assert 0 not in slt

    slt.update(range(100))

    for val in range(100):
        assert val in slt

    assert 100 not in slt

    check_sorted_key_list(slt)

    slt = SortedKeyList(range(100), key=modulo)
    slt.reset(4)
    assert all(val not in slt for val in range(100, 200))


def test_discard() -> None:
    slt = SortedKeyList(key=modulo)

    assert slt.discard(0) is None
    assert len(slt) == 0
    check_sorted_key_list(slt)

    slt = SortedKeyList([1, 2, 2, 2, 3, 3, 5], key=modulo)
    slt.reset(4)

    slt.discard(6)
    check_sorted_key_list(slt)
    slt.discard(4)
    check_sorted_key_list(slt)
    slt.discard(2)
    check_sorted_key_list(slt)
    slt.discard(11)
    slt.discard(12)
    slt.discard(13)
    slt.discard(15)

    assert all(tup[0] == tup[1] for tup in zip(slt, [1, 2, 2, 3, 3, 5], strict=False))


def test_remove() -> None:
    slt = SortedKeyList(key=modulo)

    assert slt.discard(0) is None
    assert len(slt) == 0
    check_sorted_key_list(slt)

    slt = SortedKeyList([1, 2, 2, 2, 3, 3, 5], key=modulo)
    slt.reset(4)

    slt.remove(2)
    check_sorted_key_list(slt)

    assert all(tup[0] == tup[1] for tup in zip(slt, [1, 2, 2, 3, 3, 5], strict=False))


def test_remove_valueerror1() -> None:
    slt = SortedKeyList(key=modulo)
    with pytest.raises(ValueError):
        slt.remove(0)


def test_remove_valueerror2() -> None:
    slt = SortedKeyList(range(100), key=modulo)
    slt.reset(10)
    with pytest.raises(ValueError):
        slt.remove(100)


def test_remove_valueerror3() -> None:
    slt = SortedKeyList([1, 2, 2, 2, 3, 3, 5], key=modulo)
    with pytest.raises(ValueError):
        slt.remove(4)


def test_remove_valueerror4() -> None:
    slt = SortedKeyList([1, 1, 1, 2, 2, 2], key=modulo)
    with pytest.raises(ValueError):
        slt.remove(13)


def test_remove_valueerror5() -> None:
    slt = SortedKeyList([1, 1, 1, 2, 2, 2], key=modulo)
    with pytest.raises(ValueError):
        slt.remove(12)


def test_delete() -> None:
    slt = SortedKeyList(range(20), key=modulo)
    slt.reset(4)
    check_sorted_key_list(slt)
    for val in range(20):
        slt.remove(val)
        check_sorted_key_list(slt)

    assert_sorted_list_empty(slt)


def test_getitem() -> None:
    random.seed(0)
    slt = SortedKeyList(key=modulo)
    slt.reset(17)

    slt.add(5)
    # Same story as `test_build_index`. In any case, it doesn't seem to change the outcome of the test.
    # slt.build_index()  # ruff: ignore[commented-out-code]
    check_sorted_key_list(slt)
    slt.clear()
    r = range(100)

    lst = [random.random() for _ in r]
    slt.update(lst)
    lst.sort(key=modulo)

    assert all(slt[idx] == lst[idx] for idx in r)
    assert all(slt[idx - 99] == lst[idx - 99] for idx in r)


def test_getitem_slice() -> None:
    random.seed(0)
    slt = SortedKeyList(key=modulo)
    slt.reset(17)
    vals = [-75, -25, 0, 25, 75]
    vals_small = [-5, -1, 1, 5]

    lst: list[float] = []

    for _rpt in range(100):
        val = random.random()
        slt.add(val)
        lst.append(val)

    lst.sort(key=modulo)

    assert all(slt[start:] == lst[start:] for start in vals)

    assert all(slt[:stop] == lst[:stop] for stop in vals)

    assert all(slt[::step] == lst[::step] for step in vals_small)

    assert all(slt[start:stop] == lst[start:stop] for start in vals for stop in vals)

    assert all(
        slt[:stop:step] == lst[:stop:step] for stop in vals for step in vals_small
    )

    assert all(
        slt[start::step] == lst[start::step] for start in vals for step in vals_small
    )

    assert all(
        slt[start:stop:step] == lst[start:stop:step]
        for start in vals
        for stop in vals
        for step in vals_small
    )


def test_getitem_slice_big() -> None:
    slt = SortedKeyList(range(4), key=modulo)
    lst = sorted(range(4), key=modulo)
    vals = [-6, -4, -2, 0, 2, 4, 6]

    itr = (
        (start, stop, step)
        for start in vals
        for stop in vals
        for step in [-3, -2, -1, 1, 2, 3]
    )

    for start, stop, step in itr:
        assert slt[start:stop:step] == lst[start:stop:step]


def test_getitem_slicezero() -> None:
    slt = SortedKeyList(range(100), key=modulo)
    slt.reset(17)
    with pytest.raises(ValueError):
        slt[::0]


def test_getitem_indexerror1() -> None:
    slt = SortedKeyList(key=modulo)
    with pytest.raises(IndexError):
        slt[5]


def test_getitem_indexerror2() -> None:
    slt = SortedKeyList(range(100), key=modulo)
    with pytest.raises(IndexError):
        slt[200]


def test_getitem_indexerror3() -> None:
    slt = SortedKeyList(range(100), key=modulo)
    with pytest.raises(IndexError):
        slt[-101]


def test_delitem() -> None:
    random.seed(0)

    slt = SortedKeyList(range(100), key=modulo)
    slt.reset(17)
    while len(slt) > 0:
        del slt[random.randrange(len(slt))]
        check_sorted_key_list(slt)

    slt = SortedKeyList(range(100), key=modulo)
    slt.reset(17)
    del slt[:]
    assert len(slt) == 0
    check_sorted_key_list(slt)


def test_delitem_slice() -> None:
    slt = SortedKeyList(range(100), key=modulo)
    slt.reset(17)
    del slt[10:40:1]
    del slt[10:40:-1]
    del slt[10:40:2]
    del slt[10:40:-2]


def test_iter() -> None:
    slt = SortedKeyList(range(100), key=modulo)
    itr = iter(slt)
    assert all(
        tup[0] == tup[1]
        for tup in zip(sorted(range(100), key=modulo), itr, strict=False)
    )


def test_reversed() -> None:
    slt = SortedKeyList(range(100), key=modulo)
    rev = reversed(slt)
    assert all(
        tup[0] == tup[1]
        for tup in zip(
            reversed(sorted(range(100), key=modulo)),
            rev,
            strict=False,
        )
    )


def test_reverse() -> None:
    slt = SortedKeyList(range(100), key=modulo)
    with pytest.raises(NotImplementedError):
        slt.reverse()


def test_islice() -> None:
    sl = SortedKeyList(key=modulo)
    sl.reset(7)

    assert list(sl.islice()) == []

    values = sorted(range(100), key=modulo)
    sl.update(values)

    for start in range(53):
        for stop in range(53):
            assert list(sl.islice(start, stop)) == values[start:stop]

    for start in range(53):
        for stop in range(53):
            assert (
                list(sl.islice(start, stop, reverse=True)) == values[start:stop][::-1]
            )

    for start in range(53):
        assert list(sl.islice(start=start)) == values[start:]
        assert list(sl.islice(start=start, reverse=True)) == values[start:][::-1]

    for stop in range(53):
        assert list(sl.islice(stop=stop)) == values[:stop]
        assert list(sl.islice(stop=stop, reverse=True)) == values[:stop][::-1]


IRANGE_TST_VALUES = sorted(range(100), key=modulo)


@pytest.mark.parametrize("load", range(5, 16))
def test_irange(load: int) -> None:  # ruff:ignore[complex-structure]

    slt = SortedKeyList(range(100), key=modulo)
    slt.reset(load)

    for start in range(10):
        for end in range(start, 10):
            temp = list(slt.irange(start, end))
            assert temp == IRANGE_TST_VALUES[(start * 10) : ((end + 1) * 10)]

            temp = list(slt.irange(start, end, reverse=True))
            assert temp == IRANGE_TST_VALUES[(start * 10) : ((end + 1) * 10)][::-1]

    for start in range(10):
        for end in range(start, 10):
            temp = list(slt.irange(start, end, inclusive=(True, False)))
            assert temp == IRANGE_TST_VALUES[(start * 10) : (end * 10)]

    for start in range(10):
        for end in range(start, 10):
            temp = list(slt.irange(start, end, (False, True)))
            assert temp == IRANGE_TST_VALUES[((start + 1) * 10) : ((end + 1) * 10)]

    for start in range(10):
        for end in range(start, 10):
            temp = list(slt.irange(start, end, inclusive=(False, False)))
            assert temp == IRANGE_TST_VALUES[((start + 1) * 10) : (end * 10)]

    for start in range(10):
        temp = list(slt.irange(minimum=start))
        assert temp == IRANGE_TST_VALUES[(start * 10) :]

    for end in range(10):
        temp = list(slt.irange(maximum=end))
        assert temp == IRANGE_TST_VALUES[: (end + 1) * 10]


@pytest.mark.parametrize("load", range(5, 16))
def test_irange_key(load: int) -> None:  # ruff:ignore[complex-structure]

    slt = SortedKeyList(range(100), key=modulo)
    slt.reset(load)
    r = range(10)

    for start in r:
        for end in range(start, 10):
            temp = list(slt.irange_key(start, end))
            assert temp == IRANGE_TST_VALUES[(start * 10) : ((end + 1) * 10)]

            temp = list(slt.irange_key(start, end, reverse=True))
            assert temp == IRANGE_TST_VALUES[(start * 10) : ((end + 1) * 10)][::-1]

    for start in r:
        for end in range(start, 10):
            temp = list(slt.irange_key(start, end, inclusive=(True, False)))
            assert temp == IRANGE_TST_VALUES[(start * 10) : (end * 10)]

    for start in r:
        for end in range(start, 10):
            temp = list(slt.irange_key(start, end, (False, True)))
            assert temp == IRANGE_TST_VALUES[((start + 1) * 10) : ((end + 1) * 10)]

    for start in r:
        for end in range(start, 10):
            temp = list(slt.irange_key(start, end, inclusive=(False, False)))
            assert temp == IRANGE_TST_VALUES[((start + 1) * 10) : (end * 10)]

    for start in r:
        temp = list(slt.irange_key(min_key=start))
        assert temp == IRANGE_TST_VALUES[(start * 10) :]

    for end in r:
        temp = list(slt.irange_key(max_key=end))
        assert temp == IRANGE_TST_VALUES[: (end + 1) * 10]


def test_len() -> None:
    slt = SortedKeyList(key=modulo)

    for val in range(100):
        slt.add(val)
        assert len(slt) == (val + 1)


def test_bisect_left() -> None:
    slt = SortedKeyList(key=modulo)
    assert slt.bisect_left(0) == 0
    slt = SortedKeyList(range(100), key=modulo)
    slt.reset(17)
    slt.update(range(100))
    check_sorted_key_list(slt)
    assert slt.bisect_left(50) == 0
    assert slt.bisect_left(0) == 0


def test_bisect_right() -> None:
    slt = SortedKeyList(key=modulo)
    assert slt.bisect_right(10) == 0
    slt = SortedKeyList(range(100), key=modulo)
    slt.reset(17)
    slt.update(range(100))
    check_sorted_key_list(slt)
    assert slt.bisect_right(10) == 20
    assert slt.bisect_right(0) == 20


def test_bisect_key_left() -> None:
    slt = SortedKeyList(key=modulo)
    assert slt.bisect_key_left(10) == 0
    slt = SortedKeyList(range(100), key=modulo)
    slt.reset(17)
    slt.update(range(100))
    check_sorted_key_list(slt)
    assert slt.bisect_key_left(0) == 0
    assert slt.bisect_key_left(5) == 100
    assert slt.bisect_key_left(10) == 200


def test_bisect_key_right() -> None:
    slt = SortedKeyList(key=modulo)
    assert slt.bisect_key_right(0) == 0
    slt = SortedKeyList(range(100), key=modulo)
    slt.reset(17)
    slt.update(range(100))
    check_sorted_key_list(slt)
    assert slt.bisect_key_right(0) == 20
    assert slt.bisect_key_right(5) == 120
    assert slt.bisect_key_right(10) == 200


def test_copy() -> None:
    slt = SortedKeyList(range(100), key=modulo)
    slt.reset(7)
    two = slt.copy()
    slt.add(100)
    assert len(slt) == 101
    assert len(two) == 100


def test_copy_copy() -> None:
    import copy

    slt = SortedKeyList(range(100), key=modulo)
    slt.reset(7)
    two = copy.copy(slt)
    slt.add(100)
    assert len(slt) == 101
    assert len(two) == 100


def test_count() -> None:
    slt = SortedKeyList(key=modulo)
    slt.reset(7)

    assert slt.count(0) == 0

    for iii in range(100):
        for _jjj in range(iii):
            slt.add(iii)
    check_sorted_key_list(slt)

    for iii in range(100):
        assert slt.count(iii) == iii

    slt = SortedKeyList(range(8), key=modulo)
    assert slt.count(9) == 0


def test_pop() -> None:
    slt = SortedKeyList(range(10), key=modulo)
    slt.reset(4)
    check_sorted_key_list(slt)
    assert slt.pop() == 9
    check_sorted_key_list(slt)
    assert slt.pop(0) == 0
    check_sorted_key_list(slt)
    assert slt.pop(-2) == 7
    check_sorted_key_list(slt)
    assert slt.pop(4) == 5
    check_sorted_key_list(slt)


def test_pop_indexerror1() -> None:
    slt = SortedKeyList(range(10), key=modulo)
    slt.reset(4)
    with pytest.raises(IndexError):
        _ = slt.pop(-11)


def test_pop_indexerror2() -> None:
    slt = SortedKeyList(range(10), key=modulo)
    slt.reset(4)
    with pytest.raises(IndexError):
        _ = slt.pop(10)


def test_index_enumerate() -> None:
    slt = SortedKeyList(range(100), key=modulo)
    slt.reset(7)

    for pos, val in enumerate(sorted(range(100), key=modulo)):
        assert val == slt.index(pos)

    assert slt.index(9, 0, 1000) == 90


def test_index_range() -> None:
    slt = SortedKeyList((0 for _ in range(100)), key=modulo)
    slt.reset(7)

    for start in range(100):
        for stop in range(start, 100):
            assert slt.index(0, start, stop + 1) == start

    for start in range(100):
        assert slt.index(0, -(100 - start)) == start

    assert slt.index(0, -1000) == 0


def test_index_valueerror1() -> None:
    slt = SortedKeyList([0] * 10, key=modulo)
    slt.reset(4)
    with pytest.raises(ValueError):
        _ = slt.index(0, 10)


def test_index_valueerror2() -> None:
    slt = SortedKeyList([0] * 10, key=modulo)
    slt.reset(4)
    with pytest.raises(ValueError):
        _ = slt.index(0, 0, -10)


def test_index_valueerror3() -> None:
    slt = SortedKeyList([0] * 10, key=modulo)
    slt.reset(4)
    with pytest.raises(ValueError):
        _ = slt.index(0, 7, 3)


def test_index_valueerror4() -> None:
    slt = SortedKeyList([0] * 10, key=modulo)
    slt.reset(4)
    with pytest.raises(ValueError):
        _ = slt.index(1)


def test_index_valueerror5() -> None:
    slt = SortedKeyList(key=modulo)
    with pytest.raises(ValueError):
        _ = slt.index(1)


def test_index_valueerror6() -> None:
    slt = SortedKeyList(range(100), key=modulo)
    slt.reset(4)
    with pytest.raises(ValueError):
        _ = slt.index(91, 0, 15)


def test_index_valueerror7() -> None:
    slt = SortedKeyList([0] * 10 + [1] * 10 + [2] * 10, key=modulo)
    slt.reset(4)
    with pytest.raises(ValueError):
        _ = slt.index(1, 0, 10)


def test_index_valueerror8() -> None:
    slt = SortedKeyList(range(10), key=modulo)
    slt.reset(4)
    with pytest.raises(ValueError):
        _ = slt.index(4, 5)


def test_index_valueerror9() -> None:
    slt = SortedKeyList(key=modulo)
    slt.reset(4)
    with pytest.raises(ValueError):
        _ = slt.index(5)


def test_index_valueerror10() -> None:
    slt = SortedKeyList(range(10), key=modulo)
    slt.reset(4)
    with pytest.raises(ValueError):
        _ = slt.index(19)


def test_mul() -> None:
    this = SortedKeyList(range(10), key=modulo)
    this.reset(4)
    that = this * 5
    check_sorted_key_list(this)
    check_sorted_key_list(that)
    assert this == sorted(range(10), key=modulo)
    assert that == sorted(list(range(10)) * 5, key=modulo)
    assert this != that


def test_imul() -> None:
    this = SortedKeyList(range(10), key=modulo)
    this.reset(4)
    this *= 5
    check_sorted_key_list(this)
    assert this == sorted(list(range(10)) * 5, key=modulo)


def test_op_add() -> None:
    this = SortedKeyList(range(10), key=modulo)
    this.reset(4)
    assert (this + this + this) == (this * 3)

    that = SortedKeyList(range(10), key=modulo)
    that.reset(4)
    that += that
    that += that
    assert that == (this * 4)


def test_eq() -> None:
    this = SortedKeyList(range(10), key=modulo)
    this.reset(4)
    assert this == list(range(10))
    assert this == tuple(range(10))
    assert this != list(range(9))


def test_ne() -> None:
    this = SortedKeyList(range(10, 20), key=modulo)
    this.reset(4)
    assert this != list(range(11, 21))
    assert this != tuple(range(10, 21))
    assert this != [0, 1, 2, 3, 3, 5, 6, 7, 8, 9]
    assert this != (val for val in range(10))
    assert this != set()


def test_lt() -> None:
    this = SortedKeyList(range(10, 15), key=modulo)
    this.reset(4)
    assert this < [10, 11, 13, 13, 14]
    assert this < [10, 11, 12, 13, 14, 15]
    assert this < [11]


def test_le() -> None:
    this = SortedKeyList(range(10, 15), key=modulo)
    this.reset(4)
    assert this <= [10, 11, 12, 13, 14]
    assert this <= [10, 11, 12, 13, 14, 15]
    assert this <= [10, 11, 13, 13, 14]
    assert this <= [11]


def test_gt() -> None:
    this = SortedKeyList(range(10, 15), key=modulo)
    this.reset(4)
    assert this > [10, 11, 11, 13, 14]
    assert this > [10, 11, 12, 13]
    assert this > [9]


def test_ge() -> None:
    this = SortedKeyList(range(10, 15), key=modulo)
    this.reset(4)
    assert this >= [10, 11, 12, 13, 14]
    assert this >= [10, 11, 12, 13]
    assert this >= [10, 11, 11, 13, 14]
    assert this >= [9]


def test_repr() -> None:
    this = SortedKeyList(range(10), key=modulo)
    this.reset(4)
    assert repr(this).startswith(
        "SortedKeyList([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], key=<function modulo at "
    )


@pytest.mark.skip(reason="Same reason as `test_repr_recursion` in `test_sortedlist.py`")
def test_repr_recursion() -> None:
    this: SortedKeyList[list[int], list[int]] = SortedKeyList(
        [[1], [2], [3], [4]], key=lambda val: val
    )
    # pyrefly: ignore [bad-argument-type]
    this.inner.lists[-1].append(this)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
    assert repr(this).startswith(
        "SortedKeyList([[1], [2], [3], [4], ...], key=<function "
    )


@pytest.mark.skip(reason="We don't support subclassing of concrete pyochain types ATM")
def test_repr_subclass() -> None:
    class CustomSortedKeyList[T, OT: SupportsRichComparison](SortedKeyList[T, OT]):
        pass

    this = CustomSortedKeyList(range(10), key=modulo)
    this.reset(4)
    assert repr(this).startswith(
        "CustomSortedKeyList([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], key=<function modulo at "
    )
