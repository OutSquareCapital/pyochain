"""Adapted from `sortedcontainers` test suite.

Original source:
https://github.com/grantjenks/python-sortedcontainers/blob/master/tests/test_coverage_sortedlist.py
"""

from __future__ import annotations

import random
from itertools import chain
from typing import TYPE_CHECKING

import pytest

from pyochain.collections import SortedList
from pyochain.rs import assert_sorted_list_empty, check_sorted_list

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison


def test_init() -> None:
    slt = SortedList[int]()
    check_sorted_list(slt)

    slt = SortedList[int]()
    slt.reset(10000)
    assert slt.load == 10000
    check_sorted_list(slt)

    slt = SortedList(range(100))
    assert all(tup[0] == tup[1] for tup in zip(slt, range(100), strict=False))

    slt.clear()
    assert_sorted_list_empty(slt)
    check_sorted_list(slt)


def test_add() -> None:
    random.seed(0)
    slt = SortedList[int]()
    for val in range(1000):
        slt.add(val)
        check_sorted_list(slt)

    slt = SortedList[int]()
    for val in range(1000, 0, -1):
        slt.add(val)
        check_sorted_list(slt)

    slt = SortedList[float]()
    for _ in range(1000):
        slt.add(random.random())
        check_sorted_list(slt)


def test_update() -> None:
    slt = SortedList[int]()

    slt.update(range(1000))
    assert len(slt) == 1000
    check_sorted_list(slt)

    slt.update(range(100))
    assert len(slt) == 1100
    check_sorted_list(slt)

    slt.update(range(10000))
    assert len(slt) == 11100
    check_sorted_list(slt)

    values = sorted(chain(range(1000), range(100), range(10000)))
    assert all(tup[0] == tup[1] for tup in zip(slt, values, strict=False))


def test_contains() -> None:
    slt = SortedList[int]()
    assert 0 not in slt

    slt.update(range(100))

    for val in range(100):
        assert val in slt

    assert 10000 not in slt

    check_sorted_list(slt)


def test_discard() -> None:
    slt = SortedList[int]()

    assert slt.discard(0) is None
    assert len(slt) == 0
    check_sorted_list(slt)

    slt = SortedList([1, 2, 2, 2, 3, 3, 5])
    slt.reset(4)

    slt.discard(6)
    check_sorted_list(slt)
    slt.discard(4)
    check_sorted_list(slt)
    slt.discard(2)
    check_sorted_list(slt)

    assert all(tup[0] == tup[1] for tup in zip(slt, [1, 2, 2, 3, 3, 5], strict=False))


def test_remove() -> None:
    slt = SortedList[int]()

    assert slt.discard(0) is None
    assert len(slt) == 0
    check_sorted_list(slt)

    slt = SortedList([1, 2, 2, 2, 3, 3, 5])
    slt.reset(4)

    slt.remove(2)
    check_sorted_list(slt)

    assert all(tup[0] == tup[1] for tup in zip(slt, [1, 2, 2, 3, 3, 5], strict=False))


def test_remove_valueerror1() -> None:
    slt = SortedList[int]()
    with pytest.raises(ValueError):
        slt.remove(0)


def test_remove_valueerror2() -> None:
    slt = SortedList(range(100))
    slt.reset(10)
    with pytest.raises(ValueError):
        slt.remove(100)


def test_remove_valueerror3() -> None:
    slt = SortedList([1, 2, 2, 2, 3, 3, 5])
    with pytest.raises(ValueError):
        slt.remove(4)


def test_delete() -> None:
    slt = SortedList(range(20))
    slt.reset(4)
    check_sorted_list(slt)
    for val in range(20):
        slt.remove(val)
        check_sorted_list(slt)
    assert_sorted_list_empty(slt)


def test_getitem() -> None:
    random.seed(0)
    slt = SortedList[float]()
    slt.reset(17)

    lst: list[float] = []

    for _rpt in range(100):
        val = random.random()
        slt.add(val)
        lst.append(val)

    lst.sort()

    assert all(slt[idx] == lst[idx] for idx in range(100))
    assert all(slt[idx - 99] == lst[idx - 99] for idx in range(100))


def test_getitem_slice() -> None:
    random.seed(0)
    slt = SortedList[float]()
    slt.reset(17)

    lst: list[float] = []

    for _rpt in range(100):
        val = random.random()
        slt.add(val)
        lst.append(val)

    lst.sort()

    assert all(slt[start:] == lst[start:] for start in [-75, -25, 0, 25, 75])

    assert all(slt[:stop] == lst[:stop] for stop in [-75, -25, 0, 25, 75])

    assert all(slt[::step] == lst[::step] for step in [-5, -1, 1, 5])

    assert all(
        slt[start:stop] == lst[start:stop]
        for start in [-75, -25, 0, 25, 75]
        for stop in [-75, -25, 0, 25, 75]
    )

    assert all(
        slt[:stop:step] == lst[:stop:step]
        for stop in [-75, -25, 0, 25, 75]
        for step in [-5, -1, 1, 5]
    )

    assert all(
        slt[start::step] == lst[start::step]
        for start in [-75, -25, 0, 25, 75]
        for step in [-5, -1, 1, 5]
    )

    assert all(
        slt[start:stop:step] == lst[start:stop:step]
        for start in [-75, -25, 0, 25, 75]
        for stop in [-75, -25, 0, 25, 75]
        for step in [-5, -1, 1, 5]
    )


def test_getitem_slice_big() -> None:
    slt = SortedList(range(4))
    lst = list(range(4))

    itr = (
        (start, stop, step)
        for start in [-6, -4, -2, 0, 2, 4, 6]
        for stop in [-6, -4, -2, 0, 2, 4, 6]
        for step in [-3, -2, -1, 1, 2, 3]
    )

    for start, stop, step in itr:
        assert slt[start:stop:step] == lst[start:stop:step]


def test_getitem_slicezero() -> None:
    slt = SortedList(range(100))
    slt.reset(17)
    with pytest.raises(ValueError):
        slt[::0]


def test_getitem_indexerror1() -> None:
    slt = SortedList[int]()
    with pytest.raises(IndexError):
        slt[5]


def test_getitem_indexerror2() -> None:
    slt = SortedList(range(100))
    with pytest.raises(IndexError):
        slt[200]


def test_getitem_indexerror3() -> None:
    slt = SortedList(range(100))
    with pytest.raises(IndexError):
        slt[-101]


def test_delitem() -> None:
    random.seed(0)

    slt = SortedList(range(100))
    slt.reset(17)
    while len(slt) > 0:
        pos = random.randrange(len(slt))
        del slt[pos]
        check_sorted_list(slt)

    slt = SortedList(range(100))
    slt.reset(17)
    del slt[:]
    assert len(slt) == 0
    check_sorted_list(slt)


def test_delitem_slice() -> None:
    slt = SortedList(range(100))
    slt.reset(17)
    del slt[10:40:1]
    del slt[10:40:-1]
    del slt[10:40:2]
    del slt[10:40:-2]


def test_iter() -> None:
    slt = SortedList(range(100))
    itr = iter(slt)
    assert all(tup[0] == tup[1] for tup in zip(range(100), itr, strict=False))


def test_reversed() -> None:
    slt = SortedList(range(100))
    rev = reversed(slt)
    assert all(tup[0] == tup[1] for tup in zip(range(99, -1, -1), rev, strict=False))


def test_reverse() -> None:
    slt = SortedList(range(100))
    with pytest.raises(NotImplementedError):
        slt.reverse()


def test_islice() -> None:
    sl = SortedList[int]()
    sl.reset(7)

    assert list(sl.islice()) == []

    values = list(range(53))
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


def test_irange() -> None:  # ruff:ignore[complex-structure]
    sl = SortedList[int]()
    sl.reset(7)

    assert list(sl.irange()) == []

    values = list(range(53))
    sl.update(values)

    for start in range(53):
        for end in range(start, 53):
            assert list(sl.irange(start, end)) == values[start : (end + 1)]
            assert (
                list(sl.irange(start, end, reverse=True))
                == values[start : (end + 1)][::-1]
            )

    for start in range(53):
        for end in range(start, 53):
            assert list(range(start, end)) == list(sl.irange(start, end, (True, False)))

    for start in range(53):
        for end in range(start, 53):
            assert list(range(start + 1, end + 1)) == list(
                sl.irange(start, end, (False, True))
            )

    for start in range(53):
        for end in range(start, 53):
            assert list(range(start + 1, end)) == list(
                sl.irange(start, end, (False, False))
            )

    for start in range(53):
        assert list(range(start, 53)) == list(sl.irange(start))

    for end in range(53):
        assert list(range(end)) == list(sl.irange(None, end, (True, False)))

    assert values == list(sl.irange(inclusive=(False, False)))

    assert list(sl.irange(53)) == []
    assert values == list(sl.irange(None, 53, (True, False)))


def test_len() -> None:
    slt = SortedList[int]()

    for val in range(100):
        slt.add(val)
        assert len(slt) == (val + 1)


def test_bisect_left() -> None:
    slt = SortedList[int]()
    assert slt.bisect_left(0) == 0
    slt = SortedList(range(100))
    slt.reset(17)
    slt.update(range(100))
    check_sorted_list(slt)
    assert slt.bisect_left(50) == 100
    assert slt.bisect_left(200) == 200


def test_bisect_right() -> None:
    slt = SortedList[int]()
    assert slt.bisect_right(10) == 0
    slt = SortedList(range(100))
    slt.reset(17)
    slt.update(range(100))
    check_sorted_list(slt)
    assert slt.bisect_right(10) == 22
    assert slt.bisect_right(200) == 200


def test_copy() -> None:
    alpha = SortedList(range(100))
    alpha.reset(7)
    beta = alpha.copy()
    alpha.add(100)
    assert len(alpha) == 101
    assert len(beta) == 100


def test_copy_copy() -> None:
    import copy

    alpha = SortedList(range(100))
    alpha.reset(7)
    beta = copy.copy(alpha)
    alpha.add(100)
    assert len(alpha) == 101
    assert len(beta) == 100


def test_count() -> None:
    slt = SortedList[int]()
    slt.reset(7)

    assert slt.count(0) == 0

    for iii in range(100):
        for _jjj in range(iii):
            slt.add(iii)
        check_sorted_list(slt)

    for iii in range(100):
        assert slt.count(iii) == iii

    assert slt.count(100) == 0


def test_pop() -> None:
    slt = SortedList(range(10))
    slt.reset(4)
    check_sorted_list(slt)
    assert slt.pop() == 9
    check_sorted_list(slt)
    assert slt.pop(0) == 0
    check_sorted_list(slt)
    assert slt.pop(-2) == 7
    check_sorted_list(slt)
    assert slt.pop(4) == 5
    check_sorted_list(slt)


def test_pop_indexerror1() -> None:
    slt = SortedList(range(10))
    slt.reset(4)
    with pytest.raises(IndexError):
        _ = slt.pop(-11)


def test_pop_indexerror2() -> None:
    slt = SortedList(range(10))
    slt.reset(4)
    with pytest.raises(IndexError):
        _ = slt.pop(10)


def test_pop_indexerror3() -> None:
    slt = SortedList[int]()
    with pytest.raises(IndexError):
        _ = slt.pop()


def test_index() -> None:
    slt = SortedList(range(100))
    slt.reset(17)

    for val in range(100):
        assert val == slt.index(val)

    assert slt.index(99, 0, 1000) == 99

    slt = SortedList(0 for _ in range(100))
    slt.reset(17)

    for start in range(100):
        for stop in range(start, 100):
            assert slt.index(0, start, stop + 1) == start

    for start in range(100):
        assert slt.index(0, -(100 - start)) == start

    assert slt.index(0, -1000) == 0


def test_index_valueerror1() -> None:
    slt = SortedList([0] * 10)
    slt.reset(4)
    with pytest.raises(ValueError):
        _ = slt.index(0, 10)


def test_index_valueerror2() -> None:
    slt = SortedList([0] * 10)
    slt.reset(4)
    with pytest.raises(ValueError):
        _ = slt.index(0, 0, -10)


def test_index_valueerror3() -> None:
    slt = SortedList([0] * 10)
    slt.reset(4)
    with pytest.raises(ValueError):
        _ = slt.index(0, 7, 3)


def test_index_valueerror4() -> None:
    slt = SortedList([0] * 10)
    slt.reset(4)
    with pytest.raises(ValueError):
        _ = slt.index(1)


def test_index_valueerror5() -> None:
    slt = SortedList[int]()
    with pytest.raises(ValueError):
        _ = slt.index(1)


def test_index_valueerror6() -> None:
    slt = SortedList(range(10))
    slt.reset(4)
    with pytest.raises(ValueError):
        _ = slt.index(3, 5)


def test_index_valueerror7() -> None:
    slt = SortedList([0] * 10 + [2] * 10)
    slt.reset(4)
    with pytest.raises(ValueError):
        _ = slt.index(1, 0, 10)


def test_mul() -> None:
    this = SortedList(range(10))
    this.reset(4)
    that = this * 5
    check_sorted_list(this)
    check_sorted_list(that)
    assert this == list(range(10))
    assert that == sorted(list(range(10)) * 5)
    assert this != that


def test_imul() -> None:
    this = SortedList(range(10))
    this.reset(4)
    this *= 5
    check_sorted_list(this)
    assert this == sorted(list(range(10)) * 5)


def test_op_add() -> None:
    this = SortedList(range(10))
    this.reset(4)
    assert (this + this + this) == (this * 3)

    that = SortedList(range(10))
    that.reset(4)
    that += that
    that += that
    assert that == (this * 4)


def test_eq() -> None:
    this = SortedList(range(10))
    this.reset(4)
    assert this == list(range(10))
    assert this == tuple(range(10))
    assert this != list(range(9))


def test_ne() -> None:
    this = SortedList(range(10))
    this.reset(4)
    assert this != list(range(9))
    assert this != tuple(range(11))
    assert this != [0, 1, 2, 3, 3, 5, 6, 7, 8, 9]
    assert this != (val for val in range(10))
    assert this != set()


def test_lt() -> None:
    this = SortedList(range(10, 15))
    this.reset(4)
    assert this < [10, 11, 13, 13, 14]
    assert this < [10, 11, 12, 13, 14, 15]
    assert this < [11]


def test_le() -> None:
    this = SortedList(range(10, 15))
    this.reset(4)
    assert this <= [10, 11, 12, 13, 14]
    assert this <= [10, 11, 12, 13, 14, 15]
    assert this <= [10, 11, 13, 13, 14]
    assert this <= [11]


def test_gt() -> None:
    this = SortedList(range(10, 15))
    this.reset(4)
    assert this > [10, 11, 11, 13, 14]
    assert this > [10, 11, 12, 13]
    assert this > [9]


def test_ge() -> None:
    this = SortedList(range(10, 15))
    this.reset(4)
    assert this >= [10, 11, 12, 13, 14]
    assert this >= [10, 11, 12, 13]
    assert this >= [10, 11, 11, 13, 14]
    assert this >= [9]


def test_repr() -> None:
    this = SortedList(range(10))
    this.reset(4)
    assert repr(this) == "SortedList([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])"


@pytest.mark.skip(
    reason="This precise situation of this test isn't possible anymore. Should modify to test another possible recursive case (if any)."
)
def test_repr_recursion() -> None:
    this = SortedList([[1], [2], [3], [4]])
    # pyrefly: ignore [bad-argument-type]
    this.inner.lists[-1].append(this)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
    assert repr(this) == "SortedList([[1], [2], [3], [4], ...])"


@pytest.mark.skip(reason="We don't support subclassing of concrete pyochain types ATM")
def test_repr_subclass() -> None:
    class CustomSortedList[T: SupportsRichComparison](SortedList[T]):
        pass

    this = CustomSortedList([1, 2, 3, 4])
    assert repr(this) == "CustomSortedList([1, 2, 3, 4])"


@pytest.mark.skip(reason="Pyo3 doesn't support pickling yet")
def test_pickle() -> None:
    import pickle

    alpha = SortedList(range(100))
    alpha.reset(500)
    beta: SortedList[int] = pickle.loads(pickle.dumps(alpha))  # pyright: ignore[reportAny]
    assert alpha == beta
    assert alpha.load == 500
    assert beta.load == 1000


@pytest.mark.skip(
    reason="We don't expose build_index() to python anymore, so this test is not relevant. If there's an issue, it should be caught by other tests."
)
def test_build_index() -> None:
    slt = SortedList([0])
    slt.reset(4)
    # slt.inner.build_index()  # ruff: ignore[commented-out-code]
    check_sorted_list(slt)
