"""Adapted from `sortedcontainers` test suite.

Copyright 2014-2024 Grant Jenks — Licensed under the Apache License 2.0

Original source:
https://github.com/grantjenks/python-sortedcontainers/blob/master/tests/test_coverage_sortedset.py
"""

from __future__ import annotations

import operator
from functools import partial
from typing import override

import pytest

from pyochain import Range, Seq
from pyochain.collections import SortedKeySet, SortedSet
from pyochain.collections._sorted import (  # ruff: ignore[import-private-name]
    check_sorted_set,
)

_neg_set = partial(SortedKeySet[int, int], operator.neg)


def modulo(value: int) -> int:
    return value % 10


def test_init() -> None:
    temp = SortedSet(range(100))
    temp.reset(7)
    check_sorted_set(temp)
    assert all(val == temp[val] for val in temp)


def test_init_key() -> None:
    temp = _neg_set(range(100))
    assert temp.key == operator.neg


def test_contains() -> None:
    temp = SortedSet(range(100))
    temp.reset(7)
    assert all(val in temp for val in range(100))
    assert all(val not in temp for val in range(100, 200))


def test_getitem() -> None:
    temp = SortedSet(range(100))
    temp.reset(7)
    assert all(val == temp[val] for val in temp)


def test_getitem_slice() -> None:
    vals = list(range(100))
    temp = SortedSet(vals)
    temp.reset(7)
    assert temp[20:30] == vals[20:30]


def test_getitem_key() -> None:
    temp = _neg_set(range(100))
    temp.reset(7)
    assert all(temp[val] == (99 - val) for val in range(100))


def test_delitem() -> None:
    temp = SortedSet(range(100))
    temp.reset(7)
    for val in reversed(range(50)):
        del temp[val]
    assert all(temp[pos] == (pos + 50) for pos in range(50))


def test_delitem_slice() -> None:
    vals = list(range(100))
    temp = SortedSet(vals)
    temp.reset(7)
    del vals[20:40:2]
    del temp[20:40:2]
    assert temp == set(vals)


def test_delitem_key() -> None:
    temp = SortedKeySet(modulo, range(100))
    temp.reset(7)
    values = sorted(range(100), key=modulo)
    for val in range(10):
        del temp[val]
        del values[val]
    assert list(temp) == list(values)


def test_eq() -> None:
    alpha = SortedSet(range(100))
    alpha.reset(7)
    beta = SortedSet(range(100))
    beta.reset(17)
    assert alpha == beta
    assert alpha == beta.set
    beta.add(101)
    assert alpha != beta


def test_ne() -> None:
    alpha = SortedSet(range(100))
    alpha.reset(7)
    beta = SortedSet(range(99))
    beta.reset(17)
    assert alpha != beta
    beta.add(100)
    assert alpha != beta
    assert alpha != beta.set
    assert alpha != list(range(101))


def test_lt_gt() -> None:
    temp = SortedSet(range(100))
    temp.reset(7)
    that = SortedSet(range(25, 75))
    that.reset(9)
    assert that < temp
    assert not (temp < that)
    assert that < temp.set
    assert temp > that
    assert not (that > temp)
    assert temp > that.set


def test_le_ge() -> None:
    alpha = SortedSet(range(100))
    alpha.reset(7)
    beta = SortedSet(range(101))
    beta.reset(17)
    assert alpha <= beta
    assert not (beta <= alpha)
    assert alpha <= beta.set
    assert beta >= alpha
    assert not (alpha >= beta)
    assert beta >= alpha.set


def test_iter() -> None:
    temp = SortedSet(range(100))
    temp.reset(7)
    assert all(val == temp[val] for val in iter(temp))


def test_reversed() -> None:
    temp = SortedSet(range(100))
    temp.reset(7)
    assert all(val == temp[val] for val in reversed(temp))


def test_islice() -> None:
    ss = SortedSet[int]()
    ss.reset(7)

    assert ss.islice().collect(Seq).is_empty()
    nb = 20

    values = list(range(nb))
    _ = ss.update(values)

    for start in range(nb):
        for stop in range(nb):
            assert ss.islice(start, stop).collect(list) == values[start:stop]

    for start in range(nb):
        for stop in range(nb):
            assert (
                ss.islice(start, stop, reverse=True).collect(list)
                == values[start:stop][::-1]
            )

    for start in range(nb):
        assert ss.islice(start=start).collect(list) == values[start:]
        assert (
            ss.islice(start=start, reverse=True).collect(list) == values[start:][::-1]
        )

    for stop in range(nb):
        assert ss.islice(stop=stop).collect(list) == values[:stop]
        assert ss.islice(stop=stop, reverse=True).collect(list) == values[:stop][::-1]


def test_irange() -> None:  # ruff:ignore[complex-structure]
    ss = SortedSet[int]()
    ss.reset(7)
    nb = 23

    assert list(ss.irange()) == []

    values = list(range(nb))
    _ = ss.update(values)

    for start in range(nb):
        for end in range(start, nb):
            assert ss.irange(start, end).collect(list) == values[start : (end + 1)]
            assert (
                ss.irange(start, end, reverse=True).collect(list)
                == values[start : (end + 1)][::-1]
            )
    for start in range(nb):
        for end in range(start, nb):
            assert Range(start, end).pipe(list) == ss.irange(
                start, end, (True, False)
            ).collect(list)

    for start in range(nb):
        for end in range(start, nb):
            assert list(range(start + 1, end + 1)) == ss.irange(
                start, end, (False, True)
            ).collect(list)

    for start in range(nb):
        for end in range(start, nb):
            assert list(range(start + 1, end)) == ss.irange(
                start, end, (False, False)
            ).collect(list)

    for start in range(nb):
        assert list(range(start, nb)) == ss.irange(start).collect(list)

    for end in range(nb):
        assert list(range(end)) == ss.irange(None, end, (True, False)).collect(list)

    assert values == ss.irange(inclusive=(False, False)).collect(list)

    assert list(ss.irange(nb)) == []
    assert values == ss.irange(None, nb, (True, False)).collect(list)


def test_irange_key() -> None:  # ruff:ignore[complex-structure]
    values = Range(100).iter().sort_by(modulo)

    for load in range(5, 16):
        ss = SortedKeySet(modulo, range(100))
        ss.reset(load)

        for start in range(10):
            for end in range(start, 10):
                temp = list(ss.irange_key(start, end))
                assert temp == values[(start * 10) : ((end + 1) * 10)]

                temp = list(ss.irange_key(start, end, reverse=True))
                assert temp == values[(start * 10) : ((end + 1) * 10)][::-1]

        for start in range(10):
            for end in range(start, 10):
                temp = list(ss.irange_key(start, end, inclusive=(True, False)))
                assert temp == values[(start * 10) : (end * 10)]

        for start in range(10):
            for end in range(start, 10):
                temp = list(ss.irange_key(start, end, (False, True)))
                assert temp == values[((start + 1) * 10) : ((end + 1) * 10)]

        for start in range(10):
            for end in range(start, 10):
                temp = list(ss.irange_key(start, end, inclusive=(False, False)))
                assert temp == values[((start + 1) * 10) : (end * 10)]

        for start in range(10):
            temp = list(ss.irange_key(min_key=start))
            assert temp == values[(start * 10) :]

        for end in range(10):
            temp = list(ss.irange_key(max_key=end))
            assert temp == values[: (end + 1) * 10]


def test_len() -> None:
    temp = SortedSet(range(100))
    temp.reset(7)
    assert temp.len() == 100


def test_add() -> None:
    temp = SortedSet(range(100))
    temp.reset(7)
    temp.add(100)
    temp.add(90)
    check_sorted_set(temp)
    assert Range(101).iter().all(lambda val: val == temp[val])


def test_bisect() -> None:
    r = Range(20)
    temp = SortedSet(r)
    temp.reset(7)
    assert r.iter().all(lambda val: temp.bisect_left(val) == val)
    assert r.iter().all(lambda val: temp.bisect_right(val) == (val + 1))


def test_bisect_key() -> None:
    r = Range(20)
    temp = SortedKeySet(lambda val: val, r)
    temp.reset(7)
    assert r.iter().all(lambda val: temp.bisect_key_left(val) == val)
    assert r.iter().all(lambda val: temp.bisect_key_right(val) == (val + 1))


def test_clear() -> None:
    temp = SortedSet(range(100))
    temp.reset(7)
    temp.clear()
    check_sorted_set(temp)
    assert temp.len() == 0


def test_copy() -> None:
    temp = SortedSet(range(100))
    temp.reset(7)
    that = temp.copy()
    that.add(1000)
    assert temp.len() == 100
    assert that.len() == 101


def test_copy_copy() -> None:
    import copy

    temp = SortedSet(range(100))
    temp.reset(7)
    that = copy.copy(temp)
    that.add(1000)
    assert temp.len() == 100
    assert that.len() == 101


def test_count() -> None:
    temp = SortedSet(range(100))
    temp.reset(7)
    assert all(temp.count(val) == 1 for val in range(100))
    assert temp.count(100) == 0
    assert temp.count(0) == 1
    temp.add(0)
    assert temp.count(0) == 1
    check_sorted_set(temp)


def test_sub() -> None:
    temp = SortedSet(range(100))
    temp.reset(7)
    that = temp - range(10) - range(10, 20)
    assert all(val == temp[val] for val in range(100))
    assert all((val + 20) == that[val] for val in range(80))


def test_difference() -> None:
    temp = SortedSet(range(100))
    temp.reset(7)
    that = temp.difference(range(10), range(10, 20))
    assert all(val == temp[val] for val in range(100))
    assert all((val + 20) == that[val] for val in range(80))


def test_difference_update() -> None:
    temp = SortedSet(range(100))
    temp.reset(7)
    _ = temp.difference_update(range(10), range(10, 20))
    assert all((val + 20) == temp[val] for val in range(80))


def test_isub() -> None:
    temp = SortedSet(range(100))
    temp.reset(7)
    temp -= range(10)
    temp -= range(10, 20)
    assert all((val + 20) == temp[val] for val in range(80))


def test_discard() -> None:
    temp = SortedSet(range(100))
    temp.reset(7)
    for v in (0, 99, 50, 1000):
        temp.discard(v)
    check_sorted_set(temp)
    assert len(temp) == 97


def test_index() -> None:
    temp = SortedSet(range(100))
    temp.reset(7)
    assert all(temp.index(val) == val for val in range(100))


def test_and() -> None:
    temp = SortedSet(range(100))
    temp.reset(7)
    that = temp & range(20) & range(10, 30)
    assert all(that[val] == (val + 10) for val in range(10))
    assert all(temp[val] == val for val in range(100))


def test_intersection() -> None:
    temp = SortedSet(range(100))
    temp.reset(7)
    that = temp.intersection(range(20), range(10, 30))
    assert all(that[val] == (val + 10) for val in range(10))
    assert all(temp[val] == val for val in range(100))


def test_intersection_update() -> None:
    temp = SortedSet(range(100))
    temp.reset(7)
    temp &= range(20)
    temp &= range(10, 30)
    assert all(temp[val] == (val + 10) for val in range(10))


def test_isdisjoint() -> None:
    temp = SortedSet(range(100))
    temp.reset(7)
    that = SortedSet(range(100, 200))
    that.reset(9)
    assert temp.isdisjoint(that)


def test_is_subset() -> None:
    temp = SortedSet(range(100))
    temp.reset(7)
    that = SortedSet(range(25, 75))
    that.reset(9)
    assert that.is_subset(temp)


def test_is_superset() -> None:
    temp = SortedSet(range(100))
    temp.reset(7)
    that = SortedSet(range(25, 75))
    that.reset(9)
    assert temp.is_superset(that)


def test_xor() -> None:
    temp = SortedSet(range(75))
    temp.reset(7)
    that = SortedSet(range(25, 100))
    that.reset(9)
    result = temp ^ that
    assert all(result[val] == val for val in range(25))
    assert all(result[val + 25] == (val + 75) for val in range(25))
    assert all(temp[val] == val for val in range(75))
    assert all(that[val] == (val + 25) for val in range(75))


def test_symmetric_difference() -> None:
    temp = SortedSet(range(75))
    temp.reset(7)
    that = SortedSet(range(25, 100))
    that.reset(9)
    result = temp.symmetric_difference(that)
    assert all(result[val] == val for val in range(25))
    assert all(result[val + 25] == (val + 75) for val in range(25))
    assert all(temp[val] == val for val in range(75))
    assert all(that[val] == (val + 25) for val in range(75))


def test_symmetric_difference_update() -> None:
    temp = SortedSet(range(75))
    temp.reset(7)
    that = SortedSet(range(25, 100))
    that.reset(9)
    temp ^= that
    assert all(temp[val] == val for val in range(25))
    assert all(temp[val + 25] == (val + 75) for val in range(25))


def test_pop() -> None:
    temp = SortedSet(range(100))
    temp.reset(7)
    _ = temp.pop()
    _ = temp.pop(0)
    assert all(temp[val] == (val + 1) for val in range(98))


def test_remove() -> None:
    temp = SortedSet(range(100))
    temp.reset(7)
    temp.remove(50)


def test_or() -> None:
    temp = SortedSet(range(50))
    temp.reset(7)
    that = SortedSet(range(50, 100))
    that.reset(9)
    result = temp | that
    assert all(result[val] == val for val in range(100))
    assert all(temp[val] == val for val in range(50))
    assert all(that[val] == (val + 50) for val in range(50))


def test_union() -> None:
    temp = SortedSet(range(50))
    temp.reset(7)
    that = SortedSet(range(50, 100))
    that.reset(9)
    result = temp.union(that)
    assert all(result[val] == val for val in range(100))
    assert all(temp[val] == val for val in range(50))
    assert all(that[val] == (val + 50) for val in range(50))


def test_update() -> None:
    temp = SortedSet(range(80))
    temp.reset(7)
    _ = temp.update(range(80, 90), range(90, 100))
    assert all(temp[val] == val for val in range(100))


def test_ior() -> None:
    temp = SortedSet(range(80))
    temp.reset(7)
    temp |= range(80, 90)
    temp |= range(90, 100)
    assert all(temp[val] == val for val in range(100))


class Identity:
    def __call__[T](self, value: T) -> T:
        return value

    @override
    def __repr__(self) -> str:
        return "identity"


def test_repr() -> None:
    temp = SortedKeySet(Identity(), range(10))
    temp.reset(7)
    assert repr(temp) == "SortedKeySet([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], key=identity)"


@pytest.mark.skip(reason="Pickle not supported by Pyo3")
def test_pickle() -> None:
    import pickle

    alpha = _neg_set(range(100))
    alpha.reset(500)
    data = pickle.dumps(alpha)
    beta: SortedKeySet[int, int] = pickle.loads(data)  # pyright: ignore[reportAny]
    assert alpha == beta
    assert alpha.key == beta.key
