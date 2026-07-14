"""
Tests common to Seq and Vec
"""

from __future__ import annotations

import pickle
import sys
from itertools import chain
from typing import TYPE_CHECKING, Literal, Self, override

import pytest

from pyochain import Seq, Vec

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from _typeshed import SupportsGetItem

type TestedSeq[T] = type[Seq[T] | Vec[T]]


TEST_TYPES = pytest.mark.parametrize("type2test", [Seq, Vec])


class _AlwaysEq:  # ruff:ignore[eq-without-hash]
    """
    Object that is equal to anything.
    """

    @override
    def __eq__(self, other: object) -> Literal[True]:
        return True

    @override
    def __ne__(self, other: object) -> Literal[False]:
        return False


ALWAYS_EQ = _AlwaysEq()


class _NeverEq:
    """
    Object that is not equal to anything.
    """

    @override
    def __eq__(self, other: object) -> Literal[False]:
        return False

    @override
    def __ne__(self, other: object) -> Literal[True]:
        return True

    @override
    def __hash__(self) -> Literal[1]:
        return 1


NEVER_EQ = _NeverEq()


# Various iterables
# This is used for checking the constructor (here and in test_deque.py)
def iterfunc[T](seqn: Iterable[T]) -> Iterator[T]:
    "Regular generator"
    yield from seqn


class SequenceTest[int, T]:
    "Sequence using __getitem__"

    def __init__(self, seqn: SupportsGetItem[int, T]) -> None:
        self.seqn: SupportsGetItem[int, T] = seqn

    def __getitem__(self, i: int) -> T:
        return self.seqn[i]


class IterFunc[T]:
    "Sequence using iterator protocol"

    def __init__(self, seqn: Sequence[T]) -> None:
        self.seqn: Sequence[T] = seqn
        self.i: int = 0

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        if self.i >= len(self.seqn):
            raise StopIteration
        v = self.seqn[self.i]
        self.i += 1
        return v


class IterGen[T]:
    "Sequence using iterator protocol defined with a generator"

    def __init__(self, seqn: Iterable[T]) -> None:
        self.seqn: Iterable[T] = seqn
        self.i: int = 0

    def __iter__(self) -> Iterator[T]:
        yield from self.seqn


class IterNextOnly[T]:
    "Missing __getitem__ and __iter__"

    def __init__(self, seqn: Sequence[T]) -> None:
        self.seqn: Sequence[T] = seqn
        self.i: int = 0

    def __next__(self) -> T:
        if self.i >= len(self.seqn):
            raise StopIteration
        v = self.seqn[self.i]
        self.i += 1
        return v


class IterNoNext[T]:
    "Iterator missing __next__()"

    def __init__(self, seqn: Sequence[T]) -> None:
        self.seqn: Sequence[T] = seqn
        self.i: int = 0

    def __iter__(self) -> Self:
        return self


class IterGenExc:
    "Test propagation of exceptions"

    def __init__(self, seqn: object) -> None:
        self.seqn: object = seqn
        self.i: int = 0

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> None:
        _ = 3 // 0


class IterFuncStop[T]:
    "Test immediate stop"

    def __init__(self, seqn: Sequence[T]) -> None:
        pass

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        raise StopIteration


def itermulti[T](seqn: Sequence[T]) -> chain[T]:
    "Test multiple tiers of iterators"

    def identity(x: T) -> T:
        return x

    return chain(map(identity, iterfunc(IterGen[T](SequenceTest(seqn)))))  # pyright: ignore[reportArgumentType]


@TEST_TYPES
def test_constructors(type2test: TestedSeq[object]) -> None:
    l0: list[object] = []
    l1 = [0]
    l2 = [0, 1]

    u = type2test(())
    u0 = type2test(l0)
    u1 = type2test(l1)
    u2 = type2test(l2)

    _ = type2test(u)
    _ = type2test(u0)
    _ = type2test(u1)
    _ = type2test(u2)

    _ = type2test(Seq(u))

    class OtherSeq[T]:
        def __init__(self, initseq: Sequence[T]) -> None:
            self.__data: Sequence[T] = initseq

        def __len__(self) -> int:
            return len(self.__data)

        def __getitem__(self, i: int) -> object:
            return self.__data[i]

    s = OtherSeq(u0)
    v0 = type2test(s)  # pyright: ignore[reportArgumentType]
    assert len(v0) == len(s)

    s = "this is also a sequence"
    vv = type2test(s)
    assert len(vv) == len(s)

    # Create from various iterables
    for s in ("123", "", range(1000), ("do", 1.2), range(2000, 2200, 5)):
        for g in (SequenceTest, IterFunc, IterGen, itermulti, iterfunc):
            assert type2test(g(s)) == type2test(s)  # pyright: ignore[reportArgumentType]
        assert type2test(IterFuncStop(s)) == type2test(())
        assert type2test(c for c in "123") == type2test("123")
        with pytest.raises(TypeError):
            _ = type2test(IterNextOnly(s))  # pyright: ignore[reportArgumentType]
        with pytest.raises(TypeError):
            _ = type2test(IterNoNext(s))  # pyright: ignore[reportArgumentType]
        with pytest.raises(ZeroDivisionError):
            _ = type2test(IterGenExc(s))

    with pytest.raises(TypeError):
        _ = type2test(unsupported_arg=[])  # pyright: ignore[reportCallIssue]


@TEST_TYPES
def test_truth(type2test: TestedSeq[object]) -> None:
    assert not type2test(())
    assert type2test([42])


@TEST_TYPES
def test_getitem(type2test: TestedSeq[object]) -> None:
    u = type2test([0, 1, 2, 3, 4])
    for i in range(len(u)):
        assert u[i] == i
        assert u[int(i)] == i
    for i in range(-len(u), -1):
        assert u[i] == len(u) + i
        assert u[int(i)] == len(u) + i
    with pytest.raises(IndexError):
        _ = u[-len(u) - 1]
    with pytest.raises(IndexError):
        _ = u[len(u)]
    with pytest.raises(ValueError):
        _ = u[slice(0, 10, 0)]

    u = type2test(())
    with pytest.raises(IndexError):
        _ = u[0]
    with pytest.raises(IndexError):
        _ = u[-1]

    with pytest.raises(TypeError):
        _ = u.__getitem__()  # pyright: ignore[reportCallIssue, reportUnknownVariableType]

    a = type2test([10, 11])
    assert a[0] == 10
    assert a[1] == 11
    assert a[-2] == 10
    assert a[-1] == 11
    with pytest.raises(IndexError):
        _ = a.__getitem__(-3)
    with pytest.raises(IndexError):
        _ = a.__getitem__(3)


@TEST_TYPES
def test_getslice(type2test: TestedSeq[object]) -> None:
    x = [0, 1, 2, 3, 4]
    u = type2test(x)

    assert u[0:0] == type2test(())
    assert u[1:2] == type2test([1])
    assert u[-2:-1] == type2test([3])
    assert u[-1000:1000] == u
    assert u[1000:-1000] == type2test([])
    assert u[:] == u
    assert u[1:None] == type2test([1, 2, 3, 4])
    assert u[None:3] == type2test([0, 1, 2])

    # Extended slices
    assert u[:] == u
    assert u[::2] == type2test([0, 2, 4])
    assert u[1::2] == type2test([1, 3])
    assert u[::-1] == type2test([4, 3, 2, 1, 0])
    assert u[::-2] == type2test([4, 2, 0])
    assert u[3::-2] == type2test([3, 1])
    assert u[3:3:-2] == type2test([])
    assert u[3:2:-2] == type2test([3])
    assert u[3:1:-2] == type2test([3])
    assert u[3:0:-2] == type2test([3, 1])
    assert u[::-100] == type2test([4])
    assert u[100:-100] == type2test([])
    assert u[-100:100] == u
    assert u[100:-100:-1] == u[::-1]
    assert u[-100:100:-1] == type2test([])
    assert u[-100:100:2] == type2test([0, 2, 4])

    # Test extreme cases with long ints
    a = type2test([0, 1, 2, 3, 4])
    assert a[-pow(2, 128) : 3] == type2test([0, 1, 2])
    assert a[3 : pow(2, 145)] == type2test([3, 4])
    assert a[3 :: sys.maxsize] == type2test([3])


@TEST_TYPES
def test_contains(type2test: TestedSeq[int]) -> None:
    u = type2test([0, 1, 2])
    for i in u:
        assert i in u
    for i in min(u) - 1, max(u) + 1:
        assert i not in u

    with pytest.raises(TypeError):
        _ = u.__contains__()  # pyright: ignore[reportCallIssue, reportUnknownVariableType]


@TEST_TYPES
def test_contains_fake(type2test: TestedSeq[object]) -> None:
    # Sequences must use rich comparison against each item
    # (unless "is" is true, or an earlier item answered)
    # So ALWAYS_EQ must be found in all non-empty sequences.
    assert ALWAYS_EQ not in type2test([])
    assert ALWAYS_EQ in type2test([1])
    assert 1 in type2test([ALWAYS_EQ])
    assert NEVER_EQ not in type2test([])
    assert ALWAYS_EQ not in type2test([NEVER_EQ])
    assert NEVER_EQ in type2test([ALWAYS_EQ])


@TEST_TYPES
def test_contains_order(type2test: TestedSeq[object]) -> None:
    # Sequences must test in-order.  If a rich comparison has side
    # effects, these will be visible to tests against later members.
    # In this test, the "side effect" is a short-circuiting raise.
    class DoNotTestEqError(Exception):
        pass

    class StopCompares:  # ruff:ignore[eq-without-hash]
        @override
        def __eq__(self, other: object) -> bool:
            raise DoNotTestEqError

    checkfirst = type2test([1, StopCompares()])
    assert 1 in checkfirst
    checklast = type2test([StopCompares(), 1])

    with pytest.raises(DoNotTestEqError):
        _ = checklast.__contains__(1)


@TEST_TYPES
def test_len(type2test: TestedSeq[object]) -> None:
    assert len(type2test(())) == 0
    assert len(type2test([])) == 0
    assert len(type2test([0])) == 1
    assert len(type2test([0, 1, 2])) == 3


@TEST_TYPES
def test_minmax(type2test: TestedSeq[int]) -> None:
    u = type2test([0, 1, 2])
    assert min(u) == 0
    assert max(u) == 2


@TEST_TYPES
def test_add[T](type2test: TestedSeq[int]) -> None:
    u1 = type2test([0])
    u2 = type2test([0, 1])
    assert u1 == u1 + type2test(())  # pyright: ignore[reportOperatorIssue]
    assert u1 == type2test(()) + u1  # pyright: ignore[reportOperatorIssue]
    assert u1 + type2test([1]) == u2  # pyright: ignore[reportOperatorIssue]
    assert type2test([-1]) + u1 == type2test([-1, 0])  # pyright: ignore[reportOperatorIssue]


@TEST_TYPES
def test_mul(type2test: TestedSeq[int]) -> None:
    u2 = type2test([0, 1])
    assert type2test(()) == u2 * 0
    assert type2test(()) == 0 * u2
    assert u2 == u2 * 1
    assert u2 == 1 * u2
    assert u2 + u2 == u2 * 2  # pyright: ignore[reportOperatorIssue]
    assert u2 + u2 == 2 * u2  # pyright: ignore[reportOperatorIssue]
    assert u2 + u2 + u2 == u2 * 3  # pyright: ignore[reportOperatorIssue]
    assert u2 + u2 + u2 == 3 * u2  # pyright: ignore[reportOperatorIssue]


@TEST_TYPES
def test_iadd(type2test: TestedSeq[object]) -> None:
    u = type2test([0, 1])
    u += type2test(())  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    assert u == type2test([0, 1])
    u += type2test([2, 3])  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    assert u == type2test([0, 1, 2, 3])
    u += type2test([4, 5])  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    assert u == type2test([0, 1, 2, 3, 4, 5])

    u = type2test("spam")
    u += type2test("eggs")  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    assert u == type2test("spameggs")


@TEST_TYPES
def test_imul(type2test: TestedSeq[int]) -> None:
    u = type2test([0, 1])
    u *= 3
    assert u == type2test([0, 1, 0, 1, 0, 1])
    u *= 0
    assert u == type2test([])


@TEST_TYPES
def test_repeat(type2test: TestedSeq[int]) -> None:
    for m in range(4):
        s = Seq(range(m))
        for n in range(-3, 5):
            assert type2test(s * n) == type2test(s) * n
        assert type2test(s) * -4 == type2test([])
        assert id(s) == id(s * 1)


@TEST_TYPES
def test_bigrepeat(type2test: TestedSeq[int]) -> None:
    if sys.maxsize <= 2147483647:
        x = type2test([0])
        x *= 2**16
        with pytest.raises(MemoryError):
            _ = x * 2**16
        if isinstance(x, Vec):
            with pytest.raises(MemoryError):
                _ = x.__imul__(2**16)


@TEST_TYPES
def test_subscript(type2test: TestedSeq[int]) -> None:
    a = type2test([10, 11])
    assert a.__getitem__(0) == 10
    assert a.__getitem__(1) == 11
    assert a.__getitem__(-2) == 10
    assert a.__getitem__(-1) == 11
    with pytest.raises(IndexError):
        _ = a.__getitem__(-3)
    with pytest.raises(IndexError):
        _ = a.__getitem__(3)
    assert a.__getitem__(slice(0, 1)) == type2test([10])
    assert a.__getitem__(slice(1, 2)) == type2test([11])
    assert a.__getitem__(slice(0, 2)) == type2test([10, 11])
    assert a.__getitem__(slice(0, 3)) == type2test([10, 11])
    assert a.__getitem__(slice(3, 5)) == type2test([])
    with pytest.raises(ValueError):
        _ = a.__getitem__(slice(0, 10, 0))
    with pytest.raises(TypeError):
        _ = a.__getitem__("x")  # pyright: ignore[reportCallIssue, reportArgumentType, reportUnknownVariableType]


@TEST_TYPES
def test_cmp(type2test: TestedSeq[int]) -> None:
    a = type2test([0, 1])
    _assert_cmp(a, a, 0)
    _assert_cmp(a, type2test([0, 1]), 0)
    _assert_cmp(a, type2test([0]), 1)
    _assert_cmp(a, type2test([0, 2]), -1)


def _assert_cmp[T: Seq[int] | Vec[int]](a: T, b: T, r: int) -> None:
    assert (a == b) is (r == 0)
    assert (a != b) is (r != 0)
    assert (a > b) is (r > 0)  # pyright: ignore[reportOperatorIssue]
    assert (a <= b) is (r <= 0)  # pyright: ignore[reportOperatorIssue]
    assert (a < b) is (r < 0)  # pyright: ignore[reportOperatorIssue]
    assert (a >= b) is (r >= 0)  # pyright: ignore[reportOperatorIssue]


@TEST_TYPES
def test_count(type2test: TestedSeq[object]) -> None:
    a = type2test([0, 1, 2]) * 3
    assert a.count(0) == 3
    assert a.count(1) == 3
    assert a.count(3) == 0

    assert a.count(ALWAYS_EQ) == 9
    assert type2test([ALWAYS_EQ, ALWAYS_EQ]).count(1) == 2
    assert type2test([ALWAYS_EQ, ALWAYS_EQ]).count(NEVER_EQ) == 2
    assert type2test([NEVER_EQ, NEVER_EQ]).count(ALWAYS_EQ) == 0
    with pytest.raises(TypeError):
        _ = a.count()  # pyright: ignore[reportCallIssue, reportUnknownVariableType]

    class BadExcError(Exception):
        pass

    class BadCmp:  # ruff:ignore[eq-without-hash]
        @override
        def __eq__(self, other: object) -> Literal[False]:
            if other == 2:
                raise BadExcError
            return False

    with pytest.raises(BadExcError):
        _ = a.count(BadCmp())


@TEST_TYPES
def test_index(type2test: TestedSeq[object]) -> None:
    u = type2test([0, 1])
    assert u.index(0) == 0
    assert u.index(1) == 1
    with pytest.raises(ValueError):
        _ = u.index(2)

    u = type2test([-2, -1, 0, 0, 1, 2])
    assert u.count(0) == 2
    assert u.index(0) == 2
    assert u.index(0, 2) == 2
    assert u.index(-2, -10) == 0
    assert u.index(0, 3) == 3
    assert u.index(0, 3, 4) == 3
    with pytest.raises(ValueError):
        _ = u.index(2, 0, -10)

    assert u.index(ALWAYS_EQ) == 0
    assert type2test([ALWAYS_EQ, ALWAYS_EQ]).index(1) == 0
    assert type2test([ALWAYS_EQ, ALWAYS_EQ]).index(NEVER_EQ) == 0
    with pytest.raises(ValueError):
        _ = type2test([NEVER_EQ, NEVER_EQ]).index(ALWAYS_EQ)

    with pytest.raises(TypeError):
        _ = u.index()  # pyright: ignore[reportCallIssue, reportUnknownVariableType]

    class BadExcError(Exception):
        pass

    class BadCmp:  # ruff:ignore[eq-without-hash]
        @override
        def __eq__(self, other: object) -> Literal[False]:
            if other == 2:
                raise BadExcError
            return False

    a = type2test([0, 1, 2, 3])
    with pytest.raises(BadExcError):
        _ = a.index(BadCmp())

    a = type2test([-2, -1, 0, 0, 1, 2])
    assert a.index(0) == 2
    assert a.index(0, 2) == 2
    assert a.index(0, -4) == 2
    assert a.index(-2, -10) == 0
    assert a.index(0, 3) == 3
    assert a.index(0, -3) == 3
    assert a.index(0, 3, 4) == 3
    assert a.index(0, -3, -2) == 3
    assert a.index(0, -4 * sys.maxsize, 4 * sys.maxsize) == 2
    with pytest.raises(ValueError):
        _ = a.index(0, 4 * sys.maxsize, -4 * sys.maxsize)
    with pytest.raises(ValueError):
        _ = a.index(2, 0, -10)


@pytest.mark.skip(reason="Pickle support is not implemented yet for Pyo3")
@TEST_TYPES
def test_pickle(type2test: TestedSeq[object]) -> None:
    lst = type2test([4, 5, 6, 7])
    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        lst2 = pickle.loads(pickle.dumps(lst, proto))  # pyright: ignore[reportAny]
        assert lst2 == lst
        assert id(lst2) != id(lst)  # pyright: ignore[reportAny]
