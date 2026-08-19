"""
Tests adapted from CPython `list_test.py` and `test_list.py` to test the `Vec` class.
"""

import contextlib
import pickle
import sys
from collections.abc import Iterator, MutableSequence
from functools import cmp_to_key
from typing import Self, override

import pytest

from pyochain import Vec

from .test_seq import ALWAYS_EQ, NEVER_EQ

_SKIP_INIT_TEST = pytest.mark.skip(
    reason="Pyo3 forces us to use `__new__`, making any subsequent `__init__` behavior inefficient."
)
"""Adding any overwrite/clear behavior in the `__init__` would force us to create a list, clear it, and then re-populate it, since we are forced to assign the attributes in `new`.\\
This is a niche behavior anyway, `clear` is what you *should* use."""


@_SKIP_INIT_TEST
def test_init_clear_previous_values() -> None:
    a = Vec([1, 2, 3])
    a.__init__(())  # pyright: ignore[reportCallIssue]
    assert a == Vec()


@_SKIP_INIT_TEST
def test_init_overwrite_previous_values() -> None:
    a = Vec([1, 2, 3])
    a.__init__([4, 5, 6])  # pyright: ignore[reportCallIssue]
    assert a == Vec([4, 5, 6])


def test_init_empty() -> None:
    assert Vec() == Vec()


def test_init_copy() -> None:
    a = Vec([1, 2, 3])
    # Mutables always return a new object
    b = Vec(a)
    assert id(a) != id(b)
    assert a == b


def test_getitem_error() -> None:
    a = Vec[str]([])
    msg = "list indices must be integers or slices"
    with pytest.raises(TypeError, match=msg):
        # pyrefly: ignore [bad-index]
        a["a"]  # pyright: ignore[reportCallIssue, reportArgumentType]


def test_setitem_error() -> None:
    a = Vec[str]([])
    msg = "list indices must be integers or slices"
    with pytest.raises(TypeError, match=msg):
        # pyrefly: ignore [unsupported-operation]
        a["a"] = "python"  # pyright: ignore[reportCallIssue, reportArgumentType]


@pytest.mark.skip(reason="We don't handle recursive repr yet")
def test_repr() -> None:
    l0: Vec[int] = Vec()
    l2 = Vec([0, 1, 2])
    a0 = Vec(l0)
    a2 = Vec(l2)

    assert str(a0) == str(l0)
    assert repr(a0) == repr(l0)
    assert repr(a2) == repr(l2)
    assert str(a2) == "Vec(0, 1, 2)"
    assert repr(a2) == "Vec(0, 1, 2)"

    # pyrefly: ignore [bad-argument-type]
    a2.append(a2)  # pyright: ignore[reportArgumentType]
    a2.append(3)
    assert str(a2) == "Vec(0, 1, 2, [...], 3)"
    assert repr(a2) == "Vec(0, 1, 2, [...], 3)"


def test_set_subscript() -> None:
    a = Vec(range(20))
    with pytest.raises(ValueError):
        a.__setitem__(slice(0, 10, 0), [1, 2, 3])
    with pytest.raises(TypeError):
        # pyrefly: ignore [no-matching-overload]
        a.__setitem__(slice(0, 10), 1)  # pyright: ignore[reportCallIssue, reportArgumentType]
    with pytest.raises(ValueError):
        a.__setitem__(slice(0, 10, 2), [1, 2])
    with pytest.raises(TypeError):
        # pyrefly: ignore [no-matching-overload]
        a.__getitem__("x", 1)  # pyright: ignore[reportCallIssue]
    a[slice(2, 10, 3)] = [1, 2, 3]
    assert a == Vec([
        0,
        1,
        1,
        3,
        4,
        2,
        6,
        7,
        3,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
    ])


def test_reversed() -> None:
    a = Vec(range(20))
    r = reversed(a)
    assert Vec(r) == Vec(range(19, -1, -1))
    with pytest.raises(StopIteration):
        _ = next(r)
    assert Vec(reversed(Vec[int](()))) == Vec[int](())
    # Bug 3689: make sure list-reversed-iterator doesn't have __len__
    with pytest.raises(TypeError):
        # pyrefly: ignore [bad-argument-type]
        _ = len(reversed([1, 2, 3]))  # pyright: ignore[reportArgumentType]


def test_setitem() -> None:
    a = Vec([0, 1])
    a[0] = 0
    a[1] = 100
    assert a == Vec([0, 100])
    a[-1] = 200
    assert a == Vec([0, 200])
    a[-2] = 100
    assert a == Vec([100, 200])
    with pytest.raises(IndexError):
        a.__setitem__(-3, 200)
    with pytest.raises(IndexError):
        a.__setitem__(2, 200)

    a = Vec[int]([])
    with pytest.raises(IndexError):
        a.__setitem__(0, 200)
    with pytest.raises(IndexError):
        a.__setitem__(-1, 200)
    with pytest.raises(TypeError):
        # pyrefly: ignore [no-matching-overload]
        a.__setitem__()  # pyright: ignore[reportCallIssue]

    a = Vec([0, 1, 2, 3, 4])
    a[0] = 1
    a[1] = 2
    a[2] = 3
    assert a == Vec([1, 2, 3, 3, 4])
    a[0] = 5
    a[1] = 6
    a[2] = 7
    assert a == Vec([5, 6, 7, 3, 4])
    a[-2] = 88
    a[-1] = 99
    assert a == Vec([5, 6, 7, 88, 99])
    a[-2] = 8
    a[-1] = 9
    assert a == Vec([5, 6, 7, 8, 9])

    msg = "list indices must be integers or slices"
    with pytest.raises(TypeError, match=msg):
        # pyrefly: ignore [unsupported-operation]
        a["a"] = "python"  # pyright: ignore[reportCallIssue, reportArgumentType]


def test_delitem() -> None:
    a = Vec([0, 1])
    del a[1]
    assert a == [0]
    del a[0]
    assert a == []

    a = Vec([0, 1])
    del a[-2]
    assert a == [1]
    del a[-1]
    assert a == []

    a = Vec([0, 1])
    with pytest.raises(IndexError):
        a.__delitem__(-3)
    with pytest.raises(IndexError):
        a.__delitem__(2)

    a = Vec[int]([])
    with pytest.raises(IndexError):
        a.__delitem__(0)

    with pytest.raises(TypeError):
        # pyrefly: ignore [missing-argument]
        a.__delitem__()  # pyright: ignore[reportCallIssue]


def test_setslice() -> None:
    x = [0, 1]
    a = Vec(x)

    for i in range(-3, 4):
        a[:i] = x[:i]
        assert a == x
        a2 = a[:]
        a2[:i] = a[:i]
        assert a2 == a
        a[i:] = x[i:]
        assert a == x
        a2 = a[:]
        a2[i:] = a[i:]
        assert a2 == a
        for j in range(-3, 4):
            a[i:j] = x[i:j]
            assert a == x
            a2 = a[:]
            a2[i:j] = a[i:j]
            assert a2 == a

    # pyrefly: ignore [unbound-name]
    aa2: Vec[int] = a2[:]  # pyright: ignore[reportPossiblyUnboundVariable, reportUnknownVariableType]
    aa2[:0] = [-2, -1]
    assert aa2 == [-2, -1, 0, 1]
    aa2[0:] = []
    assert aa2 == []

    a = Vec([1, 2, 3, 4, 5])
    a[:-1] = a
    assert a == Vec([1, 2, 3, 4, 5, 5])
    a = Vec([1, 2, 3, 4, 5])
    a[1:] = a
    assert a == Vec([1, 1, 2, 3, 4, 5])
    a = Vec([1, 2, 3, 4, 5])
    a[1:-1] = a
    assert a == Vec([1, 1, 2, 3, 4, 5, 5])

    a = Vec[int]([])
    a[:] = tuple(range(10))
    assert a == Vec(range(10))

    with pytest.raises(TypeError):
        # pyrefly: ignore [no-matching-overload]
        a.__setitem__(slice(0, 1, 5))  # pyright: ignore[reportCallIssue]

    with pytest.raises(TypeError):
        # pyrefly: ignore [no-matching-overload]
        a.__setitem__()  # pyright: ignore[reportCallIssue]


def test_slice_assign_iterator() -> None:
    x = Vec(range(5))
    x[0:3] = reversed(range(3))
    assert x == Vec([2, 1, 0, 3, 4])

    x[:] = reversed(range(3))
    assert x == Vec([2, 1, 0])


def test_delslice() -> None:
    a = Vec([0, 1])
    del a[1:2]
    del a[0:1]
    assert a == Vec()

    a = Vec([0, 1])
    del a[1:2]
    del a[0:1]
    assert a == Vec()

    a = Vec([0, 1])
    del a[-2:-1]
    assert a == Vec([1])

    a = Vec([0, 1])
    del a[-2:-1]
    assert a == Vec([1])

    a = Vec([0, 1])
    del a[1:]
    del a[:1]
    assert a == Vec()

    a = Vec([0, 1])
    del a[1:]
    del a[:1]
    assert a == Vec()

    a = Vec([0, 1])
    del a[-1:]
    assert a == Vec([0])

    a = Vec([0, 1])
    del a[-1:]
    assert a == Vec([0])

    a = Vec([0, 1])
    del a[:]
    assert a == Vec()


def test_append() -> None:
    a = Vec[int]([])
    a.append(0)
    a.append(1)
    a.append(2)
    assert a == Vec([0, 1, 2])

    with pytest.raises(TypeError):
        # pyrefly: ignore [missing-argument]
        a.append()  # pyright: ignore[reportCallIssue]


def test_extend() -> None:
    a1 = Vec([0])
    a2 = Vec((0, 1))
    a = a1[:]
    a.extend(a2)
    assert a == a1 + a2

    a.extend(Vec())
    assert a == a1 + a2

    a.extend(a)
    assert a == Vec([0, 0, 1, 0, 0, 1])

    a = Vec("spam")
    a.extend("eggs")
    assert a == Vec("spameggs")

    with pytest.raises(TypeError):
        # pyrefly: ignore [bad-argument-type]
        a.extend(None)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        # pyrefly: ignore [missing-argument]
        a.extend()  # pyright: ignore[reportCallIssue]

    # overflow test. issue1621
    class CustomIter:
        def __iter__(self) -> Self:
            return self

        def __next__(self) -> int:
            raise StopIteration

        def __length_hint__(self) -> int:
            return sys.maxsize

    a = Vec([1, 2, 3, 4])
    a.extend(CustomIter())
    assert a == [1, 2, 3, 4]


def test_insert() -> None:
    a = Vec[int | str]([0, 1, 2])
    a.insert(0, -2)
    a.insert(1, -1)
    a.insert(2, 0)
    assert a == [-2, -1, 0, 0, 1, 2]

    b = a[:]
    b.insert(-2, "foo")
    b.insert(-200, "left")
    b.insert(200, "right")
    assert b == Vec(["left", -2, -1, 0, 0, "foo", 1, 2, "right"])
    with pytest.raises(TypeError):
        # pyrefly: ignore [missing-argument]
        a.insert()  # pyright: ignore[reportCallIssue]


def test_pop() -> None:
    a = Vec([-1, 0, 1])
    _ = a.pop()
    assert a == [-1, 0]
    _ = a.pop(0)
    assert a == [0]
    with pytest.raises(IndexError):
        _ = a.pop(5)
    _ = a.pop(0)
    assert a == []
    with pytest.raises(IndexError):
        _ = a.pop()
    with pytest.raises(TypeError):
        # pyrefly: ignore [bad-argument-count]
        _ = a.pop(42, 42)  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
    a = Vec([0, 10, 20, 30, 40])


def test_remove() -> None:
    a = Vec([0, 0, 1])
    a.remove(1)
    assert a == [0, 0]
    a.remove(0)
    assert a == [0]
    a.remove(0)
    assert a == []

    with pytest.raises(ValueError):
        a.remove(0)

    with pytest.raises(TypeError):
        # pyrefly: ignore [bad-argument-count]
        a.remove()  # pyright: ignore[reportCallIssue]

    a = Vec[object]([1, 2])
    with pytest.raises(ValueError):
        a.remove(NEVER_EQ)
    assert a == [1, 2]
    a.remove(ALWAYS_EQ)
    assert a == [2]
    a = Vec[object]([ALWAYS_EQ])
    a.remove(1)
    assert a == []
    a = Vec[object]([ALWAYS_EQ])
    a.remove(NEVER_EQ)
    assert a == []
    a = Vec[object]([NEVER_EQ])
    with pytest.raises(ValueError):
        a.remove(ALWAYS_EQ)

    class BadExcError(Exception):
        pass

    class BadCmp:  # ruff:ignore[eq-without-hash]
        @override
        def __eq__(self, other: object) -> bool:
            if other == 2:
                raise BadExcError
            return False

    a = Vec[object]([0, 1, 2, 3])
    with pytest.raises(BadExcError):
        a.remove(BadCmp())

    class BadCmp2:  # ruff:ignore[eq-without-hash]
        @override
        def __eq__(self, other: object) -> bool:
            raise BadExcError

    d = Vec("abcdefghcij")
    d.remove("c")
    assert d == Vec("abdefghcij")
    d.remove("c")
    assert d == Vec("abdefghij")
    with pytest.raises(ValueError):
        d.remove("c")
    assert d == Vec("abdefghij")

    # Handle comparison errors
    d = Vec(["a", "b", BadCmp2(), "c"])
    e = Vec(d)
    with pytest.raises(BadExcError):
        d.remove("c")
    for x, y in zip(d, e, strict=False):
        # verify that original order and values are retained.
        assert x is y


def test_index() -> None:
    a = Vec([-2, -1, 0, 0, 1, 2])
    a.remove(0)
    with pytest.raises(ValueError):
        _ = a.index(2, 0, 4)
    assert a == Vec([-2, -1, 0, 1, 2])

    # Test modifying the list during index's iteration
    class EvilCmp[T]:  # ruff:ignore[eq-without-hash]
        def __init__(self, victim: Vec[T]) -> None:
            self.victim: Vec[T] = victim

        @override
        def __eq__(self, other: object) -> bool:
            del self.victim[:]
            return False

    a = Vec[object]()
    a[:] = [EvilCmp(a) for _ in range(100)]
    # This used to seg fault before patch #1005778
    with pytest.raises(ValueError):
        _ = a.index(None)


def test_reverse() -> None:
    u = Vec([-2, -1, 0, 1, 2])
    u2 = u[:]
    u.reverse()
    assert u == [2, 1, 0, -1, -2]
    u.reverse()
    assert u == u2

    with pytest.raises(TypeError):
        # pyrefly: ignore [bad-argument-count]
        u.reverse(42)  # pyright: ignore[reportCallIssue]


def test_clear() -> None:
    u = Vec([2, 3, 4])
    u.clear()
    assert u == []

    u = Vec[int]([])
    u.clear()
    assert u == []

    u = Vec[int]([])
    u.append(1)
    u.clear()
    u.append(2)
    assert u == [2]

    with pytest.raises(TypeError):
        # pyrefly: ignore [bad-argument-count]
        u.clear(None)  # pyright: ignore[reportCallIssue]


def test_copy() -> None:
    u = Vec([1, 2, 3])
    v = u.copy()
    assert v == [1, 2, 3]

    u = Vec[int]([])
    v = u.copy()
    assert v == []

    # test that it's indeed a copy and not a reference
    u = Vec(["a", "b"])
    v = u.copy()
    v.append("i")
    assert u == ["a", "b"]
    assert v == [*u, "i"]

    # test that it's a shallow, not a deep copy
    u = Vec([1, 2, [3, 4], 5])
    v = u.copy()
    assert u == v
    assert v[3] is u[3]

    with pytest.raises(TypeError):
        # pyrefly: ignore [bad-argument-count]
        _ = u.copy(None)  # pyright: ignore[reportCallIssue, reportUnknownVariableType]


def test_sort() -> None:
    u = Vec([1, 0])
    _ = u.sort()
    assert u == [0, 1]

    u = Vec([2, 1, 0, -1, -2])
    _ = u.sort()
    assert u == Vec([-2, -1, 0, 1, 2])

    with pytest.raises(TypeError):
        # pyrefly: ignore [bad-argument-count]
        _ = u.sort(42, 42)  # pyright: ignore[reportCallIssue, reportUnknownVariableType]

    def revcmp(a: int, b: int) -> int:
        if a == b:
            return 0
        if a < b:
            return 1
        # a > b
        return -1

    _ = u.sort_by(key=cmp_to_key(revcmp))
    assert u == Vec([2, 1, 0, -1, -2])

    # The following dumps core in unpatched Python 1.5:
    def my_comp(x: int, y: int) -> int:
        xmod, ymod = x % 3, y % 7
        if xmod == ymod:
            return 0
        if xmod < ymod:
            return -1
        # xmod > ymod
        return 1

    z = Vec(range(12))
    _ = z.sort_by(key=cmp_to_key(my_comp))

    with pytest.raises(TypeError):
        # pyrefly: ignore [bad-argument-count]
        _ = z.sort(2)  # pyright: ignore[reportCallIssue, reportUnknownVariableType]

    def self_modifying_comp(x: int, y: int) -> int:
        z.append(1)
        if x == y:
            return 0
        if x < y:
            return -1
        # x > y
        return 1

    with pytest.raises(ValueError):
        _ = z.sort_by(key=cmp_to_key(self_modifying_comp))

    with pytest.raises(TypeError):
        # pyrefly: ignore [bad-argument-count]
        _ = z.sort(42, 42, 42, 42)  # pyright: ignore[reportCallIssue, reportUnknownVariableType]


def test_slice() -> None:
    u = Vec("spam")
    u[:2] = "h"
    assert u == Vec("ham")


def test_iadd() -> None:
    u = Vec([0, 1])
    u2 = u
    u += [2, 3]
    assert u is u2

    u = Vec("spam")
    u += "eggs"
    assert u == Vec("spameggs")

    with pytest.raises(TypeError):
        # pyrefly: ignore [bad-argument-type]
        _ = u.__iadd__(None)  # pyright: ignore[reportArgumentType]


def test_imul() -> None:
    s = Vec[int]([])
    oldid = id(s)
    s *= 10
    assert id(s) == oldid


def test_extendedslicing() -> None:
    #  subscript
    a = Vec[int]([0, 1, 2, 3, 4])

    #  deletion
    del a[::2]
    assert a == Vec([1, 3])
    a = Vec(range(5))
    del a[1::2]
    assert a == Vec([0, 2, 4])
    a = Vec(range(5))
    del a[1::-2]
    assert a == Vec([0, 2, 3, 4])
    a = Vec(range(10))
    del a[::1000]
    assert a == Vec([1, 2, 3, 4, 5, 6, 7, 8, 9])
    #  assignment
    a = Vec(range(10))
    a[::2] = [-1] * 5
    assert a == Vec([-1, 1, -1, 3, -1, 5, -1, 7, -1, 9])
    a = Vec(range(10))
    a[::-4] = [10] * 3
    assert a == Vec([0, 10, 2, 3, 4, 10, 6, 7, 8, 10])
    a = Vec(range(4))
    a[::-1] = a
    assert a == Vec([3, 2, 1, 0])
    a = Vec(range(10))
    b = a[:]
    c = a[:]
    # pyrefly: ignore [unsupported-operation]
    a[2:3] = Vec(["two", "elements"])  # pyright: ignore[reportCallIssue, reportArgumentType]
    # pyrefly: ignore [unsupported-operation]
    b[slice(2, 3)] = Vec(["two", "elements"])  # pyright: ignore[reportCallIssue, reportArgumentType]
    # pyrefly: ignore [unsupported-operation]
    c[2:3:] = Vec(["two", "elements"])  # pyright: ignore[reportCallIssue, reportArgumentType]
    assert a == b
    assert a == c
    a = Vec(range(10))
    a[::2] = tuple(range(5))
    assert a == Vec([0, 1, 1, 3, 2, 5, 3, 7, 4, 9])
    # test issue7788
    a = Vec(range(10))
    del a[9 :: 1 << 333]


def test_constructor_exception_handling() -> None:
    # Bug #1242657
    class F:
        def __iter__(self) -> object:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        # pyrefly: ignore [bad-argument-type]
        _ = Vec(F())


def test_exhausted_iterator() -> None:
    a = Vec([1, 2, 3])
    exhit = iter(a)
    empit = iter(a)
    for _x in exhit:  # exhaust the iterator
        _ = next(empit)  # not exhausted
    a.append(9)
    assert Vec(exhit) == []
    assert Vec(empit) == [9]
    assert a == Vec([1, 2, 3, 9])

    # gh-115733: Crash when iterating over exhausted iterator
    exhit = iter(Vec([1, 2, 3]))
    for _ in exhit:
        _ = next(exhit, 1)


def test_basic() -> None:
    assert Vec() == []
    l0_3 = [0, 1, 2, 3]
    l0_3_bis = Vec(l0_3)
    assert l0_3 == l0_3_bis
    assert l0_3 is not l0_3_bis
    assert Vec() == []
    assert Vec((0, 1, 2, 3)) == [0, 1, 2, 3]
    assert Vec("") == []
    assert Vec("spam") == ["s", "p", "a", "m"]
    assert Vec(x for x in range(10) if x % 2) == [1, 3, 5, 7, 9]

    if sys.maxsize == 0x7FFFFFFF:
        # This test can currently only work on 32-bit machines.
        # If/when PySequence_Length() returns a ssize_t, it should be
        # re-enabled.
        # Verify clearing of bug #556025.
        # This assumes that the max data size (sys.maxint) == max
        # address size this also assumes that the address size is at
        # least 4 bytes with 8 byte addresses, the bug is not well
        # tested
        #
        # Note: This test is expected to SEGV under Cygwin 1.3.12 or
        # earlier due to a newlib bug.  See the following mailing list
        # thread for the details:

        #     http://sources.redhat.com/ml/newlib/2002/msg00369.html
        with pytest.raises(MemoryError):
            _ = Vec(range(sys.maxsize // 2))

    # This code used to segfault in Py2.4a3
    x = Vec[int]([])
    x.extend(-y for y in x)
    assert x == []


def test_keyword_args() -> None:
    with pytest.raises(TypeError, match="keyword argument"):
        # pyrefly: ignore [missing-argument, unexpected-keyword]
        _ = Vec(sequence=[])  # pyright: ignore[reportCallIssue, reportUnknownVariableType]


def test_truth() -> None:
    assert not []
    assert [42]


def test_identity() -> None:
    assert Vec() is not Vec()


def test_len() -> None:
    assert len(Vec[int]([])) == 0
    assert len(Vec([0])) == 1
    assert len(Vec([0, 1, 2])) == 3


def test_overflow() -> None:
    lst = Vec([4, 5, 6, 7])
    n = int((sys.maxsize * 2 + 2) // len(lst))

    def mul(a: Vec[int], b: int) -> Vec[int]:  # ruff:ignore[reimplemented-operator]
        return a * b

    def imul(a: Vec[int], b: int) -> None:
        a *= b

    with pytest.raises((MemoryError, OverflowError)):
        _ = mul(lst, n)
    with pytest.raises((MemoryError, OverflowError)):
        imul(lst, n)


def test_empty_slice() -> None:
    x = Vec[int]([])
    x[:] = x
    assert x == []


def test_list_resize_overflow() -> None:
    """gh-97616: test new_allocated * sizeof(PyObject*) overflow

    check in list_resize()
    """
    lst = Vec[int]([0] * 65)
    del lst[1:]
    assert len(lst) == 1

    size = sys.maxsize
    with pytest.raises((MemoryError, OverflowError)):
        _ = lst * size
    with pytest.raises((MemoryError, OverflowError)):
        lst *= size


def test_repr_mutate() -> None:
    class Obj:
        @staticmethod
        @override
        def __repr__() -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
            with contextlib.suppress(IndexError):
                _ = mylist.pop()
            return "obj"

    mylist = [Obj() for _ in range(5)]
    assert repr(mylist) == "[obj, obj, obj]"


@pytest.mark.skip(
    reason="We still haven't decided how to handle large repr. This test take quite some time by itself."
)
def test_repr_large() -> None:
    # Check the repr of large list objects
    def check(n: int) -> None:
        lst = Vec[int]([0] * n)
        s = repr(lst)
        assert s == "Vec(" + ", ".join(["0"] * n) + ")"

    check(10)  # check our checking code
    check(1000000)


@pytest.mark.skip(reason="Pyo3 does not support pickling as of now")
def test_iterator_pickle() -> None:
    orig = Vec[int]([4, 5, 6, 7])
    data = [10, 11, 12, 13, 14, 15]
    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        # initial iterator
        itorig = iter(orig)
        d = pickle.dumps((itorig, orig), proto)
        it, a = pickle.loads(d)  # pyright: ignore[reportAny]
        a[:] = data
        assert type(it) is type(itorig)  # pyright: ignore[reportAny]
        assert Vec(it) == data

        # running iterator
        _ = next(itorig)
        d = pickle.dumps((itorig, orig), proto)
        it, a = pickle.loads(d)  # pyright: ignore[reportAny]
        a[:] = data
        assert type(it) is type(itorig)  # pyright: ignore[reportAny]
        assert Vec(it) == data[1:]

        # empty iterator
        for _i in range(1, len(orig)):
            _ = next(itorig)
        d = pickle.dumps((itorig, orig), proto)
        it, a = pickle.loads(d)  # pyright: ignore[reportAny]
        a[:] = data
        assert type(it) is type(itorig)  # pyright: ignore[reportAny]
        assert Vec(it) == data[len(orig) :]

        # exhausted iterator
        with pytest.raises(StopIteration):
            _ = next(itorig)
        d = pickle.dumps((itorig, orig), proto)
        it, a = pickle.loads(d)  # pyright: ignore[reportAny]
        a[:] = data
        assert Vec(it) == []  # pyright: ignore[reportAny]


@pytest.mark.skip(reason="Pyo3 does not support pickling as of now")
def test_reversed_pickle() -> None:
    orig = Vec([4, 5, 6, 7])
    data = Vec([10, 11, 12, 13, 14, 15])
    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        # initial iterator
        itorig = reversed(orig)
        d = pickle.dumps((itorig, orig), proto)
        it, a = pickle.loads(d)  # pyright: ignore[reportAny]
        a[:] = data
        assert type(it) is type(itorig)  # pyright: ignore[reportAny]
        assert Vec(it) == data[len(orig) - 1 :: -1]

        # running iterator
        _ = next(itorig)
        d = pickle.dumps((itorig, orig), proto)
        it, a = pickle.loads(d)  # pyright: ignore[reportAny]
        a[:] = data
        assert type(it) is type(itorig)  # pyright: ignore[reportAny]
        assert Vec(it) == data[len(orig) - 2 :: -1]

        # empty iterator
        for _i in range(1, len(orig)):
            _ = next(itorig)
        d = pickle.dumps((itorig, orig), proto)
        it, a = pickle.loads(d)  # pyright: ignore[reportAny]
        a[:] = data
        assert type(it) is type(itorig)  # pyright: ignore[reportAny]
        assert Vec(it) == []

        # exhausted iterator
        with pytest.raises(StopIteration):
            _ = next(itorig)
        d = pickle.dumps((itorig, orig), proto)
        it, a = pickle.loads(d)  # pyright: ignore[reportAny]
        a[:] = data
        assert Vec(it) == []  # pyright: ignore[reportAny]


def test_step_overflow() -> None:
    a = [0, 1, 2, 3, 4]
    a[1 :: sys.maxsize] = [0]
    assert a[3 :: sys.maxsize] == [3]


def test_equal_operator_modifying_operand() -> None:
    # test fix for seg fault reported in bpo-38588 part 2.
    class X:  # ruff:ignore[eq-without-hash]
        @override
        def __eq__(self, other: object) -> bool:
            list2.clear()
            return NotImplemented

    class Y:  # ruff:ignore[eq-without-hash]
        @override
        def __eq__(self, other: object) -> bool:
            list1.clear()
            return NotImplemented

    class Z:  # ruff:ignore[eq-without-hash]
        @override
        def __eq__(self, other: object) -> bool:
            list3.clear()
            return NotImplemented

    list1 = Vec([X()])
    list2 = Vec([Y()])
    assert list1 == list2

    list3 = Vec([Z()])
    list4 = Vec([1])
    assert list3 != list4


def test_lt_operator_modifying_operand() -> None:
    # See gh-120298
    class Evil:
        def __lt__(self, other: MutableSequence[object]) -> bool:
            other.clear()
            return NotImplemented

    a = Vec([Evil()])
    with pytest.raises(TypeError):
        # pyrefly: ignore [unsupported-operation]
        _ = a[0] < a  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]


def test_list_index_modifing_operand() -> None:
    # See gh-120384
    class Evil:
        def __init__(self, lst: Vec[int]) -> None:
            self.lst: Vec[int] = lst

        def __iter__(self) -> Iterator[int]:
            yield from self.lst
            self.lst.clear()

    lst = Vec[int](range(5))
    operand = Evil(lst)
    with pytest.raises(ValueError):
        lst[::-1] = operand


def test_tier2_invalidates_iterator() -> None:
    # GH-121012
    for _ in range(100):
        a = [1, 2, 3]
        it = iter(a)
        for _ in it:
            pass
        a.append(4)
        assert Vec(it) == []
