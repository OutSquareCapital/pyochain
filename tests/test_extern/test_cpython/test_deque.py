"""This modules contains tests that for the most part have been adapted from the CPython test suite for `collections.Counter`.

Most modifications entails the conversion from `unittest` to `pytest`, the deletion of subclass/pickle tests, and the fact that `__init__` has no special handling for "resetting" the `deque`.

The original lives at:

https://github.com/python/cpython/blob/16562f1ce31ba654c42eee644f253bb31cff9f9d/Lib/test/test_deque.py#L375
"""

from __future__ import annotations

import copy
import gc
import random
import string
import weakref
from collections import deque
from typing import TYPE_CHECKING, Never, Self, override

import pytest

from pyochain import Vec
from pyochain.collections import Deque

from . import test_seq

if TYPE_CHECKING:
    from collections.abc import Sequence

BIG = 1_000


class BadCmp:  # ruff:ignore[eq-without-hash]
    @override
    def __eq__(self, other: object) -> Never:
        raise RuntimeError


class MutateCmp:  # ruff:ignore[eq-without-hash]
    def __init__(self, deque: Deque[object], result: bool) -> None:
        self.deque: Deque[object] = deque
        self.result: bool = result

    @override
    def __eq__(self, other: object) -> bool:
        self.deque.clear()
        return self.result


def test_basics() -> None:
    d = Deque(range(200))
    for i in range(200, 400):
        d.append(i)
    for i in reversed(range(-200, 0)):
        d.append_left(i)
    assert list(d) == list(range(-200, 400))
    assert len(d) == 600

    left = [d.pop_left() for _ in range(250)]
    assert left == list(range(-200, 50))
    assert list(d) == list(range(50, 400))

    right = [d.pop() for _ in range(250)]
    right.reverse()
    assert right == list(range(150, 400))
    assert list(d) == list(range(50, 150))


def test_max_length() -> None:
    with pytest.raises(ValueError):
        _ = Deque("abc", max_length=-1)
    with pytest.raises(ValueError):
        _ = Deque("abc", max_length=-2)
    it = iter(range(10))
    d = Deque(it, max_length=3)
    assert list(it) == []
    assert repr(d) == "Deque([7, 8, 9], max_length=3)"
    assert list(d) == [7, 8, 9]
    assert d == Deque(range(10), max_length=3)
    d.append(10)
    assert list(d) == [8, 9, 10]
    d.append_left(7)
    assert list(d) == [7, 8, 9]
    d.extend([10, 11])
    assert list(d) == [9, 10, 11]
    d.extend_left([8, 7])
    assert list(d) == [7, 8, 9]
    d = Deque(range(200), max_length=10)
    # pyrefly: ignore [bad-argument-type]
    d.append(d)  # pyright: ignore[reportArgumentType]
    # NOTE: In the original CPython test, the slice is [-30:], but since `max_length` is 4 chars longer,
    # we have to adjust it.
    assert repr(d)[-34:] == ", 198, 199, [...]], max_length=10)"
    d = Deque(range(10), max_length=None)
    assert repr(d) == "Deque([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])"


def test_maxlen_zero() -> None:
    it = iter(range(100))
    _ = Deque(it, max_length=0)
    assert list(it) == []

    it = iter(range(100))
    d = Deque[int](max_length=0)
    d.extend(it)
    assert list(it) == []

    it = iter(range(100))
    d = Deque[int](max_length=0)
    d.extend_left(it)
    assert list(it) == []


def test_maxlen_attribute() -> None:
    assert Deque().max_length is None
    assert Deque("abc").max_length is None
    assert Deque("abc", max_length=4).max_length == 4
    assert Deque("abc", max_length=2).max_length == 2
    assert Deque("abc", max_length=0).max_length == 0
    with pytest.raises(AttributeError):  # ruff:ignore[pytest-raises-with-multiple-statements]
        d = Deque("abc")
        # pyrefly: ignore [read-only]
        d.max_length = 10  # pyright: ignore[reportAttributeAccessIssue]


def test_count() -> None:
    for s in ("", "abracadabra", "simsalabim" * 500 + "abc"):
        inner_s = list(s)
        d = Deque(inner_s)
        for letter in string.ascii_lowercase:
            assert inner_s.count(letter) == d.count(letter), (inner_s, d, letter)
    with pytest.raises(TypeError):
        # pyrefly: ignore [bad-argument-count]
        d.count()  # too few args  # pyright: ignore[reportCallIssue]
    with pytest.raises(TypeError):
        # pyrefly: ignore [bad-argument-count, bad-argument-type]
        d.count(1, 2)  # too many args  # pyright: ignore[reportCallIssue]

    class BadCompare:  # ruff:ignore[eq-without-hash]
        @override
        def __eq__(self, other: object) -> Never:
            raise ArithmeticError

    d = Deque([1, 2, BadCompare(), 3])
    with pytest.raises(ArithmeticError):
        _ = d.count(2)
    d = Deque([1, 2, 3])
    with pytest.raises(ArithmeticError):
        # pyrefly: ignore [bad-argument-type]
        _ = d.count(BadCompare())  # pyright: ignore[reportArgumentType]

    class MutatingCompare:  # ruff:ignore[eq-without-hash]
        d: Deque[Self | int]  # pyright: ignore[reportUninitializedInstanceVariable]

        @override
        def __eq__(self, other: object) -> bool:
            _ = self.d.pop()
            return True

    m = MutatingCompare()
    d = Deque([1, 2, 3, m, 4, 5])
    m.d = d
    with pytest.raises(RuntimeError):
        _ = d.count(3)

    # test issue11004
    # block advance failed after rotation aligned elements on right side of block
    d = Deque([None] * 16)
    for _i in range(len(d)):
        _ = d.rotate(-1)
    _ = d.rotate(1)
    # pyrefly: ignore [bad-argument-type]
    assert d.count(1) == 0  # pyright: ignore[reportArgumentType]
    assert d.count(None) == 16


def test_comparisons() -> None:
    d = Deque("xabc")
    _ = d.pop_left()
    compared: list[Sequence[str]] = [d, Deque("abc"), Deque("ab"), Deque(), list(d)]
    for e in compared:
        assert (d == e) == (type(d) is type(e) and list(d) == list(e))
        assert (d != e) == (not (type(d) is type(e) and list(d) == list(e)))

    args = map(Deque, ("", "a", "b", "ab", "ba", "abc", "xba", "xabc", "cba"))
    for x in args:
        for y in args:
            assert (x == y) == (list(x) == list(y)), (x, y)
            assert (x != y) == (list(x) != list(y)), (x, y)
            assert (x < y) == (list(x) < list(y)), (x, y)
            assert (x <= y) == (list(x) <= list(y)), (x, y)
            assert (x > y) == (list(x) > list(y)), (x, y)
            assert (x >= y) == (list(x) >= list(y)), (x, y)


def test_contains() -> None:
    n = 200

    d = Deque(range(n))
    for i in range(n):
        assert i in d
    assert n + 1 not in d

    # Test detection of mutation during iteration
    d = Deque(range(n))
    # pyrefly: ignore [unsupported-operation]
    # pyrefly: ignore [bad-argument-type]
    d[n // 2] = MutateCmp(d, result=False)  # pyright: ignore[reportArgumentType]
    with pytest.raises(RuntimeError):
        _ = n in d

    # Test detection of comparison exceptions
    d = Deque(range(n))
    # pyrefly: ignore [unsupported-operation]
    d[n // 2] = BadCmp()  # pyright: ignore[reportArgumentType]
    with pytest.raises(RuntimeError):
        _ = n in d


def test_contains_count_index_stop_crashes() -> None:
    class Foo:  # ruff:ignore[eq-without-hash]
        @override
        def __eq__(self, other: object) -> bool:
            d.clear()
            return NotImplemented

    d = Deque([Foo(), Foo()])
    with pytest.raises(RuntimeError):
        _ = 3 in d
    d = Deque([Foo(), Foo()])
    with pytest.raises(RuntimeError):
        # pyrefly: ignore [bad-argument-type]
        _ = d.count(3)  # pyright: ignore[reportArgumentType]

    d = Deque([Foo()])
    with pytest.raises(RuntimeError):
        # pyrefly: ignore [bad-argument-type]
        # pyrefly: ignore [bad-argument-type]
        _ = d.index(0)  # pyright: ignore[reportArgumentType]


def test_extend() -> None:
    d = Deque("a")
    with pytest.raises(TypeError):
        # pyrefly: ignore [bad-argument-type]
        d.extend(1)  # pyright: ignore[reportArgumentType]
    d.extend("bcd")
    assert list(d) == list("abcd")
    d.extend(d)
    assert list(d) == list("abcdabcd")


def test_add() -> None:
    d = Deque[str]()
    e = Deque("abc")
    f = Deque("def")
    assert d + d == Deque()
    assert e + f == Deque("abcdef")
    assert e + e == Deque("abcabc")
    assert e + d == Deque("abc")
    assert d + e == Deque("abc")
    assert d + d is not Deque()
    assert e + d is not Deque("abc")
    assert d + e is not Deque("abc")

    g = Deque("abcdef", max_length=4)
    h = Deque("gh")
    assert g + h == Deque("efgh")

    with pytest.raises(TypeError):
        # pyrefly: ignore [unsupported-operation]
        _ = Deque("abc") + "def"  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]


def test_iadd() -> None:
    d = Deque("a")
    d += "bcd"
    assert list(d) == list("abcd")
    d += d
    assert list(d) == list("abcdabcd")


def test_extend_left() -> None:
    d = Deque("a")
    with pytest.raises(TypeError):
        # pyrefly: ignore [bad-argument-type]
        # pyrefly: ignore [bad-argument-type]
        d.extend_left(1)  # pyright: ignore[reportArgumentType]
    d.extend_left("bcd")
    assert list(d) == list(reversed("abcd"))
    d.extend_left(d)
    assert list(d) == list("abcddcba")
    d = Deque[int]()
    d.extend_left(range(100))
    assert list(d) == list(reversed(range(100)))
    with pytest.raises(SyntaxError):
        d.extend_left(fail())


def fail() -> Never:
    raise SyntaxError
    yield 1  # pyright: ignore[reportUnreachable]


def test_getitem() -> None:
    n = 200
    d = Deque(range(n))
    lst = list(range(n))
    for i in range(n):
        _ = d.pop_left()
        _ = lst.pop(0)
        if random.random() < 0.5:
            d.append(i)
            lst.append(i)
        for j in range(1 - len(lst), len(lst)):
            assert d[j] == lst[j]

    d = Deque("superman")
    assert d[0] == "s"
    assert d[-1] == "n"
    d = Deque[int]()
    with pytest.raises(IndexError):
        _ = d.__getitem__(0)
    with pytest.raises(IndexError):
        _ = d.__getitem__(-1)


def test_index() -> None:
    for n in 1, 10, 100:
        d = Deque(range(n))
        for i in range(n):
            assert d.index(i) == i

        with pytest.raises(ValueError):
            _ = d.index(n + 1)

        # Test detection of mutation during iteration
        d = Deque(range(n))
        # pyrefly: ignore [unsupported-operation]
        # pyrefly: ignore [bad-argument-type]
        d[n // 2] = MutateCmp(d, result=False)  # pyright: ignore[reportArgumentType]
        with pytest.raises(RuntimeError):
            _ = d.index(n)

        # Test detection of comparison exceptions
        d = Deque(range(n))
        # pyrefly: ignore [unsupported-operation]
        d[n // 2] = BadCmp()  # pyright: ignore[reportArgumentType]
        with pytest.raises(RuntimeError):
            _ = d.index(n)

    # Test start and stop arguments behavior matches list.index()
    elements = "ABCDEFGHI"
    d = Deque(elements * 2)
    s = list(elements * 2)
    r = range(-2 - len(s), 2 + len(s))
    for start in r:
        for stop in r:
            for element in elements + "Z":
                try:
                    target = s.index(element, start, stop)
                except ValueError:
                    with pytest.raises(ValueError):
                        _ = d.index(element, start, stop)
                else:
                    assert d.index(element, start, stop) == target

    # Test stop argument
    for elem in d:
        index = d.index(elem)
        assert index == d.index(elem, 0)
        assert index == d.index(elem, 0, len(d))
        assert index == d.index(elem, 0, len(d) + 100)

    # Test large start argument
    d = Deque(range(0, 1000, 5))
    for _step in range(10):
        i = d.index(850, 70)
        assert d[i] == 850
        # Repeat test with a different internal offset
        _ = d.rotate()


def test_index_bug_24913() -> None:
    d = Deque("A" * 3)
    with pytest.raises(ValueError):
        _ = d.index("Hello world", 0, 4)


def test_insert() -> None:
    # Test to make sure insert behaves like lists
    elements = "ABCDEFGHI"
    for i in range(-5 - len(elements) * 2, 5 + len(elements) * 2):
        d = Deque("ABCDEFGHI")
        s = list("ABCDEFGHI")
        d.insert(i, "Z")
        s.insert(i, "Z")
        assert list(d) == s


def test_insert_bug_26194() -> None:
    data = "ABC"
    d = Deque(data, max_length=len(data))
    with pytest.raises(IndexError):
        # pyrefly: ignore [bad-argument-type]
        d.insert(2, None)  # pyright: ignore[reportArgumentType]

    elements = "ABCDEFGHI"
    for i in range(-len(elements), len(elements)):
        d = Deque(elements, max_length=len(elements) + 1)
        d.insert(i, "Z")
        if i >= 0:
            assert d[i] == "Z"
        else:
            assert d[i - 1] == "Z"


def test_imul() -> None:
    for n in (-10, -1, 0, 1, 2, 10, 1000):
        d = Deque[int]()
        d *= n
        assert d == Deque()
        assert d.max_length is None

    for n in (-10, -1, 0, 1, 2, 10, 1000):
        d = Deque("a")
        d *= n
        assert d == Deque("a" * n)
        assert d.max_length is None

    for n in (-10, -1, 0, 1, 2, 10, 499, 500, 501, 1000):
        d = Deque("a", max_length=500)
        d *= n
        assert d == Deque("a" * min(n, 500))
        assert d.max_length == 500

    for n in (-10, -1, 0, 1, 2, 10, 1000):
        d = Deque("abcdef")
        d *= n
        assert d == Deque("abcdef" * n)
        assert d.max_length is None

    for n in (-10, -1, 0, 1, 2, 10, 499, 500, 501, 1000):
        d = Deque("abcdef", max_length=500)
        d *= n
        assert d == Deque(("abcdef" * n)[-500:])
        assert d.max_length == 500


def test_mul() -> None:
    d = Deque("abc")
    assert d * -5 == Deque[str]()
    assert d * 0 == Deque[str]()
    assert d * 1 == Deque("abc")
    assert d * 2 == Deque("abcabc")
    assert d * 3 == Deque("abcabcabc")
    assert d * 1 is not d

    assert Deque() * 0 == Deque()
    assert Deque() * 1 == Deque()
    assert Deque() * 5 == Deque()

    assert -5 * d == Deque[str]()
    assert 0 * d == Deque[str]()
    assert 1 * d == Deque("abc")
    assert 2 * d == Deque("abcabc")
    assert 3 * d == Deque("abcabcabc")

    d = Deque("abc", max_length=5)
    assert d * -5 == Deque[str]()
    assert d * 0 == Deque[str]()
    assert d * 1 == Deque("abc")
    assert d * 2 == Deque("bcabc")
    assert d * 30 == Deque("bcabc")


def test_setitem() -> None:
    n = 200
    d = Deque(range(n))
    for i in range(n):
        d[i] = 10 * i
    assert list(d) == [10 * i for i in range(n)]
    lst = list(d)
    for i in range(1 - n, 0, -1):
        d[i] = 7 * i
        lst[i] = 7 * i
    assert list(d) == lst


def test_delitem() -> None:
    n = 500  # O(n**2) test, don't make this too big
    d = Deque(range(n))
    with pytest.raises(IndexError):
        d.__delitem__(-n - 1)
    with pytest.raises(IndexError):
        d.__delitem__(n)
    for i in range(n):
        assert len(d) == n - i
        j = random.randrange(-len(d), len(d))
        val = d[j]
        assert val in d
        del d[j]
        assert val not in d
    assert len(d) == 0


def test_reverse() -> None:
    n = 50  # O(n**2) test, don't make this too big
    data = [random.random() for _ in range(n)]
    d = Deque[float]()
    for i in range(n):
        d = Deque(data[:i])
        r = d.reverse()
        assert list(d) == list(reversed(data[:i]))
        assert r is None
        d.reverse()
        assert list(d) == data[:i]
    with pytest.raises(TypeError):
        # pyrefly: ignore [bad-argument-count]
        d.reverse(1)  # Arity is zero  # pyright: ignore[reportCallIssue]


def test_rotate() -> None:
    s = tuple("abcde")
    n = len(s)

    d = Deque(s)
    _ = d.rotate(1)  # verify rot(1)
    assert "".join(d) == "eabcd"

    d = Deque(s)
    _ = d.rotate(-1)  # verify rot(-1)
    assert "".join(d) == "bcdea"
    _ = d.rotate()  # check default to 1
    assert tuple(d) == s

    for i in range(n * 3):
        d = Deque(s)
        e = Deque(d)
        _ = d.rotate(i)  # check vs. rot(1) n times
        for _j in range(i):
            _ = e.rotate(1)
        assert tuple(d) == tuple(e)
        _ = d.rotate(-i)  # check that it works in reverse
        assert tuple(d) == s
        _ = e.rotate(n - i)  # check that it wraps forward
        assert tuple(e) == s

    for i in range(n * 3):
        d = Deque(s)
        e = Deque(d)
        _ = d.rotate(-i)
        for _j in range(i):
            _ = e.rotate(-1)  # check vs. rot(-1) n times
        assert tuple(d) == tuple(e)
        _ = d.rotate(i)  # check that it works in reverse
        assert tuple(d) == s
        _ = e.rotate(i - n)  # check that it wraps backaround
        assert tuple(e) == s

    d = Deque(s)
    e = Deque(s)
    _ = e.rotate(BIG + 17)  # verify on long series of rotates
    dr = d.rotate
    for _ in range(BIG + 17):
        _ = dr()
    assert tuple(d) == tuple(e)

    with pytest.raises(TypeError):
        # pyrefly: ignore [bad-argument-type]
        _ = d.rotate("x")  # Wrong arg type  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        # pyrefly: ignore [bad-argument-count, unused-call-result]
        d.rotate(1, 10)  # Too many args  # pyright: ignore[reportCallIssue]

    d = Deque[int]()
    _ = d.rotate()  # rotate an empty Deque
    assert d == Deque[int]()


def test_len() -> None:
    d = Deque("ab")
    assert len(d) == 2
    _ = d.pop_left()
    assert len(d) == 1
    _ = d.pop()
    assert len(d) == 0
    with pytest.raises(IndexError):
        _ = d.pop()
    assert len(d) == 0
    d.append("c")
    assert len(d) == 1
    d.append_left("d")
    assert len(d) == 2
    d.clear()
    assert len(d) == 0


def test_underflow() -> None:
    d = Deque[int]()
    with pytest.raises(IndexError):
        _ = d.pop()
    with pytest.raises(IndexError):
        _ = d.pop_left()


def test_clear() -> None:
    d = Deque(range(100))
    assert len(d) == 100
    d.clear()
    assert len(d) == 0
    assert list(d) == []
    d.clear()  # clear an empty Deque
    assert list(d) == []


def test_remove() -> None:
    d = Deque("abcdefghcij")
    d.remove("c")
    assert d == Deque("abdefghcij")
    d.remove("c")
    assert d == Deque("abdefghij")
    with pytest.raises(ValueError):
        d.remove("c")
    assert d == Deque("abdefghij")

    # Handle comparison errors
    d = Deque(["a", "b", BadCmp(), "c"])
    e = Deque(d)
    with pytest.raises(RuntimeError):
        d.remove("c")
    for x, y in zip(d, e, strict=False):
        # verify that original order and values are retained.
        assert x is y

    # Handle evil mutator
    for match in (True, False):
        d = Deque(["ab"])
        # pyrefly: ignore [bad-argument-type]
        # pyrefly: ignore [bad-argument-type]
        d.extend([MutateCmp(d, result=match), "c"])  # pyright: ignore[reportArgumentType]
        with pytest.raises(IndexError):
            d.remove("c")
        assert d == Deque()


def test_repr() -> None:
    d = Deque(range(200))
    e = eval(repr(d))  # pyright: ignore[reportAny]
    assert list(d) == list(e)  # pyright: ignore[reportAny]
    # pyrefly: ignore [bad-argument-type]
    # pyrefly: ignore [bad-argument-type]
    d.append(d)  # pyright: ignore[reportArgumentType]
    assert repr(d)[-20:] == "7, 198, 199, [...]])"


def test_init() -> None:
    # NOTE: here we differ from cpython, since our constructor is more flexible
    # Both test raise TypeError for a collections.deque
    assert Deque[int | str]("abc", 2, 3) == Deque(["abc", 2, 3])
    assert Deque[int](1) == Deque([1])


def test_hash() -> None:
    with pytest.raises(TypeError):
        _ = hash(Deque("abc"))


def test_long_steadystate_queue_pop_left() -> None:
    for size in (0, 1, 2, 100, 1000):
        d = Deque(range(size))
        append, pop = d.append, d.pop_left
        for i in range(size, BIG):
            append(i)
            x = pop()
            if x != i - size:
                assert x == i - size
        assert list(d) == list(range(BIG - size, BIG))


def test_long_steadystate_queue_popright() -> None:
    for size in (0, 1, 2, 100, 1000):
        d = Deque(reversed(range(size)))
        append, pop = d.append_left, d.pop
        for i in range(size, BIG):
            append(i)
            x = pop()
            if x != i - size:
                assert x == i - size
        assert list(reversed(list(d))) == list(range(BIG - size, BIG))


def test_big_queue_pop_left() -> None:
    d = Deque[int]()
    append, pop = d.append, d.pop_left
    for i in range(BIG):
        append(i)
    for i in range(BIG):
        x = pop()
        if x != i:
            assert x == i


def test_big_queue_popright() -> None:
    d = Deque[int]()
    append, pop = d.append_left, d.pop
    for i in range(BIG):
        append(i)
    for i in range(BIG):
        x = pop()
        if x != i:
            assert x == i


def test_big_stack_right() -> None:
    d = Deque[int]()
    append, pop = d.append, d.pop
    for i in range(BIG):
        append(i)
    for i in reversed(range(BIG)):
        x = pop()
        if x != i:
            assert x == i
    assert len(d) == 0


def test_big_stack_left() -> None:
    d = Deque[int]()
    append, pop = d.append_left, d.pop_left
    for i in range(BIG):
        append(i)
    for i in reversed(range(BIG)):
        x = pop()
        if x != i:
            assert x == i
    assert len(d) == 0


def test_roundtrip_iter_init() -> None:
    d = Deque(range(200))
    e = Deque(d)
    assert id(d) != id(e)
    assert list(d) == list(e)


@pytest.mark.skip(
    reason="Pyo3 doesn't support pickling, and copy::deepcopy call `__reduce_ex__`, which for some reason do not return None, so it tries to reduce it, which subsequently fails."
)
def test_deepcopy() -> None:
    mut = [10]
    d = Deque([mut])
    e = copy.deepcopy(d)
    assert list(d) == list(e)
    mut[0] = 11
    assert id(d) != id(e)
    assert list(d) != list(e)


def test_copy() -> None:
    mut = [10]
    d = Deque([mut])
    e = copy.copy(d)
    assert list(d) == list(e)
    mut[0] = 11
    assert id(d) != id(e)
    assert list(d) == list(e)

    for i in range(5):
        for maxlen in range(-1, 6):
            s = [random.random() for _ in range(i)]
            d = Deque(s) if maxlen == -1 else Deque(s, maxlen)
            e = d.copy()
            assert d == e
            assert d.max_length == e.max_length
            assert all(x is y for x, y in zip(d, e, strict=False))


def test_copy_method() -> None:
    mut = [10]
    d = Deque([mut])
    e = d.copy()
    assert list(d) == list(e)
    mut[0] = 11
    assert id(d) != id(e)
    assert list(d) == list(e)


def test_reversed() -> None:
    for s in ("abcd", range(20)):
        assert list(reversed(Deque(s))) == list(reversed(s))


def test_reversed_new() -> None:
    klass = type(reversed(Deque[str | int]()))
    # NOTE: klass is the dedicated reversed deque iterator type
    for s in ("abcd", range(20)):
        # pyrefly: ignore [bad-argument-count, bad-instantiation]
        assert Vec(klass(deque(s))) == Vec(reversed(s))  # pyright: ignore[reportCallIssue]


@pytest.mark.skip(
    reason="Unless we start fucking around with traverse and clear, it's simpler to just skip this test as it is quite slow"
)
def test_gc_doesnt_blowup() -> None:
    import gc

    # This used to assert-fail in deque_traverse() under a debug
    # build, or run wild with a NULL pointer in a release build.
    d = Deque[int]()
    for _i in range(100):
        d.append(1)
        _ = gc.collect()


def test_container_iterator() -> None:
    # Bug #3680: tp_traverse was not implemented for deque iterator objects
    class C:
        pass

    for i in range(2):
        obj = C()
        ref = weakref.ref(obj)
        container = Deque([obj, 1]) if i == 0 else reversed(Deque([obj, 1]))
        # pyrefly: ignore [missing-attribute]
        obj.x = iter(container)  # pyright: ignore[reportAttributeAccessIssue]
        del obj, container
        _ = gc.collect()
        assert ref() is None, "Cycle was not collected"


def test_constructor() -> None:
    for s in ("123", "", range(100), ("do", 1.2), range(200, 220, 5)):
        for g in (
            test_seq.SequenceTest[int, str],
            test_seq.IterFunc[str | int],
            test_seq.IterGen[str | int],
            test_seq.IterFuncStop[str | int],
            test_seq.itermulti,
            test_seq.iterfunc,
        ):
            # pyrefly: ignore [bad-argument-type]
            # pyrefly: ignore [bad-argument-type]
            assert list(Deque(g(s))) == list(g(s))  # pyright: ignore[reportArgumentType]
        # NOTE: Here we diff from CPython, since our constructor is more flexible.
        _ = Deque(test_seq.IterNextOnly(s))
        with pytest.raises(TypeError):
            # pyrefly: ignore [bad-argument-type]
            _ = Deque(test_seq.IterNoNext(s))
        with pytest.raises(ZeroDivisionError):
            _ = Deque(test_seq.IterGenExc(s))


def test_iter_with_altered_data() -> None:
    d = Deque("abcdefg")
    it = iter(d)
    _ = d.pop()
    with pytest.raises(RuntimeError):
        _ = next(it)


def test_runtime_error_on_empty_deque() -> None:
    d = Deque[int]()
    it = iter(d)
    d.append(10)
    with pytest.raises(RuntimeError):
        _ = next(it)
