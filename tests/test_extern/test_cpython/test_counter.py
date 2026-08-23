"""This modules contains tests that for the most part have been adapted from the CPython test suite for `collections.Counter`.

Most modifications entails the conversion from `unittest` to `pytest`, the metaclass behavior, and the `dict` subclassing differences.

There are also various performance optimizations. The original tests are a bit innefficient.

The original lives at:

https://github.com/python/cpython/blob/2cd5b79284dd2331cbd9e11afabbfbf7e906103d/Lib/test/test_collections.py#L2076
"""

import copy
import itertools
import pickle
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from random import randrange
from typing import Any, override

import pytest

from pyochain import Seq
from pyochain.abc import PyoMutableMapping
from pyochain.collections import PyoCounter

type Ops[T] = list[
    tuple[
        Callable[[PyoCounter[str], PyoCounter[str]], PyoCounter[str]],
        Callable[[T, T], T],
    ]
]

CP = PyoCounter({"a": 1, "b": 0, "c": 1})
CQ = PyoCounter({"c": 1, "d": 0, "e": 1})
SP = {"a", "c"}
SQ = {"c", "e"}
FUNCS = {
    "add equals union": lambda: set(CP + CQ) == SP | SQ,
    "sub": lambda: set(CP - CQ) == SP - SQ,
    "union": lambda: set(CP | CQ) == SP | SQ,
    "intersection": lambda: set(CP & CQ) == SP & SQ,
    "equals": lambda: (CP == CQ) == (SP == SQ),
    "not equals": lambda: (CP != CQ) == (SP != SQ),
    "less than or equal": lambda: (CP <= CQ) == (SP <= SQ),
    "less than": lambda: (CP < CQ) == (SP < SQ),
    "greater than or equal": lambda: (CP >= CQ) == (SP >= SQ),
    "greater than": lambda: (CP > CQ) == (SP > SQ),
}


@pytest.mark.parametrize("fn", FUNCS.values(), ids=FUNCS.keys())
def test_counter(fn: Callable[[], bool]) -> None:
    assert fn()


def test_basics() -> None:
    c = PyoCounter("abcaba")
    assert c == PyoCounter({"a": 3, "b": 2, "c": 1})
    assert c == PyoCounter(a=3, b=2, c=1)
    assert isinstance(c, PyoMutableMapping)
    assert isinstance(c, MutableMapping)
    assert isinstance(c, Mapping)
    assert issubclass(PyoCounter, PyoMutableMapping)
    assert issubclass(PyoCounter, MutableMapping)
    assert issubclass(PyoCounter, Mapping)
    assert len(c) == 3
    assert sum(c.values()) == 6
    assert list(c.values()) == [3, 2, 1]
    assert list(c.keys()) == ["a", "b", "c"]
    assert list(c) == ["a", "b", "c"]
    assert list(c.items()) == [("a", 3), ("b", 2), ("c", 1)]
    assert c["b"] == 2
    assert c["z"] == 0
    assert "c" in c
    assert "z" not in c
    assert c.get("b", 10) == 2
    assert c.get("z", 10) == 10
    assert c == {"a": 3, "b": 2, "c": 1}
    assert repr(c) == "PyoCounter({'a': 3, 'b': 2, 'c': 1})"
    assert c.most_common() == [("a", 3), ("b", 2), ("c", 1)]
    for i in range(5):
        assert c.most_common(i) == [("a", 3), ("b", 2), ("c", 1)][:i]
    assert "".join(c.elements()) == "aaabbc"
    c["a"] += 1  # increment an existing value
    c["b"] -= 2  # sub existing value to zero
    del c["c"]  # remove an entry
    del c["c"]  # make sure that del doesn't raise KeyError
    c["d"] -= 2  # sub from a missing value
    c["e"] = -5  # directly assign a missing value
    c["f"] += 4  # add to a missing value
    assert c == {"a": 4, "b": 0, "d": -2, "e": -5, "f": 4}
    assert "".join(c.elements()) == "aaaaffff"
    assert c.pop("f") == 4
    assert "f" not in c
    for _ in range(3):
        elem, _cnt = c.popitem()
        assert elem not in c
    c.clear()
    assert c == {}
    assert repr(c) == "PyoCounter()"
    with pytest.raises(TypeError):
        _ = hash(c)
    c.update({"a": 5, "b": 3})
    c.update(c=1)
    c.update(PyoCounter("a" * 50 + "b" * 30))
    c.update()  # test case with no args


@pytest.mark.skip(
    reason="We don't support the __init__ reinitialization behavior of CPython's Counter"
)
def test_init_reinitialization() -> None:
    c = PyoCounter({"a": 5, "b": 3, "c": 1})
    c.__init__("a" * 500 + "b" * 300)  # pyright: ignore[reportCallIssue]
    c.__init__("cdc")  # pyright: ignore[reportCallIssue]
    c.__init__()
    assert c == {"a": 555, "b": 333, "c": 3, "d": 1}
    assert c.setdefault("d", 5) == 1
    assert c["d"] == 1
    assert c.setdefault("e", 5) == 5
    assert c["e"] == 5


def test_update_reentrant_add_clears_counter() -> None:
    c = PyoCounter[object]()
    key = object()

    class Evil(int):
        @override
        def __add__(self, other: object) -> object:  # pyright: ignore[reportIncompatibleMethodOverride]
            c.clear()
            return NotImplemented

    c[key] = Evil()
    c.update([key])
    assert c[key] == 1


def test_init() -> None:
    assert list(PyoCounter(self=42).items()) == [("self", 42)]
    assert list(PyoCounter(iterable=42).items()) == [("iterable", 42)]
    # pyrefly: ignore [bad-argument-type]
    assert list(PyoCounter[str](iterable=None).items()) == [("iterable", None)]  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        # pyrefly: ignore [no-matching-overload]
        _ = PyoCounter[object](42)  # pyright: ignore[reportCallIssue, reportArgumentType]
    with pytest.raises(TypeError):
        # pyrefly: ignore [no-matching-overload]
        _ = PyoCounter[object]((), ())  # pyright: ignore[reportCallIssue]
    with pytest.raises(TypeError):
        # pyrefly: ignore [no-matching-overload]
        _ = PyoCounter[object].__init__()  # pyright: ignore[reportCallIssue, reportUnknownVariableType]


def test_total() -> None:
    assert PyoCounter(a=10, b=5, c=0).total() == 15


def test_order_preservation() -> None:
    # Input order dictates items() order
    assert list(PyoCounter("abracadabra").items()) == [
        ("a", 5),
        ("b", 2),
        ("r", 2),
        ("c", 1),
        ("d", 1),
    ]

    # Verify retention of order even when all counts are equal
    assert list(PyoCounter("xyzpdqqdpzyx").items()) == [
        ("x", 2),
        ("y", 2),
        ("z", 2),
        ("p", 2),
        ("d", 2),
        ("q", 2),
    ]

    # Input order dictates elements() order
    assert list(PyoCounter("abracadabra simsalabim").elements()) == [
        "a",
        "a",
        "a",
        "a",
        "a",
        "a",
        "a",
        "b",
        "b",
        "b",
        "r",
        "r",
        "c",
        "d",
        " ",
        "s",
        "s",
        "i",
        "i",
        "m",
        "m",
        "l",
    ]

    # Math operations order first by the order encountered in the left
    # operand and then by the order encountered in the right operand.
    ps = "aaabbcdddeefggghhijjjkkl"
    qs = "abbcccdeefffhkkllllmmnno"
    order = {letter: i for i, letter in enumerate(dict.fromkeys(ps + qs))}

    def correctly_ordered(seq: Iterable[str]) -> bool:
        """Return true if the letters occur in the expected order."""
        positions = [order[letter] for letter in seq]
        return positions == sorted(positions)

    p, q = PyoCounter(ps), PyoCounter(qs)
    assert correctly_ordered(+p)
    assert correctly_ordered(-p)
    assert correctly_ordered(p + q)
    assert correctly_ordered(p - q)
    assert correctly_ordered(p | q)
    assert correctly_ordered(p & q)
    assert correctly_ordered(p ^ q)

    p, q = PyoCounter(ps), PyoCounter(qs)
    p += q
    assert correctly_ordered(p)

    p, q = PyoCounter(ps), PyoCounter(qs)
    p -= q
    assert correctly_ordered(p)

    p, q = PyoCounter(ps), PyoCounter(qs)
    p |= q
    assert correctly_ordered(p)

    p, q = PyoCounter(ps), PyoCounter(qs)
    p &= q
    assert correctly_ordered(p)

    p, q = PyoCounter(ps), PyoCounter(qs)
    p ^= q
    assert correctly_ordered(p)

    p, q = PyoCounter(ps), PyoCounter(qs)
    p.update(q)
    assert correctly_ordered(p)

    p, q = PyoCounter(ps), PyoCounter(qs)
    p.subtract(q)
    assert correctly_ordered(p)


def test_update() -> None:
    c = PyoCounter[object]()
    c.update(self=42)
    assert list(c.items()) == [("self", 42)]
    c = PyoCounter[object]()
    c.update(iterable=42)
    assert list(c.items()) == [("iterable", 42)]
    c = PyoCounter[object]()
    # pyrefly: ignore [bad-argument-type]
    c.update(iterable=None)  # pyright: ignore[reportArgumentType]
    assert list(c.items()) == [("iterable", None)]
    with pytest.raises(TypeError):
        # pyrefly: ignore [no-matching-overload]
        c.update(42)  # pyright: ignore[reportCallIssue, reportArgumentType]
    with pytest.raises(TypeError):
        # pyrefly: ignore [no-matching-overload]
        c.update({}, {})  # pyright: ignore[reportCallIssue]
    with pytest.raises(TypeError):
        # pyrefly: ignore [no-matching-overload]
        PyoCounter.update()  # pyright: ignore[reportCallIssue, reportUnknownMemberType]


def test_copying() -> None:
    # Check that counters are copyable, deepcopyable, picklable, and
    # have a repr/eval round-trip
    words = PyoCounter(["which", "witch", "had", "which", "witches", "wrist", "watch"])
    _check_words(words, words.copy())
    _check_words(words, copy.copy(words))


@pytest.mark.skip(reason="Pickling and deepcopying are not supported by pyo3 currently")
def test_pickling_and_deepcopy() -> None:
    words = PyoCounter(["which", "witch", "had", "which", "witches", "wrist", "watch"])
    _check_words(words, copy.deepcopy(words))
    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        _check_words(words, pickle.loads(pickle.dumps(words, proto)))
    _check_words(words, eval(repr(words)))
    update_test = PyoCounter[object]()
    update_test.update(words)
    _check_words(words, update_test)
    _check_words(words, PyoCounter(words))


def _check_words(words: PyoCounter[str], dup: Any) -> None:  # ruff:ignore[any-type]  # pyright: ignore[reportAny]
    msg = f"\ncopy: {dup}\nwords: {words}"
    assert dup is not words, msg
    assert dup == words


def test_conversions() -> None:
    # Convert to: set, list, dict
    s = "she sells sea shells by the sea shore"
    assert sorted(PyoCounter(s).elements()) == sorted(s)
    assert sorted(PyoCounter(s)) == sorted(set(s))
    assert dict(PyoCounter(s)) == dict(PyoCounter(s).items())
    assert set(PyoCounter(s)) == set(s)


def test_invariant_for_the_in_operator() -> None:
    c = PyoCounter(a=10, b=-2, c=0)
    for elem in c:
        assert elem in c
        assert elem in c


def test_multiset_operations() -> None:
    # Verify that adding a zero counter will strip zeros and negatives
    c = PyoCounter(a=10, b=-2, c=0) + PyoCounter[str]()
    assert dict(c) == {"a": 10}

    ops_1: Ops[int] = [
        (PyoCounter[str].__add__, lambda x, y: max(0, x + y)),
        (PyoCounter[str].__sub__, lambda x, y: max(0, x - y)),
        (PyoCounter[str].__or__, lambda x, y: max(0, x, y)),
        (PyoCounter[str].__and__, lambda x, y: max(0, min(x, y))),
        (PyoCounter[str].__xor__, lambda x, y: max(0, max(x, y) - min(x, y))),
    ]
    elements = "abcd"
    for _i in range(250):
        # test random pairs of multisets
        p = PyoCounter({elem: randrange(-2, 4) for elem in elements})
        p.update(e=1, f=-1, g=0)
        q = PyoCounter({elem: randrange(-2, 4) for elem in elements})
        q.update(h=1, i=-1, j=0)
        for counterop, numberop in ops_1:
            result = counterop(p, q)
            for x in elements:
                assert numberop(p[x], q[x]) == result[x], (counterop, x, p, q)
            # verify that results exclude non-positive counts
            assert all(x > 0 for x in result.values())

    elements = "abcdef"
    ops_2: Ops[set[str]] = [
        (PyoCounter[str].__sub__, set[str].__sub__),
        (PyoCounter[str].__or__, set[str].__or__),
        (PyoCounter[str].__and__, set[str].__and__),
        (PyoCounter[str].__xor__, set[str].__xor__),
    ]
    for _i in range(100):
        # verify that random multisets with no repeats are exactly like sets
        p = PyoCounter({elem: randrange(0, 2) for elem in elements})
        q = PyoCounter({elem: randrange(0, 2) for elem in elements})
        for counterop, setop in ops_2:
            counter_result = counterop(p, q)
            set_result = setop(set(p.elements()), set(q.elements()))
            assert counter_result == dict.fromkeys(set_result, 1)


IN_PLACE_OPS = Seq((
    (PyoCounter[str].__iadd__, PyoCounter[str].__add__),
    (PyoCounter[str].__isub__, PyoCounter[str].__sub__),
    (PyoCounter[str].__ior__, PyoCounter[str].__or__),
    (PyoCounter[str].__iand__, PyoCounter[str].__and__),
    (PyoCounter[str].__ixor__, PyoCounter[str].__xor__),
))
type PyoCounterFn = Callable[[PyoCounter[str], PyoCounter[str]], PyoCounter[str]]


@pytest.mark.parametrize(
    "ops",
    IN_PLACE_OPS,
    ids=IN_PLACE_OPS
    .iter()
    .map_star(lambda f, f1: f.__name__ + " and " + f1.__name__)
    .collect(tuple),
)
def test_inplace_operations(ops: tuple[PyoCounterFn, PyoCounterFn]) -> None:
    elements = "abcd"
    inplace_op, regular_op = ops

    for _i in range(50):
        # test random pairs of multisets
        p = PyoCounter({elem: randrange(-2, 4) for elem in elements})
        p.update(e=1, f=-1, g=0)
        q = PyoCounter({elem: randrange(-2, 4) for elem in elements})
        q.update(h=1, i=-1, j=0)
        c = p.copy()
        c_id = id(c)
        regular_result = regular_op(c, q)
        inplace_result = inplace_op(c, q)
        assert inplace_result == regular_result
        assert id(inplace_result) == c_id


def test_subtract() -> None:
    c = PyoCounter(a=-5, b=0, c=5, d=10, e=15, g=40)
    c.subtract(a=1, b=2, c=-3, d=10, e=20, f=30, h=-50)
    assert c == PyoCounter(a=-6, b=-2, c=8, d=0, e=-5, f=-30, g=40, h=50)
    c = PyoCounter(a=-5, b=0, c=5, d=10, e=15, g=40)
    c.subtract(PyoCounter(a=1, b=2, c=-3, d=10, e=20, f=30, h=-50))
    assert c == PyoCounter(a=-6, b=-2, c=8, d=0, e=-5, f=-30, g=40, h=50)
    c = PyoCounter("aaabbcd")
    c.subtract("aaaabbcce")
    assert c == PyoCounter(a=-1, b=0, c=-1, d=1, e=-1)

    c = PyoCounter[str]()
    c.subtract(self=42)
    assert list(c.items()) == [("self", -42)]
    c = PyoCounter[str]()
    c.subtract(iterable=42)
    assert list(c.items()) == [("iterable", -42)]
    with pytest.raises(TypeError):
        # pyrefly: ignore [no-matching-overload]
        c.subtract(42)  # pyright: ignore[reportCallIssue, reportArgumentType]
    with pytest.raises(TypeError):
        # pyrefly: ignore [no-matching-overload]
        c.subtract({}, {})  # pyright: ignore[reportCallIssue]
    with pytest.raises(TypeError):
        # pyrefly: ignore [no-matching-overload]
        PyoCounter[str].subtract()  # pyright: ignore[reportCallIssue]


def test_unary() -> None:
    c = PyoCounter(a=-5, b=0, c=5, d=10, e=15, g=40)
    assert dict(+c) == {"c": 5, "d": 10, "e": 15, "g": 40}
    assert dict(-c) == {"a": 5}


def test_repr_nonsortable() -> None:
    # pyrefly: ignore [bad-argument-type]
    c = PyoCounter[str](a=2, b=None)  # pyright: ignore[reportArgumentType]
    r = repr(c)
    assert "'a': 2" in r
    assert "'b': None" in r


def test_multiset_operations_equivalent_to_set_operations() -> None:
    # When the multiplicities are all zero or one, multiset operations
    # are guaranteed to be equivalent to the corresponding operations
    # for regular sets.
    s = list(itertools.product(("a", "b"), range(2)))
    powerset = itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )
    counters = [PyoCounter(dict(groups)) for groups in powerset]
    for cp, cq in itertools.product(counters, repeat=2):
        sp = set(cp.elements())
        sq = set(cq.elements())
        assert set(cp + cq) == sp | sq
        assert set(cp - cq) == sp - sq
        assert set(cp | cq) == sp | sq
        assert set(cp & cq) == sp & sq
        assert set(cp ^ cq) == sp ^ sq
        assert (cp == cq) == (sp == sq)
        assert (cp != cq) == (sp != sq)
        assert (cp <= cq) == (sp <= sq)
        assert (cp >= cq) == (sp >= sq)
        assert (cp < cq) == (sp < sq)
        assert (cp > cq) == (sp > sq)


def test_eq() -> None:
    assert PyoCounter(a=3, b=2, c=0) == PyoCounter("ababa")
    assert PyoCounter(a=3, b=2) != PyoCounter("babab")


def test_le() -> None:
    assert PyoCounter(a=3, b=2, c=0) <= PyoCounter("ababa")
    assert PyoCounter() <= PyoCounter(c=1)
    assert not PyoCounter() <= PyoCounter(c=-1)
    assert not PyoCounter(a=3, b=2) <= PyoCounter("babab")


def test_lt() -> None:
    assert PyoCounter(a=3, b=1, c=0) < PyoCounter("ababa")
    assert not PyoCounter(a=3, b=2, c=0) < PyoCounter("ababa")


def test_ge() -> None:
    assert PyoCounter(a=2, b=1, c=0) >= PyoCounter("aab")
    assert PyoCounter() >= PyoCounter(c=-1)
    assert not PyoCounter() >= PyoCounter(c=1)
    assert not PyoCounter(a=3, b=2, c=0) >= PyoCounter("aabd")


def test_gt() -> None:
    assert PyoCounter(a=3, b=2, c=0) > PyoCounter("aab")
    assert not PyoCounter(a=2, b=1, c=0) > PyoCounter("aab")


def test_symmetric_difference() -> None:
    population = (-4, -1, 0, 1, 4)

    for a, b1, b2, c in itertools.product(population, repeat=4):
        p = PyoCounter(a=a, b=b1)
        q = PyoCounter(b=b2, c=c)
        r = p ^ q

        # Elementwise invariants
        for k in ("a", "b", "c"):
            assert r[k] == max(p[k], q[k]) - min(p[k], q[k])
            assert r[k] == abs(p[k] - q[k])

        # Invariant for all positive, negative, and zero counts
        assert r == p - q | q - p

        # Invariant for non-negative counts
        if a >= 0 and b1 >= 0 and b2 >= 0 and c >= 0:
            assert r == (p | q) - (p & q)

        # Zeros and negatives eliminated
        assert all(value > 0 for value in r.values())

        # Output preserves input order:  p first and then q
        keys = list(p) + list(q)
        indices = [keys.index(k) for k in r]
        assert indices == sorted(indices)

        # Inplace operation matches binary operation
        pp = PyoCounter(p)
        qq = PyoCounter(q)
        pp ^= qq
        assert pp == r
