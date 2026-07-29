from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, override

import pytest

from pyochain import Dict, Set, SetMut, Vec
from pyochain.abc import PyoSet
from pyochain.collections import SortedSet

from ._utils import validate_abstract_methods, validate_comparison

if TYPE_CHECKING:
    from collections.abc import Iterator

type ConcreteSet[T] = Set[T] | SetMut[T] | SortedSet[Any]


@pytest.mark.parametrize("x", (Set, SetMut, SortedSet))
def test_set_ok(x: type[ConcreteSet[Any]]) -> None:
    assert isinstance(x(()), PyoSet)
    assert issubclass(x, PyoSet)


@pytest.mark.skip(reason="No metaclass behavior in Pyo3 yet")
def test_abstract_methods() -> None:
    validate_abstract_methods(PyoSet, "__contains__", "__iter__", "__len__")


def test_subclass_comparison() -> None:
    class MySet(PyoSet[object]):
        @override
        def __contains__(self, x: object) -> bool:
            return False

        @override
        def __len__(self) -> int:
            return 0

        @override
        def __iter__(self) -> Iterator[object]:
            return iter([])

    validate_comparison(MySet())


def test_hash_set() -> None:
    class OneTwoThreeSet(PyoSet[int]):
        def __init__(self) -> None:
            self.contents: list[int] = [1, 2, 3]

        @override
        def __contains__(self, x: object) -> bool:
            return x in self.contents

        @override
        def __len__(self) -> int:
            return len(self.contents)

        @override
        def __iter__(self) -> Iterator[int]:
            return iter(self.contents)

        @override
        def __hash__(self) -> int:
            return self._hash()

    a, b = OneTwoThreeSet(), OneTwoThreeSet()
    assert hash(a) == hash(b)


def test_isdisjoint_set() -> None:
    class MySet(PyoSet[object]):
        def __init__(self, itr: Iterable[object]) -> None:
            self.contents: Iterable[object] = itr

        @override
        def __contains__(self, x: object) -> bool:
            return x in self.contents

        @override
        def __iter__(self) -> Iterator[object]:
            return iter(self.contents)

        @override
        def __len__(self) -> int:
            return len(list(self.contents))

    s1 = MySet((1, 2, 3))
    s2 = MySet((4, 5, 6))
    s3 = MySet((1, 5, 6))
    assert s1.isdisjoint(s2)
    assert not s1.isdisjoint(s3)


def test_equality_set() -> None:
    class MySet(PyoSet[object]):
        def __init__(self, itr: Iterable[object]) -> None:
            self.contents: Iterable[object] = itr

        @override
        def __contains__(self, x: object) -> bool:
            return x in self.contents

        @override
        def __iter__(self) -> Iterator[object]:
            return iter(self.contents)

        @override
        def __len__(self) -> int:
            return len(list(self.contents))

    s1 = MySet((1,))
    s2 = MySet((1, 2))
    s3 = MySet((3, 4))
    s4 = MySet((3, 4))
    assert s2 > s1
    assert s1 < s2
    assert not s2 <= s1
    assert not s2 <= s3
    assert not s1 >= s2
    assert s3 == s4
    assert s2 != s3


def test_arithmetic_set() -> None:
    class MySet(PyoSet[object]):
        def __init__(self, itr: Iterable[object]) -> None:
            self.contents: Iterable[object] = itr

        @override
        def __contains__(self, x: object) -> bool:
            return x in self.contents

        @override
        def __iter__(self) -> Iterator[object]:
            return iter(self.contents)

        @override
        def __len__(self) -> int:
            return len(list(self.contents))

    s1 = MySet((1, 2, 3))
    s2 = MySet((3, 4, 5))
    s3 = s1 & s2
    assert s3 == MySet((3,))


def test_issue16373() -> None:
    # Recursion error comparing comparable and noncomparable
    # PyoSet instances
    class MyComparableSet(PyoSet[object]):
        @override
        def __contains__(self, x: object) -> bool:
            return False

        @override
        def __len__(self) -> int:
            return 0

        @override
        def __iter__(self) -> Iterator[object]:
            return iter([])

    class MyNonComparableSet(PyoSet[object]):
        @override
        def __contains__(self, x: object) -> bool:
            return False

        @override
        def __len__(self) -> int:
            return 0

        @override
        def __iter__(self) -> Iterator[object]:
            return iter([])

        @override
        def __le__(self, x: object) -> bool:
            return NotImplemented

        @override
        def __lt__(self, x: object) -> bool:
            return NotImplemented

    cs = MyComparableSet()
    ncs = MyNonComparableSet()
    assert not ncs < cs
    assert ncs <= cs
    assert not ncs > cs
    assert ncs >= cs


def test_pyoset_interoperability_with_real_sets() -> None:  # ruff:ignore[too-many-statements]
    """Issue: 8743."""

    class ListSet(PyoSet[object]):
        def __init__(self, elements: Iterable[object] = ()) -> None:
            self.data: list[object] = []
            for elem in elements:
                if elem not in self.data:
                    self.data.append(elem)

        @override
        def __contains__(self, elem: object) -> bool:
            return elem in self.data

        @override
        def __iter__(self) -> Iterator[object]:
            return iter(self.data)

        @override
        def __len__(self) -> int:
            return len(self.data)

        @override
        def __repr__(self) -> str:
            return f"PyoSet({self.data!r})"

    r1 = Set("abc")
    r2 = Set("bcd")
    r3 = Set("abcde")
    f1 = ListSet("abc")
    f2 = ListSet("bcd")
    f3 = ListSet("abcde")
    l1 = Vec("abccba")
    l2 = Vec("bcddcb")
    l3 = Vec("abcdeedcba")

    target = r1 & r2
    assert f1 & f2 == target
    assert f1 & r2 == target
    assert r2 & f1 == target
    assert f1 & l2 == target  # pyright: ignore[reportOperatorIssue]

    target = r1 | r2
    assert f1 | f2 == target
    assert f1 | r2 == target
    assert r2 | f1 == target
    assert f1 | l2 == target  # pyright: ignore[reportOperatorIssue]

    fwd_target = r1 - r2
    rev_target = r2 - r1
    assert f1 - f2 == fwd_target
    assert f2 - f1 == rev_target
    assert f1 - r2 == fwd_target
    assert f2 - r1 == rev_target
    assert r1 - f2 == fwd_target
    assert r2 - f1 == rev_target
    assert f1 - l2 == fwd_target  # pyright: ignore[reportOperatorIssue]
    assert f2 - l1 == rev_target  # pyright: ignore[reportOperatorIssue]

    target = r1 ^ r2
    assert f1 ^ f2 == target
    assert f1 ^ r2 == target
    assert r2 ^ f1 == target
    assert f1 ^ l2 == target  # pyright: ignore[reportOperatorIssue]

    # Don't change the following to use assertLess or other
    # "more specific" unittest assertions.  The current
    # assertTrue/assertFalse style makes the pattern of test
    # case combinations clear and allows us to know for sure
    # the exact operator being invoked.

    # proper subset
    assert f1 < f3
    assert not f1 < f1
    assert not f1 < f2
    assert r1 < f3
    assert not r1 < f1
    assert not r1 < f2
    assert r1 < r3
    assert not r1 < r1
    assert not r1 < r2
    with pytest.raises(TypeError):
        _ = f1 < l3  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    with pytest.raises(TypeError):
        _ = f1 < l1  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    with pytest.raises(TypeError):
        _ = f1 < l2  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]

    # any subset
    assert f1 <= f3
    assert f1 <= f1
    assert not f1 <= f2
    assert r1 <= f3
    assert r1 <= f1
    assert not r1 <= f2
    assert r1 <= r3
    assert r1 <= r1
    assert not r1 <= r2
    with pytest.raises(TypeError):
        _ = f1 <= l3  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    with pytest.raises(TypeError):
        _ = f1 <= l1  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    with pytest.raises(TypeError):
        _ = f1 <= l2  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]

    # proper superset
    assert f3 > f1
    assert not f1 > f1
    assert not f2 > f1
    assert r3 > r1
    assert not f1 > r1
    assert not f2 > r1
    assert r3 > r1
    assert not r1 > r1
    assert not r2 > r1
    with pytest.raises(TypeError):
        _ = f1 > l3  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    with pytest.raises(TypeError):
        _ = f1 > l1  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    with pytest.raises(TypeError):
        _ = f1 > l2  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]

    # any superset
    assert f3 >= f1
    assert f1 >= f1
    assert not f2 >= f1
    assert r3 >= r1
    assert f1 >= r1
    assert not f2 >= r1
    assert r3 >= r1
    assert r1 >= r1
    assert not r2 >= r1
    with pytest.raises(TypeError):
        _ = f1 >= l3  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    with pytest.raises(TypeError):
        _ = f1 >= l1  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    with pytest.raises(TypeError):
        _ = f1 >= l2  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]

    # equality
    assert f1 == f1
    assert r1 == f1
    assert f1 == r1
    assert f1 != f3
    assert r1 != f3
    assert f1 != r3
    assert f1 != l3
    assert f1 != l1
    assert f1 != l2

    # inequality
    assert f1 == f1
    assert r1 == f1
    assert f1 == r1
    assert f1 != f3
    assert r1 != f3
    assert f1 != r3
    assert f1 != l3
    assert f1 != l1
    assert f1 != l2


SETS = Dict[str, Iterable[object]]({
    "empty dict": {},
    "set with 1": {1},
    "set with None": {None},
    "set with -1": {-1},
    "set with 0.0": {0.0},
    "set with 'abc'": {"abc"},
    "set with 1, 2, 3": {1, 2, 3},
    "set with large numbers": {10**100, 10**101},
    "set with strings": {"a", "b", "ab", ""},
    "set with booleans": {False, True},
    "set with objects": {object(), object(), object()},
    "set with nan": {float("nan")},
    "empty frozenset": {frozenset[object]()},
    "range 0-999": {*range(1000)},
    "range 0-999 excluding 100, 200, 300": {*range(1000)} - {100, 200, 300},
    "range near sys.maxsize": {*range(sys.maxsize - 10, sys.maxsize + 10)},
})


@pytest.mark.parametrize("s", SETS.values(), ids=SETS.keys())
def test_set_hash_matches_frozenset(s: Iterable[object]) -> None:
    fs = Set(s)
    assert hash(fs) == PyoSet._hash(fs), s  # pyright: ignore[reportUnknownMemberType]
