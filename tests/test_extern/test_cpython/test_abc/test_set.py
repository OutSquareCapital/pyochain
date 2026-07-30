from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, NamedTuple, override

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


class InteropData(NamedTuple):
    """Issue: 8743."""

    r1: Set[str]
    r2: Set[str]
    r3: Set[str]
    f1: ListSet
    f2: ListSet
    f3: ListSet
    l1: Vec[str]
    l2: Vec[str]
    l3: Vec[str]


OP = InteropData(
    Set("abc"),
    Set("bcd"),
    Set("abcde"),
    ListSet("abc"),
    ListSet("bcd"),
    ListSet("abcde"),
    Vec("abccba"),
    Vec("bcddcb"),
    Vec("abcdeedcba"),
)


def test_interop_and() -> None:
    target = OP.r1 & OP.r2
    assert OP.f1 & OP.f2 == target
    assert OP.f1 & OP.r2 == target
    assert OP.r2 & OP.f1 == target
    assert OP.f1 & OP.l2 == target


def test_interop_or() -> None:
    target = OP.r1 | OP.r2
    assert OP.f1 | OP.f2 == target
    assert OP.f1 | OP.r2 == target
    assert OP.r2 | OP.f1 == target
    assert OP.f1 | OP.l2 == target


def test_interop_sub() -> None:
    fwd_target = OP.r1 - OP.r2
    rev_target = OP.r2 - OP.r1
    assert OP.f1 - OP.f2 == fwd_target
    assert OP.f2 - OP.f1 == rev_target
    assert OP.f1 - OP.r2 == fwd_target
    assert OP.f2 - OP.r1 == rev_target
    assert OP.r1 - OP.f2 == fwd_target
    assert OP.r2 - OP.f1 == rev_target
    assert OP.f1 - OP.l2 == fwd_target
    assert OP.f2 - OP.l1 == rev_target


def test_interop_xor() -> None:
    target = OP.r1 ^ OP.r2
    assert OP.f1 ^ OP.f2 == target
    assert OP.f1 ^ OP.r2 == target
    assert OP.r2 ^ OP.f1 == target
    assert OP.f1 ^ OP.l2 == target


def test_interop_lt() -> None:
    # Don't change the following to use assertLess or other
    # "more specific" unittest assertions.  The current
    # assertTrue/assertFalse style makes the pattern of test
    # case combinations clear and allows us to know for sure
    # the exact operator being invoked.

    # proper subset
    assert OP.f1 < OP.f3
    assert not OP.f1 < OP.f1
    assert not OP.f1 < OP.f2
    assert OP.r1 < OP.f3
    assert not OP.r1 < OP.f1
    assert not OP.r1 < OP.f2
    assert OP.r1 < OP.r3
    assert not OP.r1 < OP.r1
    assert not OP.r1 < OP.r2
    with pytest.raises(TypeError):
        # pyrefly: ignore [unsupported-operation]
        _ = OP.f1 < OP.l3  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    with pytest.raises(TypeError):
        # pyrefly: ignore [unsupported-operation]
        _ = OP.f1 < OP.l1  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    with pytest.raises(TypeError):
        # pyrefly: ignore [unsupported-operation]
        _ = OP.f1 < OP.l2  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]


def test_interop_le() -> None:
    # any subset
    assert OP.f1 <= OP.f3
    assert OP.f1 <= OP.f1
    assert not OP.f1 <= OP.f2
    assert OP.r1 <= OP.f3
    assert OP.r1 <= OP.f1
    assert not OP.r1 <= OP.f2
    assert OP.r1 <= OP.r3
    assert OP.r1 <= OP.r1
    assert not OP.r1 <= OP.r2
    with pytest.raises(TypeError):
        # pyrefly: ignore [unsupported-operation]
        _ = OP.f1 <= OP.l3  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    with pytest.raises(TypeError):
        # pyrefly: ignore [unsupported-operation]
        _ = OP.f1 <= OP.l1  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    with pytest.raises(TypeError):
        # pyrefly: ignore [unsupported-operation]
        _ = OP.f1 <= OP.l2  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]


def test_interop_gt() -> None:
    # proper superset
    assert OP.f3 > OP.f1
    assert not OP.f1 > OP.f1
    assert not OP.f2 > OP.f1
    assert OP.r3 > OP.r1
    assert not OP.f1 > OP.r1
    assert not OP.f2 > OP.r1
    assert OP.r3 > OP.r1
    assert not OP.r1 > OP.r1
    assert not OP.r2 > OP.r1
    with pytest.raises(TypeError):
        # pyrefly: ignore [unsupported-operation]
        _ = OP.f1 > OP.l3  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    with pytest.raises(TypeError):
        # pyrefly: ignore [unsupported-operation]
        _ = OP.f1 > OP.l1  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    with pytest.raises(TypeError):
        # pyrefly: ignore [unsupported-operation]
        _ = OP.f1 > OP.l2  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]


def test_interop_ge() -> None:
    # any superset
    assert OP.f3 >= OP.f1
    assert OP.f1 >= OP.f1
    assert not OP.f2 >= OP.f1
    assert OP.r3 >= OP.r1
    assert OP.f1 >= OP.r1
    assert not OP.f2 >= OP.r1
    assert OP.r3 >= OP.r1
    assert OP.r1 >= OP.r1
    assert not OP.r2 >= OP.r1
    with pytest.raises(TypeError):
        # pyrefly: ignore [unsupported-operation]
        _ = OP.f1 >= OP.l3  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    with pytest.raises(TypeError):
        # pyrefly: ignore [unsupported-operation]
        _ = OP.f1 >= OP.l1  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    with pytest.raises(TypeError):
        # pyrefly: ignore [unsupported-operation]
        _ = OP.f1 >= OP.l2  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]


def test_interop_eq() -> None:
    # equality
    assert OP.f1 == OP.f1
    assert OP.r1 == OP.f1
    assert OP.f1 == OP.r1
    assert OP.f1 != OP.f3
    assert OP.r1 != OP.f3
    assert OP.f1 != OP.r3
    assert OP.f1 != OP.l3
    assert OP.f1 != OP.l1
    assert OP.f1 != OP.l2


def test_interop_ne() -> None:
    # inequality
    assert OP.f1 == OP.f1
    assert OP.r1 == OP.f1
    assert OP.f1 == OP.r1
    assert OP.f1 != OP.f3
    assert OP.r1 != OP.f3
    assert OP.f1 != OP.r3
    assert OP.f1 != OP.l3
    assert OP.f1 != OP.l1
    assert OP.f1 != OP.l2


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
