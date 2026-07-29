from __future__ import annotations

from typing import TYPE_CHECKING, Self, override

import pytest

from pyochain import Set, SetMut
from pyochain.abc import PyoMutableSet

from ._utils import WithSet, validate_abstract_methods

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


def test_mutable_set() -> None:
    assert isinstance(SetMut(()), PyoMutableSet)
    assert issubclass(SetMut, PyoMutableSet)
    assert not isinstance(Set(()), PyoMutableSet)
    assert not issubclass(Set, PyoMutableSet)


@pytest.mark.skip(reason="No metaclass behavior in Pyo3 yet")
def test_abstract_methods() -> None:
    validate_abstract_methods(
        PyoMutableSet, "__contains__", "__iter__", "__len__", "add", "discard"
    )


def test_issue_5647() -> None:
    # PyoMutableSet.__iand__ mutated the set during iteration
    s = WithSet("abcd")
    s &= WithSet("cdef")  # This used to fail
    assert set(s) == set("cd")


def test_issue_4920() -> None:
    # PyoMutableSet.pop() method did not work
    class MySet(PyoMutableSet[object]):
        __slots__ = ["__s"]  # pyright: ignore[reportUnannotatedClassAttribute]

        def __init__(self, items: Iterable[object] | None = None) -> None:
            if items is None:
                items = []
            self.__s: set[object] = set(items)

        @override
        def __contains__(self, v: object) -> bool:
            return v in self.__s

        @override
        def __iter__(self) -> Iterator[object]:
            return iter(self.__s)

        @override
        def __len__(self) -> int:
            return len(self.__s)

        @override
        def add(self, v: object) -> bool:  # pyright: ignore[reportIncompatibleMethodOverride]
            result = v not in self.__s
            self.__s.add(v)
            return result

        @override
        def discard(self, v: object) -> bool:  # pyright: ignore[reportIncompatibleMethodOverride]
            result = v in self.__s
            self.__s.discard(v)
            return result

        @override
        def __repr__(self) -> str:
            return f"MySet({list(self)!r})"

    items = [5, 43, 2, 1]
    s = MySet(items)
    r = s.pop()
    assert len(s) == len(items) - 1
    assert r not in s
    assert r in items


def test_issue8750() -> None:
    empty = WithSet()
    full = WithSet(range(10))
    s = WithSet(full)
    s -= s
    assert s == empty
    s = WithSet(full)
    s ^= s
    assert s == empty
    s = WithSet(full)
    s &= s
    assert s == full
    s |= s
    assert s == full


def test_set_from_iterable() -> None:
    """Verify _from_iterable overridden to an instance method works."""

    class SetUsingInstanceFromPyoIterable(PyoMutableSet[object]):
        def __init__(self, values: Iterable[object], created_by: str) -> None:
            if not created_by:
                msg = "created_by must be specified"
                raise ValueError(msg)
            self.created_by: str = created_by
            self._values: set[object] = set(values)

        @override
        def _from_iterable(self, values: Iterable[object]) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
            return type(self)(values, "from_iterable")

        @override
        def __contains__(self, value: object) -> bool:
            return value in self._values

        @override
        def __iter__(self) -> Iterator[object]:
            yield from self._values

        @override
        def __len__(self) -> int:
            return len(self._values)

        @override
        def add(self, value: object) -> None:
            self._values.add(value)

        @override
        def discard(self, value: object) -> None:
            self._values.discard(value)

    impl = SetUsingInstanceFromPyoIterable([1, 2, 3], "test")

    actual = impl - {1}
    assert isinstance(actual, SetUsingInstanceFromPyoIterable)
    assert actual.created_by == "from_iterable"
    assert {2, 3} == actual

    actual = impl | {4}
    assert isinstance(actual, SetUsingInstanceFromPyoIterable)
    assert actual.created_by == "from_iterable"
    assert {1, 2, 3, 4} == actual

    actual = impl & {2}
    assert isinstance(actual, SetUsingInstanceFromPyoIterable)
    assert actual.created_by == "from_iterable"
    assert {2} == actual

    actual = impl ^ {3, 4}
    assert isinstance(actual, SetUsingInstanceFromPyoIterable)
    assert actual.created_by == "from_iterable"
    assert {1, 2, 4} == actual

    # NOTE: ixor'ing with a list is important here: internally, __ixor__
    # only calls _from_iterable if the other value isn't already a PyoSet.
    impl ^= [3, 4]  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    assert isinstance(impl, SetUsingInstanceFromPyoIterable)
    assert impl.created_by == "test"
    assert {1, 2, 4} == impl
