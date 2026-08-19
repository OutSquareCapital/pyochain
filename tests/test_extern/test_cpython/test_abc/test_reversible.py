from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, override

import pytest

from pyochain import Dict, Seq, Set, SetMut, Vec
from pyochain.abc import PyoMapping, PyoMutableMapping, PyoReversible, PyoSequence
from pyochain.collections import PyoCounter, SortedDict

from ._utils import validate_abstract_methods

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.mark.parametrize(
    "x", (None, 42, math.pi, 1j, Set[object](()), SetMut[object](()))
)
def test_non_reversible(x: object) -> None:
    # Check some non-reversibles
    assert not isinstance(x, PyoReversible)
    assert not issubclass(type(x), PyoReversible)


@pytest.mark.parametrize("x", (Vec[object]().iter(), Vec[object]().rev()))
def test_non_reversible_iterables(x: object) -> None:
    # Check some non-reversible iterables
    assert not isinstance(x, PyoReversible)
    assert not issubclass(type(x), PyoReversible)


type Reversibles = Seq[PyoReversible[object]]

# pyrefly: ignore [bad-assignment]
REVERSIBLES: Reversibles = Seq((  # pyright: ignore[reportAssignmentType]
    Seq[object](()),
    Vec[object]([]),
    SortedDict[Any, object](()),
    PyoCounter[object](()),
    Dict[object, object](()),
    PyoCounter[object]().keys(),
    PyoCounter[object]().items(),
    PyoCounter[object]().values(),
    SortedDict[Any, object]().keys(),
    SortedDict[Any, object]().items(),
    SortedDict[Any, object]().values(),
    Dict[object, object](()).keys(),
    Dict[object, object](()).items(),
    Dict[object, object](()).values(),
))


@pytest.mark.skip(reason="TODO: Correctly handle reversible for subclassing")
@pytest.mark.parametrize(
    "x", REVERSIBLES, ids=REVERSIBLES.iter().map(lambda x: x.__class__.__name__)
)
def test_reversible_iterable(x: PyoReversible[object]) -> None:
    # Check some reversible iterables
    assert isinstance(x, PyoReversible)
    assert issubclass(type(x), PyoReversible)


@pytest.mark.skip(reason="Same as test above")
def test_is_reversible_abc() -> None:
    # Check also PyoMapping, PyoMutableMapping, and PyoSequence
    assert issubclass(PyoSequence, PyoReversible)


def test_not_reversible_abc() -> None:
    assert not issubclass(PyoMapping, PyoReversible)
    assert not issubclass(PyoMutableMapping, PyoReversible)


def test_direct_subclassing() -> None:
    # Check direct subclassing

    class R(PyoReversible[object]):
        @override
        def __iter__(self) -> Iterator[object]:
            return iter([])

        @override
        def __reversed__(self) -> Iterator[object]:
            return iter([])

    assert list(reversed(R())) == []
    assert not issubclass(float, R)


@pytest.mark.skip(reason="No metaclass behavior in Pyo3 yet")
def test_abstract_methods() -> None:
    validate_abstract_methods(PyoReversible, "__reversed__", "__iter__")


class RevNoIter:
    def __reversed__(self) -> Iterator[object]:
        return reversed([])


class RevPlusIter(RevNoIter):
    def __iter__(self) -> Iterator[object]:
        return iter([])


@pytest.mark.skip(reason="Same as abc and direct subclassing tests")
def test_is_reversible_non_iterable() -> None:
    # Check reversible non-iterable (which is not PyoReversible)

    assert issubclass(RevPlusIter, PyoReversible)
    assert isinstance(RevPlusIter(), PyoReversible)


def test_not_reversible_non_iterable() -> None:
    assert not issubclass(RevNoIter, PyoReversible)
    assert not isinstance(RevNoIter(), PyoReversible)


class Rev:
    def __iter__(self) -> Iterator[object]:
        return iter([])

    def __reversed__(self) -> Iterator[object]:
        return reversed([])


class RevItBlocked(Rev):
    # pyrefly: ignore [bad-override-mutable-attribute, implicit-any-attribute]
    __iter__ = None  # pyright: ignore[reportUnannotatedClassAttribute, reportAssignmentType]


class RevRevBlocked(Rev):
    # pyrefly: ignore [bad-override-mutable-attribute, implicit-any-attribute]
    __reversed__ = None  # pyright: ignore[reportUnannotatedClassAttribute, reportAssignmentType]


@pytest.mark.skip(reason="Same as abc and direct subclassing tests")
def test_is_reversible_blocked() -> None:
    # Check None blocking
    assert issubclass(Rev, PyoReversible)
    assert isinstance(Rev(), PyoReversible)


def test_is_not_reversible_blocked() -> None:
    assert not issubclass(RevItBlocked, PyoReversible)
    assert not isinstance(RevItBlocked(), PyoReversible)
    assert not issubclass(RevRevBlocked, PyoReversible)
    assert not isinstance(RevRevBlocked(), PyoReversible)
