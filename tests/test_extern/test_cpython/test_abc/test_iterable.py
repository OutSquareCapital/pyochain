from __future__ import annotations

import math
from typing import TYPE_CHECKING, override

import pytest

from pyochain import Dict, Seq, Set, SetMut, Vec
from pyochain.abc import PyoIterable

from ._utils import validate_abstract_methods, validate_isinstance

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


@pytest.mark.parametrize("obj", (None, 42, math.pi, 1j))
def test_non_iterable(obj: object) -> None:
    # Check some non-iterables
    assert not isinstance(obj, PyoIterable)
    assert not issubclass(type(obj), PyoIterable)


ITERABLES: tuple[PyoIterable[object], ...] = (
    Seq[object](()),
    Vec[object]([]),
    SetMut[object](()),
    Set[object](()),
    Dict[object, object](()),
    Dict[object, object](()).keys(),
    Dict[object, object](()).items(),
    Dict[object, object](()).values(),
    Vec[object](()).iter(),
)


@pytest.mark.parametrize("obj", ITERABLES)
def test_iterable(obj: Iterable[object]) -> None:
    # Check some iterables
    assert isinstance(obj, PyoIterable)
    assert issubclass(type(obj), PyoIterable)


def test_subclassing() -> None:
    # Check direct subclassing
    class ItSubclass(PyoIterable[object]):
        @override
        def __iter__(self) -> Iterator[object]:
            return iter([])

    assert list(ItSubclass()) == []
    assert not issubclass(str, ItSubclass)


@pytest.mark.skip(reason="No metaclass behavior in Pyo3 yet")
def test_abstract_methods() -> None:
    validate_abstract_methods(PyoIterable, "__iter__")


@pytest.mark.skip(reason="No metaclass behavior in Pyo3 yet")
def test_isinstance() -> None:
    validate_isinstance(PyoIterable, "__iter__")


@pytest.mark.skip(reason="No metaclass behavior in Pyo3 yet")
def test_none_blocking() -> None:
    # Check None blocking
    class It:
        def __iter__(self) -> Iterator[object]:
            return iter([])

    class ItBlocked(It):
        __iter__ = None  # pyright: ignore[reportAssignmentType, reportUnannotatedClassAttribute]

    assert issubclass(It, PyoIterable)
    assert isinstance(It(), PyoIterable)
    assert not issubclass(ItBlocked, PyoIterable)
    assert not isinstance(ItBlocked(), PyoIterable)
