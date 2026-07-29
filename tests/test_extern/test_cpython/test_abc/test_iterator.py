from __future__ import annotations

import math

import pytest

from pyochain import Dict, Iter, Seq, Set, SetMut, Vec
from pyochain.abc import PyoIterator

from ._utils import validate_abstract_methods

ERRS: list[object] = [None, 42, math.pi, 1j, b"", "", (), [], {}, set()]


@pytest.mark.parametrize("x", ERRS)
def test_iterators_errs(x: object) -> None:
    assert not isinstance(x, PyoIterator)
    assert not issubclass(type(x), PyoIterator)


OKS: tuple[PyoIterator[object], ...] = (
    Seq(()).iter(),
    Vec(()).iter(),
    Dict[object, object](()).iter(),
    Set(()).iter(),
    SetMut(()).iter(),
    Dict[object, object](()).keys().iter(),
    Dict[object, object](()).items().iter(),
    Dict[object, object](()).values().iter(),
    Iter(x for x in list[object]()),
)


@pytest.mark.parametrize("x", OKS)
def test_iterators_ok(x: PyoIterator[object]) -> None:
    assert isinstance(x, PyoIterator)
    assert issubclass(type(x), PyoIterator)


@pytest.mark.skip(reason="No metaclass behavior in Pyo3 yet")
def test_abstract_methods() -> None:
    validate_abstract_methods(PyoIterator, "__next__")


def test_issue_10565() -> None:
    # Issue 10565
    class NextOnly:
        def __next__(self) -> object:
            yield 1

    assert not isinstance(NextOnly(), PyoIterator)
