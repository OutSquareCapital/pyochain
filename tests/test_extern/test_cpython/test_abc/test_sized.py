from __future__ import annotations

import math

import pytest

from pyochain import Dict, Seq, Set, SetMut, Vec
from pyochain.abc import PyoSized

from ._utils import validate_abstract_methods, validate_isinstance

ERRS: tuple[object, ...] = (None, 42, math.pi, 1j, (x for x in ()))


@pytest.mark.parametrize("x", ERRS)
def test_sized_err(x: object) -> None:
    assert not isinstance(x, PyoSized)
    assert not issubclass(type(x), PyoSized)


OKS: Seq[PyoSized] = Seq((
    Seq[object](),
    Vec[object](),
    Set[object](()),
    SetMut[object](()),
    Dict[object, object](()),
    Dict[object, object](()).keys(),
    Dict[object, object](()).items(),
    Dict[object, object](()).values(),
))


@pytest.mark.skip(reason="Same issue as with `Reversible` subclassing tests")
@pytest.mark.parametrize("x", OKS, ids=OKS.iter().map(lambda x: x.__class__.__name__))
def test_sized_ok(x: PyoSized) -> None:
    assert isinstance(x, PyoSized)
    assert issubclass(type(x), PyoSized)


@pytest.mark.skip(reason="No metaclass behavior in Pyo3 yet")
def test_abstract_methods() -> None:
    validate_abstract_methods(PyoSized, "__len__")
    validate_isinstance(PyoSized, "__len__")
