from collections.abc import Sized

from pyochain.abc import PyoSized

from . import checks
from ._utils import ImplSized


class _PyFail(Sized): ...  # pyright: ignore[reportImplicitAbstractClass]


class _PyOk(ImplSized, Sized): ...


class _PyoFail(PyoSized): ...  # pyright: ignore[reportImplicitAbstractClass]


class _PyoOk(ImplSized, PyoSized): ...


def test_sized() -> None:
    checks.init_fail(_PyFail)
    checks.len_fail(_PyoFail())  # pyright: ignore[reportAbstractUsage]


def test_len() -> None:
    pyo = _PyoOk()
    py = _PyOk()
    assert len(py) == len(pyo)
    assert len(py) == pyo.len()
    assert len(py) == pyo.__len__()
