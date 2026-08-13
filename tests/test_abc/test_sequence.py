from collections.abc import Sequence

from pyochain.abc import PyoSequence

from . import checks
from ._utils import ImplSequence, assert_iter_eq


class _PyFail(Sequence[int]): ...  # pyright: ignore[reportImplicitAbstractClass]


class _PyOk(ImplSequence, Sequence[int]): ...


class _PyoFail(PyoSequence[int]): ...  # pyright: ignore[reportImplicitAbstractClass]


class _PyoOk(ImplSequence, PyoSequence[int]): ...


def test_sequence() -> None:

    checks.init_fail(_PyFail)
    fail = _PyoFail()  # pyright: ignore[reportAbstractUsage]
    checks.getitem_fail(fail)
    checks.len_fail(fail)
    py_ok = _PyOk()
    pyo_ok = _PyoOk()
    assert py_ok[0] == pyo_ok[0]
    assert len(py_ok) == len(pyo_ok)
    assert_iter_eq(py_ok, pyo_ok)
    assert 1 in py_ok
    assert 1 in pyo_ok
    assert_iter_eq(reversed(py_ok), reversed(pyo_ok))
    assert py_ok.index(2) == pyo_ok.index(2)
    assert py_ok.count(2) == pyo_ok.count(2)
