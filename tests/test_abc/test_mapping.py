from collections.abc import Mapping

from pyochain.abc import PyoMapping

from . import checks
from ._utils import ImplMapping, assert_iter_eq


class _PyFail(Mapping[int, int]): ...  # pyright: ignore[reportImplicitAbstractClass]


class _PyOk(ImplMapping, Mapping[int, int]): ...


class _PyoFail(PyoMapping[int, int]): ...  # pyright: ignore[reportImplicitAbstractClass]


class _PyoOk(ImplMapping, PyoMapping[int, int]): ...


def test_mapping() -> None:
    checks.init_fail(_PyFail)
    fail = _PyoFail()  # pyright: ignore[reportAbstractUsage]
    checks.getitem_fail(fail)
    checks.len_fail(fail)
    checks.iter_fail(fail)
    py_ok = _PyOk()
    pyo_ok = _PyoOk()
    assert py_ok[0] == pyo_ok[0]
    assert len(py_ok) == len(pyo_ok)
    assert_iter_eq(py_ok, pyo_ok)
    assert 1 in py_ok
    assert 1 in pyo_ok
    assert_iter_eq(py_ok.keys(), pyo_ok.keys())
    assert_iter_eq(py_ok.values(), pyo_ok.values())
    assert_iter_eq(py_ok.items(), pyo_ok.items())
    assert py_ok.get(0) == pyo_ok.get(0)
    assert py_ok == pyo_ok
    assert not (py_ok != pyo_ok)  # ruff:ignore[negate-not-equal-op]
