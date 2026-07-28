from collections.abc import MutableMapping

from pyochain.abc import PyoMutableMapping

from . import checks
from ._utils import ImplMutableMapping, assert_iter_eq


class _PyFail(MutableMapping[int, int]): ...  # pyright: ignore[reportImplicitAbstractClass]


class _PyOk(ImplMutableMapping, MutableMapping[int, int]): ...


class _PyoFail(PyoMutableMapping[int, int]): ...  # pyright: ignore[reportImplicitAbstractClass]


class _PyoOk(ImplMutableMapping, PyoMutableMapping[int, int]): ...


def test_mutable_mapping() -> None:

    checks.init_fail(_PyFail)
    fail = _PyoFail()  # pyright: ignore[reportAbstractUsage]
    checks.getitem_fail(fail)
    checks.len_fail(fail)
    checks.iter_fail(fail)
    checks.setitem_fail(fail)
    checks.delitem_fail(fail)
    py_ok = _PyOk()
    pyo_ok = _PyoOk()
    assert py_ok[0] == pyo_ok[0]
    assert len(py_ok) == len(pyo_ok)
    assert_iter_eq(py_ok, pyo_ok)
    pyo_ok[0] = 100
    py_ok[0] = 100
    assert py_ok[0] == pyo_ok[0]
    del pyo_ok[0]
    del py_ok[0]
    assert py_ok == pyo_ok
    assert 1 in py_ok
    assert 1 in pyo_ok
    assert_iter_eq(py_ok.keys(), pyo_ok.keys())
    assert_iter_eq(py_ok.values(), pyo_ok.values())
    assert_iter_eq(py_ok.items(), pyo_ok.items())
    assert py_ok.get(0) == pyo_ok.get(0)
    assert not (py_ok != pyo_ok)  # ruff:ignore[negate-not-equal-op]
    assert py_ok.pop(1) == pyo_ok.pop(1)
    assert py_ok.popitem() == pyo_ok.popitem()
    assert py_ok.setdefault(0, 0) == pyo_ok.setdefault(0, 0)
    assert py_ok.update({0: 0}) == pyo_ok.update({0: 0})
    py_ok.clear()
    pyo_ok.clear()
    py_ok.update({0: 0})
    pyo_ok.update({0: 0})
    assert py_ok == pyo_ok
