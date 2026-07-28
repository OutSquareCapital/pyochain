from collections.abc import MutableSequence

from pyochain.abc import PyoMutableSequence

from . import checks
from ._utils import ImplMutableSequence, assert_iter_eq


class _PyFail(MutableSequence[int]): ...  # pyright: ignore[reportImplicitAbstractClass]


class _PyOk(ImplMutableSequence, MutableSequence[int]): ...


class _PyoFail(PyoMutableSequence[int]): ...  # pyright: ignore[reportImplicitAbstractClass]


class _PyoOk(ImplMutableSequence, PyoMutableSequence[int]): ...


def test_mutable_sequence() -> None:
    checks.init_fail(_PyFail)
    fail = _PyoFail()  # pyright: ignore[reportAbstractUsage]
    checks.getitem_fail(fail)
    checks.len_fail(fail)
    checks.setitem_fail(fail)
    checks.delitem_fail(fail)
    checks.insert_fail(fail)
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
    py_ok.append(4)
    pyo_ok.append(4)
    assert_iter_eq(py_ok, pyo_ok)
    additional = [5, 6]
    pyo_ok.extend(additional)
    py_ok.extend(additional)
    assert pyo_ok.pop() == py_ok.pop()
    pyo_ok.remove(5)
    py_ok.remove(5)
    assert_iter_eq(py_ok, pyo_ok)
    pyo_ok.reverse()
    py_ok.reverse()
    assert_iter_eq(py_ok, pyo_ok)
    pyo_ok.clear()
    py_ok.clear()
    pyo_ok.insert(0, 1)
    py_ok.insert(0, 1)
    assert_iter_eq(py_ok, pyo_ok)
    pyo_ok += additional
    py_ok += additional
    assert_iter_eq(py_ok, pyo_ok)
