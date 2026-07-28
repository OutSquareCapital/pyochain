from collections.abc import Set as AbstractSet

import pytest

from pyochain.abc import PyoSet
from tests.test_abc._utils import ImplContainer, ImplSized

from . import checks
from ._utils import ImplCollection, ImplIter, assert_iter_eq


class _PyFail(AbstractSet[int]): ...  # pyright: ignore[reportImplicitAbstractClass]


class _PyOk(ImplCollection, AbstractSet[int]): ...


class _PyoFail(PyoSet[int]): ...  # pyright: ignore[reportImplicitAbstractClass]


class _PyoOk(ImplCollection, PyoSet[int]): ...


def test_set() -> None:
    checks.init_fail(_PyFail)
    fail = _PyoFail()  # pyright: ignore[reportAbstractUsage]
    checks.len_fail(fail)
    checks.contains_fail(fail)
    checks.iter_fail(fail)
    py_ok = _PyOk()
    pyo_ok = _PyoOk()
    assert len(py_ok) == len(pyo_ok)
    assert 1 in py_ok
    assert 1 in pyo_ok
    assert_iter_eq(py_ok, pyo_ok)
    assert py_ok <= pyo_ok
    assert py_ok >= pyo_ok
    assert py_ok == pyo_ok
    assert not (py_ok != pyo_ok)  # ruff:ignore[negate-not-equal-op]
    assert not (py_ok < pyo_ok)
    assert not (py_ok > pyo_ok)
    assert pyo_ok & py_ok == {1, 2, 3}
    assert pyo_ok | py_ok == {1, 2, 3}
    assert pyo_ok - py_ok == frozenset() == py_ok - pyo_ok
    assert pyo_ok ^ py_ok == frozenset() == py_ok ^ pyo_ok
    assert not pyo_ok.isdisjoint(py_ok)
    assert pyo_ok.isdisjoint(_PyOk([4, 5, 6]))


def test_set_from_iterable_note() -> None:
    class _NoIterableInit(ImplSized, ImplContainer, ImplIter): ...

    class _Bad(_NoIterableInit, PyoSet[int]): ...

    a = _Bad()
    b = _Bad()
    with pytest.raises(TypeError) as exc_info:
        _ = a | b
    assert exc_info.value.__notes__
    assert "PyoSet" in exc_info.value.__notes__[0]
