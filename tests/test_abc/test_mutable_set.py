from __future__ import annotations

from collections.abc import Callable, Iterable
from collections.abc import MutableSet as AbstractMutableSet

import pytest

from pyochain import SetMut
from pyochain.abc import PyoMutableSet
from pyochain.collections import SortedSet

from . import checks
from ._utils import ImplCollection, assert_iter_eq


class ImplMutableSet(ImplCollection):
    def __init__(self, it: Iterable[int] | None = None) -> None:
        self._data: set[int] = {1, 2, 3} if it is None else set(it)  # pyright: ignore[reportIncompatibleVariableOverride]

    def add(self, item: int) -> None:
        self._data.add(item)

    def discard(self, item: int) -> None:
        self._data.discard(item)


class _PyFail(AbstractMutableSet[int]): ...  # pyright: ignore[reportImplicitAbstractClass]


class _PyOk(ImplMutableSet, AbstractMutableSet[int]): ...


class _PyoFail(PyoMutableSet[int]): ...  # pyright: ignore[reportImplicitAbstractClass]


class _PyoOk(ImplMutableSet, PyoMutableSet[int]): ...


def test_mutable_set() -> None:
    checks.init_fail(_PyFail)
    fail = _PyoFail()  # pyright: ignore[reportAbstractUsage]
    checks.len_fail(fail)
    checks.contains_fail(fail)
    checks.iter_fail(fail)
    checks.add_fail(fail)
    checks.discard_fail(fail)
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
    assert pyo_ok.isdisjoint(_PyOk({4, 5, 6}))
    pyo_ok.add(4)
    py_ok.add(4)
    assert 4 in pyo_ok
    assert 4 in py_ok
    pyo_ok.remove(4)
    py_ok.remove(4)
    assert 4 not in pyo_ok
    assert 4 not in py_ok
    pyo_ok.discard(5)
    py_ok.discard(5)
    additional = {4, 5}
    pyo_ok |= additional
    py_ok |= additional
    assert pyo_ok == py_ok
    pyo_ok &= {1, 2}
    py_ok &= {1, 2}
    assert pyo_ok == py_ok
    pyo_ok ^= {2, 3}
    py_ok ^= {2, 3}
    assert pyo_ok == py_ok
    pyo_ok -= {1}
    py_ok -= {1}
    assert pyo_ok == py_ok


@pytest.mark.parametrize("cls", (_PyoOk, SetMut, SortedSet))
def test_clear(cls: Callable[[Iterable[int]], PyoMutableSet[int]]) -> None:
    data = (1, 2, 3)
    py = _PyOk(data)
    pyo = cls(data)
    py.clear()
    pyo.clear()
    assert py == pyo
    assert len(py) == 0
    assert pyo.len() == 0
